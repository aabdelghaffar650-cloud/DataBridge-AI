"""DataBridge AI Stage 20.4 replayable feature-derivation contract verification.

Run from project root:
    python tests/test_stage20_4_replayable_feature_derivations.py
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

fake_streamlit = types.ModuleType("streamlit")
fake_streamlit.session_state = {}
def _cache_data(*args, **kwargs):
    if args and callable(args[0]) and len(args) == 1 and not kwargs:
        return args[0]
    return lambda func: func
fake_streamlit.cache_data = _cache_data
sys.modules.setdefault("streamlit", fake_streamlit)

from core.feature_derivation import (
    FeatureDerivationError,
    append_derivation_step,
    build_formula_derivation_step,
    replay_feature_derivations,
    source_columns_for_pipeline,
)
from modules.data_mapper import analyze_dataframe
from modules.feature_pipeline import create_default_feature_pipeline_spec
from modules.ml_engine import run_supervised_experiment
from modules.model_package import (
    ModelPackageError,
    create_signed_model_package,
    load_signed_model_package,
    prepare_model_input_frame,
    run_batch_prediction,
    validate_prediction_frame,
)

SIGNING_KEY = b"R" * 32


def _raw_frame(rows: int = 260) -> pd.DataFrame:
    rng = np.random.default_rng(204)
    amount = rng.normal(480, 90, rows).round(2)
    units = rng.integers(1, 10, rows)
    segment = np.resize(["Retail", "SME", "Enterprise"], rows)
    score = amount + units * 20 + (segment == "Enterprise") * 55 + rng.normal(0, 30, rows)
    target = np.where(score > 645, "yes", "no")
    frame = pd.DataFrame(
        {
            "record_id": np.arange(5000, 5000 + rows),
            "amount": amount,
            "units": units,
            "segment": segment,
            "target": target,
        }
    )
    frame.loc[7, "amount"] = np.nan
    return frame


def _training_with_recipe():
    raw = _raw_frame()
    recipe = []
    step1 = build_formula_derivation_step(raw, "TotalValue", "amount + units * 20")
    recipe = append_derivation_step(recipe, step1)
    derived1 = replay_feature_derivations(raw, recipe)
    step2 = build_formula_derivation_step(derived1, "HighValue", "TotalValue > 620")
    recipe = append_derivation_step(recipe, step2)
    train = replay_feature_derivations(raw, recipe)
    return raw, train, recipe


def _train_model(train: pd.DataFrame):
    analysis = analyze_dataframe(train)
    spec = create_default_feature_pipeline_spec(
        train,
        analysis["profiles"],
        "target",
        task="classification",
        configured_revision=4,
        configured_fingerprint="stage20-4-fp",
    )
    # Ensure both replayed columns are active inputs regardless of heuristic defaults.
    selected = list(spec.get("feature_columns", []))
    for column in ["TotalValue", "HighValue"]:
        if column not in selected:
            selected.append(column)
    spec["feature_columns"] = selected
    groups = spec.setdefault("groups", {})
    for group_name, columns in list(groups.items()):
        groups[group_name] = [column for column in columns if column not in {"TotalValue", "HighValue"}]
    groups.setdefault("numeric", []).append("TotalValue")
    groups.setdefault("boolean", []).append("HighValue")

    result = run_supervised_experiment(
        train,
        spec,
        dataset_revision=4,
        dataset_fingerprint="stage20-4-fp",
        split_strategy="stratified",
        cv_folds=3,
        random_state=42,
        model_names=["Logistic Regression"],
        max_rows=None,
        include_xgboost=False,
    )
    return analysis, spec, result


def test_formula_recipe_supports_transitive_dependencies() -> None:
    raw, train, recipe = _training_with_recipe()
    assert [step["output_column"] for step in recipe] == ["TotalValue", "HighValue"]
    assert recipe[0]["inputs"] == ["amount", "units"]
    assert recipe[1]["inputs"] == ["TotalValue"]
    replayed = replay_feature_derivations(raw, recipe)
    pd.testing.assert_series_equal(replayed["TotalValue"], train["TotalValue"])
    pd.testing.assert_series_equal(replayed["HighValue"], train["HighValue"])
    source = source_columns_for_pipeline(recipe, ["segment", "TotalValue", "HighValue"])
    assert source == ["segment", "amount", "units"], source


def test_signed_package_accepts_raw_source_and_replays_derivations() -> None:
    raw, train, recipe = _training_with_recipe()
    analysis, spec, result = _train_model(train)
    build = create_signed_model_package(
        result,
        train,
        spec,
        semantic_profiles=analysis["profiles"],
        feature_derivation_recipe=recipe,
        signing_key=SIGNING_KEY,
    )
    manifest = build.manifest
    assert manifest["derivation_contract"]["available"] is True
    assert manifest["derivation_contract"]["step_count"] == 2
    assert len(manifest["derivation_contract"]["recipe_fingerprint"]) == 64

    # Fresh retraining/source data may not contain the derived columns yet; package
    # creation must be able to reconstruct the same signed contract from raw inputs.
    raw_build = create_signed_model_package(
        result,
        raw,
        spec,
        semantic_profiles=analysis["profiles"],
        feature_derivation_recipe=recipe,
        signing_key=SIGNING_KEY,
    )
    raw_package = load_signed_model_package(raw_build.package_bytes, signing_key=SIGNING_KEY)
    assert raw_package.required_columns == load_signed_model_package(build.package_bytes, signing_key=SIGNING_KEY).required_columns

    package = load_signed_model_package(build.package_bytes, signing_key=SIGNING_KEY)
    assert package.pipeline_required_columns
    assert "TotalValue" in package.pipeline_required_columns
    assert "HighValue" in package.pipeline_required_columns
    assert "TotalValue" not in package.required_columns
    assert "HighValue" not in package.required_columns
    assert len(package.feature_derivation_recipe) == 2

    scoring = raw.drop(columns=["target"]).head(40).copy(deep=True)
    before = scoring.copy(deep=True)
    schema = validate_prediction_frame(scoring, package, max_rows=100)
    assert schema["valid"] is True, schema
    assert "TotalValue" not in schema["missing_columns"]
    pred = run_batch_prediction(package, scoring, max_rows=100)
    assert len(pred.output) == len(scoring)
    pd.testing.assert_frame_equal(scoring, before)

    prepared = prepare_model_input_frame(package, scoring)
    assert "TotalValue" in prepared.columns and "HighValue" in prepared.columns

    # A caller cannot spoof the derived columns: signed replay overwrites them internally.
    spoofed = scoring.copy(deep=True)
    spoofed["TotalValue"] = 999999.0
    spoofed["HighValue"] = False
    pred_spoofed = run_batch_prediction(package, spoofed, max_rows=100)
    assert pred.output[pred.prediction_column].tolist() == pred_spoofed.output[pred_spoofed.prediction_column].tolist()


def test_missing_raw_dependency_blocks_and_stale_recipe_cannot_be_exported() -> None:
    raw, train, recipe = _training_with_recipe()
    analysis, spec, result = _train_model(train)
    build = create_signed_model_package(
        result,
        train,
        spec,
        semantic_profiles=analysis["profiles"],
        feature_derivation_recipe=recipe,
        signing_key=SIGNING_KEY,
    )
    package = load_signed_model_package(build.package_bytes, signing_key=SIGNING_KEY)
    missing = raw.drop(columns=["target", "amount"]).head(20)
    report = validate_prediction_frame(missing, package, max_rows=100)
    assert report["valid"] is False
    assert "amount" in report["missing_columns"]

    tampered_train = train.copy(deep=True)
    tampered_train["TotalValue"] = tampered_train["TotalValue"].fillna(0) + 1
    try:
        create_signed_model_package(
            result,
            tampered_train,
            spec,
            semantic_profiles=analysis["profiles"],
            feature_derivation_recipe=recipe,
            signing_key=SIGNING_KEY,
        )
        raise AssertionError("A training dataframe that diverges from its recorded recipe must be rejected.")
    except ModelPackageError as exc:
        assert "no longer matches" in str(exc).lower(), exc


def test_existing_column_redefinition_is_blocked() -> None:
    raw = _raw_frame()
    try:
        build_formula_derivation_step(raw, "amount", "amount + units")
        raise AssertionError("Replayable derivations must not overwrite existing source columns.")
    except FeatureDerivationError as exc:
        assert "already exists" in str(exc).lower()


def main() -> None:
    test_formula_recipe_supports_transitive_dependencies()
    test_signed_package_accepts_raw_source_and_replays_derivations()
    test_missing_raw_dependency_blocks_and_stale_recipe_cannot_be_exported()
    test_existing_column_redefinition_is_blocked()
    print(
        "PASS: Stage 20.4 records deterministic computed-feature recipes, signs the effective recipe with the model package, accepts raw source schemas at prediction time, replays transitive derivations automatically without mutating input data, ignores spoofed derived values, and blocks missing dependencies or training data that diverges from its recorded recipe."
    )


if __name__ == "__main__":
    main()
