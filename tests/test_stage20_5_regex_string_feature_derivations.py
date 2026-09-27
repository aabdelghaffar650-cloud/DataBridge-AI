"""DataBridge AI Stage 20.5 replayable regex/string derivation verification.

Run from project root:
    python tests/test_stage20_5_regex_string_feature_derivations.py
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
    build_regex_extract_derivation_step,
    build_value_map_derivation_step,
    replay_feature_derivations,
    source_columns_for_pipeline,
)
from modules.data_mapper import analyze_dataframe
from modules.feature_pipeline import create_default_feature_pipeline_spec
from modules.ml_engine import run_supervised_experiment
from modules.model_package import (
    create_signed_model_package,
    load_signed_model_package,
    prepare_model_input_frame,
    run_batch_prediction,
    validate_prediction_frame,
)

SIGNING_KEY = b"S" * 32
TITLE_PATTERN = r",\s*([^.]*)\."
TITLE_MAPPING = {
    "Mlle": "Miss",
    "Ms": "Miss",
    "Mme": "Mrs",
    "Dr": "Rare",
    "Rev": "Rare",
    "Col": "Rare",
    "Major": "Rare",
    "Don": "Rare",
    "Lady": "Rare",
    "Sir": "Rare",
    "Capt": "Rare",
    "Countess": "Rare",
    "Jonkheer": "Rare",
}


def _raw_titanic_like(rows: int = 360) -> pd.DataFrame:
    rng = np.random.default_rng(205)
    titles = np.resize(
        ["Mr", "Miss", "Mrs", "Master", "Dr", "Rev", "Mlle", "Mme", "Lady", "Jonkheer"],
        rows,
    )
    sex = np.where(np.isin(titles, ["Mrs", "Miss", "Mlle", "Mme", "Lady"]), "female", "male")
    age = np.clip(rng.normal(31, 13, rows), 1, 79).round(1)
    pclass = rng.integers(1, 4, rows)
    survived_score = (
        (sex == "female") * 2.0
        + np.isin(titles, ["Master", "Miss", "Mrs", "Mlle", "Mme", "Lady"]) * 0.8
        - (pclass - 1) * 0.45
        - (age > 58) * 0.5
        + rng.normal(0, 0.55, rows)
    )
    survived = (survived_score > 0.65).astype(int)
    names = [f"Surname{i}, {title}. Person {i}" for i, title in enumerate(titles)]
    return pd.DataFrame(
        {
            "PassengerId": np.arange(1, rows + 1),
            "Pclass": pclass,
            "Name": names,
            "Sex": sex,
            "Age": age,
            "Fare": np.maximum(5, rng.normal(40, 28, rows)).round(2),
            "Survived": survived,
        }
    )


def _derive_titles(raw: pd.DataFrame):
    recipe = []
    extract = build_regex_extract_derivation_step(
        raw,
        "Title",
        "Name",
        TITLE_PATTERN,
        capture_group=1,
        no_match_value="Unknown",
    )
    recipe = append_derivation_step(recipe, extract)
    with_title = replay_feature_derivations(raw, recipe)
    normalise = build_value_map_derivation_step(
        with_title,
        "Title_Normalized",
        "Title",
        TITLE_MAPPING,
        unmatched_strategy="preserve",
    )
    recipe = append_derivation_step(recipe, normalise)
    return replay_feature_derivations(raw, recipe), recipe


def _train_model(train: pd.DataFrame):
    analysis = analyze_dataframe(train)
    spec = create_default_feature_pipeline_spec(
        train,
        analysis["profiles"],
        "Survived",
        task="classification",
        configured_revision=5,
        configured_fingerprint="stage20-5-fp",
    )
    # Use normalized title and ordinary tabular inputs; exclude raw Name and extracted intermediate Title.
    selected = ["Pclass", "Sex", "Age", "Fare", "Title_Normalized"]
    spec["feature_columns"] = selected
    groups = spec.setdefault("groups", {})
    for group_name in list(groups):
        groups[group_name] = [c for c in groups[group_name] if c in selected]
    for c in selected:
        for group_name in list(groups):
            groups[group_name] = [x for x in groups[group_name] if x != c]
    groups.setdefault("numeric", []).extend(["Pclass", "Age", "Fare"])
    groups.setdefault("categorical", []).extend(["Sex", "Title_Normalized"])

    result = run_supervised_experiment(
        train,
        spec,
        dataset_revision=5,
        dataset_fingerprint="stage20-5-fp",
        split_strategy="stratified",
        cv_folds=3,
        random_state=42,
        model_names=["Logistic Regression"],
        max_rows=None,
        include_xgboost=False,
    )
    return analysis, spec, result


def test_titanic_title_extract_and_mapping_are_replayable() -> None:
    raw = _raw_titanic_like(40)
    train, recipe = _derive_titles(raw)
    assert train.loc[0, "Title"] == "Mr"
    assert train.loc[1, "Title"] == "Miss"
    assert train.loc[4, "Title_Normalized"] == "Rare"
    assert train.loc[6, "Title_Normalized"] == "Miss"
    assert train.loc[7, "Title_Normalized"] == "Mrs"
    assert train.loc[3, "Title_Normalized"] == "Master"
    assert source_columns_for_pipeline(recipe, ["Title_Normalized"]) == ["Name"]


def test_signed_package_replays_title_recipe_from_raw_name() -> None:
    raw = _raw_titanic_like()
    train, recipe = _derive_titles(raw)
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
    assert len(package.feature_derivation_recipe) == 2
    assert [step["kind"] for step in package.feature_derivation_recipe] == [
        "regex_extract",
        "value_map",
    ]
    assert "Name" in package.required_columns
    assert "Title" not in package.required_columns
    assert "Title_Normalized" not in package.required_columns

    scoring = raw.drop(columns=["Survived"]).head(55).copy(deep=True)
    before = scoring.copy(deep=True)
    report = validate_prediction_frame(scoring, package, max_rows=100)
    assert report["valid"] is True, report
    prepared = prepare_model_input_frame(package, scoring)
    assert prepared.loc[4, "Title_Normalized"] == "Rare"
    result1 = run_batch_prediction(package, scoring, max_rows=100)
    pd.testing.assert_frame_equal(scoring, before)

    spoofed = scoring.copy(deep=True)
    spoofed["Title"] = "Master"
    spoofed["Title_Normalized"] = "Master"
    result2 = run_batch_prediction(package, spoofed, max_rows=100)
    assert result1.output[result1.prediction_column].tolist() == result2.output[result2.prediction_column].tolist()


def test_regex_safety_and_mapping_validation_fail_closed() -> None:
    raw = _raw_titanic_like(20)
    unsafe = r"(.*)+"
    try:
        build_regex_extract_derivation_step(raw, "Unsafe", "Name", unsafe)
        raise AssertionError("Nested/unbounded regex quantifiers must be rejected.")
    except FeatureDerivationError as exc:
        assert "quantifier" in str(exc).lower(), exc

    try:
        build_regex_extract_derivation_step(raw, "NoCapture", "Name", r"Mr\.")
        raise AssertionError("Regex Extract without a capture group must be rejected.")
    except FeatureDerivationError as exc:
        assert "capture group" in str(exc).lower(), exc

    with_title, _ = _derive_titles(raw)
    try:
        build_value_map_derivation_step(
            with_title,
            "BadMap",
            "Title",
            {},
        )
        raise AssertionError("Empty Value Mapping must be rejected.")
    except FeatureDerivationError as exc:
        assert "at least one" in str(exc).lower(), exc


def test_ui_exposes_preview_and_replayable_string_feature_controls() -> None:
    page = (PROJECT_ROOT / "_pages" / "replace_values.py").read_text(encoding="utf-8")
    required = [
        "Text / Regex Feature",
        "Regex Extract",
        "Preview Regex Extraction",
        "Add Regex Extract Feature",
        "Value Mapping",
        "Preview Value Mapping",
        "Add Value-Mapped Feature",
        "replayable regex recipe",
        "replayable value-mapping recipe",
    ]
    for marker in required:
        assert marker in page, marker


def main() -> None:
    test_titanic_title_extract_and_mapping_are_replayable()
    test_signed_package_replays_title_recipe_from_raw_name()
    test_regex_safety_and_mapping_validation_fail_closed()
    test_ui_exposes_preview_and_replayable_string_feature_controls()
    print(
        "PASS: Stage 20.5 adds safe replayable Regex Extract and exact Value Mapping feature derivations, supports Titanic-style Title normalization as a transitive signed recipe, accepts raw Name-only prediction schemas, overwrites spoofed derived values internally, and blocks unsafe regex/mapping contracts."
    )


if __name__ == "__main__":
    main()
