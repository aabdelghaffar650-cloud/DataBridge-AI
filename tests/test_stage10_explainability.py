"""DataBridge AI Stage 10 Explainability verification.

Run from project root:
    python tests/test_stage10_explainability.py
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


class AttrDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


fake_streamlit = types.ModuleType("streamlit")
fake_streamlit.session_state = AttrDict()


def _cache_data(*args, **kwargs):
    if args and callable(args[0]) and len(args) == 1 and not kwargs:
        return args[0]
    return lambda func: func


fake_streamlit.cache_data = _cache_data
sys.modules.setdefault("streamlit", fake_streamlit)

from modules.data_mapper import analyze_dataframe
from modules.feature_pipeline import (
    create_default_feature_pipeline_spec,
    feature_pipeline_spec_fingerprint,
    validate_feature_pipeline_spec,
)
from modules.ml_engine import (
    SPLIT_STRATIFIED,
    run_supervised_experiment,
)
from modules.model_explainability import (
    ExplainabilityError,
    build_error_analysis,
    compute_segment_performance,
    reconcile_explainability_report,
    run_global_explainability,
    run_local_sensitivity,
)


def _classification_frame(rows: int = 360) -> pd.DataFrame:
    rng = np.random.default_rng(20260810)
    amount = rng.normal(500, 100, rows)
    units = rng.integers(1, 10, rows)
    noise = rng.normal(0, 1, rows)
    segment = np.resize(["Retail", "SME", "Enterprise", "Public"], rows)
    signal = amount * 0.07 + units * 4.0 + (segment == "Enterprise") * 12.0
    outcome = np.where(signal + rng.normal(0, 4, rows) > 68, "won", "lost")
    frame = pd.DataFrame(
        {
            "amount": amount,
            "units": units,
            "noise": noise,
            "segment": segment,
            "outcome": outcome,
        }
    )
    frame.loc[5, "amount"] = np.nan
    frame.loc[11, "segment"] = None
    return frame


def _regression_frame(rows: int = 320) -> pd.DataFrame:
    rng = np.random.default_rng(20260811)
    x1 = rng.normal(0, 1, rows)
    x2 = rng.normal(5, 2, rows)
    noise = rng.normal(0, 1, rows)
    category = np.resize(["A", "B", "C"], rows)
    target = 10 + 7.0 * x1 - 3.0 * x2 + (category == "C") * 4 + rng.normal(0, 0.35, rows)
    frame = pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "noise": noise,
            "category": category,
            "target_value": target,
        }
    )
    frame.loc[8, "x2"] = np.nan
    return frame


def _spec(df: pd.DataFrame, target: str, fingerprint: str, task: str) -> dict:
    analysis = analyze_dataframe(df)
    spec = create_default_feature_pipeline_spec(
        df,
        analysis["profiles"],
        target,
        task=task,
        configured_revision=1,
        configured_fingerprint=fingerprint,
    )
    report = validate_feature_pipeline_spec(
        df,
        spec,
        current_revision=1,
        current_fingerprint=fingerprint,
    )
    assert report["valid"] is True, report
    return spec


def test_classification_global_and_local_explanations_are_holdout_only() -> None:
    df = _classification_frame()
    before = df.copy(deep=True)
    fingerprint = "stage10-classification-fingerprint"
    spec = _spec(df, "outcome", fingerprint, "classification")
    result = run_supervised_experiment(
        df,
        spec,
        dataset_revision=1,
        dataset_fingerprint=fingerprint,
        split_strategy=SPLIT_STRATIFIED,
        holdout_size=0.25,
        cv_folds=3,
        random_state=31,
        model_names=["Logistic Regression"],
        max_rows=None,
        include_xgboost=False,
    )

    global_result = run_global_explainability(
        df,
        result,
        spec,
        current_revision=1,
        current_fingerprint=fingerprint,
        max_rows=120,
        n_repeats=3,
        random_state=31,
    )
    assert global_result.experiment_id == result.experiment_id
    assert global_result.rows_used <= result.holdout_rows
    assert global_result.rows_used <= 120
    assert set(global_result.source_importance["Feature"]) == set(spec["feature_columns"])
    assert np.isfinite(global_result.source_importance["Importance Mean"]).all()
    assert global_result.source_importance.iloc[0]["Feature"] in {"amount", "units", "segment"}
    assert not global_result.native_importance.empty
    assert set(global_result.native_importance["Source Feature"]) - {"Derived / Unmapped"}
    report = global_result.report()
    assert report["status"] == "Completed"
    assert report["artifact_available"] is True

    source_index = result.predictions.index[0]
    local = run_local_sensitivity(
        df,
        result,
        spec,
        source_index=source_index,
        current_revision=1,
        current_fingerprint=fingerprint,
        background_rows=80,
        random_state=31,
    )
    assert local.experiment_id == result.experiment_id
    assert local.source_index == source_index
    assert set(local.sensitivity["Feature"]) == set(spec["feature_columns"])
    assert np.isfinite(local.sensitivity["Sensitivity"]).all()
    assert local.score_name in {
        "Predicted-class probability",
        "Predicted-class decision score",
        "Predicted-class match indicator",
    }
    pd.testing.assert_frame_equal(df, before)


def test_regression_explanation_error_and_segment_analysis() -> None:
    df = _regression_frame()
    before = df.copy(deep=True)
    fingerprint = "stage10-regression-fingerprint"
    spec = _spec(df, "target_value", fingerprint, "regression")
    result = run_supervised_experiment(
        df,
        spec,
        dataset_revision=1,
        dataset_fingerprint=fingerprint,
        split_strategy="random",
        holdout_size=0.20,
        cv_folds=3,
        random_state=19,
        model_names=["Ridge Regression"],
        max_rows=None,
        include_xgboost=False,
    )

    global_result = run_global_explainability(
        df,
        result,
        spec,
        current_revision=1,
        current_fingerprint=fingerprint,
        max_rows=100,
        n_repeats=3,
        random_state=19,
    )
    assert global_result.permutation_metric == "RMSE increase"
    assert global_result.source_importance.iloc[0]["Feature"] in {"x1", "x2"}

    source_index = result.predictions.index[-1]
    local = run_local_sensitivity(
        df,
        result,
        spec,
        source_index=source_index,
        current_revision=1,
        current_fingerprint=fingerprint,
        background_rows=60,
        random_state=19,
    )
    assert local.score_name == "Predicted value"
    assert np.isclose(local.predicted_value, result.predictions.loc[source_index, "Predicted"])

    errors = build_error_analysis(result, limit=12)
    assert len(errors) == 12
    assert errors["Absolute Error"].is_monotonic_decreasing
    segment = compute_segment_performance(
        df,
        result,
        segment_column="category",
        min_group_rows=2,
    )
    assert not segment.empty
    assert {"Segment", "Rows", "MAE", "RMSE", "R²"}.issubset(segment.columns)
    assert int(segment["Rows"].sum()) == result.holdout_rows
    pd.testing.assert_frame_equal(df, before)


def test_staleness_and_safety_guards() -> None:
    df = _classification_frame(220)
    fingerprint = "stage10-stale-fingerprint"
    spec = _spec(df, "outcome", fingerprint, "classification")
    result = run_supervised_experiment(
        df,
        spec,
        dataset_revision=1,
        dataset_fingerprint=fingerprint,
        split_strategy=SPLIT_STRATIFIED,
        cv_folds=3,
        random_state=7,
        model_names=["Logistic Regression"],
        max_rows=None,
        include_xgboost=False,
    )
    try:
        run_global_explainability(
            df,
            result,
            spec,
            current_revision=2,
            current_fingerprint=fingerprint,
        )
    except ExplainabilityError as exc:
        assert "older dataset revision" in str(exc)
    else:
        raise AssertionError("Stale dataset revision was not blocked")

    current = {
        "status": "Completed",
        "dataset_revision": 1,
        "dataset_fingerprint": fingerprint,
        "experiment_id": result.experiment_id,
        "pipeline_spec_fingerprint": feature_pipeline_spec_fingerprint(spec),
    }
    valid = reconcile_explainability_report(
        current,
        current_revision=1,
        current_fingerprint=fingerprint,
        current_experiment_id=result.experiment_id,
        current_pipeline_spec_fingerprint=feature_pipeline_spec_fingerprint(spec),
        artifact_available=True,
    )
    assert valid["status"] == "Completed"
    stale = reconcile_explainability_report(
        current,
        current_revision=2,
        current_fingerprint="changed",
        current_experiment_id="new-experiment",
        current_pipeline_spec_fingerprint="changed-pipeline",
        artifact_available=False,
    )
    assert stale["status"] == "Stale"
    assert stale["stale"] is True
    assert len(stale["stale_reasons"]) >= 4



def test_unified_state_invalidates_explanations_after_dataset_change() -> None:
    import importlib

    dataset_module = importlib.import_module("core.dataset")
    session_module = importlib.import_module("core.session")
    dataset_module.st = fake_streamlit
    session_module.st = fake_streamlit
    dataset_module.run_quality_engine = lambda frame: {
        "quality_score": 100.0,
        "total_cells": int(frame.shape[0] * frame.shape[1]),
    }

    fake_streamlit.session_state.clear()
    session_module.init_session_state()
    df = _classification_frame(240)
    dataset_module.activate_dataset(
        df,
        {"source_type": "test", "source_name": "stage10"},
        display_name="stage10.csv",
    )
    state = fake_streamlit.session_state.dataset_state
    fingerprint = state.working_fingerprint
    spec = _spec(fake_streamlit.session_state.df, "outcome", fingerprint, "classification")
    # The helper fixes revision=1, which is the first activated dataset revision.
    pipeline_report = validate_feature_pipeline_spec(
        fake_streamlit.session_state.df,
        spec,
        current_revision=state.revision,
        current_fingerprint=fingerprint,
    )
    dataset_module.update_dataset_context(
        expected_revision=state.revision,
        feature_pipeline_spec=spec,
        feature_pipeline_report=pipeline_report,
    )
    result = run_supervised_experiment(
        fake_streamlit.session_state.df,
        spec,
        dataset_revision=state.revision,
        dataset_fingerprint=fingerprint,
        split_strategy=SPLIT_STRATIFIED,
        cv_folds=3,
        random_state=13,
        model_names=["Logistic Regression"],
        max_rows=None,
        include_xgboost=False,
    )
    explanation = run_global_explainability(
        fake_streamlit.session_state.df,
        result,
        spec,
        current_revision=state.revision,
        current_fingerprint=fingerprint,
        max_rows=100,
        n_repeats=2,
        random_state=13,
    )
    fake_streamlit.session_state.ml_experiment_artifact = result
    fake_streamlit.session_state.model_explainability_artifact = explanation
    fake_streamlit.session_state.local_explanation_result = object()
    dataset_module.update_dataset_context(
        expected_revision=state.revision,
        ml_experiment_report=result.report(),
        explainability_report=explanation.report(),
    )

    dataset_module.apply_dataset_change(
        "Modify one value",
        lambda frame: frame.assign(amount=frame["amount"].fillna(0) + 1.0),
        expected_revision=state.revision,
    )
    assert fake_streamlit.session_state.ml_experiment_artifact is None
    assert fake_streamlit.session_state.model_explainability_artifact is None
    assert fake_streamlit.session_state.local_explanation_result is None
    assert fake_streamlit.session_state.explainability_report["status"] == "Stale"
    assert dataset_module.dataset_state_health(deep=True)["ok"] is True

def main() -> None:
    test_classification_global_and_local_explanations_are_holdout_only()
    test_regression_explanation_error_and_segment_analysis()
    test_staleness_and_safety_guards()
    test_unified_state_invalidates_explanations_after_dataset_change()
    print(
        "PASS: Stage 10 Explainability Studio uses untouched holdout rows for model-agnostic source permutation importance, maps native transformed importance safely, provides non-causal local sensitivity, supports error/segment analysis, preserves source data, and invalidates stale explanations."
    )


if __name__ == "__main__":
    main()
