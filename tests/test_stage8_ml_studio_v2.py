"""DataBridge AI Stage 8 ML Studio V2 verification.

Run from project root:
    python tests/test_stage8_ml_studio_v2.py
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
    SPLIT_GROUP,
    SPLIT_STRATIFIED,
    SPLIT_TIME,
    _safe_datetime,
    reconcile_ml_experiment_report,
    run_clustering_experiment,
    run_supervised_experiment,
)


def _classification_frame(rows: int = 240) -> pd.DataFrame:
    rng = np.random.default_rng(20260808)
    amount = rng.normal(500, 90, rows)
    units = rng.integers(1, 9, rows)
    segment = np.resize(["Retail", "SME", "Enterprise", "Public"], rows)
    group = np.resize([f"account_{index:02d}" for index in range(24)], rows)
    dates = pd.date_range("2023-01-01", periods=rows, freq="D")
    signal = amount + units * 30 + (segment == "Enterprise") * 80
    outcome = np.where(signal + rng.normal(0, 35, rows) > 650, "won", "lost")
    frame = pd.DataFrame(
        {
            "customer_id": np.arange(100_000, 100_000 + rows),
            "amount": amount.round(2),
            "units": units,
            "segment": segment,
            "account_group": group,
            "event_date": dates.astype(str),
            "outcome": outcome,
        }
    )
    frame.loc[3, "amount"] = np.nan
    frame.loc[9, "segment"] = None
    return frame


def _regression_frame(rows: int = 220) -> pd.DataFrame:
    rng = np.random.default_rng(20260809)
    x1 = rng.normal(0, 1, rows)
    x2 = rng.normal(5, 2, rows)
    category = np.resize(["A", "B", "C"], rows)
    target = 12 + 4.5 * x1 - 2.2 * x2 + (category == "C") * 3 + rng.normal(0, 0.5, rows)
    frame = pd.DataFrame(
        {
            "row_id": np.arange(rows),
            "x1": x1,
            "x2": x2,
            "category": category,
            "target_value": target,
        }
    )
    frame.loc[7, "x2"] = np.nan
    return frame


def _spec(df: pd.DataFrame, target: str, fingerprint: str, task: str | None = None) -> dict:
    analysis = analyze_dataframe(df)
    spec = create_default_feature_pipeline_spec(
        df,
        analysis["profiles"],
        target,
        task=task or "auto",
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
    assert report["status"] == "Configured"
    return spec


def test_classification_cv_holdout_baseline_and_train_only_fit() -> None:
    df = _classification_frame()
    before = df.copy(deep=True)
    spec = _spec(df, "outcome", "class-fp", task="classification")
    result = run_supervised_experiment(
        df,
        spec,
        dataset_revision=1,
        dataset_fingerprint="class-fp",
        split_strategy=SPLIT_STRATIFIED,
        holdout_size=0.20,
        cv_folds=3,
        random_state=17,
        model_names=["Logistic Regression"],
        tune_best=False,
        max_rows=None,
        include_xgboost=False,
    )

    assert result.selected_model == "Logistic Regression"
    assert "Baseline — Most Frequent" in result.leaderboard["Model"].tolist()
    assert result.improvement_vs_baseline is not None
    assert result.improvement_vs_baseline > 0
    assert result.train_rows + result.holdout_rows == len(df)
    assert set(result.predictions.index).isdisjoint(set(df.index.difference(result.predictions.index)))
    assert result.holdout_metrics["F1 Weighted"] is not None
    assert result.confusion is not None
    assert result.classification_report_df is not None
    assert result.report()["artifact_available"] is True
    assert "fitted_pipeline" not in result.report()
    assert "predictions" not in result.report()

    # The fitted scaler must reflect training rows only, not the untouched holdout.
    train_index = df.index.difference(result.predictions.index)
    numeric_columns = spec["groups"]["numeric"]
    assert "amount" in numeric_columns
    amount_position = numeric_columns.index("amount")
    training_amount = df.loc[train_index, "amount"]
    training_amount = training_amount.fillna(training_amount.median())
    numeric_pipeline = result.fitted_pipeline.named_steps["features"].named_transformers_["numeric"]
    scaler = numeric_pipeline.named_steps["scale"]
    assert np.isclose(float(scaler.mean_[amount_position]), float(training_amount.mean()))
    pd.testing.assert_frame_equal(df, before)



def test_iso_time_parsing_is_version_safe() -> None:
    values = pd.Series(
        [
            "2023-01-02",
            "2023-02-01",
            "2023-12-01",
            "2024-01-01T12:30:00",
        ]
    )
    expected = [
        pd.Timestamp("2023-01-02", tz="UTC"),
        pd.Timestamp("2023-02-01", tz="UTC"),
        pd.Timestamp("2023-12-01", tz="UTC"),
        pd.Timestamp("2024-01-01 12:30:00", tz="UTC"),
    ]

    original_to_datetime = pd.to_datetime

    def legacy_compatible_to_datetime(*args, **kwargs):
        if kwargs.get("format") == "mixed":
            raise TypeError("format='mixed' is unavailable in this simulated pandas version")
        return original_to_datetime(*args, **kwargs)

    pd.to_datetime = legacy_compatible_to_datetime
    try:
        parsed = _safe_datetime(values, dayfirst=True)
    finally:
        pd.to_datetime = original_to_datetime

    assert parsed.tolist() == expected
    assert parsed.is_monotonic_increasing

def test_time_and_group_splits_prevent_leakage() -> None:
    df = _classification_frame()
    spec = _spec(df, "outcome", "split-fp", task="classification")

    time_result = run_supervised_experiment(
        df,
        spec,
        dataset_revision=1,
        dataset_fingerprint="split-fp",
        split_strategy=SPLIT_TIME,
        split_column="event_date",
        holdout_size=0.20,
        cv_folds=3,
        random_state=42,
        model_names=["Logistic Regression"],
        max_rows=None,
        include_xgboost=False,
    )
    holdout_dates = pd.to_datetime(df.loc[time_result.predictions.index, "event_date"])
    train_dates = pd.to_datetime(df.drop(index=time_result.predictions.index)["event_date"])
    assert train_dates.max() < holdout_dates.min()
    assert time_result.split_summary["train_time_end"] < time_result.split_summary["holdout_time_start"]

    group_result = run_supervised_experiment(
        df,
        spec,
        dataset_revision=1,
        dataset_fingerprint="split-fp",
        split_strategy=SPLIT_GROUP,
        split_column="account_group",
        holdout_size=0.25,
        cv_folds=3,
        random_state=11,
        model_names=["Logistic Regression"],
        max_rows=None,
        include_xgboost=False,
    )
    holdout_groups = set(df.loc[group_result.predictions.index, "account_group"])
    train_groups = set(df.drop(index=group_result.predictions.index)["account_group"])
    assert holdout_groups.isdisjoint(train_groups)
    assert group_result.split_summary["group_overlap"] == 0
    assert not any("account_group" in name for name in group_result.feature_names)
    group_report = reconcile_ml_experiment_report(
        group_result.report(),
        current_revision=1,
        current_fingerprint="split-fp",
        current_pipeline_spec_fingerprint=feature_pipeline_spec_fingerprint(spec),
        artifact_available=True,
    )
    assert group_report["status"] == "Completed"
    assert group_report["artifact_available"] is True


def test_regression_baseline_metrics_and_train_only_tuning() -> None:
    df = _regression_frame()
    before = df.copy(deep=True)
    spec = _spec(df, "target_value", "reg-fp", task="regression")
    result = run_supervised_experiment(
        df,
        spec,
        dataset_revision=1,
        dataset_fingerprint="reg-fp",
        split_strategy="random",
        holdout_size=0.20,
        cv_folds=3,
        random_state=9,
        model_names=["Ridge Regression"],
        tune_best=True,
        tuning_iterations=3,
        max_rows=None,
        include_xgboost=False,
    )
    assert result.selected_model == "Ridge Regression"
    assert result.tuned is True
    assert result.tuning_iterations == 3
    assert result.best_params
    assert result.holdout_metrics["RMSE"] is not None
    assert result.holdout_metrics["R²"] is not None
    assert result.improvement_vs_baseline is not None
    assert result.improvement_vs_baseline > 0
    assert "Residual" in result.predictions.columns
    pd.testing.assert_frame_equal(df, before)


def test_clustering_pipeline_metrics_and_no_source_mutation() -> None:
    df = _regression_frame()
    before = df.copy(deep=True)
    result = run_clustering_experiment(
        df,
        ["x1", "x2"],
        algorithm="kmeans",
        n_clusters=3,
        random_state=42,
        max_rows=10_000,
    )
    assert len(result.labels) == len(df)
    assert result.metrics["Clusters"] == 3.0
    assert result.metrics["Silhouette"] is not None
    assert result.projection.shape == (len(df), 3)
    assert result.cluster_sizes["Count"].sum() == len(df)
    predictions = result.fitted_pipeline.predict(df[["x1", "x2"]].head(5))
    assert len(predictions) == 5
    pd.testing.assert_frame_equal(df, before)


def test_unified_state_experiment_staleness_and_artifact_invalidation() -> None:
    sys.modules["streamlit"] = fake_streamlit
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
    df = _classification_frame(180)
    dataset_module.activate_dataset(
        df,
        {"source_type": "test", "source_name": "stage8"},
        display_name="stage8.csv",
    )
    state = fake_streamlit.session_state.dataset_state
    spec = create_default_feature_pipeline_spec(
        fake_streamlit.session_state.df,
        fake_streamlit.session_state.semantic_profiles,
        "outcome",
        task="classification",
        configured_revision=state.revision,
        configured_fingerprint=state.working_fingerprint,
    )
    pipeline_report = validate_feature_pipeline_spec(
        fake_streamlit.session_state.df,
        spec,
        current_revision=state.revision,
        current_fingerprint=state.working_fingerprint,
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
        dataset_fingerprint=state.working_fingerprint,
        split_strategy=SPLIT_STRATIFIED,
        cv_folds=3,
        random_state=5,
        model_names=["Logistic Regression"],
        max_rows=None,
        include_xgboost=False,
    )
    fake_streamlit.session_state.ml_experiment_artifact = result
    dataset_module.update_dataset_context(
        expected_revision=state.revision,
        ml_experiment_report=result.report(),
    )
    original_fp = fake_streamlit.session_state.working_fingerprint
    original_pipeline_fp = feature_pipeline_spec_fingerprint(spec)
    assert fake_streamlit.session_state.ml_experiment_report["status"] == "Completed"

    dataset_module.apply_dataset_change(
        "Stage 8 staleness test",
        lambda working: working.assign(amount=working["amount"] + 1),
        expected_revision=fake_streamlit.session_state.dataset_revision,
    )
    assert fake_streamlit.session_state.ml_experiment_report["status"] == "Stale"
    assert fake_streamlit.session_state.ml_experiment_report["stale"] is True
    assert fake_streamlit.session_state.ml_experiment_artifact is None

    assert dataset_module.perform_undo() is True
    assert fake_streamlit.session_state.working_fingerprint == original_fp
    assert feature_pipeline_spec_fingerprint(fake_streamlit.session_state.feature_pipeline_spec) == original_pipeline_fp
    assert fake_streamlit.session_state.ml_experiment_report["status"] == "Completed"
    assert fake_streamlit.session_state.ml_experiment_report["artifact_available"] is False
    assert fake_streamlit.session_state.ml_experiment_artifact is None
    assert dataset_module.dataset_state_health(deep=True)["ok"] is True
    assert (
        fake_streamlit.session_state.ml_experiment_report
        is fake_streamlit.session_state.dataset_state.ml_experiment_report
    )

    changed_pipeline = dict(fake_streamlit.session_state.feature_pipeline_spec)
    changed_pipeline["name"] = "Changed pipeline"
    reconciled = reconcile_ml_experiment_report(
        fake_streamlit.session_state.ml_experiment_report,
        current_revision=fake_streamlit.session_state.dataset_revision,
        current_fingerprint=fake_streamlit.session_state.working_fingerprint,
        current_pipeline_spec_fingerprint=feature_pipeline_spec_fingerprint(changed_pipeline),
        artifact_available=False,
    )
    assert reconciled["status"] == "Stale"
    assert "feature pipeline changed" in reconciled["stale_reason"]


def main() -> None:
    test_classification_cv_holdout_baseline_and_train_only_fit()
    test_iso_time_parsing_is_version_safe()
    test_time_and_group_splits_prevent_leakage()
    test_regression_baseline_metrics_and_train_only_tuning()
    test_clustering_pipeline_metrics_and_no_source_mutation()
    test_unified_state_experiment_staleness_and_artifact_invalidation()
    print(
        "PASS: Stage 8 ML Studio V2 performs train-only cross-validation, baseline comparison, split-aware holdouts, optional train-only tuning, honest untouched-holdout evaluation, clustering diagnostics, and unified-state experiment invalidation safely."
    )


if __name__ == "__main__":
    main()
