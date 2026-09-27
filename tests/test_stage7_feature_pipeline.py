"""DataBridge AI Stage 7 leakage-safe feature pipeline verification.

Run from project root:
    python tests/test_stage7_feature_pipeline.py
"""
from __future__ import annotations

import copy
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression

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
    build_model_pipeline,
    create_default_feature_pipeline_spec,
    feature_pipeline_spec_fingerprint,
    normalise_feature_pipeline_spec,
    safe_training_preview,
    validate_feature_pipeline_spec,
)
import core.dataset as dataset_module
import core.session as session_module


def _sample_dataframe(rows: int = 160) -> pd.DataFrame:
    rng = np.random.default_rng(20260806)
    frame = pd.DataFrame(
        {
            "customer_id": np.arange(50000, 50000 + rows),
            "amount": rng.normal(500.0, 80.0, rows).round(2),
            "units": rng.integers(1, 8, rows),
            "segment": np.resize(["Retail", "SME", "Enterprise", "Public"], rows),
            "priority": np.resize(["Low", "Medium", "High", "Medium"], rows),
            "paid": np.resize(["yes", "no"], rows),
            "event_date": [f"2025-{(index % 12) + 1:02d}-{(index % 27) + 1:02d}" for index in range(rows)],
            "description": [
                f"Operational service note number {index % 15} عربي data quality"
                for index in range(rows)
            ],
            "email": [f"person{index}@example.org" for index in range(rows)],
            "prediction": rng.random(rows),
            "outcome": np.resize(["won", "lost"], rows),
        }
    )
    frame.loc[5, "amount"] = np.nan
    frame.loc[9, "segment"] = None
    frame.loc[11, "event_date"] = "bad-date-kept-as-text"
    return frame


def _default_spec(df: pd.DataFrame) -> tuple[dict, dict]:
    analysis = analyze_dataframe(df)
    spec = create_default_feature_pipeline_spec(
        df,
        analysis["profiles"],
        "outcome",
        configured_revision=1,
        configured_fingerprint="dataset-v1",
    )
    return spec, analysis


def test_safe_defaults_and_no_source_mutation() -> None:
    df = _sample_dataframe()
    before = df.copy(deep=True)
    spec, analysis = _default_spec(df)

    assert spec["task"] == "classification"
    assert "customer_id" in spec["excluded_columns"]
    assert "email" in spec["excluded_columns"]
    assert "prediction" in spec["excluded_columns"]
    assert "amount" in spec["groups"]["numeric"]
    assert "segment" in spec["groups"]["categorical"]
    assert "priority" in spec["groups"]["ordinal"]
    assert "paid" in spec["groups"]["boolean"]
    assert "event_date" in spec["groups"]["datetime"]
    assert "description" not in spec["feature_columns"], "Free text must require explicit TF-IDF approval."

    report = validate_feature_pipeline_spec(
        df,
        spec,
        current_revision=1,
        current_fingerprint="dataset-v1",
    )
    assert report["valid"] is True
    assert report["status"] == "Configured"
    assert report["estimated_features"]["total"] > 0
    pd.testing.assert_frame_equal(df, before)
    assert analysis["profiles"]["event_date"]["signals"]["date_ratio"] > 0.95


def test_training_only_fit_and_unknown_categories() -> None:
    df = _sample_dataframe()
    before = df.copy(deep=True)
    spec, _ = _default_spec(df)
    spec["numeric"]["add_missing_indicator"] = False
    spec = normalise_feature_pipeline_spec(spec)

    preview = safe_training_preview(df, spec, random_state=17)
    fitted = preview["fitted"]
    assert preview["train_rows"] + preview["holdout_rows"] == len(df)
    assert preview["output_features"] == len(preview["feature_names"])
    assert preview["train_matrix_shape"][1] == preview["holdout_matrix_shape"][1]

    numeric_pipeline = fitted.preprocessor.named_transformers_["numeric"]
    scaler = numeric_pipeline.named_steps["scale"]
    train_amount_mean = df.loc[preview["train_indices"], "amount"].median()
    # Median imputation happens before scaling, so the scaler mean must be based
    # only on the training rows after train-only median fill.
    train_amount = df.loc[preview["train_indices"], "amount"].fillna(train_amount_mean)
    assert np.isclose(float(scaler.mean_[0]), float(train_amount.mean()))

    new_rows = df.head(3).copy(deep=True)
    new_rows["segment"] = "Never-Seen-Category"
    new_rows["priority"] = "Unknown-Priority"
    transformed = fitted.transform(new_rows.drop(columns=["outcome"]))
    assert transformed.shape == (3, preview["output_features"])
    assert sparse.issparse(transformed) or isinstance(transformed, np.ndarray)
    pd.testing.assert_frame_equal(df, before)


def test_datetime_ordinal_text_and_reusable_model_pipeline() -> None:
    df = _sample_dataframe()
    spec, _ = _default_spec(df)
    spec["groups"]["text"] = ["description"]
    spec["text"]["enabled"] = True
    spec["text"]["max_features"] = 120
    spec["text"]["ngram_min"] = 1
    spec["text"]["ngram_max"] = 2
    spec["ordinal"]["orders"] = {"priority": ["Low", "Medium", "High"]}
    spec = normalise_feature_pipeline_spec(spec)

    report = validate_feature_pipeline_spec(df, spec)
    assert report["valid"] is True

    incomplete_order = copy.deepcopy(spec)
    incomplete_order["ordinal"]["orders"] = {"priority": ["Low", "Medium"]}
    incomplete_report = validate_feature_pipeline_spec(df, incomplete_order)
    assert incomplete_report["valid"] is False
    assert any("does not include observed values" in item for item in incomplete_report["blockers"])

    preview = safe_training_preview(df, spec, random_state=42)
    names = preview["feature_names"]
    assert any("event_date_elapsed_days" in name for name in names)
    assert any("event_date_month_sin" in name for name in names)
    assert any("priority" in name for name in names)
    assert any("description" in name for name in names)

    modelling = df.loc[df["outcome"].notna()].copy()
    model_pipeline = build_model_pipeline(
        spec,
        LogisticRegression(max_iter=1000, random_state=42),
    )
    model_pipeline.fit(modelling[spec["feature_columns"]], modelling["outcome"])
    predictions = model_pipeline.predict(modelling[spec["feature_columns"]].head(8))
    assert len(predictions) == 8


def _configure_state() -> None:
    fake_streamlit.session_state.clear()
    session_module.st = fake_streamlit
    dataset_module.st = fake_streamlit
    dataset_module.run_quality_engine = lambda frame: {
        "quality_score": 100.0,
        "total_cells": int(frame.shape[0] * frame.shape[1]),
        "columns": tuple(frame.columns),
    }
    session_module.init_session_state()
    dataset_module.activate_dataset(
        _sample_dataframe(),
        {"source_type": "test", "source_name": "stage7"},
        display_name="stage7.csv",
    )


def test_unified_state_pipeline_context_and_staleness() -> None:
    _configure_state()
    session = fake_streamlit.session_state
    state = session.dataset_state
    spec = create_default_feature_pipeline_spec(
        session.df,
        session.semantic_profiles,
        "outcome",
        configured_revision=state.revision,
        configured_fingerprint=state.working_fingerprint,
    )
    report = validate_feature_pipeline_spec(
        session.df,
        spec,
        current_revision=state.revision,
        current_fingerprint=state.working_fingerprint,
    )
    dataset_module.update_dataset_context(
        expected_revision=state.revision,
        feature_pipeline_spec=spec,
        feature_pipeline_report=report,
    )

    assert session.feature_pipeline_spec is session.dataset_state.feature_pipeline_spec
    assert session.feature_pipeline_report is session.dataset_state.feature_pipeline_report
    assert session.feature_pipeline_report["status"] == "Configured"
    original_spec_fp = feature_pipeline_spec_fingerprint(session.feature_pipeline_spec)
    original_dataset_fp = session.working_fingerprint

    result = dataset_module.apply_dataset_change(
        "Shift amount for pipeline staleness test",
        lambda working: working.assign(amount=working["amount"] + 10),
        expected_revision=session.dataset_revision,
    )
    assert result.changed is True
    assert session.feature_pipeline_report["valid"] is True
    assert session.feature_pipeline_report["status"] == "Stale"
    assert session.feature_pipeline_report["stale"] is True

    assert dataset_module.perform_undo() is True
    assert session.working_fingerprint == original_dataset_fp
    assert feature_pipeline_spec_fingerprint(session.feature_pipeline_spec) == original_spec_fp
    assert session.feature_pipeline_report["status"] == "Configured"
    assert session.feature_pipeline_report["stale"] is False
    assert dataset_module.dataset_state_health(deep=True)["ok"] is True

    # Removing a configured feature keeps the specification for auditability but
    # marks it invalid instead of silently changing the modelling contract.
    selected = session.feature_pipeline_spec["feature_columns"][0]
    dataset_module.apply_dataset_change(
        "Drop configured feature for validation test",
        lambda working: working.drop(columns=[selected]),
        expected_revision=session.dataset_revision,
    )
    assert session.feature_pipeline_report["status"] == "Invalid"
    assert session.feature_pipeline_report["valid"] is False
    assert any("missing" in message.lower() for message in session.feature_pipeline_report["blockers"])


def main() -> None:
    test_safe_defaults_and_no_source_mutation()
    test_training_only_fit_and_unknown_categories()
    test_datetime_ordinal_text_and_reusable_model_pipeline()
    test_unified_state_pipeline_context_and_staleness()
    print(
        "PASS: Stage 7 leakage-safe feature specification, train-only fitting, unknown-category handling, datetime/ordinal/text features, reusable sklearn pipelines, and unified-state staleness protection are functioning."
    )


if __name__ == "__main__":
    main()
