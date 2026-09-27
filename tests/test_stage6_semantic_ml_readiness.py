"""DataBridge AI Stage 6 semantic typing and ML readiness verification.

Run from project root:
    python tests/test_stage6_semantic_ml_readiness.py
"""
from __future__ import annotations

import copy
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

from modules.data_mapper import analyze_dataframe, apply_manual_mappings, auto_map_columns
import core.dataset as dataset_module
import core.session as session_module


def _sample_dataframe(rows: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    dates = [f"2025-03-{(index % 28) + 1:02d}" for index in range(rows)]
    dates[-1] = "bad-date-kept-as-text"
    return pd.DataFrame(
        {
            "customer_id": np.arange(10000, 10000 + rows),
            "event_date": dates,
            "paid": np.resize(["yes", "no"], rows),
            "email_address": [f"user{index}@example.org" for index in range(rows)],
            "phone": [f"+20 (10) 555-{index:04d}" for index in range(rows)],
            "amount": rng.normal(250.0, 35.0, rows).round(2),
            "units": rng.integers(1, 7, rows),
            "priority": np.resize(["Low", "Medium", "High"], rows),
            "category": np.resize(["A", "B", "C", "D"], rows),
            "description": [
                f"Long operational description for record {index} with multiple words and review notes"
                for index in range(rows)
            ],
            "outcome": np.resize(["won", "lost"], rows),
            "candidate": [f"Person {index}" for index in range(rows)],
            "segment_code": [f"SEG-{index:04d}" for index in range(rows)],
            "segment_name": [f"Segment {index:04d}" for index in range(rows)],
            "prediction": rng.random(rows),
        }
    )


def test_content_aware_semantic_typing() -> None:
    df = _sample_dataframe()
    before = df.copy(deep=True)
    analysis = analyze_dataframe(df)
    mappings = analysis["mappings"]
    profiles = analysis["profiles"]

    assert mappings["customer_id"][0] == "Identifier"
    assert mappings["event_date"][0] == "Datetime"
    assert mappings["paid"][0] == "Boolean", "The token 'id' inside 'paid' must not cause an ID match."
    assert mappings["email_address"][0] == "Email"
    assert mappings["phone"][0] == "Phone"
    assert mappings["amount"][0] == "Currency"
    assert mappings["units"][0] == "Numeric Discrete"
    assert mappings["priority"][0] == "Ordinal"
    assert mappings["category"][0] == "Categorical"
    assert mappings["description"][0] == "Free Text"
    assert mappings["candidate"][0] != "Datetime", "A name containing 'date' as letters must not become a date."
    assert profiles["segment_name"]["high_cardinality"] is True
    assert profiles["prediction"]["leakage_risk"] is True
    assert profiles["event_date"]["signals"]["date_ratio"] > 0.95
    assert df.loc[len(df) - 1, "event_date"] == "bad-date-kept-as-text"
    pd.testing.assert_frame_equal(df, before)

    # Compatibility name-only mode is deliberately capped and cannot claim
    # high content confidence.
    name_only = auto_map_columns(["event_date", "paid"])
    assert name_only["event_date"][1] <= 0.65
    assert name_only["paid"][0] != "Identifier"


def test_ml_readiness_and_target_candidates() -> None:
    analysis = analyze_dataframe(_sample_dataframe())
    report = analysis["readiness"]

    assert report["status"] in {"Ready", "Needs Review"}
    assert report["training_blocked"] is False
    assert report["counts"]["usable_features"] >= 5
    assert report["counts"]["high_cardinality"] >= 1
    assert report["counts"]["leakage_risks"] >= 3
    assert report["target_candidates"], "At least one target candidate should be proposed."
    top = report["target_candidates"][0]
    assert top["column"] == "outcome"
    assert top["task"] == "classification"
    assert top["score"] >= 0.85
    assert "customer_id" in report["excluded_features"]
    assert "email_address" in report["excluded_features"]
    assert "phone" in report["excluded_features"]
    assert "event_date" in report["preprocessing_plan"]["datetime"]
    assert "description" in report["preprocessing_plan"]["text"]


def test_human_mapping_review_rebuilds_readiness() -> None:
    df = _sample_dataframe()
    analysis = analyze_dataframe(df)
    mappings = {column: semantic_type for column, (semantic_type, _) in analysis["mappings"].items()}
    mappings["segment_name"] = "Identifier"
    mappings["candidate"] = "Free Text"

    profiles, readiness = apply_manual_mappings(df, analysis["profiles"], mappings)
    assert profiles["segment_name"]["manual_override"] is True
    assert profiles["segment_name"]["effective_semantic_type"] == "Identifier"
    assert profiles["candidate"]["effective_semantic_type"] == "Free Text"
    assert "segment_name" in readiness["excluded_features"]
    assert "candidate" in readiness["preprocessing_plan"]["text"]
    assert readiness["mappings_reviewed"] is True


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
        {"source_type": "test", "source_name": "stage6"},
        display_name="stage6.csv",
    )


def test_unified_state_refresh_and_history_restore() -> None:
    _configure_state()
    session = fake_streamlit.session_state
    state = session.dataset_state

    assert state.semantic_profiles is session.semantic_profiles
    assert state.ml_readiness_report is session.ml_readiness_report
    assert set(state.semantic_profiles) == set(map(str, session.df.columns))
    initial_profiles = copy.deepcopy(state.semantic_profiles)
    initial_readiness = copy.deepcopy(state.ml_readiness_report)
    revision = state.revision

    result = dataset_module.apply_dataset_change(
        "Add constant test feature",
        lambda working: working.assign(constant_feature=1),
        expected_revision=revision,
    )
    assert result.changed is True
    assert "constant_feature" in session.semantic_profiles
    assert session.semantic_profiles["constant_feature"]["is_constant"] is True
    assert session.ml_readiness_report["counts"]["constant"] >= 1
    assert dataset_module.dataset_state_health(deep=True)["ok"] is True

    assert dataset_module.perform_undo() is True
    assert "constant_feature" not in session.df.columns
    assert set(session.semantic_profiles) == set(initial_profiles)
    assert session.ml_readiness_report == initial_readiness
    assert dataset_module.dataset_state_health(deep=True)["ok"] is True

    assert dataset_module.perform_redo() is True
    assert "constant_feature" in session.semantic_profiles
    assert session.semantic_profiles["constant_feature"]["is_constant"] is True


def main() -> None:
    test_content_aware_semantic_typing()
    test_ml_readiness_and_target_candidates()
    test_human_mapping_review_rebuilds_readiness()
    test_unified_state_refresh_and_history_restore()
    print(
        "PASS: Stage 6 content-aware semantic typing, target/readiness assessment, human overrides, unified-state refresh, and Undo/Redo restoration are functioning."
    )


if __name__ == "__main__":
    main()
