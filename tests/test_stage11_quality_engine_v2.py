"""DataBridge AI Stage 11 Quality Engine V2 verification.

Run from project root:
    python tests/test_stage11_quality_engine_v2.py
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

from modules.quality_engine import (  # noqa: E402
    compare_quality_reports,
    normalise_quality_policy,
    run_quality_engine,
)
import core.dataset as dataset_module  # noqa: E402
import core.session as session_module  # noqa: E402


def _profiles() -> dict:
    return {
        "record_id": {
            "effective_semantic_type": "Identifier",
            "semantic_type": "Identifier",
            "business_role": "ID / Identifier",
            "target_candidate_score": 0.0,
        },
        "amount": {
            "effective_semantic_type": "Numeric Continuous",
            "semantic_type": "Numeric Continuous",
            "business_role": "Value / Amount",
            "target_candidate_score": 0.0,
        },
        "event_date": {
            "effective_semantic_type": "Datetime",
            "semantic_type": "Datetime",
            "business_role": "Date",
            "target_candidate_score": 0.0,
        },
        "outcome": {
            "effective_semantic_type": "Categorical",
            "semantic_type": "Categorical",
            "business_role": "Score / Result",
            "target_candidate_score": 0.92,
        },
    }


def test_future_dates_and_invalid_values_reduce_score_without_double_counting() -> None:
    frame = pd.DataFrame(
        {
            "record_id": [1, 2, 3, 4],
            "amount": ["10", "bad", "inf", "40"],
            "event_date": ["2024-01-01", "not-a-date", "2099-01-01", "2024-04-01"],
            "outcome": ["yes", "no", "yes", None],
        }
    )
    before = frame.copy(deep=True)
    report = run_quality_engine(
        frame,
        policy={
            "auto_critical_columns": True,
            "critical_multiplier": 2.0,
        },
        semantic_profiles=_profiles(),
        dataset_revision=7,
        dataset_fingerprint="stage11-fingerprint",
    )

    assert report["engine_version"] == "11.0"
    assert report["quality_score"] < 100.0
    assert report["date_errors"]["event_date"]["invalid_dates"] == 1
    assert report["date_errors"]["event_date"]["future_dates"] == 1
    assert report["type_errors"]["amount"] == 1
    assert report["non_finite_by_col"]["amount"] == 1
    assert report["unique_invalid_cells"] == 4
    assert report["raw_validity_events"] == 4
    assert report["overlap_avoided"] == 0
    assert report["unique_defect_cells"] == 5
    assert "record_id" in report["critical_columns"]
    assert "event_date" in report["critical_columns"]
    assert "outcome" in report["critical_columns"]
    assert report["dimension_scores"]["validity"] < 100.0
    pd.testing.assert_frame_equal(frame, before)


def test_critical_weighting_and_future_date_policy_are_effective() -> None:
    frame = pd.DataFrame(
        {
            "feature": [1.0, np.nan, 3.0, 4.0],
            "target": [1.0, np.nan, 3.0, 4.0],
            "forecast_date": ["2024-01-01", "2024-02-01", "2099-01-01", "2024-04-01"],
        }
    )
    profiles = {
        "feature": {"effective_semantic_type": "Numeric Continuous"},
        "target": {
            "effective_semantic_type": "Numeric Continuous",
            "target_candidate_score": 0.9,
        },
        "forecast_date": {
            "effective_semantic_type": "Datetime",
            "business_role": "Date",
        },
    }
    unweighted = run_quality_engine(
        frame,
        policy={
            "auto_critical_columns": False,
            "critical_columns": [],
            "critical_multiplier": 1.0,
        },
        semantic_profiles=profiles,
    )
    critical = run_quality_engine(
        frame,
        policy={
            "auto_critical_columns": False,
            "critical_columns": ["target"],
            "critical_multiplier": 4.0,
        },
        semantic_profiles=profiles,
    )
    assert critical["dimension_scores"]["completeness"] < unweighted["dimension_scores"]["completeness"]
    assert critical["quality_score"] < unweighted["quality_score"]

    allowed = run_quality_engine(
        frame,
        policy={
            "auto_critical_columns": False,
            "future_dates_allowed": ["forecast_date"],
        },
        semantic_profiles=profiles,
    )
    assert allowed["quality_score"] > unweighted["quality_score"]
    assert "forecast_date" not in allowed["date_errors"]


def test_policy_normalisation_and_report_comparison() -> None:
    frame = pd.DataFrame({"a": [1, 2], "b": [None, 2]})
    policy = normalise_quality_policy(
        frame,
        {
            "critical_columns": ["b", "missing-column"],
            "critical_multiplier": 99,
            "future_dates_allowed": ["missing-column"],
            "dimension_weights": {"completeness": 2, "validity": 1, "uniqueness": 1},
            "auto_critical_columns": False,
        },
    )
    assert policy["critical_columns"] == ["b"]
    assert policy["critical_multiplier"] == 5.0
    assert policy["future_dates_allowed"] == []
    assert abs(sum(policy["dimension_weights"].values()) - 1.0) < 1e-12

    before = run_quality_engine(frame, policy=policy)
    after = run_quality_engine(frame.fillna(0), policy=policy)
    comparison = compare_quality_reports(before, after)
    assert comparison["available"] is True
    assert comparison["score_delta"] > 0
    assert comparison["null_delta"] == -1
    assert comparison["dimension_deltas"]["completeness"] > 0


def _configure_dataset_state() -> None:
    fake_streamlit.session_state.clear()
    session_module.st = fake_streamlit
    dataset_module.st = fake_streamlit
    session_module.init_session_state()

    frame = pd.DataFrame(
        {
            "record_id": [1, 2, 3, 4, 4],
            "amount": [10.0, np.nan, 30.0, 40.0, 40.0],
            "event_date": pd.to_datetime(
                ["2024-01-01", "2024-02-01", "2024-03-01", "2099-01-01", "2099-01-01"]
            ),
            "outcome": ["won", "lost", "won", "lost", "lost"],
        }
    )
    dataset_module.activate_dataset(
        frame,
        {"source_type": "test", "source_name": "stage11"},
        display_name="stage11.csv",
    )


def test_unified_state_tracks_baseline_previous_policy_and_undo() -> None:
    _configure_dataset_state()
    session = fake_streamlit.session_state
    initial = session.quality_report
    assert initial["engine_version"] == "11.0"
    assert session.quality_baseline_report["quality_score"] == initial["quality_score"]
    assert session.dataset_state.quality_report is session.quality_report
    assert session.dataset_state.quality_policy is session.quality_policy
    assert dataset_module.dataset_state_health(deep=True)["ok"] is True

    revision = dataset_module.current_dataset_revision()
    fingerprint = session.working_fingerprint
    raw_before = session.raw_df.copy(deep=True)
    report_after_policy = dataset_module.refresh_quality_report(
        policy={
            "auto_critical_columns": False,
            "critical_columns": ["amount"],
            "critical_multiplier": 3.0,
            "future_dates_allowed": ["event_date"],
            "dimension_weights": {"completeness": 0.5, "validity": 0.35, "uniqueness": 0.15},
        },
        expected_revision=revision,
    )
    assert dataset_module.current_dataset_revision() == revision
    assert session.working_fingerprint == fingerprint
    assert report_after_policy["policy"]["critical_columns"] == ["amount"]
    assert "event_date" not in report_after_policy["date_errors"]
    pd.testing.assert_frame_equal(session.raw_df, raw_before)

    baseline_score = session.quality_baseline_report["quality_score"]
    previous_score = session.quality_report["quality_score"]
    dataset_module.apply_dataset_change(
        "Repair Stage 11 test data",
        lambda working: working.assign(
            amount=working["amount"].fillna(working["amount"].median())
        ).drop_duplicates().reset_index(drop=True),
        expected_revision=revision,
    )
    assert session.quality_previous_report["quality_score"] == previous_score
    assert session.quality_baseline_report["quality_score"] == baseline_score
    assert session.quality_report["quality_score"] > previous_score
    assert session.quality_report["duplicate_count"] == 0
    assert session.quality_report["total_nulls"] == 0
    assert dataset_module.dataset_state_health(deep=True)["ok"] is True

    assert dataset_module.perform_undo() is True
    assert session.quality_report["duplicate_count"] == 1
    assert session.quality_report["total_nulls"] == 1
    assert session.quality_policy["critical_columns"] == ["amount"]
    assert session.quality_baseline_report["quality_score"] == baseline_score
    assert dataset_module.dataset_state_health(deep=True)["ok"] is True


def main() -> None:
    test_future_dates_and_invalid_values_reduce_score_without_double_counting()
    test_critical_weighting_and_future_date_policy_are_effective()
    test_policy_normalisation_and_report_comparison()
    test_unified_state_tracks_baseline_previous_policy_and_undo()
    print(
        "PASS: Stage 11 Quality Engine V2 includes date defects, deduplicates cell-level validity events, "
        "weights critical columns and quality dimensions, preserves a true import baseline and previous scan, "
        "keeps policy/state synchronized through atomic changes and Undo, and never mutates source data."
    )


if __name__ == "__main__":
    main()
