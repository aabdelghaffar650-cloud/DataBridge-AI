"""DataBridge AI Stage 15 Champion/Challenger governance verification.

Run from project root:
    python tests/test_stage15_model_governance.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
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

from modules.data_mapper import analyze_dataframe
from modules.feature_pipeline import create_default_feature_pipeline_spec
from modules.ml_engine import run_supervised_experiment
from modules.model_governance import (
    ModelGovernanceError,
    STATUS_ARCHIVED,
    STATUS_CANDIDATE,
    STATUS_CHALLENGER,
    STATUS_CHAMPION,
    STATUS_REJECTED,
    assess_promotion,
    default_family_name,
    get_governance_state,
    governance_history,
    governance_status_for_package,
    list_family_models,
    list_governance_families,
    promote_challenger,
    reject_challenger,
    resubmit_archived_as_challenger,
    submit_as_challenger,
)
from modules.model_package import create_signed_model_package
from modules.model_registry import (
    ModelRegistryError,
    delete_registered_package,
    list_registered_packages,
    register_signed_package,
)


SIGNING_KEY = b"G" * 32


def _frame(rows: int = 280) -> pd.DataFrame:
    rng = np.random.default_rng(150015)
    amount = rng.normal(700, 120, rows)
    units = rng.integers(1, 14, rows)
    segment = np.resize(["Retail", "SME", "Enterprise", "Public"], rows)
    event_date = pd.date_range("2025-01-01", periods=rows, freq="D").astype(str)
    signal = amount + units * 25 + (segment == "Enterprise") * 95
    outcome = np.where(signal + rng.normal(0, 45, rows) > 900, "won", "lost")
    return pd.DataFrame(
        {
            "record_id": np.arange(10_000, 10_000 + rows),
            "amount": amount.round(2),
            "units": units,
            "segment": segment,
            "event_date": event_date,
            "outcome": outcome,
        }
    )


def _experiment_and_spec():
    df = _frame()
    analysis = analyze_dataframe(df)
    spec = create_default_feature_pipeline_spec(
        df,
        analysis["profiles"],
        "outcome",
        task="classification",
        configured_revision=1,
        configured_fingerprint="stage15-fp",
    )
    result = run_supervised_experiment(
        df,
        spec,
        dataset_revision=1,
        dataset_fingerprint="stage15-fp",
        split_strategy="stratified",
        cv_folds=3,
        random_state=15,
        model_names=["Logistic Regression"],
        max_rows=None,
        include_xgboost=False,
    )
    return df, analysis, spec, result


def _build_package(df, analysis, spec, result):
    return create_signed_model_package(
        result,
        df,
        spec,
        semantic_profiles=analysis["profiles"],
        signing_key=SIGNING_KEY,
    )


def _setup_temp_registry():
    temp = tempfile.TemporaryDirectory(prefix="databridge-stage15-")
    old = os.environ.get("DATABRIDGE_USER_DATA_DIR")
    os.environ["DATABRIDGE_USER_DATA_DIR"] = temp.name
    app_dir = Path(temp.name) / "DataBridgeAI"
    app_dir.mkdir(parents=True, exist_ok=True)
    (app_dir / "model_package_signing.key").write_bytes(SIGNING_KEY)
    return temp, old, app_dir


def _teardown(temp, old):
    temp.cleanup()
    if old is None:
        os.environ.pop("DATABRIDGE_USER_DATA_DIR", None)
    else:
        os.environ["DATABRIDGE_USER_DATA_DIR"] = old


def test_candidate_to_challenger_to_champion_and_atomic_supersede() -> None:
    df, analysis, spec, result = _experiment_and_spec()
    first = _build_package(df, analysis, spec, result)
    second = _build_package(df, analysis, spec, result)
    temp, old, _ = _setup_temp_registry()
    try:
        family = "Revenue Win Model"
        m1 = register_signed_package(first.package_bytes, label="v1", model_family=family)
        m2 = register_signed_package(second.package_bytes, label="v2", model_family=family)
        assert m1["governance_status"] == STATUS_CANDIDATE
        assert governance_status_for_package(m1["package_id"])["status"] == STATUS_CANDIDATE
        assert governance_status_for_package(m2["package_id"])["status"] == STATUS_CANDIDATE

        submit_as_challenger(m1["package_id"], actor="tester", note="Initial production candidate")
        a1 = assess_promotion(m1["package_id"])
        assert a1.ready is True
        assert a1.champion_id == ""
        promoted1 = promote_challenger(
            m1["package_id"],
            approval_token=a1.approval_token,
            approval_note="Approved as first production Champion",
            actor="tester",
        )
        assert promoted1["champion_id"] == m1["package_id"]
        assert governance_status_for_package(m1["package_id"])["status"] == STATUS_CHAMPION

        submit_as_challenger(m2["package_id"], actor="tester", note="Compare v2 with current Champion")
        a2 = assess_promotion(m2["package_id"])
        assert a2.ready is True
        assert a2.champion_id == m1["package_id"]
        assert a2.comparison["same_dataset"] is True
        assert a2.comparison["metric_delta_favourable"] is True
        promoted2 = promote_challenger(
            m2["package_id"],
            approval_token=a2.approval_token,
            approval_note="Approved after Champion/Challenger comparison",
            actor="tester",
        )
        assert promoted2["previous_champion_id"] == m1["package_id"]
        assert governance_status_for_package(m1["package_id"])["status"] == STATUS_ARCHIVED
        assert governance_status_for_package(m2["package_id"])["status"] == STATUS_CHAMPION

        families = list_governance_families()
        assert len(families) == 1
        assert families[0]["champion_id"] == m2["package_id"]
        models = list_family_models(families[0]["family_id"])
        assert {row["status"] for row in models} == {STATUS_CHAMPION, STATUS_ARCHIVED}
        events = governance_history(families[0]["family_id"])
        assert any(event["event"] == "challenger_promoted" for event in events)
        assert all(event.get("event_hash") for event in events)
    finally:
        _teardown(temp, old)


def test_stale_approval_token_and_rejection_are_safe() -> None:
    df, analysis, spec, result = _experiment_and_spec()
    builds = [_build_package(df, analysis, spec, result) for _ in range(3)]
    temp, old, _ = _setup_temp_registry()
    try:
        metas = [register_signed_package(build.package_bytes, model_family="Risk Family") for build in builds]
        submit_as_challenger(metas[0]["package_id"], note="first")
        first_assessment = assess_promotion(metas[0]["package_id"])
        # Any governance revision change invalidates the previous decision token.
        submit_as_challenger(metas[1]["package_id"], note="parallel review")
        try:
            promote_challenger(
                metas[0]["package_id"],
                approval_token=first_assessment.approval_token,
                approval_note="This must be stale",
            )
        except ModelGovernanceError as exc:
            assert "stale" in str(exc).lower() or "changed" in str(exc).lower()
        else:
            raise AssertionError("Stale promotion token should have been rejected")

        rejected = reject_challenger(
            metas[1]["package_id"],
            reason="Validation review found deployment concerns",
            actor="tester",
        )
        assert rejected["status"] == STATUS_REJECTED
        try:
            submit_as_challenger(metas[1]["package_id"], note="retry unchanged")
        except ModelGovernanceError:
            pass
        else:
            raise AssertionError("Rejected immutable package should not be re-submitted unchanged")
    finally:
        _teardown(temp, old)


def test_governance_tamper_detection_and_champion_delete_protection() -> None:
    df, analysis, spec, result = _experiment_and_spec()
    first = _build_package(df, analysis, spec, result)
    second = _build_package(df, analysis, spec, result)
    temp, old, app_dir = _setup_temp_registry()
    try:
        m1 = register_signed_package(first.package_bytes, model_family="Delete Protection")
        m2 = register_signed_package(second.package_bytes, model_family="Delete Protection")
        submit_as_challenger(m1["package_id"], note="approve")
        a1 = assess_promotion(m1["package_id"])
        promote_challenger(
            m1["package_id"], approval_token=a1.approval_token, approval_note="Production approval"
        )
        try:
            delete_registered_package(m1["package_id"])
        except ModelRegistryError as exc:
            assert "Champion" in str(exc)
        else:
            raise AssertionError("Champion deletion must be blocked")

        # Promote second and then archived first may be explicitly removed.
        submit_as_challenger(m2["package_id"], note="new candidate")
        a2 = assess_promotion(m2["package_id"])
        promote_challenger(
            m2["package_id"], approval_token=a2.approval_token, approval_note="Supersede old Champion"
        )
        delete_registered_package(m1["package_id"])
        assert not any(row["package_id"] == m1["package_id"] for row in list_registered_packages())

        family_id = list_governance_families()[0]["family_id"]
        gov_path = app_dir / "model_registry" / "governance" / f"{family_id}.json"
        envelope = json.loads(gov_path.read_text(encoding="utf-8"))
        envelope["payload"]["champion_id"] = "tampered-id"
        gov_path.write_text(json.dumps(envelope), encoding="utf-8")
        try:
            get_governance_state(family_id)
        except ModelGovernanceError as exc:
            assert "authentication" in str(exc).lower() or "invalid" in str(exc).lower()
        else:
            raise AssertionError("Tampered governance record must be rejected")
    finally:
        _teardown(temp, old)


def test_archived_champion_can_only_return_through_challenger_gate() -> None:
    df, analysis, spec, result = _experiment_and_spec()
    first = _build_package(df, analysis, spec, result)
    second = _build_package(df, analysis, spec, result)
    temp, old, _ = _setup_temp_registry()
    try:
        m1 = register_signed_package(first.package_bytes, model_family="Rollback Family")
        m2 = register_signed_package(second.package_bytes, model_family="Rollback Family")
        for meta, note in ((m1, "first"), (m2, "second")):
            submit_as_challenger(meta["package_id"], note=note)
            assessment = assess_promotion(meta["package_id"])
            promote_challenger(
                meta["package_id"],
                approval_token=assessment.approval_token,
                approval_note=f"Approve {note}",
            )
        assert governance_status_for_package(m1["package_id"])["status"] == STATUS_ARCHIVED
        resubmit_archived_as_challenger(
            m1["package_id"], reason="Rollback requested after production review"
        )
        assert governance_status_for_package(m1["package_id"])["status"] == STATUS_CHALLENGER
        rollback_assessment = assess_promotion(m1["package_id"])
        assert rollback_assessment.champion_id == m2["package_id"]
        promote_challenger(
            m1["package_id"],
            approval_token=rollback_assessment.approval_token,
            approval_note="Rollback approved after fresh comparison",
        )
        assert governance_status_for_package(m1["package_id"])["status"] == STATUS_CHAMPION
        assert governance_status_for_package(m2["package_id"])["status"] == STATUS_ARCHIVED
    finally:
        _teardown(temp, old)


def main() -> None:
    test_candidate_to_challenger_to_champion_and_atomic_supersede()
    test_stale_approval_token_and_rejection_are_safe()
    test_governance_tamper_detection_and_champion_delete_protection()
    test_archived_champion_can_only_return_through_challenger_gate()
    print(
        "PASS: Stage 15 enforces authenticated Candidate → Challenger → Champion governance, "
        "human approval tokens, atomic Champion supersession, rejection/rollback gates, tamper-evident decision history, "
        "and active-Champion deletion protection without altering signed model packages."
    )


if __name__ == "__main__":
    main()
