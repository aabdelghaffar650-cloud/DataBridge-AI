"""DataBridge AI Stage 16 safe auto-retraining verification.

Run from project root:
    python tests/test_stage16_safe_auto_retraining.py
"""
from __future__ import annotations

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

from core.dataset import dataframe_fingerprint
from modules.data_mapper import analyze_dataframe
from modules.feature_pipeline import create_default_feature_pipeline_spec
from modules.ml_engine import run_supervised_experiment
from modules.model_governance import (
    STATUS_CHALLENGER,
    STATUS_CHAMPION,
    assess_promotion,
    governance_status_for_package,
    promote_challenger,
    submit_as_challenger,
)
from modules.model_package import create_signed_model_package, load_signed_model_package
from modules.model_registry import register_signed_package
from modules.retraining_workflow import (
    RetrainingWorkflowError,
    assess_retraining_need,
    create_and_register_retraining_challenger,
    feature_logic_fingerprint,
    run_safe_retraining_experiment,
)


SIGNING_KEY = b"R" * 32


def _frame(rows: int = 320, *, shift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(160016 + int(shift * 10))
    amount = rng.normal(720 + shift, 125, rows)
    units = rng.integers(1, 14, rows)
    segment = np.resize(["Retail", "SME", "Enterprise", "Public"], rows)
    event_date = pd.date_range("2025-01-01", periods=rows, freq="D").astype(str)
    signal = amount + units * 25 + (segment == "Enterprise") * 100
    outcome = np.where(signal + rng.normal(0, 48, rows) > 930 + shift * 0.2, "won", "lost")
    return pd.DataFrame(
        {
            "record_id": np.arange(50_000, 50_000 + rows),
            "amount": amount.round(2),
            "units": units,
            "segment": segment,
            "event_date": event_date,
            "outcome": outcome,
        }
    )


def _setup_temp_registry():
    temp = tempfile.TemporaryDirectory(prefix="databridge-stage16-")
    old = os.environ.get("DATABRIDGE_USER_DATA_DIR")
    os.environ["DATABRIDGE_USER_DATA_DIR"] = temp.name
    app_dir = Path(temp.name) / "DataBridgeAI"
    app_dir.mkdir(parents=True, exist_ok=True)
    (app_dir / "model_package_signing.key").write_bytes(SIGNING_KEY)
    return temp, old


def _teardown(temp, old):
    temp.cleanup()
    if old is None:
        os.environ.pop("DATABRIDGE_USER_DATA_DIR", None)
    else:
        os.environ["DATABRIDGE_USER_DATA_DIR"] = old


def _champion_package():
    df = _frame()
    fingerprint = dataframe_fingerprint(df)
    analysis = analyze_dataframe(df)
    spec = create_default_feature_pipeline_spec(
        df,
        analysis["profiles"],
        "outcome",
        task="classification",
        configured_revision=1,
        configured_fingerprint=fingerprint,
    )
    result = run_supervised_experiment(
        df,
        spec,
        dataset_revision=1,
        dataset_fingerprint=fingerprint,
        split_strategy="stratified",
        holdout_size=0.20,
        cv_folds=3,
        random_state=16,
        model_names=["Logistic Regression", "Random Forest"],
        class_weight_mode="balanced",
        tune_best=False,
        tuning_iterations=5,
        max_rows=None,
        include_xgboost=False,
    )
    build = create_signed_model_package(
        result,
        df,
        spec,
        semantic_profiles=analysis["profiles"],
        signing_key=SIGNING_KEY,
    )
    loaded = load_signed_model_package(build.package_bytes, signing_key=SIGNING_KEY)
    assert loaded.retraining_contract
    return df, analysis, spec, result, build, loaded


def _promote_first(build) -> str:
    meta = register_signed_package(build.package_bytes, label="stage16 champion", model_family="Safe Retraining Family")
    package_id = meta["package_id"]
    submit_as_challenger(package_id, actor="tester", note="Initial Champion for retraining test")
    assessment = assess_promotion(package_id)
    assert assessment.ready
    promote_challenger(
        package_id,
        approval_token=assessment.approval_token,
        approval_note="Approved initial Champion for Stage 16 test",
        actor="tester",
    )
    assert governance_status_for_package(package_id)["status"] == STATUS_CHAMPION
    return package_id


def test_critical_monitoring_replays_locked_contract_and_stops_at_challenger() -> None:
    base_df, _, base_spec, base_result, build, _ = _champion_package()
    temp, old = _setup_temp_registry()
    try:
        champion_id = _promote_first(build)
        champion = load_signed_model_package(build.package_bytes)
        new_df = _frame(360, shift=95.0)
        untouched = new_df.copy(deep=True)
        new_fp = dataframe_fingerprint(new_df)
        monitor = {
            "package_id": champion_id,
            "task": "classification",
            "target": "outcome",
            "overall_status": "Critical",
            "schema": {"valid": True},
            "observed_performance": {"available": False},
        }

        gate = assess_retraining_need(champion, new_df, monitoring_report=monitor)
        assert gate.eligible is True
        assert gate.recommended is True
        assert gate.monitoring_status == "Critical"

        result, rebound, fresh_gate = run_safe_retraining_experiment(
            champion,
            new_df,
            dataset_revision=2,
            dataset_fingerprint=new_fp,
            monitoring_report=monitor,
        )
        assert new_df.equals(untouched)
        assert result.dataset_fingerprint == new_fp
        assert result.split_strategy == base_result.split_strategy
        assert result.random_state == base_result.random_state
        assert result.experiment_config["candidate_models"] == base_result.experiment_config["candidate_models"]
        assert result.experiment_config["class_weight_mode"] == "balanced"
        assert feature_logic_fingerprint(rebound) == feature_logic_fingerprint(base_spec)
        assert rebound["configured_fingerprint"] == new_fp
        assert rebound["configured_revision"] == 2

        semantic = analyze_dataframe(new_df)["profiles"]
        candidate = create_and_register_retraining_challenger(
            champion,
            new_df,
            result,
            rebound,
            fresh_gate,
            semantic_profiles=semantic,
            actor="tester",
            note="Critical drift retraining candidate",
        )
        challenger_id = candidate.registry_metadata["package_id"]
        assert challenger_id != champion_id
        assert candidate.governance["status"] == STATUS_CHALLENGER
        assert governance_status_for_package(champion_id)["status"] == STATUS_CHAMPION
        assert governance_status_for_package(challenger_id)["status"] == STATUS_CHALLENGER
        assert new_df.equals(untouched)

        challenger = load_signed_model_package(candidate.package_build.package_bytes)
        assert challenger.retraining_contract
        assert challenger.retraining_contract["safety"]["auto_promotion_allowed"] is False
        assert challenger.retraining_contract["experiment"]["split_strategy"] == base_result.split_strategy
    finally:
        _teardown(temp, old)


def test_stable_monitoring_requires_explicit_manual_override() -> None:
    _, _, _, _, build, _ = _champion_package()
    temp, old = _setup_temp_registry()
    try:
        champion_id = _promote_first(build)
        champion = load_signed_model_package(build.package_bytes)
        new_df = _frame(330, shift=10.0)
        fp = dataframe_fingerprint(new_df)
        stable = {
            "package_id": champion_id,
            "task": "classification",
            "target": "outcome",
            "overall_status": "Stable",
            "schema": {"valid": True},
            "observed_performance": {"available": False},
        }
        gate = assess_retraining_need(champion, new_df, monitoring_report=stable)
        assert gate.eligible is True
        assert gate.recommended is False
        try:
            run_safe_retraining_experiment(
                champion,
                new_df,
                dataset_revision=3,
                dataset_fingerprint=fp,
                monitoring_report=stable,
                manual_override=False,
            )
        except RetrainingWorkflowError as exc:
            assert "not currently recommended" in str(exc)
        else:
            raise AssertionError("Stable monitoring must not auto-trigger retraining without explicit override")

        result, rebound, _ = run_safe_retraining_experiment(
            champion,
            new_df,
            dataset_revision=3,
            dataset_fingerprint=fp,
            monitoring_report=stable,
            manual_override=True,
        )
        assert result.train_rows > 0 and result.holdout_rows > 0
        assert rebound["configured_fingerprint"] == fp
    finally:
        _teardown(temp, old)


def test_missing_contract_columns_and_non_champion_are_blocked() -> None:
    _, _, _, _, build, _ = _champion_package()
    temp, old = _setup_temp_registry()
    try:
        meta = register_signed_package(build.package_bytes, model_family="Not Champion Yet")
        package = load_signed_model_package(build.package_bytes)
        frame = _frame().drop(columns=["amount"])
        gate = assess_retraining_need(
            package,
            frame,
            monitoring_report={
                "package_id": meta["package_id"],
                "task": "classification",
                "target": "outcome",
                "overall_status": "Critical",
                "schema": {"valid": False, "missing_columns": ["amount"]},
            },
        )
        assert gate.eligible is False
        assert any("Champion" in item for item in gate.blockers)
        assert any("missing contract columns" in item for item in gate.blockers)
    finally:
        _teardown(temp, old)


def main() -> None:
    test_critical_monitoring_replays_locked_contract_and_stops_at_challenger()
    test_stable_monitoring_requires_explicit_manual_override()
    test_missing_contract_columns_and_non_champion_are_blocked()
    print(
        "PASS: Stage 16 Safe Auto Retraining replays an authenticated Champion training contract on new data, "
        "uses training-only CV plus an untouched holdout, triggers from drift/performance evidence or explicit manual override, "
        "preserves source data, signs/registers the result as Challenger only, and never auto-promotes over the active Champion."
    )


if __name__ == "__main__":
    main()
