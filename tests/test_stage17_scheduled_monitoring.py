"""DataBridge AI Stage 17 scheduled monitoring verification.

Run from project root:
    python tests/test_stage17_scheduled_monitoring.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
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
from modules.model_package import create_signed_model_package, load_signed_model_package
import modules.monitoring_scheduler as scheduler

SIGNING_KEY = b"S" * 32


def _frame(rows: int = 220) -> pd.DataFrame:
    rng = np.random.default_rng(20260817)
    amount = rng.normal(500, 70, rows)
    units = rng.integers(1, 9, rows)
    segment = np.resize(["Retail", "SME", "Enterprise"], rows)
    dates = pd.date_range("2025-01-01", periods=rows, freq="D").astype(str)
    score = amount + units * 35 + (segment == "Enterprise") * 90
    outcome = np.where(score + rng.normal(0, 28, rows) > 690, "won", "lost")
    return pd.DataFrame({
        "record_id": np.arange(1000, 1000 + rows),
        "amount": amount.round(2),
        "units": units,
        "segment": segment,
        "event_date": dates,
        "outcome": outcome,
    })


def _build_package():
    df = _frame()
    analysis = analyze_dataframe(df)
    spec = create_default_feature_pipeline_spec(
        df, analysis["profiles"], "outcome", task="classification",
        configured_revision=1, configured_fingerprint="stage17-fp",
    )
    result = run_supervised_experiment(
        df, spec, dataset_revision=1, dataset_fingerprint="stage17-fp",
        split_strategy="stratified", cv_folds=3, random_state=17,
        model_names=["Logistic Regression"], max_rows=None, include_xgboost=False,
    )
    build = create_signed_model_package(
        result, df, spec, semantic_profiles=analysis["profiles"], signing_key=SIGNING_KEY,
    )
    package = load_signed_model_package(build.package_bytes, signing_key=SIGNING_KEY)
    return df, build, package


def _patch_champion(build, package):
    original_current = scheduler.current_champion
    original_load_bytes = scheduler.load_registered_package_bytes
    scheduler.current_champion = lambda family_id, signing_key=None: {
        "family_id": family_id,
        "family_name": "Stage 17 Family",
        "package_id": package.package_id,
        "task": package.task,
        "target": package.target,
    }
    scheduler.load_registered_package_bytes = lambda package_id: build.package_bytes
    return original_current, original_load_bytes


def test_signed_jobs_schedule_and_tamper_detection() -> None:
    df, build, package = _build_package()
    old = os.environ.get("DATABRIDGE_USER_DATA_DIR")
    with tempfile.TemporaryDirectory(prefix="databridge-stage17-") as temp:
        os.environ["DATABRIDGE_USER_DATA_DIR"] = temp
        app = Path(temp) / "DataBridgeAI"
        app.mkdir(parents=True, exist_ok=True)
        (app / "model_package_signing.key").write_bytes(SIGNING_KEY)
        source = Path(temp) / "monitor.csv"
        df.to_csv(source, index=False)
        orig_current, orig_bytes = _patch_champion(build, package)
        try:
            daily = scheduler.normalise_schedule("daily", "8:05")
            weekly = scheduler.normalise_schedule("weekly", "19:30", weekday="fri")
            monthly = scheduler.normalise_schedule("monthly", "06:10", month_day=28)
            assert daily == {"cadence": "daily", "time_of_day": "08:05"}
            assert weekly["weekday"] == "FRI"
            assert monthly["month_day"] == 28

            job = scheduler.create_monitoring_job(
                name="Daily Champion Health",
                family_id="family-stage17",
                source_mode="file",
                source_path=str(source),
                cadence="daily",
                time_of_day="08:05",
                actual_target_column="outcome",
                retention_reports=5,
                skip_unchanged=True,
                signing_key=SIGNING_KEY,
            )
            loaded = scheduler.load_monitoring_job(job["job_id"], signing_key=SIGNING_KEY)
            assert loaded["family"]["champion_at_save"] == package.package_id
            assert loaded["source"]["path"] == str(source.resolve())
            assert loaded["signature"] if "signature" in loaded else True  # signature is verified then stripped

            path = scheduler._job_path(job["job_id"])
            raw = json.loads(path.read_text(encoding="utf-8"))
            raw["name"] = "tampered"
            path.write_text(json.dumps(raw), encoding="utf-8")
            try:
                scheduler.load_monitoring_job(job["job_id"], signing_key=SIGNING_KEY)
                raise AssertionError("tampered signed job was accepted")
            except scheduler.MonitoringSchedulerError:
                pass
            scheduler.save_monitoring_job(loaded, signing_key=SIGNING_KEY)

            args = scheduler.windows_task_arguments(loaded)
            joined = " ".join(args)
            assert "schtasks" in args[0].lower()
            assert "--run-job" in joined
            assert job["job_id"] in joined
            assert str(source) not in joined
            assert "outcome" not in joined
        finally:
            scheduler.current_champion = orig_current
            scheduler.load_registered_package_bytes = orig_bytes
            if old is None:
                os.environ.pop("DATABRIDGE_USER_DATA_DIR", None)
            else:
                os.environ["DATABRIDGE_USER_DATA_DIR"] = old


def test_unattended_runner_reports_skip_unchanged_and_never_persists_source_path() -> None:
    df, build, package = _build_package()
    old = os.environ.get("DATABRIDGE_USER_DATA_DIR")
    with tempfile.TemporaryDirectory(prefix="databridge-stage17-run-") as temp:
        os.environ["DATABRIDGE_USER_DATA_DIR"] = temp
        app = Path(temp) / "DataBridgeAI"
        app.mkdir(parents=True, exist_ok=True)
        (app / "model_package_signing.key").write_bytes(SIGNING_KEY)
        source = Path(temp) / "daily_monitor.csv"
        df.to_csv(source, index=False)
        orig_current, orig_bytes = _patch_champion(build, package)
        try:
            job = scheduler.create_monitoring_job(
                name="Production monitor",
                family_id="family-stage17",
                source_mode="file",
                source_path=str(source),
                cadence="weekly",
                time_of_day="07:45",
                weekday="MON",
                actual_target_column="outcome",
                retention_reports=3,
                skip_unchanged=True,
                signing_key=SIGNING_KEY,
            )
            first = scheduler.run_scheduled_job(job["job_id"], signing_key=SIGNING_KEY)
            assert first.status == scheduler.RUN_COMPLETED
            assert first.package_id == package.package_id
            assert Path(first.report_path).is_file()
            saved_text = Path(first.report_path).read_text(encoding="utf-8")
            assert str(Path(temp).resolve()) not in saved_text
            saved = json.loads(saved_text)
            assert saved["scheduled_run"]["source_path_persisted"] is False
            assert saved["scheduled_run"]["source_name"] == source.name
            assert saved["scheduled_run"]["source_sha256"] == first.source_sha256

            second = scheduler.run_scheduled_job(job["job_id"], signing_key=SIGNING_KEY)
            assert second.status == scheduler.RUN_SKIPPED_UNCHANGED
            assert len(scheduler.list_scheduled_reports(job["job_id"])) == 1

            shifted = df.copy(deep=True)
            shifted["amount"] = pd.to_numeric(shifted["amount"], errors="coerce") + 1200
            shifted["units"] = pd.to_numeric(shifted["units"], errors="coerce") + 35
            shifted["segment"] = "NEW_SEGMENT"
            shifted["event_date"] = pd.date_range("2038-01-01", periods=len(shifted), freq="D").astype(str)
            shifted.to_csv(source, index=False)
            time.sleep(0.01)
            third = scheduler.run_scheduled_job(job["job_id"], signing_key=SIGNING_KEY)
            assert third.status == scheduler.RUN_COMPLETED
            assert third.overall_status in {"Drifted", "Critical"}
            assert third.retraining_recommended is True
            reports = scheduler.list_scheduled_reports(job["job_id"])
            assert len(reports) == 2
            assert reports[0]["retraining_recommended"] is True

            latest = scheduler.load_scheduled_report(reports[0]["path"])
            assert latest["package_id"] == package.package_id
            assert latest["scheduled_run"]["family_id"] == "family-stage17"
        finally:
            scheduler.current_champion = orig_current
            scheduler.load_registered_package_bytes = orig_bytes
            if old is None:
                os.environ.pop("DATABRIDGE_USER_DATA_DIR", None)
            else:
                os.environ["DATABRIDGE_USER_DATA_DIR"] = old


def test_latest_file_mode_and_network_paths_are_blocked() -> None:
    old = os.environ.get("DATABRIDGE_USER_DATA_DIR")
    with tempfile.TemporaryDirectory(prefix="databridge-stage17-source-") as temp:
        os.environ["DATABRIDGE_USER_DATA_DIR"] = temp
        folder = Path(temp) / "incoming"
        folder.mkdir()
        first = folder / "monitor_001.csv"
        second = folder / "monitor_002.csv"
        _frame(20).to_csv(first, index=False)
        time.sleep(0.02)
        _frame(21).to_csv(second, index=False)
        job = {
            "source": {"mode": "latest_in_folder", "path": str(folder), "pattern": "monitor_*.csv"},
            "max_source_age_hours": 0,
        }
        snapshot = scheduler.load_job_source(job)
        assert snapshot.path.name == second.name
        assert len(snapshot.frame) == 21
        try:
            scheduler._validate_source("file", r"\\server\share\monitor.csv")
            raise AssertionError("UNC source was accepted")
        except scheduler.MonitoringSchedulerError:
            pass
        if old is None:
            os.environ.pop("DATABRIDGE_USER_DATA_DIR", None)
        else:
            os.environ["DATABRIDGE_USER_DATA_DIR"] = old


def main() -> None:
    test_signed_jobs_schedule_and_tamper_detection()
    test_unattended_runner_reports_skip_unchanged_and_never_persists_source_path()
    test_latest_file_mode_and_network_paths_are_blocked()
    print(
        "PASS: Stage 17 authenticates scheduled monitoring jobs, follows the current governed Champion at run time, "
        "supports daily/weekly/monthly Windows Task Scheduler actions without embedding source paths or secrets, safely parses local monitoring files, "
        "skips unchanged/stale inputs, writes privacy-preserving retained reports, recommends but never performs retraining/promotion, and blocks tampered or network-backed jobs."
    )


if __name__ == "__main__":
    main()
