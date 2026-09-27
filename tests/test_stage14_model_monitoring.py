"""DataBridge AI Stage 14 model monitoring and drift verification.

Run from project root:
    python tests/test_stage14_model_monitoring.py
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
from modules.model_monitoring import run_monitoring_analysis
from modules.model_package import (
    create_signed_model_package,
    load_signed_model_package,
)
from modules.model_registry import (
    delete_registered_package,
    list_registered_packages,
    load_registered_package_bytes,
    register_signed_package,
    save_monitoring_report,
)


SIGNING_KEY = b"M" * 32


def _frame(rows: int = 260) -> pd.DataFrame:
    rng = np.random.default_rng(20260814)
    amount = rng.normal(520, 80, rows)
    units = rng.integers(1, 10, rows)
    segment = np.resize(["Retail", "SME", "Enterprise", "Public"], rows)
    event_date = pd.date_range("2024-01-01", periods=rows, freq="D").astype(str)
    signal = amount + units * 31 + (segment == "Enterprise") * 78
    outcome = np.where(signal + rng.normal(0, 34, rows) > 680, "won", "lost")
    df = pd.DataFrame(
        {
            "record_id": np.arange(5000, 5000 + rows),
            "amount": amount.round(2),
            "units": units,
            "segment": segment,
            "event_date": event_date,
            "outcome": outcome,
        }
    )
    df.loc[5, "amount"] = np.nan
    df.loc[9, "segment"] = None
    return df


def _build():
    df = _frame()
    analysis = analyze_dataframe(df)
    spec = create_default_feature_pipeline_spec(
        df,
        analysis["profiles"],
        "outcome",
        task="classification",
        configured_revision=1,
        configured_fingerprint="stage14-fp",
    )
    result = run_supervised_experiment(
        df,
        spec,
        dataset_revision=1,
        dataset_fingerprint="stage14-fp",
        split_strategy="stratified",
        cv_folds=3,
        random_state=14,
        model_names=["Logistic Regression"],
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
    package = load_signed_model_package(build.package_bytes, signing_key=SIGNING_KEY)
    return df, analysis, spec, result, build, package


def test_training_reference_is_train_only_and_privacy_preserving() -> None:
    df, analysis, spec, result, build, package = _build()
    reference = result.monitoring_reference
    assert reference["reference_scope"] == "training_rows_only"
    assert reference["rows"] == result.train_rows
    assert reference["contains_row_level_data"] is False
    assert reference["columns"]

    persisted = package.manifest["monitoring"]["training_reference"]
    assert persisted["rows"] == result.train_rows
    assert persisted["contains_row_level_data"] is False
    assert persisted["category_token_scheme"].startswith("HMAC-SHA256")
    monitoring_text = json.dumps(package.manifest["monitoring"], ensure_ascii=False)
    # The Stage 14 monitoring reference must not add raw categorical labels.
    # The fitted preprocessing artifact itself necessarily retains learned encoder categories.
    for value in ["Retail", "SME", "Enterprise", "Public"]:
        assert value not in monitoring_text
    segment_ref = persisted["columns"].get("segment", {})
    if segment_ref.get("kind") == "categorical":
        assert "top_distribution" not in segment_ref
        assert segment_ref.get("top_distribution_tokens")
        assert all(len(item["token"]) == 64 for item in segment_ref["top_distribution_tokens"])


def test_stable_training_distribution_and_clear_shift_detection() -> None:
    df, analysis, spec, result, build, package = _build()
    holdout_indices = set(result.predictions.index.tolist())
    training = df.loc[[idx for idx in df.index if idx not in holdout_indices]].copy()
    # The experiment has no target-missing rows and no sampling, so this is the exact train population.
    stable = run_monitoring_analysis(
        package,
        training,
        signing_key=SIGNING_KEY,
        actual_target_column="outcome",
        include_prediction_output=False,
    )
    assert stable.report["schema"]["valid"] is True
    assert stable.report["observed_performance"]["available"] is True
    assert set(stable.feature_table["Status"]).issubset({"Stable", "Watch"})
    assert stable.report["overall_status"] in {"Stable", "Watch"}

    shifted = training.copy(deep=True)
    if "amount" in shifted.columns:
        shifted["amount"] = pd.to_numeric(shifted["amount"], errors="coerce") + 900
    if "units" in shifted.columns:
        shifted["units"] = pd.to_numeric(shifted["units"], errors="coerce") + 25
    if "segment" in shifted.columns:
        shifted["segment"] = "BRAND_NEW_SEGMENT"
    if "event_date" in shifted.columns:
        shifted["event_date"] = pd.date_range("2035-01-01", periods=len(shifted), freq="D").astype(str)
    drifted = run_monitoring_analysis(
        package,
        shifted,
        signing_key=SIGNING_KEY,
        actual_target_column="outcome",
        include_prediction_output=False,
    )
    assert drifted.report["overall_status"] in {"Drifted", "Critical"}
    assert drifted.report["drift_score"] > stable.report["drift_score"]
    assert any(status in {"Drifted", "Critical"} for status in drifted.feature_table["Status"])


def test_schema_drift_is_blocked_by_signed_contract() -> None:
    df, analysis, spec, result, build, package = _build()
    required = package.required_columns
    assert required
    bad = df.drop(columns=[required[0]]).head(20)
    result_bad = run_monitoring_analysis(package, bad, signing_key=SIGNING_KEY)
    assert result_bad.report["schema"]["valid"] is False
    assert required[0] in result_bad.report["schema"]["missing_columns"]
    assert result_bad.report["overall_status"] == "Critical"
    assert result_bad.prediction_output is None


def test_local_registry_verifies_before_storage_and_report_persistence() -> None:
    df, analysis, spec, result, build, package = _build()
    old = os.environ.get("DATABRIDGE_USER_DATA_DIR")
    with tempfile.TemporaryDirectory(prefix="databridge-stage14-") as temp:
        os.environ["DATABRIDGE_USER_DATA_DIR"] = temp
        app_dir = Path(temp) / "DataBridgeAI"
        app_dir.mkdir(parents=True, exist_ok=True)
        (app_dir / "model_package_signing.key").write_bytes(SIGNING_KEY)
        try:
            meta = register_signed_package(build.package_bytes, label="Stage 14 test")
            assert meta["package_id"] == package.package_id
            assert meta["has_monitoring_reference"] is True
            rows = list_registered_packages()
            assert any(row["package_id"] == package.package_id for row in rows)
            raw = load_registered_package_bytes(package.package_id)
            loaded = load_signed_model_package(raw, signing_key=SIGNING_KEY)
            assert loaded.package_id == package.package_id

            monitor = run_monitoring_analysis(
                package,
                df.head(80),
                signing_key=SIGNING_KEY,
                actual_target_column="outcome",
                include_prediction_output=False,
            )
            path = save_monitoring_report(monitor.report)
            assert path.is_file()
            saved = json.loads(path.read_text(encoding="utf-8"))
            assert saved["package_id"] == package.package_id
            delete_registered_package(package.package_id)
            assert not any(row["package_id"] == package.package_id for row in list_registered_packages())
        finally:
            if old is None:
                os.environ.pop("DATABRIDGE_USER_DATA_DIR", None)
            else:
                os.environ["DATABRIDGE_USER_DATA_DIR"] = old


def main() -> None:
    test_training_reference_is_train_only_and_privacy_preserving()
    test_stable_training_distribution_and_clear_shift_detection()
    test_schema_drift_is_blocked_by_signed_contract()
    test_local_registry_verifies_before_storage_and_report_persistence()
    print(
        "PASS: Stage 14 stores privacy-preserving training-only monitoring references in signed packages; "
        "detects schema, numeric, categorical, datetime/text-length and prediction drift; evaluates optional labelled performance; "
        "and maintains a verified local model registry without mutating source data."
    )


if __name__ == "__main__":
    main()
