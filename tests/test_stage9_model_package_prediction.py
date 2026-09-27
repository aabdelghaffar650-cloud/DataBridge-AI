"""DataBridge AI Stage 9 signed model package and Prediction Studio verification.

Run from project root:
    python tests/test_stage9_model_package_prediction.py
"""
from __future__ import annotations

import builtins
import io
import json
import pickle
import sys
import tempfile
import types
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# The verification script exercises pure engines and can run even when the
# project venv does not expose Streamlit to this interpreter.
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
from modules.model_package import (
    ALLOWED_MEMBERS,
    MANIFEST_MEMBER,
    MODEL_MEMBER,
    README_MEMBER,
    SIGNATURE_MEMBER,
    ModelPackageError,
    create_signed_model_package,
    load_signed_model_package,
    run_batch_prediction,
    validate_prediction_frame,
)


SIGNING_KEY = b"D" * 32
OTHER_KEY = b"X" * 32


def _classification_frame(rows: int = 210) -> pd.DataFrame:
    rng = np.random.default_rng(20260810)
    amount = rng.normal(500, 85, rows)
    units = rng.integers(1, 10, rows)
    segment = np.resize(["Retail", "SME", "Enterprise", "Public"], rows)
    dates = pd.date_range("2024-01-01", periods=rows, freq="D").astype(str)
    signal = amount + units * 32 + (segment == "Enterprise") * 75
    outcome = np.where(signal + rng.normal(0, 32, rows) > 665, "won", "lost")
    frame = pd.DataFrame(
        {
            "record_id": np.arange(1000, 1000 + rows),
            "amount": amount.round(2),
            "units": units,
            "segment": segment,
            "event_date": dates,
            "outcome": outcome,
        }
    )
    frame.loc[3, "amount"] = np.nan
    frame.loc[8, "segment"] = None
    return frame


def _regression_frame(rows: int = 190) -> pd.DataFrame:
    rng = np.random.default_rng(20260811)
    x1 = rng.normal(0, 1, rows)
    x2 = rng.normal(5, 2, rows)
    category = np.resize(["A", "B", "C"], rows)
    target = 20 + 4.2 * x1 - 2.5 * x2 + (category == "C") * 2.7 + rng.normal(0, 0.45, rows)
    frame = pd.DataFrame(
        {
            "row_id": np.arange(rows),
            "x1": x1,
            "x2": x2,
            "category": category,
            "target_value": target,
        }
    )
    frame.loc[4, "x2"] = np.nan
    return frame


def _train(df: pd.DataFrame, target: str, task: str):
    analysis = analyze_dataframe(df)
    spec = create_default_feature_pipeline_spec(
        df,
        analysis["profiles"],
        target,
        task=task,
        configured_revision=1,
        configured_fingerprint=f"{task}-fp",
    )
    model_name = "Logistic Regression" if task == "classification" else "Ridge Regression"
    result = run_supervised_experiment(
        df,
        spec,
        dataset_revision=1,
        dataset_fingerprint=f"{task}-fp",
        split_strategy="stratified" if task == "classification" else "random",
        cv_folds=3,
        random_state=14,
        model_names=[model_name],
        max_rows=None,
        include_xgboost=False,
    )
    return analysis, spec, result


def _read_zip(raw: bytes) -> dict[str, bytes]:
    with zipfile.ZipFile(io.BytesIO(raw), "r") as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def _write_zip(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def test_classification_package_roundtrip_and_unknown_categories() -> None:
    df = _classification_frame()
    analysis, spec, result = _train(df, "outcome", "classification")
    build = create_signed_model_package(
        result,
        df,
        spec,
        semantic_profiles=analysis["profiles"],
        signing_key=SIGNING_KEY,
    )
    assert build.file_name.endswith(".dbmlpkg")
    assert build.manifest["security"]["contains_raw_dataset"] is False
    assert build.manifest["security"]["contains_row_level_predictions"] is False
    assert "holdout_rows_data" not in build.manifest
    assert "prediction_rows" not in build.manifest
    assert set(_read_zip(build.package_bytes)) == ALLOWED_MEMBERS

    package = load_signed_model_package(build.package_bytes, signing_key=SIGNING_KEY)
    assert package.trusted is True
    assert package.task == "classification"
    assert package.target == "outcome"
    assert package.required_columns

    scoring = df.drop(columns=["outcome"]).head(24).copy(deep=True)
    scoring.loc[scoring.index[0], "segment"] = "NEW_UNSEEN_SEGMENT"
    before = scoring.copy(deep=True)
    schema = validate_prediction_frame(scoring, package, max_rows=1000)
    assert schema["valid"] is True, schema
    pred = run_batch_prediction(package, scoring, max_rows=1000, include_probabilities=True)
    assert len(pred.output) == len(scoring)
    assert pred.prediction_column in pred.output.columns
    assert set(pred.output[pred.prediction_column]).issubset(set(package.classes))
    assert any(name.startswith("Prediction_Confidence") for name in pred.probability_columns)
    pd.testing.assert_frame_equal(scoring, before)


def test_schema_gate_missing_and_unparseable_columns() -> None:
    df = _classification_frame()
    analysis, spec, result = _train(df, "outcome", "classification")
    build = create_signed_model_package(
        result, df, spec, semantic_profiles=analysis["profiles"], signing_key=SIGNING_KEY
    )
    package = load_signed_model_package(build.package_bytes, signing_key=SIGNING_KEY)

    missing = df.drop(columns=["outcome", "amount"]).head(10)
    report = validate_prediction_frame(missing, package, max_rows=100)
    assert report["valid"] is False
    assert "amount" in report["missing_columns"]

    invalid = df.drop(columns=["outcome"]).head(10).copy()
    invalid["amount"] = "not-a-number"
    report = validate_prediction_frame(invalid, package, max_rows=100)
    assert report["valid"] is False
    assert any("amount" in message for message in report["blockers"])


def test_tamper_wrong_signer_raw_pickle_and_path_traversal_are_rejected() -> None:
    df = _classification_frame()
    analysis, spec, result = _train(df, "outcome", "classification")
    build = create_signed_model_package(
        result, df, spec, semantic_profiles=analysis["profiles"], signing_key=SIGNING_KEY
    )

    try:
        load_signed_model_package(build.package_bytes, signing_key=OTHER_KEY)
        raise AssertionError("different signer should have been rejected")
    except ModelPackageError as exc:
        assert "different" in str(exc).lower()

    try:
        load_signed_model_package(pickle.dumps({"unsafe": True}), signing_key=SIGNING_KEY)
        raise AssertionError("raw pickle should have been rejected")
    except ModelPackageError as exc:
        assert "raw .pkl" in str(exc).lower()

    members = _read_zip(build.package_bytes)
    damaged = bytearray(members[MODEL_MEMBER])
    damaged[-1] ^= 0x01
    members[MODEL_MEMBER] = bytes(damaged)
    try:
        load_signed_model_package(_write_zip(members), signing_key=SIGNING_KEY)
        raise AssertionError("tampered model should have been rejected")
    except ModelPackageError as exc:
        assert "hash" in str(exc).lower()

    traversal = _read_zip(build.package_bytes)
    traversal["../outside.txt"] = b"unsafe"
    try:
        load_signed_model_package(_write_zip(traversal), signing_key=SIGNING_KEY)
        raise AssertionError("path traversal should have been rejected")
    except ModelPackageError as exc:
        assert "invalid package contents" in str(exc).lower() or "unsafe archive" in str(exc).lower()


def test_signature_is_verified_before_pickle_execution() -> None:
    df = _classification_frame()
    analysis, spec, result = _train(df, "outcome", "classification")
    build = create_signed_model_package(
        result, df, spec, semantic_profiles=analysis["profiles"], signing_key=SIGNING_KEY
    )
    members = _read_zip(build.package_bytes)

    with tempfile.TemporaryDirectory() as tmp:
        marker = Path(tmp) / "pickle_executed.txt"

        class Exploit:
            def __reduce__(self):
                code = f"from pathlib import Path; Path({str(marker)!r}).write_text('executed', encoding='utf-8')"
                return (builtins.exec, (code,))

        malicious = pickle.dumps(Exploit(), protocol=pickle.HIGHEST_PROTOCOL)
        manifest = json.loads(members[MANIFEST_MEMBER].decode("utf-8"))
        import hashlib

        manifest["artifact"]["sha256"] = hashlib.sha256(malicious).hexdigest()
        manifest["artifact"]["size_bytes"] = len(malicious)
        members[MANIFEST_MEMBER] = json.dumps(
            manifest, indent=2, sort_keys=True, ensure_ascii=False
        ).encode("utf-8")
        members[MODEL_MEMBER] = malicious
        # Keep the old signature intentionally. The hash now passes, but HMAC must
        # fail before pickle.loads can execute Exploit.
        try:
            load_signed_model_package(_write_zip(members), signing_key=SIGNING_KEY)
            raise AssertionError("invalid HMAC should have been rejected")
        except ModelPackageError as exc:
            assert "authentication" in str(exc).lower()
        assert not marker.exists(), "pickle executed before signature verification"


def test_regression_package_prediction_roundtrip() -> None:
    df = _regression_frame()
    analysis, spec, result = _train(df, "target_value", "regression")
    build = create_signed_model_package(
        result, df, spec, semantic_profiles=analysis["profiles"], signing_key=SIGNING_KEY
    )
    package = load_signed_model_package(build.package_bytes, signing_key=SIGNING_KEY)
    scoring = df.drop(columns=["target_value"]).tail(20).copy()
    prediction = run_batch_prediction(package, scoring, max_rows=100)
    assert prediction.task == "regression"
    assert prediction.output[prediction.prediction_column].notna().all()
    assert pd.api.types.is_numeric_dtype(prediction.output[prediction.prediction_column])


def main() -> None:
    test_classification_package_roundtrip_and_unknown_categories()
    test_schema_gate_missing_and_unparseable_columns()
    test_tamper_wrong_signer_raw_pickle_and_path_traversal_are_rejected()
    test_signature_is_verified_before_pickle_execution()
    test_regression_package_prediction_roundtrip()
    print(
        "PASS: Stage 9 signed model packages verify archive structure, SHA-256, local signer, and HMAC before deserialization; preserve the complete fitted preprocessing/model contract; reject raw or tampered pickle payloads; validate prediction schemas; handle unknown categories; and run classification/regression batch predictions without mutating source data."
    )


if __name__ == "__main__":
    main()
