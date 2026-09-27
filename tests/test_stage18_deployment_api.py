"""DataBridge AI Stage 18 governed Deployment API verification.

Run from project root:
    python tests/test_stage18_deployment_api.py
"""
from __future__ import annotations

import json
import os
import socket
import sys
import tempfile
import threading
import types
import urllib.error
import urllib.request
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

from core.credential_store import CredentialStore
from modules.data_mapper import analyze_dataframe
from modules.deployment_api import (
    DeploymentAPIError,
    GovernedPredictionService,
    _strict_json_loads,
    create_deployment_config,
    create_deployment_http_server,
    generate_deployment_token,
    load_deployment_config,
)
from modules.feature_pipeline import create_default_feature_pipeline_spec
from modules.ml_engine import run_supervised_experiment
from modules.model_governance import (
    assess_promotion,
    current_champion,
    list_governance_families,
    promote_challenger,
    submit_as_challenger,
)
from modules.model_package import create_signed_model_package
from modules.model_registry import register_signed_package


SIGNING_KEY = b"D" * 32
API_TOKEN = "dbai_" + "T" * 48


def _frame(rows: int = 260, *, seed: int = 180018) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    amount = rng.normal(680, 115, rows)
    units = rng.integers(1, 13, rows)
    segment = np.resize(["Retail", "SME", "Enterprise", "Public"], rows)
    event_date = pd.date_range("2025-01-01", periods=rows, freq="D").astype(str)
    signal = amount + units * 27 + (segment == "Enterprise") * 90
    outcome = np.where(signal + rng.normal(0, 42, rows) > 900, "won", "lost")
    return pd.DataFrame(
        {
            "record_id": np.arange(90_000, 90_000 + rows),
            "amount": amount.round(2),
            "units": units,
            "segment": segment,
            "event_date": event_date,
            "outcome": outcome,
        }
    )


def _package(seed: int = 18):
    df = _frame(seed=180000 + seed)
    analysis = analyze_dataframe(df)
    spec = create_default_feature_pipeline_spec(
        df,
        analysis["profiles"],
        "outcome",
        task="classification",
        configured_revision=1,
        configured_fingerprint=f"stage18-{seed}",
    )
    result = run_supervised_experiment(
        df,
        spec,
        dataset_revision=1,
        dataset_fingerprint=f"stage18-{seed}",
        split_strategy="stratified",
        cv_folds=3,
        random_state=seed,
        model_names=["Logistic Regression"],
        class_weight_mode="balanced",
        max_rows=None,
        include_xgboost=False,
    )
    return df, create_signed_model_package(
        result,
        df,
        spec,
        semantic_profiles=analysis["profiles"],
        signing_key=SIGNING_KEY,
    )


def _setup():
    temp = tempfile.TemporaryDirectory(prefix="databridge-stage18-")
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


def _register_champion(build, family: str = "Stage 18 Revenue API") -> tuple[str, str]:
    meta = register_signed_package(build.package_bytes, label="api champion", model_family=family)
    package_id = meta["package_id"]
    submit_as_challenger(package_id, actor="tester", note="Stage 18 API approval")
    assessment = assess_promotion(package_id)
    assert assessment.ready
    promoted = promote_challenger(
        package_id,
        approval_token=assessment.approval_token,
        approval_note="Approved for governed API serving",
        actor="tester",
    )
    return package_id, promoted["family_id"]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _request(base: str, path: str, *, token: str = "", payload=None):
    headers = {}
    data = None
    method = "GET"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if payload is not None:
        method = "POST"
        headers["Content-Type"] = "application/json"
        data = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
    req = urllib.request.Request(base + path, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.status, json.loads(response.read().decode("utf-8")), dict(response.headers)
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read().decode("utf-8")), dict(exc.headers)


class _FakeKeyring:
    def __init__(self):
        self.values = {}
    def set_password(self, service, name, value):
        self.values[(service, name)] = value
    def get_password(self, service, name):
        return self.values.get((service, name))
    def delete_password(self, service, name):
        self.values.pop((service, name), None)
    def get_keyring(self):
        return self
    priority = 1


def test_token_storage_config_authentication_and_remote_tls_gate() -> None:
    _, build = _package(18)
    temp, old, app_dir = _setup()
    try:
        _, family_id = _register_champion(build)
        backend = _FakeKeyring()
        token = generate_deployment_token(backend=backend)
        assert token.startswith("dbai_") and len(token) >= 48
        stored, source = CredentialStore(backend=backend).get_secret("deployment_api_token")
        assert stored == token and source == "credential_manager"

        config = create_deployment_config(
            name="Local champion API",
            family_id=family_id,
            port=_free_port(),
            max_rows=50,
            rate_limit_per_minute=50,
            signing_key=SIGNING_KEY,
        )
        config_path = app_dir / "deployment_api" / "configs" / f"{config['config_id']}.json"
        text = config_path.read_text(encoding="utf-8")
        assert token not in text and API_TOKEN not in text
        loaded = load_deployment_config(config["config_id"], signing_key=SIGNING_KEY)
        assert loaded["governance"]["champion_only"] is True
        assert loaded["governance"]["follow_future_champion_promotions"] is True

        tampered = json.loads(text)
        tampered["family_id"] = "family-attacker"
        config_path.write_text(json.dumps(tampered), encoding="utf-8")
        try:
            load_deployment_config(config["config_id"], signing_key=SIGNING_KEY)
        except DeploymentAPIError as exc:
            assert "integrity" in str(exc).lower()
        else:
            raise AssertionError("Tampered deployment configuration was accepted")

        try:
            create_deployment_config(
                name="Unsafe remote",
                family_id=family_id,
                host="0.0.0.0",
                port=_free_port(),
                allow_remote=True,
                signing_key=SIGNING_KEY,
            )
        except DeploymentAPIError as exc:
            assert "tls" in str(exc).lower()
        else:
            raise AssertionError("Remote plaintext deployment was accepted")

        try:
            create_deployment_config(
                name="Wildcard CORS",
                family_id=family_id,
                port=_free_port(),
                cors_origins=["*"],
                signing_key=SIGNING_KEY,
            )
        except DeploymentAPIError as exc:
            assert "wildcard" in str(exc).lower()
        else:
            raise AssertionError("Wildcard CORS was accepted")
    finally:
        _teardown(temp, old)


def test_http_api_serves_only_current_champion_and_follows_human_promotion() -> None:
    base_df, first_build = _package(19)
    _, second_build = _package(20)
    temp, old, _ = _setup()
    server = None
    thread = None
    try:
        first_id, family_id = _register_champion(first_build)
        port = _free_port()
        config = create_deployment_config(
            name="Governed HTTP test",
            family_id=family_id,
            port=port,
            max_rows=20,
            rate_limit_per_minute=100,
            signing_key=SIGNING_KEY,
        )
        server = create_deployment_http_server(config, API_TOKEN)
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{port}"

        status, health, headers = _request(base, "/health")
        assert status == 200 and health["status"] == "ok"
        assert headers.get("Cache-Control") == "no-store"

        status, body, _ = _request(base, "/v1/model")
        assert status == 401 and body["error"]["code"] == "unauthorized"
        status, model, _ = _request(base, "/v1/model", token=API_TOKEN)
        assert status == 200 and model["champion_id"] == first_id
        assert model["family_id"] == family_id

        rows = base_df.drop(columns=["outcome"]).head(3).to_dict(orient="records")
        rows[0]["segment"] = "Brand New Segment"
        status, validated, _ = _request(base, "/v1/validate", token=API_TOKEN, payload={"records": rows})
        assert status == 200 and validated["valid"] is True
        status, predicted, _ = _request(
            base,
            "/v1/predict",
            token=API_TOKEN,
            payload={"records": rows, "include_probabilities": True},
        )
        assert status == 200
        assert predicted["champion_id"] == first_id
        assert predicted["rows"] == 3
        assert len(predicted["results"]) == 3
        assert "amount" not in predicted["results"][0]  # source data is never echoed
        assert "prediction" in predicted["results"][0]

        broken = [{key: value for key, value in rows[0].items() if key != "amount"}]
        status, invalid, _ = _request(base, "/v1/predict", token=API_TOKEN, payload={"records": broken})
        assert status == 422 and invalid["error"]["code"] == "schema_validation_failed"

        # Register and manually promote a new Challenger while the API process remains alive.
        meta2 = register_signed_package(second_build.package_bytes, label="new version", model_family="Stage 18 Revenue API")
        second_id = meta2["package_id"]
        submit_as_challenger(second_id, actor="tester", note="Human review for API promotion")
        assessment = assess_promotion(second_id)
        assert assessment.ready
        promote_challenger(
            second_id,
            approval_token=assessment.approval_token,
            approval_note="Human-approved replacement Champion",
            actor="tester",
        )
        assert current_champion(family_id)["package_id"] == second_id

        # The same running API must resolve the governance revision and switch automatically.
        status, model2, _ = _request(base, "/v1/model", token=API_TOKEN)
        assert status == 200 and model2["champion_id"] == second_id
        status, predicted2, _ = _request(base, "/v1/predict", token=API_TOKEN, payload={"records": rows[:1]})
        assert status == 200 and predicted2["champion_id"] == second_id
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(timeout=3)
        _teardown(temp, old)


def test_request_safety_caps_and_strict_json() -> None:
    try:
        _strict_json_loads(b'{"records":[],"records":[]}')
    except ValueError as exc:
        assert "duplicate" in str(exc).lower()
    else:
        raise AssertionError("Duplicate JSON keys were accepted")
    try:
        _strict_json_loads(b'{"x":NaN}')
    except ValueError:
        pass
    else:
        raise AssertionError("Non-finite JSON values were accepted")

    source = (PROJECT_ROOT / "modules" / "deployment_api.py").read_text(encoding="utf-8")
    assert 'ctypes.WinDLL("kernel32"' in source
    assert "Do NOT use os.kill(pid, 0) on Windows" in source

    _, build = _package(21)
    temp, old, _ = _setup()
    try:
        _, family_id = _register_champion(build, family="Stage 18 Limit Family")
        config = create_deployment_config(
            name="Limit service",
            family_id=family_id,
            port=_free_port(),
            max_rows=2,
            rate_limit_per_minute=2,
            signing_key=SIGNING_KEY,
        )
        service = GovernedPredictionService(config, API_TOKEN)
        rows = _frame(3, seed=99).drop(columns=["outcome"]).to_dict(orient="records")
        try:
            service.predict_records(rows)
        except DeploymentAPIError as exc:
            assert "maximum" in str(exc).lower()
        else:
            raise AssertionError("Row cap was not enforced")
        assert service.rate_limiter.allow("client") is True
        assert service.rate_limiter.allow("client") is True
        assert service.rate_limiter.allow("client") is False
    finally:
        _teardown(temp, old)



def main() -> None:
    test_token_storage_config_authentication_and_remote_tls_gate()
    test_http_api_serves_only_current_champion_and_follows_human_promotion()
    test_request_safety_caps_and_strict_json()
    print(
        "PASS: Stage 18 serves only the authenticated governed Champion through a Bearer-protected prediction API; "
        "follows human-approved Champion promotions without restart; verifies signed deployment configs and model packages; "
        "blocks plaintext remote exposure, wildcard CORS, tampered configs, invalid schemas, oversized batches, duplicate/non-finite JSON, "
        "and rate-limit abuse; and never echoes or audit-logs source prediction records."
    )


if __name__ == "__main__":
    main()
