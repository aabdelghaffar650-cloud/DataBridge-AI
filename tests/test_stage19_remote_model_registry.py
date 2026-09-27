"""DataBridge AI Stage 19 Remote Model Registry verification.

Run from project root:
    python tests/test_stage19_remote_model_registry.py
"""
from __future__ import annotations

import os
import socket
import sys
import tempfile
import threading
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
    assess_promotion,
    current_champion,
    get_governance_state,
    promote_challenger,
    submit_as_challenger,
)
from modules.model_package import create_signed_model_package
from modules.model_registry import list_registered_packages, register_signed_package
from modules.remote_model_registry import (
    RemoteRegistryClient,
    RemoteRegistryError,
    client_from_profile,
    create_remote_registry_client_profile,
    create_remote_registry_http_server,
    create_remote_registry_server_config,
    generate_remote_registry_token,
    pull_family_from_remote,
    push_family_to_remote,
)

SIGNING_KEY = b"R" * 32


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


def _frame(seed: int, rows: int = 220) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    amount = rng.normal(700, 120, rows)
    units = rng.integers(1, 14, rows)
    segment = np.resize(["Retail", "SME", "Enterprise", "Public"], rows)
    signal = amount + units * 24 + (segment == "Enterprise") * 90
    outcome = np.where(signal + rng.normal(0, 40, rows) > 900, "won", "lost")
    return pd.DataFrame(
        {
            "record_id": np.arange(30_000, 30_000 + rows),
            "amount": amount.round(2),
            "units": units,
            "segment": segment,
            "event_date": pd.date_range("2025-01-01", periods=rows, freq="D").astype(str),
            "outcome": outcome,
        }
    )


def _build(seed: int):
    df = _frame(seed)
    analysis = analyze_dataframe(df)
    spec = create_default_feature_pipeline_spec(
        df,
        analysis["profiles"],
        "outcome",
        task="classification",
        configured_revision=1,
        configured_fingerprint=f"stage19-{seed}",
    )
    result = run_supervised_experiment(
        df,
        spec,
        dataset_revision=1,
        dataset_fingerprint=f"stage19-{seed}",
        split_strategy="stratified",
        cv_folds=3,
        random_state=seed,
        model_names=["Logistic Regression"],
        max_rows=None,
        include_xgboost=False,
    )
    return create_signed_model_package(
        result,
        df,
        spec,
        semantic_profiles=analysis["profiles"],
        signing_key=SIGNING_KEY,
    )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _activate_root(root: Path) -> None:
    os.environ["DATABRIDGE_USER_DATA_DIR"] = str(root)
    app = root / "DataBridgeAI"
    app.mkdir(parents=True, exist_ok=True)
    (app / "model_package_signing.key").write_bytes(SIGNING_KEY)


def _register_first_champion(build, family: str) -> tuple[str, str]:
    meta = register_signed_package(build.package_bytes, label="v1", model_family=family)
    package_id = meta["package_id"]
    submit_as_challenger(package_id, actor="stage19-test", note="Initial central Champion")
    assessment = assess_promotion(package_id)
    assert assessment.ready
    result = promote_challenger(
        package_id,
        approval_token=assessment.approval_token,
        approval_note="Approved for Stage 19 registry test",
        actor="stage19-test",
    )
    return package_id, result["family_id"]


def _promote_new(build, family: str) -> str:
    meta = register_signed_package(build.package_bytes, label="new", model_family=family)
    package_id = meta["package_id"]
    submit_as_challenger(package_id, actor="stage19-test", note="New version")
    assessment = assess_promotion(package_id)
    assert assessment.ready
    promote_challenger(
        package_id,
        approval_token=assessment.approval_token,
        approval_note="Human-approved replacement",
        actor="stage19-test",
    )
    return package_id


def main() -> None:
    original_root = os.environ.get("DATABRIDGE_USER_DATA_DIR")
    source = tempfile.TemporaryDirectory(prefix="databridge-stage19-source-")
    target = tempfile.TemporaryDirectory(prefix="databridge-stage19-target-")
    wrong = tempfile.TemporaryDirectory(prefix="databridge-stage19-wrong-")
    server_data = tempfile.TemporaryDirectory(prefix="databridge-stage19-server-")
    backend = _FakeKeyring()
    server = None
    thread = None
    try:
        _activate_root(Path(source.name))
        first_build = _build(19)
        second_build = _build(20)
        third_build = _build(21)
        first_id, family_id = _register_first_champion(first_build, "Stage 19 Revenue Family")

        token = generate_remote_registry_token(backend=backend)
        port = _free_port()
        config = create_remote_registry_server_config(
            name="Stage 19 central registry",
            host="127.0.0.1",
            port=port,
            signing_key=SIGNING_KEY,
        )
        server = create_remote_registry_http_server(
            config,
            token,
            signing_key=SIGNING_KEY,
            storage_root=Path(server_data.name),
        )
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        thread.start()
        base_url = f"http://127.0.0.1:{port}"

        source_profile = create_remote_registry_client_profile(
            name="Stage19 source",
            base_url=base_url,
            signing_key=SIGNING_KEY,
        )
        client = RemoteRegistryClient(source_profile, token)
        health = client.health()
        assert health["status"] == "ok"
        assert health["signer_key_id"]
        pushed = push_family_to_remote(source_profile["profile_id"], family_id, backend=backend)
        assert pushed.packages_transferred == 1
        assert pushed.champion_id == first_id
        assert client.list_families()[0]["champion_id"] == first_id
        assert len(client.list_packages()) == 1

        # A second trusted installation with the SAME model trust key can pull the
        # signed package and exact governance envelope, preserving Champion state.
        _activate_root(Path(target.name))
        target_profile = create_remote_registry_client_profile(
            name="Stage19 target",
            base_url=base_url,
            signing_key=SIGNING_KEY,
        )
        pulled = pull_family_from_remote(target_profile["profile_id"], family_id, backend=backend)
        assert pulled.packages_transferred == 1
        assert current_champion(family_id)["package_id"] == first_id
        assert len(list_registered_packages()) == 1

        # Governance can advance on the trusted target and fast-forward centrally.
        second_id = _promote_new(second_build, "Stage 19 Revenue Family")
        pushed2 = push_family_to_remote(target_profile["profile_id"], family_id, backend=backend)
        assert pushed2.champion_id == second_id
        assert pushed2.packages_transferred == 1

        # A stale source may not silently overwrite the newer central history.
        _activate_root(Path(source.name))
        source_profile2 = create_remote_registry_client_profile(
            name="Stage19 source",
            base_url=base_url,
            signing_key=SIGNING_KEY,
        )
        register_signed_package(third_build.package_bytes, label="divergent", model_family="Stage 19 Revenue Family")
        try:
            push_family_to_remote(source_profile2["profile_id"], family_id, backend=backend)
        except RemoteRegistryError as exc:
            assert "fast-forward" in str(exc).lower() or "pull" in str(exc).lower()
        else:
            raise AssertionError("Divergent stale governance was allowed to overwrite the central registry")
        try:
            pull_family_from_remote(source_profile2["profile_id"], family_id, backend=backend)
        except RemoteRegistryError as exc:
            assert "fast-forward" in str(exc).lower() or "silent" in str(exc).lower()
        else:
            raise AssertionError("Divergent local governance was silently overwritten by remote state")

        # Different installations cannot join merely by knowing the Bearer token;
        # artifact/governance trust also requires the shared model signing key.
        os.environ["DATABRIDGE_USER_DATA_DIR"] = wrong.name
        wrong_app = Path(wrong.name) / "DataBridgeAI"
        wrong_app.mkdir(parents=True, exist_ok=True)
        (wrong_app / "model_package_signing.key").write_bytes(b"X" * 32)
        wrong_profile = create_remote_registry_client_profile(
            name="Wrong trust",
            base_url=base_url,
            signing_key=b"X" * 32,
        )
        try:
            client_from_profile(wrong_profile["profile_id"], backend=backend)
        except RemoteRegistryError as exc:
            assert "different model trust key" in str(exc).lower()
        else:
            raise AssertionError("Bearer token alone bypassed the shared model trust key requirement")

        # Plaintext remote exposure is blocked both for server and client profiles.
        try:
            create_remote_registry_server_config(
                name="Unsafe",
                host="0.0.0.0",
                port=_free_port(),
                allow_remote=True,
                signing_key=b"X" * 32,
            )
        except RemoteRegistryError as exc:
            assert "tls" in str(exc).lower()
        else:
            raise AssertionError("Non-loopback registry server was allowed without TLS")
        try:
            create_remote_registry_client_profile(
                name="Unsafe remote",
                base_url="http://192.168.1.20:8890",
                signing_key=b"X" * 32,
            )
        except RemoteRegistryError as exc:
            assert "https" in str(exc).lower()
        else:
            raise AssertionError("Plain HTTP remote registry client profile was accepted")

        print(
            "PASS: Stage 19 Remote Model Registry requires Bearer authentication plus a shared model trust key; "
            "verifies signed packages and authenticated Champion/Challenger governance centrally; synchronizes families across trusted installations; "
            "uses optimistic fast-forward conflict protection so stale/divergent governance is never silently overwritten; and blocks non-loopback plaintext transport."
        )
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(timeout=3)
        source.cleanup()
        target.cleanup()
        wrong.cleanup()
        server_data.cleanup()
        if original_root is None:
            os.environ.pop("DATABRIDGE_USER_DATA_DIR", None)
        else:
            os.environ["DATABRIDGE_USER_DATA_DIR"] = original_root


if __name__ == "__main__":
    main()
