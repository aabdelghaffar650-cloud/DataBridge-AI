"""DataBridge AI Stage 12 Security Finalization verification.

Run from project root:
    python tests/test_stage12_security_finalization.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import types
from pathlib import Path

import pandas as pd


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
fake_streamlit.cache_data = lambda *args, **kwargs: (args[0] if args and callable(args[0]) else (lambda func: func))
sys.modules.setdefault("streamlit", fake_streamlit)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ai.base import AIEngineStrategy  # noqa: E402
from ai.orchestrator import AIPrivacyError, DataBridgeAIEngine  # noqa: E402
from core.audit import audit_csv_bytes, audit_jsonl_bytes, sanitise_audit_event  # noqa: E402
from core.credential_store import CredentialStore  # noqa: E402
from core.security import (  # noqa: E402
    anonymise_df_for_ai,
    is_loopback_url,
    redact_sensitive_text,
    safe_html,
    validate_http_endpoint,
)
from modules.model_package import (  # noqa: E402
    ModelPackageError,
    create_encrypted_signing_key_backup,
    get_or_create_signing_key,
    restore_encrypted_signing_key_backup,
    rotate_signing_key,
    signing_key_id,
)


class MemoryCredentialBackend:
    def __init__(self):
        self.values = {}

    def get_password(self, service, username):
        return self.values.get((service, username))

    def set_password(self, service, username, password):
        self.values[(service, username)] = password

    def delete_password(self, service, username):
        del self.values[(service, username)]


class CaptureStrategy(AIEngineStrategy):
    def __init__(self, engine_type: str):
        self.engine_type = engine_type
        self.context = ""

    def generate_insights(self, context: str, prompt: str, history: list) -> str:
        self.context = context
        return "ok"

    def get_engine_type(self) -> str:
        return self.engine_type


def test_os_credential_wrapper_has_no_plaintext_fallback() -> None:
    backend = MemoryCredentialBackend()
    store = CredentialStore(backend=backend)
    store.set_secret("anthropic_api_key", "sk-ant-stage12-test-secret")
    value, source = store.get_secret("anthropic_api_key")
    assert value == "sk-ant-stage12-test-secret"
    assert source == "credential_manager"
    assert store.status("anthropic_api_key").configured is True
    assert store.delete_secret("anthropic_api_key") is True
    assert store.get_secret("anthropic_api_key")[0] is None


def _privacy_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_name": ["Ahmed Hassan", "Mona Ali"],
            "contact": ["ahmed@example.com", "+20 100 123 4567"],
            "notes": [
                "Card 4111 1111 1111 1111 from 192.168.1.10",
                "safe note",
            ],
            "amount": [10.0, 20.0],
        }
    )


def test_pii_masking_is_value_aware_and_non_mutating() -> None:
    frame = _privacy_frame()
    before = frame.copy(deep=True)
    masked, report = anonymise_df_for_ai(frame, return_report=True)
    pd.testing.assert_frame_equal(frame, before)
    assert set(report.masked_columns) >= {"customer_name", "contact"}
    combined = " ".join(masked.astype(str).to_numpy().reshape(-1))
    assert "Ahmed Hassan" not in combined
    assert "ahmed@example.com" not in combined
    assert "4111 1111 1111 1111" not in combined
    assert "192.168.1.10" not in combined
    assert report.value_redactions >= 4


def test_ai_privacy_modes_and_remote_endpoint_guard() -> None:
    frame = _privacy_frame()

    metadata_strategy = CaptureStrategy("cloud")
    metadata_engine = DataBridgeAIEngine(metadata_strategy, privacy_mode="metadata")
    assert metadata_engine.process_task(frame, "summary", []) == "ok"
    assert "Ahmed Hassan" not in metadata_strategy.context
    assert "Not included in metadata-only mode" in metadata_strategy.context

    masked_strategy = CaptureStrategy("cloud")
    masked_engine = DataBridgeAIEngine(masked_strategy, privacy_mode="masked")
    masked_engine.process_task(frame, "summary", [])
    assert "Ahmed Hassan" not in masked_strategy.context
    assert "ahmed@example.com" not in masked_strategy.context
    assert "***REDACTED***" in masked_strategy.context

    raw_strategy = CaptureStrategy("cloud")
    try:
        DataBridgeAIEngine(raw_strategy, privacy_mode="raw").process_task(frame, "summary", [])
    except AIPrivacyError:
        pass
    else:
        raise AssertionError("Raw cloud transfer was not blocked without confirmation.")

    approved = DataBridgeAIEngine(
        raw_strategy,
        privacy_mode="raw",
        raw_data_confirmed=True,
    )
    approved.process_task(frame, "summary", [])
    assert "Ahmed Hassan" in raw_strategy.context

    remote_strategy = CaptureStrategy("remote")
    try:
        DataBridgeAIEngine(remote_strategy, privacy_mode="metadata").process_task(frame, "summary", [])
    except AIPrivacyError:
        pass
    else:
        raise AssertionError("Remote endpoint was not blocked without approval.")
    DataBridgeAIEngine(
        remote_strategy,
        privacy_mode="metadata",
        allow_remote_endpoint=True,
    ).process_task(frame, "summary", [])


def test_endpoint_validation_and_secret_redaction() -> None:
    assert is_loopback_url("http://localhost:11434") is True
    assert is_loopback_url("http://127.0.0.1:11434") is True
    assert is_loopback_url("http://10.0.0.5:11434") is False
    assert validate_http_endpoint("https://ollama.example.com/") == "https://ollama.example.com"
    try:
        validate_http_endpoint("http://user:pass@example.com")
    except ValueError:
        pass
    else:
        raise AssertionError("Embedded URL credentials were not rejected.")

    text = "key=sk-ant-supersecretvalue https://user:pass@db.example/x bearer abcdefghijklmnop"
    redacted = redact_sensitive_text(text)
    assert "supersecretvalue" not in redacted
    assert "user:pass" not in redacted
    assert "abcdefghijklmnop" not in redacted
    assert safe_html('<img src=x onerror="alert(1)">').startswith("&lt;img")


def test_audit_export_redacts_fields_and_token_patterns() -> None:
    event = {
        "event": "connection",
        "api_key": "AIzaThisMustNeverAppear1234567890",
        "details": {
            "connection_url": "postgresql://admin:secret@localhost/db",
            "message": "Bearer abcdefghijklmnop",
        },
    }
    clean = sanitise_audit_event(event)
    encoded = json.dumps(clean)
    assert "ThisMustNeverAppear" not in encoded
    assert "admin:secret" not in encoded
    assert "abcdefghijklmnop" not in encoded
    assert clean["api_key"] == "***REDACTED***"
    assert b"ThisMustNeverAppear" not in audit_jsonl_bytes([event])
    assert b"ThisMustNeverAppear" not in audit_csv_bytes([event])


def test_encrypted_signing_key_backup_restore_and_rotation() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        key_path = Path(temp_dir) / "model_package_signing.key"
        key = get_or_create_signing_key(key_path)
        key_id = signing_key_id(key)
        backup = create_encrypted_signing_key_backup(
            "stage12-strong-backup-passphrase",
            key_path=key_path,
        )
        assert key not in backup
        assert b"stage12-strong-backup-passphrase" not in backup

        try:
            restore_encrypted_signing_key_backup(
                backup,
                "wrong-passphrase-value",
                overwrite=True,
                key_path=Path(temp_dir) / "wrong.key",
            )
        except ModelPackageError:
            pass
        else:
            raise AssertionError("Wrong backup passphrase was accepted.")

        restored_path = Path(temp_dir) / "restored.key"
        result = restore_encrypted_signing_key_backup(
            backup,
            "stage12-strong-backup-passphrase",
            key_path=restored_path,
        )
        assert result["restored"] is True
        assert signing_key_id(restored_path.read_bytes()) == key_id

        rotation = rotate_signing_key(key_id, key_path=key_path)
        assert rotation["old_key_id"] == key_id
        assert rotation["new_key_id"] != key_id
        assert Path(rotation["archived_key_path"]).exists()
        assert key_path.exists()


def test_tauri_hardening_and_no_remote_font_import() -> None:
    config_path = PROJECT_ROOT / "desktop" / "src-tauri" / "tauri.conf.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    security = config["app"]["security"]
    assert security["csp"] is not None
    assert security["csp"]["object-src"] == "'none'"
    assert security["headers"]["X-Content-Type-Options"] == "nosniff"
    cargo = (PROJECT_ROOT / "desktop" / "src-tauri" / "Cargo.toml").read_text(encoding="utf-8")
    rust = (PROJECT_ROOT / "desktop" / "src-tauri" / "src" / "main.rs").read_text(encoding="utf-8")
    styles = (PROJECT_ROOT / "ui" / "styles.py").read_text(encoding="utf-8")
    assert "tauri-plugin-shell" not in cargo
    assert "tauri_plugin_shell" not in rust
    assert "fonts.googleapis.com" not in styles


def main() -> None:
    test_os_credential_wrapper_has_no_plaintext_fallback()
    test_pii_masking_is_value_aware_and_non_mutating()
    test_ai_privacy_modes_and_remote_endpoint_guard()
    test_endpoint_validation_and_secret_redaction()
    test_audit_export_redacts_fields_and_token_patterns()
    test_encrypted_signing_key_backup_restore_and_rotation()
    test_tauri_hardening_and_no_remote_font_import()
    print(
        "PASS: Stage 12 Security Finalization stores AI secrets through the OS credential backend without plaintext fallback; enforces metadata/masked/raw transfer policy and remote Ollama approval; masks PII values without mutating source data; redacts audit/error output; protects signing-key backup, restore, and rotation; escapes HTML; and hardens the Tauri desktop shell."
    )


if __name__ == "__main__":
    main()
