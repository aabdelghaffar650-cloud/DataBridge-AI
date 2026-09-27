# ════════════════════════════════════════════════════════
# DataBridge AI — Governed Deployment API
# Stage 18: Champion-only authenticated prediction service
# ════════════════════════════════════════════════════════
from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import math
import os
import re
import secrets
import signal
import ssl
import subprocess
import sys
import tempfile
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from core.audit import write_persistent_audit_event
from core.credential_store import SecretStoreError, get_credential_store
from core.security import safe_error_message
from core.user_paths import deployment_api_dir
from modules.model_governance import ModelGovernanceError, current_champion
from modules.model_package import (
    LoadedModelPackage,
    ModelPackageError,
    load_signed_model_package,
    run_batch_prediction,
    validate_prediction_frame,
    get_or_create_signing_key,
)
from modules.model_registry import ModelRegistryError, load_registered_package_bytes


DEPLOYMENT_CONFIG_FORMAT = "DataBridgeAI Governed Deployment API"
DEPLOYMENT_CONFIG_VERSION = 1
DEPLOYMENT_CONTEXT = b"DataBridgeAI-Deployment-API-v1\0"
DEPLOYMENT_TOKEN_SECRET = "deployment_api_token"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_MAX_ROWS = 10_000
MAX_MAX_ROWS = 50_000
DEFAULT_MAX_BODY_BYTES = 8 * 1024 * 1024
MAX_BODY_BYTES_HARD_CAP = 32 * 1024 * 1024
DEFAULT_RATE_LIMIT_PER_MINUTE = 120
MAX_RATE_LIMIT_PER_MINUTE = 5_000
MAX_CONFIG_BYTES = 256 * 1024
MAX_COLUMNS = 500
MAX_STRING_CHARS = 65_536
MAX_CORS_ORIGINS = 20


class DeploymentAPIError(ValueError):
    """Raised when a deployment configuration or request is unsafe."""


@dataclass(frozen=True)
class DeploymentProcessStatus:
    config_id: str
    running: bool
    pid: Optional[int]
    scheme: str
    host: str
    port: int
    family_id: str
    started_at: str
    last_error: str = ""

    @property
    def base_url(self) -> str:
        host = "127.0.0.1" if self.host == "0.0.0.0" else self.host
        return f"{self.scheme}://{host}:{self.port}"


@dataclass(frozen=True)
class _ChampionRuntime:
    family_id: str
    package_id: str
    package: LoadedModelPackage
    loaded_at: float
    governance_revision: int


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_text(value: Any, *, limit: int = 500) -> str:
    return str(value or "").replace("\x00", " ").strip()[:limit]


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _config_hmac(payload: Mapping[str, Any], key: bytes) -> str:
    return hmac.new(key, DEPLOYMENT_CONTEXT + _canonical_json_bytes(payload), hashlib.sha256).hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
        try:
            os.chmod(temp_name, 0o600)
        except OSError:
            pass
        os.replace(temp_name, path)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    finally:
        try:
            Path(temp_name).unlink(missing_ok=True)
        except Exception:
            pass


def _safe_config_id(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "")).strip("._-")
    if not clean:
        raise DeploymentAPIError("Invalid deployment configuration identifier.")
    return clean[:120]


def _configs_dir() -> Path:
    path = deployment_api_dir() / "configs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _runtime_dir() -> Path:
    path = deployment_api_dir() / "runtime"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _config_path(config_id: str) -> Path:
    return _configs_dir() / f"{_safe_config_id(config_id)}.json"


def _status_path(config_id: str) -> Path:
    return _runtime_dir() / f"{_safe_config_id(config_id)}.json"


def _is_loopback_host(host: str) -> bool:
    text = str(host or "").strip().casefold()
    if text == "localhost":
        return True
    try:
        return ipaddress.ip_address(text).is_loopback
    except ValueError:
        return False


def _validate_bind_host(host: str, *, allow_remote: bool) -> str:
    clean = str(host or DEFAULT_HOST).strip()
    if clean.casefold() == "localhost":
        clean = DEFAULT_HOST
    try:
        ip = ipaddress.ip_address(clean)
    except ValueError as exc:
        raise DeploymentAPIError("Deployment host must be an explicit IPv4 address or localhost.") from exc
    if ip.version != 4:
        raise DeploymentAPIError("Stage 18 deployment currently supports IPv4 bind addresses only.")
    if ip.is_loopback:
        return str(ip)
    if not allow_remote:
        raise DeploymentAPIError("Non-loopback binding requires explicit Remote API approval.")
    return str(ip)


def _validate_port(port: int) -> int:
    try:
        value = int(port)
    except Exception as exc:
        raise DeploymentAPIError("Deployment port must be an integer.") from exc
    if value < 1024 or value > 65535:
        raise DeploymentAPIError("Deployment port must be between 1024 and 65535.")
    return value


def _validate_local_file(path_value: str, *, label: str) -> str:
    text = str(path_value or "").strip()
    if not text:
        raise DeploymentAPIError(f"{label} is required for remote TLS deployment.")
    if text.startswith("\\\\") or text.startswith("//"):
        raise DeploymentAPIError(f"{label} must be a local file, not a network/UNC path.")
    path = Path(text).expanduser()
    if not path.is_absolute():
        raise DeploymentAPIError(f"{label} must use an absolute local path.")
    if not path.is_file():
        raise DeploymentAPIError(f"{label} does not exist.")
    if path.stat().st_size <= 0 or path.stat().st_size > 4 * 1024 * 1024:
        raise DeploymentAPIError(f"{label} has an unsafe file size.")
    return str(path.resolve())


def _normalise_cors_origins(origins: Iterable[str] | None) -> list[str]:
    values: list[str] = []
    for item in origins or []:
        text = str(item or "").strip()
        if not text:
            continue
        if text == "*":
            raise DeploymentAPIError("Wildcard CORS origins are not allowed for an authenticated prediction API.")
        if not re.fullmatch(r"https?://[^\s/]+(?::\d{1,5})?", text, flags=re.IGNORECASE):
            raise DeploymentAPIError(f"Invalid CORS origin: {text}")
        if text not in values:
            values.append(text)
        if len(values) > MAX_CORS_ORIGINS:
            raise DeploymentAPIError(f"At most {MAX_CORS_ORIGINS} CORS origins are allowed.")
    return values


def create_deployment_config(
    *,
    name: str,
    family_id: str,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    allow_remote: bool = False,
    tls_cert_path: str = "",
    tls_key_path: str = "",
    cors_origins: Sequence[str] | None = None,
    max_rows: int = DEFAULT_MAX_ROWS,
    max_body_bytes: int = DEFAULT_MAX_BODY_BYTES,
    rate_limit_per_minute: int = DEFAULT_RATE_LIMIT_PER_MINUTE,
    signing_key: Optional[bytes] = None,
) -> Dict[str, Any]:
    """Create an authenticated deployment configuration without storing API tokens."""
    family = _safe_text(family_id, limit=120)
    if not family:
        raise DeploymentAPIError("A governed model family is required.")
    champion = current_champion(family, signing_key=signing_key)
    if not champion:
        raise DeploymentAPIError("The selected model family has no approved Champion.")

    bind_host = _validate_bind_host(host, allow_remote=bool(allow_remote))
    bind_port = _validate_port(port)
    remote = not _is_loopback_host(bind_host)
    cert = ""
    key_path = ""
    if remote:
        if not allow_remote:
            raise DeploymentAPIError("Remote deployment requires explicit approval.")
        cert = _validate_local_file(tls_cert_path, label="TLS certificate")
        key_path = _validate_local_file(tls_key_path, label="TLS private key")
        if Path(cert).resolve() == Path(key_path).resolve():
            raise DeploymentAPIError("TLS certificate and private-key files must be distinct.")
    elif tls_cert_path or tls_key_path:
        # Loopback TLS is allowed, but both files must be present together.
        cert = _validate_local_file(tls_cert_path, label="TLS certificate")
        key_path = _validate_local_file(tls_key_path, label="TLS private key")

    try:
        rows_cap = int(max_rows)
        body_cap = int(max_body_bytes)
        rate_cap = int(rate_limit_per_minute)
    except Exception as exc:
        raise DeploymentAPIError("Deployment safety limits must be integers.") from exc
    if rows_cap < 1 or rows_cap > MAX_MAX_ROWS:
        raise DeploymentAPIError(f"Prediction row cap must be between 1 and {MAX_MAX_ROWS:,}.")
    if body_cap < 64 * 1024 or body_cap > MAX_BODY_BYTES_HARD_CAP:
        raise DeploymentAPIError(
            f"Request-body cap must be between 64 KB and {MAX_BODY_BYTES_HARD_CAP // (1024 * 1024)} MB."
        )
    if rate_cap < 1 or rate_cap > MAX_RATE_LIMIT_PER_MINUTE:
        raise DeploymentAPIError(
            f"Rate limit must be between 1 and {MAX_RATE_LIMIT_PER_MINUTE:,} requests/minute."
        )

    allowed_origins = _normalise_cors_origins(cors_origins)
    config_id = secrets.token_hex(12)
    payload: Dict[str, Any] = {
        "format": DEPLOYMENT_CONFIG_FORMAT,
        "version": DEPLOYMENT_CONFIG_VERSION,
        "config_id": config_id,
        "name": _safe_text(name or "Champion Prediction API", limit=120),
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "family_id": family,
        "bind": {
            "host": bind_host,
            "port": bind_port,
            "allow_remote": bool(allow_remote),
        },
        "tls": {
            "enabled": bool(cert and key_path),
            "cert_path": cert,
            "key_path": key_path,
        },
        "cors_origins": allowed_origins,
        "limits": {
            "max_rows": rows_cap,
            "max_body_bytes": body_cap,
            "rate_limit_per_minute": rate_cap,
        },
        "governance": {
            "champion_only": True,
            "configured_champion_id": str(champion.get("package_id", "")),
            "configured_revision": int(champion.get("governance_revision", 0)),
            "follow_future_champion_promotions": True,
        },
    }
    secret = bytes(signing_key) if signing_key is not None else get_or_create_signing_key()
    envelope = dict(payload)
    envelope["hmac_sha256"] = _config_hmac(payload, secret)
    data = _canonical_json_bytes(envelope)
    if len(data) > MAX_CONFIG_BYTES:
        raise DeploymentAPIError("Deployment configuration is unexpectedly large.")
    _atomic_write(_config_path(config_id), data)
    write_persistent_audit_event(
        {
            "event": "deployment_api_config_created",
            "config_id": config_id,
            "family_id": family,
            "remote": remote,
            "tls_enabled": bool(cert and key_path),
            "port": bind_port,
        }
    )
    return payload


def load_deployment_config(
    config_id: str,
    *,
    signing_key: Optional[bytes] = None,
) -> Dict[str, Any]:
    path = _config_path(config_id)
    if not path.is_file():
        raise DeploymentAPIError("Deployment configuration not found.")
    if path.stat().st_size <= 0 or path.stat().st_size > MAX_CONFIG_BYTES:
        raise DeploymentAPIError("Deployment configuration has an unsafe size.")
    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(envelope, dict):
            raise ValueError
        signature = str(envelope.pop("hmac_sha256"))
    except Exception as exc:
        raise DeploymentAPIError("Deployment configuration format is invalid.") from exc
    if envelope.get("format") != DEPLOYMENT_CONFIG_FORMAT or int(envelope.get("version", 0)) != DEPLOYMENT_CONFIG_VERSION:
        raise DeploymentAPIError("Deployment configuration format/version is not supported.")
    if str(envelope.get("config_id", "")) != _safe_config_id(config_id):
        raise DeploymentAPIError("Deployment configuration identifier mismatch.")
    key = bytes(signing_key) if signing_key is not None else get_or_create_signing_key()
    expected = _config_hmac(envelope, key)
    if not hmac.compare_digest(signature, expected):
        raise DeploymentAPIError("Deployment configuration failed integrity verification.")

    bind = envelope.get("bind") or {}
    host = _validate_bind_host(str(bind.get("host", "")), allow_remote=bool(bind.get("allow_remote")))
    _validate_port(int(bind.get("port", 0)))
    remote = not _is_loopback_host(host)
    tls = envelope.get("tls") or {}
    if remote and not bool(tls.get("enabled")):
        raise DeploymentAPIError("Remote deployment configuration is invalid because TLS is disabled.")
    if bool(tls.get("enabled")):
        _validate_local_file(str(tls.get("cert_path", "")), label="TLS certificate")
        _validate_local_file(str(tls.get("key_path", "")), label="TLS private key")
    _normalise_cors_origins(envelope.get("cors_origins") or [])

    limits = envelope.get("limits") or {}
    max_rows = int(limits.get("max_rows", 0))
    max_body = int(limits.get("max_body_bytes", 0))
    rate = int(limits.get("rate_limit_per_minute", 0))
    if not (1 <= max_rows <= MAX_MAX_ROWS):
        raise DeploymentAPIError("Deployment row cap failed verification.")
    if not (64 * 1024 <= max_body <= MAX_BODY_BYTES_HARD_CAP):
        raise DeploymentAPIError("Deployment body cap failed verification.")
    if not (1 <= rate <= MAX_RATE_LIMIT_PER_MINUTE):
        raise DeploymentAPIError("Deployment rate limit failed verification.")
    if not bool((envelope.get("governance") or {}).get("champion_only")):
        raise DeploymentAPIError("Deployment configuration does not enforce Champion-only serving.")
    return envelope


def list_deployment_configs(*, signing_key: Optional[bytes] = None) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for path in _configs_dir().glob("*.json"):
        try:
            row = load_deployment_config(path.stem, signing_key=signing_key)
            status = read_deployment_status(row["config_id"])
            row["runtime"] = {
                "running": status.running,
                "pid": status.pid,
                "scheme": status.scheme,
                "base_url": status.base_url,
                "last_error": status.last_error,
            }
            rows.append(row)
        except Exception:
            continue
    rows.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
    return rows


def delete_deployment_config(config_id: str) -> None:
    status = read_deployment_status(config_id)
    if status.running:
        raise DeploymentAPIError("Stop the deployment API before deleting its configuration.")
    _config_path(config_id).unlink(missing_ok=True)
    _status_path(config_id).unlink(missing_ok=True)
    write_persistent_audit_event({"event": "deployment_api_config_deleted", "config_id": _safe_config_id(config_id)})


def generate_deployment_token(*, backend: Any | None = None) -> str:
    """Generate and persist a high-entropy bearer token. Return it once to the caller."""
    token = "dbai_" + secrets.token_urlsafe(36)
    store = get_credential_store() if backend is None else None
    if backend is None:
        assert store is not None
        store.set_secret(DEPLOYMENT_TOKEN_SECRET, token)
    else:
        # Test hook: construct the same CredentialStore without importing internals in tests.
        from core.credential_store import CredentialStore
        CredentialStore(backend=backend).set_secret(DEPLOYMENT_TOKEN_SECRET, token)
    write_persistent_audit_event({"event": "deployment_api_token_rotated"})
    return token


def deployment_token_status() -> Dict[str, Any]:
    status = get_credential_store().status(DEPLOYMENT_TOKEN_SECRET)
    return {
        "configured": bool(status.configured),
        "source": status.source,
        "backend": status.backend,
    }


def _load_deployment_token() -> str:
    try:
        token, source = get_credential_store().get_secret(DEPLOYMENT_TOKEN_SECRET)
    except SecretStoreError as exc:
        raise DeploymentAPIError("Secure deployment API credential storage is unavailable.") from exc
    token = str(token or "").strip()
    if len(token) < 32:
        raise DeploymentAPIError("Deployment API token is not configured or is too short.")
    return token


def _status_payload(config: Mapping[str, Any], *, pid: int, running: bool, last_error: str = "") -> Dict[str, Any]:
    tls = config.get("tls") or {}
    bind = config.get("bind") or {}
    return {
        "config_id": str(config.get("config_id", "")),
        "pid": int(pid) if pid else None,
        "running": bool(running),
        "scheme": "https" if bool(tls.get("enabled")) else "http",
        "host": str(bind.get("host", DEFAULT_HOST)),
        "port": int(bind.get("port", DEFAULT_PORT)),
        "family_id": str(config.get("family_id", "")),
        "started_at": _utc_now() if running else "",
        "last_error": safe_error_message(last_error) if str(last_error or "").strip() else "",
        "updated_at": _utc_now(),
    }


def _write_status(config: Mapping[str, Any], *, pid: int, running: bool, last_error: str = "") -> None:
    payload = _status_payload(config, pid=pid, running=running, last_error=last_error)
    _atomic_write(_status_path(str(config["config_id"])), _canonical_json_bytes(payload))


def _pid_alive(pid: Optional[int]) -> bool:
    if not pid or pid <= 0:
        return False
    if os.name == "nt":
        # Do NOT use os.kill(pid, 0) on Windows: non-console signals are
        # implemented with TerminateProcess and can kill the process.
        try:
            import ctypes
            from ctypes import wintypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            STILL_ACTIVE = 259
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            kernel32.OpenProcess.restype = wintypes.HANDLE
            kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
            kernel32.GetExitCodeProcess.restype = wintypes.BOOL
            kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
            kernel32.CloseHandle.restype = wintypes.BOOL
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
            if not handle:
                return False
            try:
                code = wintypes.DWORD()
                if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                    return False
                return int(code.value) == STILL_ACTIVE
            finally:
                kernel32.CloseHandle(handle)
        except Exception:
            return False
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ProcessLookupError, PermissionError):
        return False


def read_deployment_status(config_id: str) -> DeploymentProcessStatus:
    config_id = _safe_config_id(config_id)
    config: Dict[str, Any] = {}
    try:
        config = load_deployment_config(config_id)
    except Exception:
        pass
    path = _status_path(config_id)
    payload: Dict[str, Any] = {}
    if path.is_file() and path.stat().st_size <= 64 * 1024:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                payload = raw
        except Exception:
            payload = {}
    pid_raw = payload.get("pid")
    try:
        pid = int(pid_raw) if pid_raw else None
    except Exception:
        pid = None
    running = bool(payload.get("running")) and _pid_alive(pid)
    bind = config.get("bind") or {}
    tls = config.get("tls") or {}
    host = str(payload.get("host") or bind.get("host") or DEFAULT_HOST)
    port = int(payload.get("port") or bind.get("port") or DEFAULT_PORT)
    scheme = str(payload.get("scheme") or ("https" if tls.get("enabled") else "http"))
    return DeploymentProcessStatus(
        config_id=config_id,
        running=running,
        pid=pid,
        scheme=scheme,
        host=host,
        port=port,
        family_id=str(payload.get("family_id") or config.get("family_id") or ""),
        started_at=str(payload.get("started_at") or ""),
        last_error=str(payload.get("last_error") or ""),
    )


def _server_command(config_id: str) -> list[str]:
    return [sys.executable, "-m", "modules.deployment_api", "serve", "--config", _safe_config_id(config_id)]


def start_deployment_process(config_id: str) -> DeploymentProcessStatus:
    config = load_deployment_config(config_id)
    _load_deployment_token()  # fail before spawning when credential is absent
    current = read_deployment_status(config_id)
    if current.running:
        return current
    project_root = Path(__file__).resolve().parents[1]
    creationflags = 0
    kwargs: Dict[str, Any] = {
        "cwd": str(project_root),
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "close_fds": True,
    }
    if os.name == "nt":
        creationflags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0)) | int(getattr(subprocess, "DETACHED_PROCESS", 0))
        kwargs["creationflags"] = creationflags
    else:
        kwargs["start_new_session"] = True
    try:
        proc = subprocess.Popen(_server_command(config_id), **kwargs)
    except Exception as exc:
        _write_status(config, pid=0, running=False, last_error=exc)
        raise DeploymentAPIError(f"Could not start deployment API: {safe_error_message(exc)}") from exc
    _write_status(config, pid=proc.pid, running=True)
    # Give the child a short moment to fail fast on port/TLS/config errors.
    time.sleep(0.25)
    status = read_deployment_status(config_id)
    if not status.running:
        raise DeploymentAPIError(
            "Deployment API process exited during startup. Check the configuration, port, TLS files, and credential backend."
        )
    write_persistent_audit_event(
        {
            "event": "deployment_api_started",
            "config_id": config_id,
            "family_id": str(config.get("family_id", "")),
            "port": int((config.get("bind") or {}).get("port", DEFAULT_PORT)),
            "remote": not _is_loopback_host(str((config.get("bind") or {}).get("host", DEFAULT_HOST))),
        }
    )
    return status


def stop_deployment_process(config_id: str) -> DeploymentProcessStatus:
    config = load_deployment_config(config_id)
    status = read_deployment_status(config_id)
    if status.pid and _pid_alive(status.pid):
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(status.pid), "/T", "/F"],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=10,
                    check=False,
                    creationflags=int(getattr(subprocess, "CREATE_NO_WINDOW", 0)),
                )
            else:
                os.killpg(os.getpgid(status.pid), signal.SIGTERM)
        except Exception as exc:
            raise DeploymentAPIError(f"Could not stop deployment API safely: {safe_error_message(exc)}") from exc
    _write_status(config, pid=0, running=False)
    write_persistent_audit_event({"event": "deployment_api_stopped", "config_id": config_id})
    return read_deployment_status(config_id)


def _strict_json_loads(raw: bytes) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON number is not allowed: {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> Dict[str, Any]:
        obj: Dict[str, Any] = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError(f"Duplicate JSON key: {key}")
            obj[str(key)] = value
        return obj

    return json.loads(raw.decode("utf-8"), parse_constant=reject_constant, object_pairs_hook=unique_object)


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, bool, int, float))


def _normalise_records(value: Any, *, max_rows: int) -> pd.DataFrame:
    if not isinstance(value, list):
        raise DeploymentAPIError("'records' must be a JSON array of objects.")
    if not value:
        raise DeploymentAPIError("'records' must contain at least one row.")
    if len(value) > max_rows:
        raise DeploymentAPIError(f"Request has {len(value):,} rows; maximum is {max_rows:,}.")
    keys: set[str] = set()
    clean_records: list[Dict[str, Any]] = []
    for row_index, row in enumerate(value):
        if not isinstance(row, dict):
            raise DeploymentAPIError(f"Record {row_index} is not a JSON object.")
        clean: Dict[str, Any] = {}
        if len(row) > MAX_COLUMNS:
            raise DeploymentAPIError(f"Record {row_index} exceeds the {MAX_COLUMNS}-column safety cap.")
        for raw_key, item in row.items():
            key = str(raw_key)
            if not key or len(key) > 256:
                raise DeploymentAPIError("Input contains an empty or excessively long column name.")
            if not _is_scalar(item):
                raise DeploymentAPIError(
                    f"Nested arrays/objects are not allowed in prediction field '{key}'."
                )
            if isinstance(item, float) and not math.isfinite(item):
                raise DeploymentAPIError(f"Non-finite numeric value is not allowed in field '{key}'.")
            if isinstance(item, str) and len(item) > MAX_STRING_CHARS:
                raise DeploymentAPIError(f"Value in field '{key}' exceeds the per-value size limit.")
            clean[key] = item
            keys.add(key)
        clean_records.append(clean)
    if len(keys) > MAX_COLUMNS:
        raise DeploymentAPIError(f"Prediction payload exceeds the {MAX_COLUMNS}-column safety cap.")
    return pd.DataFrame.from_records(clean_records)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    return str(value)


class _RateLimiter:
    def __init__(self, requests_per_minute: int):
        self.limit = max(1, int(requests_per_minute))
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        cutoff = now - 60.0
        with self._lock:
            queue = self._events[str(key)]
            while queue and queue[0] < cutoff:
                queue.popleft()
            if len(queue) >= self.limit:
                return False
            queue.append(now)
            if len(self._events) > 10_000:
                stale = [name for name, values in self._events.items() if not values or values[-1] < cutoff]
                for name in stale[:5_000]:
                    self._events.pop(name, None)
            return True


class GovernedPredictionService:
    """Thread-safe Champion resolver and prediction facade used by the HTTP layer."""

    def __init__(self, config: Mapping[str, Any], token: str):
        self.config = dict(config)
        self.family_id = str(config.get("family_id", ""))
        self.token = str(token)
        self.max_rows = int((config.get("limits") or {}).get("max_rows", DEFAULT_MAX_ROWS))
        self.max_body_bytes = int((config.get("limits") or {}).get("max_body_bytes", DEFAULT_MAX_BODY_BYTES))
        self.cors_origins = tuple(map(str, config.get("cors_origins") or []))
        self.rate_limiter = _RateLimiter(int((config.get("limits") or {}).get("rate_limit_per_minute", DEFAULT_RATE_LIMIT_PER_MINUTE)))
        self._runtime: Optional[_ChampionRuntime] = None
        self._runtime_lock = threading.RLock()

    def authenticate(self, authorization: str) -> bool:
        header = str(authorization or "")
        if not header.startswith("Bearer "):
            return False
        supplied = header[7:].strip()
        if not supplied:
            return False
        try:
            return hmac.compare_digest(supplied.encode("utf-8"), self.token.encode("utf-8"))
        except Exception:
            return False

    def champion(self) -> _ChampionRuntime:
        try:
            summary = current_champion(self.family_id)
        except ModelGovernanceError as exc:
            raise DeploymentAPIError(f"Governance verification failed: {safe_error_message(exc)}") from exc
        if not summary:
            raise DeploymentAPIError("The configured model family currently has no approved Champion.")
        package_id = str(summary.get("package_id", ""))
        revision = int(summary.get("governance_revision", 0))
        with self._runtime_lock:
            if self._runtime and self._runtime.package_id == package_id and self._runtime.governance_revision == revision:
                return self._runtime
            try:
                raw = load_registered_package_bytes(package_id)
                package = load_signed_model_package(raw)
            except (ModelRegistryError, ModelPackageError) as exc:
                raise DeploymentAPIError(f"Champion package verification failed: {safe_error_message(exc)}") from exc
            if package.package_id != package_id:
                raise DeploymentAPIError("Governance Champion and verified package identifiers do not match.")
            runtime = _ChampionRuntime(
                family_id=self.family_id,
                package_id=package_id,
                package=package,
                loaded_at=time.time(),
                governance_revision=revision,
            )
            self._runtime = runtime
            return runtime

    def model_info(self) -> Dict[str, Any]:
        runtime = self.champion()
        manifest = runtime.package.manifest
        return {
            "family_id": self.family_id,
            "champion_id": runtime.package_id,
            "governance_revision": runtime.governance_revision,
            "task": runtime.package.task,
            "target": runtime.package.target,
            "selected_model": str((manifest.get("model") or {}).get("selected_model", "")),
            "required_columns": list(runtime.package.required_columns),
            "created_at": str(manifest.get("created_at", "")),
        }

    def validate_records(self, records: Any) -> tuple[pd.DataFrame, Dict[str, Any], _ChampionRuntime]:
        frame = _normalise_records(records, max_rows=self.max_rows)
        runtime = self.champion()
        report = validate_prediction_frame(frame, runtime.package, max_rows=self.max_rows)
        return frame, report, runtime

    def predict_records(self, records: Any, *, include_probabilities: bool = True) -> Dict[str, Any]:
        frame, report, runtime = self.validate_records(records)
        if not report.get("valid"):
            raise _SchemaBlocked(report)
        result = run_batch_prediction(
            runtime.package,
            frame,
            max_rows=self.max_rows,
            include_probabilities=bool(include_probabilities),
        )
        generated = [result.prediction_column, *result.probability_columns]
        response_rows: list[Dict[str, Any]] = []
        for index, (_, row) in enumerate(result.output[generated].iterrows()):
            item: Dict[str, Any] = {"row": index, "prediction": _json_safe(row[result.prediction_column])}
            confidence = next((name for name in result.probability_columns if name.startswith("Prediction_Confidence")), None)
            if confidence:
                item["confidence"] = _json_safe(row[confidence])
            probabilities: Dict[str, Any] = {}
            for name in result.probability_columns:
                if name.startswith("Probability_"):
                    probabilities[name[len("Probability_"):]] = _json_safe(row[name])
            if probabilities:
                item["probabilities"] = probabilities
            response_rows.append(item)
        write_persistent_audit_event(
            {
                "event": "deployment_api_prediction",
                "config_id": str(self.config.get("config_id", "")),
                "family_id": self.family_id,
                "package_id": runtime.package_id,
                "rows": int(len(frame)),
                "probabilities": bool(include_probabilities),
            }
        )
        return {
            "family_id": self.family_id,
            "champion_id": runtime.package_id,
            "task": runtime.package.task,
            "target": runtime.package.target,
            "rows": len(response_rows),
            "warnings": list(result.warnings),
            "results": response_rows,
        }


class _SchemaBlocked(DeploymentAPIError):
    def __init__(self, report: Mapping[str, Any]):
        super().__init__("Prediction schema validation failed.")
        self.report = dict(report)


class _DeploymentHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, address: tuple[str, int], handler, service: GovernedPredictionService, *, tls_enabled: bool):
        super().__init__(address, handler)
        self.service = service
        self.tls_enabled = bool(tls_enabled)


class _DeploymentRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "DataBridgeAI-DeploymentAPI"
    sys_version = ""

    @property
    def service(self) -> GovernedPredictionService:
        return self.server.service  # type: ignore[attr-defined]

    def log_message(self, format: str, *args: Any) -> None:
        # Never log payloads, bearer tokens, or request headers through BaseHTTPRequestHandler.
        return

    def _cors_origin(self) -> str:
        origin = str(self.headers.get("Origin") or "").strip()
        return origin if origin and origin in self.service.cors_origins else ""

    def _send_json(self, status: int, payload: Mapping[str, Any], *, request_id: str = "") -> None:
        body_payload = dict(payload)
        if request_id:
            body_payload.setdefault("request_id", request_id)
        try:
            body = json.dumps(body_payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
        except Exception:
            body = b'{"error":{"code":"serialization_error","message":"Response serialization failed safely."}}'
            status = 500
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Pragma", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        if bool(getattr(self.server, "tls_enabled", False)):
            self.send_header("Strict-Transport-Security", "max-age=31536000")
        origin = self._cors_origin()
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _error(self, status: int, code: str, message: str, *, request_id: str, details: Any = None) -> None:
        payload: Dict[str, Any] = {"error": {"code": code, "message": _safe_text(message, limit=400)}}
        if details is not None:
            payload["error"]["details"] = details
        self._send_json(status, payload, request_id=request_id)

    def _rate_allowed(self) -> bool:
        client = str(self.client_address[0] if self.client_address else "unknown")
        return self.service.rate_limiter.allow(client)

    def _authorised(self, request_id: str) -> bool:
        if not self._rate_allowed():
            self._error(429, "rate_limited", "Too many requests. Retry later.", request_id=request_id)
            return False
        if not self.service.authenticate(str(self.headers.get("Authorization") or "")):
            payload = {"error": {"code": "unauthorized", "message": "A valid Bearer token is required."}, "request_id": request_id}
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self.send_response(401)
            self.send_header("WWW-Authenticate", "Bearer")
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            try:
                self.wfile.write(body)
            except Exception:
                pass
            write_persistent_audit_event({"event": "deployment_api_auth_failed", "route": self.path.split("?", 1)[0]})
            return False
        return True

    def _read_json_body(self, request_id: str) -> Optional[Dict[str, Any]]:
        content_type = str(self.headers.get("Content-Type") or "").split(";", 1)[0].strip().casefold()
        if content_type != "application/json":
            self._error(415, "unsupported_media_type", "Content-Type must be application/json.", request_id=request_id)
            return None
        try:
            content_length = int(self.headers.get("Content-Length") or "0")
        except ValueError:
            content_length = -1
        if content_length <= 0:
            self._error(400, "empty_body", "A JSON request body is required.", request_id=request_id)
            return None
        if content_length > self.service.max_body_bytes:
            self._error(413, "payload_too_large", "Request body exceeds the configured safety cap.", request_id=request_id)
            return None
        raw = self.rfile.read(content_length)
        if len(raw) != content_length:
            self._error(400, "incomplete_body", "Request body was incomplete.", request_id=request_id)
            return None
        try:
            payload = _strict_json_loads(raw)
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
            self._error(400, "invalid_json", safe_error_message(exc), request_id=request_id)
            return None
        if not isinstance(payload, dict):
            self._error(400, "invalid_payload", "JSON body must be an object.", request_id=request_id)
            return None
        return payload

    def do_OPTIONS(self) -> None:  # noqa: N802
        request_id = secrets.token_hex(8)
        origin = self._cors_origin()
        if not origin:
            self._error(403, "cors_denied", "Origin is not allowed.", request_id=request_id)
            return
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", origin)
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Authorization,Content-Type")
        self.send_header("Access-Control-Max-Age", "600")
        self.send_header("Vary", "Origin")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        request_id = secrets.token_hex(8)
        path = self.path.split("?", 1)[0]
        if path == "/health":
            self._send_json(200, {"status": "ok"}, request_id=request_id)
            return
        if path != "/v1/model":
            self._error(404, "not_found", "Endpoint not found.", request_id=request_id)
            return
        if not self._authorised(request_id):
            return
        try:
            self._send_json(200, self.service.model_info(), request_id=request_id)
        except DeploymentAPIError as exc:
            self._error(409, "champion_unavailable", safe_error_message(exc), request_id=request_id)
        except Exception as exc:
            self._error(500, "internal_error", "Request failed safely.", request_id=request_id)
            write_persistent_audit_event({"event": "deployment_api_error", "route": path, "error": safe_error_message(exc)})

    def do_POST(self) -> None:  # noqa: N802
        request_id = secrets.token_hex(8)
        path = self.path.split("?", 1)[0]
        if path not in {"/v1/validate", "/v1/predict"}:
            self._error(404, "not_found", "Endpoint not found.", request_id=request_id)
            return
        if not self._authorised(request_id):
            return
        payload = self._read_json_body(request_id)
        if payload is None:
            return
        if "records" not in payload:
            self._error(400, "missing_records", "JSON body must contain a 'records' array.", request_id=request_id)
            return
        try:
            if path == "/v1/validate":
                _, report, runtime = self.service.validate_records(payload["records"])
                response = {
                    "valid": bool(report.get("valid")),
                    "status": report.get("status"),
                    "family_id": self.service.family_id,
                    "champion_id": runtime.package_id,
                    "rows": report.get("rows"),
                    "required_columns": report.get("required_columns", []),
                    "missing_columns": report.get("missing_columns", []),
                    "extra_columns": report.get("extra_columns", []),
                    "blockers": report.get("blockers", []),
                    "warnings": report.get("warnings", []),
                    "column_checks": report.get("column_checks", []),
                }
                self._send_json(200 if response["valid"] else 422, response, request_id=request_id)
                return
            include_probabilities = bool(payload.get("include_probabilities", True))
            result = self.service.predict_records(payload["records"], include_probabilities=include_probabilities)
            self._send_json(200, result, request_id=request_id)
        except _SchemaBlocked as exc:
            report = exc.report
            self._error(
                422,
                "schema_validation_failed",
                "Prediction input does not satisfy the Champion schema contract.",
                request_id=request_id,
                details={
                    "blockers": report.get("blockers", []),
                    "warnings": report.get("warnings", []),
                    "missing_columns": report.get("missing_columns", []),
                },
            )
        except DeploymentAPIError as exc:
            message = safe_error_message(exc)
            status = 413 if "maximum" in message.casefold() and "rows" in message.casefold() else 400
            self._error(status, "invalid_request", message, request_id=request_id)
        except Exception as exc:
            self._error(500, "internal_error", "Request failed safely.", request_id=request_id)
            write_persistent_audit_event({"event": "deployment_api_error", "route": path, "error": safe_error_message(exc)})



def build_api_contract(config: Mapping[str, Any]) -> Dict[str, Any]:
    bind = config.get("bind") or {}
    tls = config.get("tls") or {}
    scheme = "https" if tls.get("enabled") else "http"
    host = str(bind.get("host", DEFAULT_HOST))
    display_host = "127.0.0.1" if host == "0.0.0.0" else host
    base = f"{scheme}://{display_host}:{int(bind.get('port', DEFAULT_PORT))}"
    return {
        "base_url": base,
        "authentication": "Authorization: Bearer <deployment-token>",
        "endpoints": [
            {"method": "GET", "path": "/health", "auth": False, "purpose": "Minimal liveness check"},
            {"method": "GET", "path": "/v1/model", "auth": True, "purpose": "Current governed Champion contract"},
            {"method": "POST", "path": "/v1/validate", "auth": True, "purpose": "Validate records without scoring"},
            {"method": "POST", "path": "/v1/predict", "auth": True, "purpose": "Score records with the current Champion"},
        ],
        "champion_only": True,
        "follows_promotions": True,
    }


def create_deployment_http_server(config: Mapping[str, Any], token: str) -> ThreadingHTTPServer:
    """Create a bound HTTP(S) server from an already verified configuration."""
    service = GovernedPredictionService(config, token)
    bind = config["bind"]
    host = str(bind["host"])
    port = int(bind["port"])
    tls = config.get("tls") or {}
    server = _DeploymentHTTPServer((host, port), _DeploymentRequestHandler, service, tls_enabled=bool(tls.get("enabled")))
    if bool(tls.get("enabled")):
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.load_cert_chain(certfile=str(tls["cert_path"]), keyfile=str(tls["key_path"]))
        server.socket = context.wrap_socket(server.socket, server_side=True)
    return server


def run_deployment_server(config_id: str) -> None:
    config = load_deployment_config(config_id)
    token = _load_deployment_token()
    server = create_deployment_http_server(config, token)
    service = server.service  # type: ignore[attr-defined]
    bind = config["bind"]
    host = str(bind["host"])
    port = int(bind["port"])
    tls = config.get("tls") or {}
    _write_status(config, pid=os.getpid(), running=True)
    write_persistent_audit_event(
        {
            "event": "deployment_api_server_ready",
            "config_id": config_id,
            "family_id": service.family_id,
            "port": port,
            "tls_enabled": bool(tls.get("enabled")),
        }
    )
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            server.server_close()
        finally:
            _write_status(config, pid=0, running=False)


def _main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="DataBridge AI governed Champion deployment API")
    sub = parser.add_subparsers(dest="command", required=True)
    serve = sub.add_parser("serve", help="Serve a signed deployment configuration")
    serve.add_argument("--config", required=True, dest="config_id")
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "serve":
            run_deployment_server(args.config_id)
            return 0
    except Exception as exc:
        # Never print secrets or request payloads. This is only a bounded operational error.
        sys.stderr.write("Deployment API failed safely: " + safe_error_message(exc) + "\n")
        try:
            config = load_deployment_config(getattr(args, "config_id", ""))
            _write_status(config, pid=0, running=False, last_error=exc)
        except Exception:
            pass
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(_main())
