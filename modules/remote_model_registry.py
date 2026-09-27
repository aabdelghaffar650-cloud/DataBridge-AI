# ════════════════════════════════════════════════════════
# DataBridge AI — Remote Model Registry
# Stage 19: authenticated central package + governance synchronization
# ════════════════════════════════════════════════════════
from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import os
import re
import secrets
import ssl
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from core.credential_store import SecretStoreError, get_credential_store
from core.user_paths import remote_model_registry_dir
from modules.model_governance import (
    MAX_GOVERNANCE_BYTES,
    ModelGovernanceError,
    export_governance_envelope,
    get_governance_state,
    governance_is_fast_forward,
    governance_state_digest,
    import_governance_envelope,
    list_governance_families,
    validate_governance_envelope_bytes,
)
from modules.model_package import (
    MAX_PACKAGE_BYTES,
    ModelPackageError,
    get_or_create_signing_key,
    load_signed_model_package,
    signing_key_id,
)
from modules.model_registry import (
    ModelRegistryError,
    load_registered_package_bytes,
    register_signed_package,
)


REMOTE_REGISTRY_FORMAT = "DataBridgeAI Remote Model Registry"
REMOTE_REGISTRY_VERSION = 1
REMOTE_CONTEXT = b"DataBridgeAI-Remote-Registry-v1\0"
REMOTE_TOKEN_SECRET = "remote_registry_token"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8890
MAX_CONFIG_BYTES = 256 * 1024
MAX_METADATA_BYTES = 512 * 1024
MAX_RESPONSE_BYTES = max(MAX_PACKAGE_BYTES + 1024 * 1024, 8 * 1024 * 1024)


class RemoteRegistryError(ValueError):
    """Raised when remote-registry transport, trust, or synchronization is unsafe."""


@dataclass(frozen=True)
class RemoteRegistryProcessStatus:
    config_id: str
    running: bool
    pid: Optional[int]
    scheme: str
    host: str
    port: int
    started_at: str
    last_error: str = ""

    @property
    def base_url(self) -> str:
        host = "127.0.0.1" if self.host == "0.0.0.0" else self.host
        return f"{self.scheme}://{host}:{self.port}"


@dataclass(frozen=True)
class RemoteSyncResult:
    family_id: str
    family_name: str
    direction: str
    packages_transferred: int
    governance_revision: int
    champion_id: str
    message: str


# ── Generic helpers ──────────────────────────────────────────────────────────
def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _config_hmac(payload: Mapping[str, Any], key: bytes) -> str:
    return hmac.new(key, REMOTE_CONTEXT + _canonical_json_bytes(payload), hashlib.sha256).hexdigest()


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _safe_id(value: str, *, label: str = "identifier") -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "")).strip("._-")
    if not clean:
        raise RemoteRegistryError(f"Invalid {label}.")
    return clean[:120]


def _safe_text(value: Any, *, limit: int = 500) -> str:
    return str(value or "").replace("\x00", " ").strip()[:limit]


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


def _server_configs_dir() -> Path:
    path = remote_model_registry_dir() / "server_configs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _client_profiles_dir() -> Path:
    path = remote_model_registry_dir() / "client_profiles"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _runtime_dir() -> Path:
    path = remote_model_registry_dir() / "runtime"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _server_store_dir() -> Path:
    path = remote_model_registry_dir() / "server_store"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _server_package_dir(root: Path) -> Path:
    path = root / "packages"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _server_governance_dir(root: Path) -> Path:
    path = root / "governance"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _server_metadata_dir(root: Path) -> Path:
    path = root / "metadata"
    path.mkdir(parents=True, exist_ok=True)
    return path


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
        raise RemoteRegistryError("Registry host must be an explicit IPv4 address or localhost.") from exc
    if ip.version != 4:
        raise RemoteRegistryError("Remote registry currently supports IPv4 bind addresses only.")
    if not ip.is_loopback and not allow_remote:
        raise RemoteRegistryError("Non-loopback registry binding requires explicit remote-server approval.")
    return str(ip)


def _validate_port(port: int) -> int:
    try:
        value = int(port)
    except Exception as exc:
        raise RemoteRegistryError("Registry port must be an integer.") from exc
    if value < 1024 or value > 65535:
        raise RemoteRegistryError("Registry port must be between 1024 and 65535.")
    return value


def _validate_local_file(path_value: str, *, label: str) -> str:
    text = str(path_value or "").strip()
    if not text:
        raise RemoteRegistryError(f"{label} is required.")
    if text.startswith("\\\\") or text.startswith("//"):
        raise RemoteRegistryError(f"{label} must be a local file, not a UNC/network path.")
    path = Path(text).expanduser()
    if not path.is_absolute() or not path.is_file():
        raise RemoteRegistryError(f"{label} must be an existing absolute local file.")
    if path.stat().st_size <= 0 or path.stat().st_size > 4 * 1024 * 1024:
        raise RemoteRegistryError(f"{label} has an unsafe size.")
    return str(path.resolve())


def _validate_remote_base_url(base_url: str) -> str:
    text = str(base_url or "").strip().rstrip("/")
    try:
        parsed = urllib.parse.urlsplit(text)
    except Exception as exc:
        raise RemoteRegistryError("Remote registry URL is invalid.") from exc
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise RemoteRegistryError("Registry URL must be a plain http(s) URL without embedded credentials.")
    if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
        raise RemoteRegistryError("Registry URL must contain only scheme, host, and optional port.")
    if parsed.scheme == "http" and not _is_loopback_host(parsed.hostname):
        raise RemoteRegistryError("Remote registry connections outside loopback require HTTPS/TLS.")
    return text


# ── OS credential token ─────────────────────────────────────────────────────
def generate_remote_registry_token(*, backend: Any | None = None) -> str:
    token = "dbri_" + secrets.token_urlsafe(48)
    store = get_credential_store() if backend is None else __import__(
        "core.credential_store", fromlist=["CredentialStore"]
    ).CredentialStore(backend=backend)
    store.set_secret(REMOTE_TOKEN_SECRET, token)
    return token


def save_remote_registry_token(token: str, *, backend: Any | None = None) -> None:
    value = str(token or "").strip()
    if len(value) < 48:
        raise RemoteRegistryError("Remote registry token is too short.")
    store = get_credential_store() if backend is None else __import__(
        "core.credential_store", fromlist=["CredentialStore"]
    ).CredentialStore(backend=backend)
    store.set_secret(REMOTE_TOKEN_SECRET, value)


def remote_registry_token_status() -> Dict[str, Any]:
    status = get_credential_store().status(REMOTE_TOKEN_SECRET)
    return {
        "configured": bool(status.configured),
        "source": status.source,
        "backend": status.backend,
    }


def _load_remote_registry_token(*, backend: Any | None = None) -> str:
    store = get_credential_store() if backend is None else __import__(
        "core.credential_store", fromlist=["CredentialStore"]
    ).CredentialStore(backend=backend)
    try:
        token, _ = store.get_secret(REMOTE_TOKEN_SECRET)
    except SecretStoreError as exc:
        raise RemoteRegistryError(str(exc)) from exc
    if not token:
        raise RemoteRegistryError("Remote registry Bearer token is not configured.")
    return token


# ── Signed local server configuration ───────────────────────────────────────
def _config_path(config_id: str) -> Path:
    return _server_configs_dir() / f"{_safe_id(config_id, label='server config ID')}.json"


def _profile_path(profile_id: str) -> Path:
    return _client_profiles_dir() / f"{_safe_id(profile_id, label='profile ID')}.json"


def _status_path(config_id: str) -> Path:
    return _runtime_dir() / f"{_safe_id(config_id, label='server config ID')}.json"


def _write_signed_json(path: Path, payload: Mapping[str, Any], key: bytes) -> None:
    body = dict(payload)
    envelope = {"payload": body, "signature": _config_hmac(body, key)}
    raw = json.dumps(envelope, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
    if len(raw) > MAX_CONFIG_BYTES:
        raise RemoteRegistryError("Signed registry configuration is unexpectedly large.")
    _atomic_write(path, raw)


def _read_signed_json(path: Path, key: bytes, *, expected_kind: str) -> Dict[str, Any]:
    if not path.is_file() or path.stat().st_size <= 0 or path.stat().st_size > MAX_CONFIG_BYTES:
        raise RemoteRegistryError("Registry configuration was not found or has an unsafe size.")
    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RemoteRegistryError("Registry configuration is unreadable.") from exc
    if not isinstance(envelope, dict) or set(envelope) != {"payload", "signature"}:
        raise RemoteRegistryError("Registry configuration envelope is invalid.")
    payload = envelope.get("payload")
    signature = str(envelope.get("signature", ""))
    if not isinstance(payload, dict) or payload.get("format") != REMOTE_REGISTRY_FORMAT:
        raise RemoteRegistryError("Registry configuration payload is invalid.")
    if int(payload.get("version", 0)) != REMOTE_REGISTRY_VERSION or payload.get("kind") != expected_kind:
        raise RemoteRegistryError("Unsupported registry configuration version or kind.")
    if str(payload.get("signer_key_id", "")) != signing_key_id(key):
        raise RemoteRegistryError("Registry configuration belongs to a different model trust key.")
    if not hmac.compare_digest(signature, _config_hmac(payload, key)):
        raise RemoteRegistryError("Registry configuration integrity verification failed.")
    return dict(payload)


def create_remote_registry_server_config(
    *,
    name: str,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    allow_remote: bool = False,
    tls_cert_path: str = "",
    tls_key_path: str = "",
    signing_key: Optional[bytes] = None,
) -> Dict[str, Any]:
    key = bytes(signing_key) if signing_key is not None else get_or_create_signing_key()
    bind_host = _validate_bind_host(host, allow_remote=allow_remote)
    bind_port = _validate_port(port)
    remote = not _is_loopback_host(bind_host)
    cert = _validate_local_file(tls_cert_path, label="TLS certificate") if remote else ""
    private_key = _validate_local_file(tls_key_path, label="TLS private key") if remote else ""
    if remote and (not cert or not private_key):
        raise RemoteRegistryError("Remote registry exposure requires TLS certificate and private key files.")
    clean_name = _safe_text(name or "Remote Model Registry", limit=120)
    config_id = "registry-" + hashlib.sha256(
        f"{clean_name}|{bind_host}|{bind_port}".encode("utf-8")
    ).hexdigest()[:16]
    payload = {
        "format": REMOTE_REGISTRY_FORMAT,
        "version": REMOTE_REGISTRY_VERSION,
        "kind": "server",
        "config_id": config_id,
        "name": clean_name,
        "host": bind_host,
        "port": bind_port,
        "scheme": "https" if remote else "http",
        "allow_remote": bool(remote),
        "tls_cert_path": cert,
        "tls_key_path": private_key,
        "signer_key_id": signing_key_id(key),
        "created_at": _utc_now(),
    }
    _write_signed_json(_config_path(config_id), payload, key)
    return payload


def load_remote_registry_server_config(config_id: str, *, signing_key: Optional[bytes] = None) -> Dict[str, Any]:
    key = bytes(signing_key) if signing_key is not None else get_or_create_signing_key()
    return _read_signed_json(_config_path(config_id), key, expected_kind="server")


def list_remote_registry_server_configs(*, signing_key: Optional[bytes] = None) -> list[Dict[str, Any]]:
    key = bytes(signing_key) if signing_key is not None else get_or_create_signing_key()
    rows: list[Dict[str, Any]] = []
    for path in _server_configs_dir().glob("*.json"):
        try:
            rows.append(_read_signed_json(path, key, expected_kind="server"))
        except Exception:
            continue
    rows.sort(key=lambda row: str(row.get("name", "")).casefold())
    return rows


def create_remote_registry_client_profile(
    *,
    name: str,
    base_url: str,
    ca_cert_path: str = "",
    signing_key: Optional[bytes] = None,
) -> Dict[str, Any]:
    key = bytes(signing_key) if signing_key is not None else get_or_create_signing_key()
    url = _validate_remote_base_url(base_url)
    ca_path = ""
    if ca_cert_path:
        ca_path = _validate_local_file(ca_cert_path, label="Registry CA certificate")
    clean_name = _safe_text(name or "Remote Registry", limit=120)
    profile_id = "remote-" + hashlib.sha256(f"{clean_name}|{url}".encode("utf-8")).hexdigest()[:16]
    payload = {
        "format": REMOTE_REGISTRY_FORMAT,
        "version": REMOTE_REGISTRY_VERSION,
        "kind": "client",
        "profile_id": profile_id,
        "name": clean_name,
        "base_url": url,
        "ca_cert_path": ca_path,
        "signer_key_id": signing_key_id(key),
        "created_at": _utc_now(),
    }
    _write_signed_json(_profile_path(profile_id), payload, key)
    return payload


def load_remote_registry_client_profile(profile_id: str, *, signing_key: Optional[bytes] = None) -> Dict[str, Any]:
    key = bytes(signing_key) if signing_key is not None else get_or_create_signing_key()
    return _read_signed_json(_profile_path(profile_id), key, expected_kind="client")


def list_remote_registry_client_profiles(*, signing_key: Optional[bytes] = None) -> list[Dict[str, Any]]:
    key = bytes(signing_key) if signing_key is not None else get_or_create_signing_key()
    rows: list[Dict[str, Any]] = []
    for path in _client_profiles_dir().glob("*.json"):
        try:
            rows.append(_read_signed_json(path, key, expected_kind="client"))
        except Exception:
            continue
    rows.sort(key=lambda row: str(row.get("name", "")).casefold())
    return rows


# ── Server-side central store ────────────────────────────────────────────────
class RemoteRegistryStore:
    """Central artifact store. It never accepts an unsigned package or governance state."""

    def __init__(self, root: Path, signing_key: bytes):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.key = bytes(signing_key)
        self.signer_key_id = signing_key_id(self.key)

    def _package_path(self, package_id: str) -> Path:
        return _server_package_dir(self.root) / f"{_safe_id(package_id, label='package ID')}.dbmlpkg"

    def _metadata_path(self, package_id: str) -> Path:
        return _server_metadata_dir(self.root) / f"{_safe_id(package_id, label='package ID')}.json"

    def _governance_path(self, family_id: str) -> Path:
        return _server_governance_dir(self.root) / f"{_safe_id(family_id, label='family ID')}.json"

    def store_package(self, raw: bytes) -> Dict[str, Any]:
        if not isinstance(raw, (bytes, bytearray)) or not raw or len(raw) > MAX_PACKAGE_BYTES:
            raise RemoteRegistryError("Uploaded model package has an unsafe size.")
        try:
            package = load_signed_model_package(bytes(raw), signing_key=self.key)
        except ModelPackageError as exc:
            raise RemoteRegistryError(f"Uploaded package failed shared-trust verification: {exc}") from exc
        package_id = _safe_id(package.package_id, label="package ID")
        manifest = package.manifest
        metadata = {
            "package_id": package.package_id,
            "sha256": _sha256(bytes(raw)),
            "size_bytes": len(raw),
            "task": package.task,
            "target": package.target,
            "selected_model": str(manifest.get("model", {}).get("selected_model", "")),
            "created_at": str(manifest.get("created_at", "")),
            "application_version": str(manifest.get("application", {}).get("version", "")),
            "signer_key_id": package.signer_key_id,
            "stored_at": _utc_now(),
        }
        _atomic_write(self._package_path(package_id), bytes(raw))
        _atomic_write(
            self._metadata_path(package_id),
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8"),
        )
        return metadata

    def get_package(self, package_id: str) -> bytes:
        path = self._package_path(package_id)
        if not path.is_file() or path.stat().st_size <= 0 or path.stat().st_size > MAX_PACKAGE_BYTES:
            raise RemoteRegistryError("Remote package was not found or has an unsafe size.")
        raw = path.read_bytes()
        try:
            package = load_signed_model_package(raw, signing_key=self.key)
        except ModelPackageError as exc:
            raise RemoteRegistryError(f"Stored remote package failed verification: {exc}") from exc
        if package.package_id != package_id:
            raise RemoteRegistryError("Remote package identifier does not match its signed payload.")
        return raw

    def list_packages(self) -> list[Dict[str, Any]]:
        rows: list[Dict[str, Any]] = []
        for path in _server_metadata_dir(self.root).glob("*.json"):
            try:
                if path.stat().st_size > MAX_METADATA_BYTES:
                    continue
                row = json.loads(path.read_text(encoding="utf-8"))
                package_id = str(row.get("package_id", ""))
                if package_id and self._package_path(package_id).is_file():
                    rows.append(row)
            except Exception:
                continue
        rows.sort(key=lambda row: str(row.get("created_at", "")), reverse=True)
        return rows

    def get_governance_raw(self, family_id: str) -> bytes:
        path = self._governance_path(family_id)
        if not path.is_file() or path.stat().st_size <= 0 or path.stat().st_size > MAX_GOVERNANCE_BYTES:
            raise RemoteRegistryError("Remote governance family was not found.")
        raw = path.read_bytes()
        state = validate_governance_envelope_bytes(raw, signing_key=self.key)
        if str(state.get("family_id")) != str(family_id):
            raise RemoteRegistryError("Remote governance family ID mismatch.")
        return raw

    def get_governance_state(self, family_id: str) -> Optional[Dict[str, Any]]:
        path = self._governance_path(family_id)
        if not path.is_file():
            return None
        return validate_governance_envelope_bytes(path.read_bytes(), signing_key=self.key)

    def put_governance(
        self,
        raw: bytes,
        *,
        expected_revision: int,
        expected_digest: str,
    ) -> Dict[str, Any]:
        try:
            incoming = validate_governance_envelope_bytes(raw, signing_key=self.key)
        except ModelGovernanceError as exc:
            raise RemoteRegistryError(str(exc)) from exc
        family_id = str(incoming["family_id"])
        current = self.get_governance_state(family_id)
        current_revision = int(current.get("revision", 0)) if current is not None else -1
        current_digest = governance_state_digest(current) if current is not None else ""
        if int(expected_revision) != current_revision or str(expected_digest or "") != current_digest:
            raise RemoteRegistryError(
                "Remote governance changed since the client read it. Pull the latest family before pushing again."
            )
        if current is not None and not governance_is_fast_forward(current, incoming):
            raise RemoteRegistryError("Incoming governance history diverges from the central registry.")

        for package_id, row in (incoming.get("models") or {}).items():
            if str((row or {}).get("status", "")) == "Removed":
                continue
            self.get_package(str(package_id))

        _atomic_write(self._governance_path(family_id), bytes(raw))
        return {
            "family_id": family_id,
            "family_name": incoming.get("family_name", ""),
            "revision": int(incoming.get("revision", 0)),
            "champion_id": str(incoming.get("champion_id", "") or ""),
            "digest": governance_state_digest(incoming),
        }

    def list_families(self) -> list[Dict[str, Any]]:
        rows: list[Dict[str, Any]] = []
        for path in _server_governance_dir(self.root).glob("*.json"):
            try:
                state = validate_governance_envelope_bytes(path.read_bytes(), signing_key=self.key)
                counts: Dict[str, int] = {}
                for row in (state.get("models") or {}).values():
                    status = str((row or {}).get("status", "Unknown"))
                    counts[status] = counts.get(status, 0) + 1
                rows.append(
                    {
                        "family_id": state["family_id"],
                        "family_name": state["family_name"],
                        "task": state["task"],
                        "target": state["target"],
                        "revision": int(state.get("revision", 0)),
                        "champion_id": str(state.get("champion_id", "") or ""),
                        "model_counts": counts,
                        "digest": governance_state_digest(state),
                        "updated_at": state.get("updated_at", ""),
                    }
                )
            except Exception:
                continue
        rows.sort(key=lambda row: str(row.get("family_name", "")).casefold())
        return rows


# ── HTTP server ──────────────────────────────────────────────────────────────
class _RegistryHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    def __init__(self, address, handler, *, store: RemoteRegistryStore, token: str):
        super().__init__(address, handler)
        self.registry_store = store
        self.registry_token = token


class _RegistryHandler(BaseHTTPRequestHandler):
    server_version = "DataBridgeRegistry/1.0"
    protocol_version = "HTTP/1.1"

    @property
    def registry_server(self) -> _RegistryHTTPServer:
        return self.server  # type: ignore[return-value]

    def log_message(self, format: str, *args: Any) -> None:  # no request bodies/tokens in logs
        return

    def _send_bytes(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, payload: Mapping[str, Any]) -> None:
        body = json.dumps(dict(payload), ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
        self._send_bytes(status, body, "application/json; charset=utf-8")

    def _error(self, status: int, code: str, message: str) -> None:
        self._send_json(status, {"error": {"code": code, "message": _safe_text(message, limit=500)}})

    def _authenticated(self) -> bool:
        header = str(self.headers.get("Authorization", ""))
        if not header.startswith("Bearer "):
            return False
        supplied = header[7:].strip()
        return bool(supplied) and hmac.compare_digest(supplied, self.registry_server.registry_token)

    def _require_auth(self) -> bool:
        if self._authenticated():
            return True
        self._error(401, "unauthorized", "A valid registry Bearer token is required.")
        return False

    def _read_body(self, *, max_bytes: int) -> bytes:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise RemoteRegistryError("Invalid Content-Length.") from exc
        if length <= 0 or length > max_bytes:
            raise RemoteRegistryError("Request body has an unsafe size.")
        raw = self.rfile.read(length)
        if len(raw) != length:
            raise RemoteRegistryError("Request body was truncated.")
        return raw

    def do_GET(self) -> None:  # noqa: N802
        path = urllib.parse.urlsplit(self.path).path
        try:
            if path == "/health":
                self._send_json(
                    200,
                    {
                        "status": "ok",
                        "service": "DataBridge AI Remote Model Registry",
                        "version": REMOTE_REGISTRY_VERSION,
                        "signer_key_id": self.registry_server.registry_store.signer_key_id,
                    },
                )
                return
            if not self._require_auth():
                return
            if path == "/v1/packages":
                self._send_json(200, {"packages": self.registry_server.registry_store.list_packages()})
                return
            if path == "/v1/families":
                self._send_json(200, {"families": self.registry_server.registry_store.list_families()})
                return
            match = re.fullmatch(r"/v1/packages/([A-Za-z0-9._-]+)", path)
            if match:
                raw = self.registry_server.registry_store.get_package(match.group(1))
                self._send_bytes(200, raw, "application/vnd.databridge.model-package")
                return
            match = re.fullmatch(r"/v1/families/([A-Za-z0-9._-]+)/governance", path)
            if match:
                raw = self.registry_server.registry_store.get_governance_raw(match.group(1))
                self._send_bytes(200, raw, "application/vnd.databridge.governance+json")
                return
            self._error(404, "not_found", "Registry endpoint not found.")
        except RemoteRegistryError as exc:
            self._error(404, "not_found", str(exc))
        except Exception:
            self._error(500, "internal_error", "Registry request failed safely.")

    def do_PUT(self) -> None:  # noqa: N802
        path = urllib.parse.urlsplit(self.path).path
        if not self._require_auth():
            return
        try:
            match = re.fullmatch(r"/v1/packages/([A-Za-z0-9._-]+)", path)
            if match:
                raw = self._read_body(max_bytes=MAX_PACKAGE_BYTES)
                meta = self.registry_server.registry_store.store_package(raw)
                if str(meta.get("package_id")) != match.group(1):
                    raise RemoteRegistryError("URL package ID does not match the signed package.")
                self._send_json(200, {"stored": meta})
                return
            match = re.fullmatch(r"/v1/families/([A-Za-z0-9._-]+)/governance", path)
            if match:
                raw = self._read_body(max_bytes=MAX_GOVERNANCE_BYTES)
                expected_revision = int(self.headers.get("X-Expected-Revision", "-2"))
                expected_digest = str(self.headers.get("X-Expected-Digest", ""))
                result = self.registry_server.registry_store.put_governance(
                    raw,
                    expected_revision=expected_revision,
                    expected_digest=expected_digest,
                )
                if str(result.get("family_id")) != match.group(1):
                    raise RemoteRegistryError("URL family ID does not match the signed governance state.")
                self._send_json(200, {"governance": result})
                return
            self._error(404, "not_found", "Registry endpoint not found.")
        except RemoteRegistryError as exc:
            message = str(exc)
            status = 409 if "changed" in message.lower() or "diverge" in message.lower() else 422
            self._error(status, "registry_conflict" if status == 409 else "invalid_artifact", message)
        except Exception:
            self._error(500, "internal_error", "Registry write failed safely.")


def create_remote_registry_http_server(
    config: Mapping[str, Any],
    token: str,
    *,
    signing_key: Optional[bytes] = None,
    storage_root: Optional[Path] = None,
) -> ThreadingHTTPServer:
    key = bytes(signing_key) if signing_key is not None else get_or_create_signing_key()
    if str(config.get("signer_key_id", "")) != signing_key_id(key):
        raise RemoteRegistryError("Registry server config and model trust key do not match.")
    host = _validate_bind_host(str(config.get("host", DEFAULT_HOST)), allow_remote=bool(config.get("allow_remote")))
    port = _validate_port(int(config.get("port", DEFAULT_PORT)))
    token_value = str(token or "").strip()
    if len(token_value) < 48:
        raise RemoteRegistryError("Registry Bearer token is missing or too short.")
    store = RemoteRegistryStore(Path(storage_root) if storage_root else _server_store_dir(), key)
    server = _RegistryHTTPServer((host, port), _RegistryHandler, store=store, token=token_value)
    if str(config.get("scheme")) == "https":
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.minimum_version = ssl.TLSVersion.TLSv1_2
        context.load_cert_chain(str(config.get("tls_cert_path")), str(config.get("tls_key_path")))
        server.socket = context.wrap_socket(server.socket, server_side=True)
    elif not _is_loopback_host(host):
        server.server_close()
        raise RemoteRegistryError("Plain HTTP registry is allowed on loopback only.")
    return server


# ── Client ──────────────────────────────────────────────────────────────────
class RemoteRegistryClient:
    def __init__(
        self,
        profile: Mapping[str, Any],
        token: str,
        *,
        timeout: float = 20.0,
    ) -> None:
        self.profile = dict(profile)
        self.base_url = _validate_remote_base_url(str(profile.get("base_url", "")))
        self.token = str(token or "").strip()
        if len(self.token) < 48:
            raise RemoteRegistryError("Remote registry Bearer token is missing or too short.")
        self.timeout = min(max(float(timeout), 2.0), 120.0)
        ca_path = str(profile.get("ca_cert_path", "") or "")
        if self.base_url.startswith("https://"):
            self.ssl_context = ssl.create_default_context(cafile=ca_path or None)
            self.ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
        else:
            self.ssl_context = None

    def _request(
        self,
        method: str,
        path: str,
        *,
        body: Optional[bytes] = None,
        content_type: str = "application/json",
        headers: Optional[Mapping[str, str]] = None,
        max_response: int = MAX_RESPONSE_BYTES,
    ) -> tuple[int, bytes, Mapping[str, str]]:
        request_headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/json, application/octet-stream",
        }
        if body is not None:
            request_headers["Content-Type"] = content_type
        request_headers.update({str(k): str(v) for k, v in (headers or {}).items()})
        req = urllib.request.Request(self.base_url + path, data=body, headers=request_headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout, context=self.ssl_context) as response:
                length_text = response.headers.get("Content-Length")
                if length_text and int(length_text) > max_response:
                    raise RemoteRegistryError("Remote registry response exceeds the safe size limit.")
                raw = response.read(max_response + 1)
                if len(raw) > max_response:
                    raise RemoteRegistryError("Remote registry response exceeds the safe size limit.")
                return int(response.status), raw, dict(response.headers)
        except urllib.error.HTTPError as exc:
            raw = exc.read(512 * 1024)
            try:
                payload = json.loads(raw.decode("utf-8"))
                message = str((payload.get("error") or {}).get("message") or "Remote registry request failed.")
            except Exception:
                message = "Remote registry request failed."
            raise RemoteRegistryError(f"Remote registry HTTP {exc.code}: {message}") from exc
        except urllib.error.URLError as exc:
            raise RemoteRegistryError("Could not connect to the remote registry securely.") from exc

    @staticmethod
    def _json(raw: bytes) -> Dict[str, Any]:
        try:
            value = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            raise RemoteRegistryError("Remote registry returned invalid JSON.") from exc
        if not isinstance(value, dict):
            raise RemoteRegistryError("Remote registry returned an invalid response shape.")
        return value

    def health(self) -> Dict[str, Any]:
        # Health does not require auth server-side, but the token is harmless and never logged.
        _, raw, _ = self._request("GET", "/health", max_response=256 * 1024)
        return self._json(raw)

    def list_packages(self) -> list[Dict[str, Any]]:
        _, raw, _ = self._request("GET", "/v1/packages", max_response=4 * 1024 * 1024)
        rows = self._json(raw).get("packages") or []
        return [dict(row) for row in rows if isinstance(row, Mapping)]

    def list_families(self) -> list[Dict[str, Any]]:
        _, raw, _ = self._request("GET", "/v1/families", max_response=4 * 1024 * 1024)
        rows = self._json(raw).get("families") or []
        return [dict(row) for row in rows if isinstance(row, Mapping)]

    def get_package(self, package_id: str) -> bytes:
        safe = _safe_id(package_id, label="package ID")
        _, raw, _ = self._request("GET", f"/v1/packages/{safe}", max_response=MAX_PACKAGE_BYTES)
        return raw

    def put_package(self, package_id: str, raw: bytes) -> Dict[str, Any]:
        safe = _safe_id(package_id, label="package ID")
        _, response, _ = self._request(
            "PUT",
            f"/v1/packages/{safe}",
            body=raw,
            content_type="application/vnd.databridge.model-package",
            max_response=512 * 1024,
        )
        return self._json(response).get("stored") or {}

    def get_governance(self, family_id: str) -> bytes:
        safe = _safe_id(family_id, label="family ID")
        _, raw, _ = self._request(
            "GET", f"/v1/families/{safe}/governance", max_response=MAX_GOVERNANCE_BYTES
        )
        return raw

    def put_governance(
        self,
        family_id: str,
        raw: bytes,
        *,
        expected_revision: int,
        expected_digest: str,
    ) -> Dict[str, Any]:
        safe = _safe_id(family_id, label="family ID")
        _, response, _ = self._request(
            "PUT",
            f"/v1/families/{safe}/governance",
            body=raw,
            content_type="application/vnd.databridge.governance+json",
            headers={
                "X-Expected-Revision": str(int(expected_revision)),
                "X-Expected-Digest": str(expected_digest or ""),
            },
            max_response=512 * 1024,
        )
        return self._json(response).get("governance") or {}


def client_from_profile(profile_id: str, *, backend: Any | None = None) -> RemoteRegistryClient:
    profile = load_remote_registry_client_profile(profile_id)
    token = _load_remote_registry_token(backend=backend)
    client = RemoteRegistryClient(profile, token)
    health = client.health()
    local_key_id = signing_key_id(get_or_create_signing_key())
    remote_key_id = str(health.get("signer_key_id", ""))
    if remote_key_id != local_key_id:
        raise RemoteRegistryError(
            "Remote registry uses a different model trust key. Restore the same encrypted signing-key backup on every trusted registry node before synchronization."
        )
    return client


def push_family_to_remote(
    profile_id: str,
    family_id: str,
    *,
    backend: Any | None = None,
) -> RemoteSyncResult:
    client = client_from_profile(profile_id, backend=backend)
    local_state = get_governance_state(family_id)
    local_raw = export_governance_envelope(family_id)
    remote_family = next((row for row in client.list_families() if str(row.get("family_id")) == family_id), None)
    if remote_family:
        expected_revision = int(remote_family.get("revision", 0))
        expected_digest = str(remote_family.get("digest", ""))
        try:
            remote_raw = client.get_governance(family_id)
            remote_state = validate_governance_envelope_bytes(remote_raw)
        except Exception as exc:
            raise RemoteRegistryError(f"Could not verify current remote governance before push: {exc}") from exc
        if governance_state_digest(remote_state) == governance_state_digest(local_state):
            return RemoteSyncResult(
                family_id=family_id,
                family_name=str(local_state.get("family_name", "")),
                direction="push",
                packages_transferred=0,
                governance_revision=int(local_state.get("revision", 0)),
                champion_id=str(local_state.get("champion_id", "") or ""),
                message="Remote family is already synchronized.",
            )
        if not governance_is_fast_forward(remote_state, local_state):
            raise RemoteRegistryError(
                "Local governance is not a fast-forward of the remote family. Pull the remote family and resolve the divergence before pushing."
            )
    else:
        expected_revision = -1
        expected_digest = ""

    transferred = 0
    remote_packages = {str(row.get("package_id")) for row in client.list_packages()}
    for package_id, row in (local_state.get("models") or {}).items():
        if str((row or {}).get("status", "")) == "Removed":
            continue
        if str(package_id) in remote_packages:
            continue
        raw = load_registered_package_bytes(str(package_id))
        # Verify with the local shared trust key before transport.
        package = load_signed_model_package(raw)
        if package.package_id != str(package_id):
            raise RemoteRegistryError("Local registry package ID mismatch blocked remote upload.")
        client.put_package(str(package_id), raw)
        transferred += 1

    result = client.put_governance(
        family_id,
        local_raw,
        expected_revision=expected_revision,
        expected_digest=expected_digest,
    )
    return RemoteSyncResult(
        family_id=family_id,
        family_name=str(local_state.get("family_name", "")),
        direction="push",
        packages_transferred=transferred,
        governance_revision=int(result.get("revision", local_state.get("revision", 0))),
        champion_id=str(result.get("champion_id", local_state.get("champion_id", "")) or ""),
        message="Family pushed to the authenticated remote registry.",
    )


def pull_family_from_remote(
    profile_id: str,
    family_id: str,
    *,
    backend: Any | None = None,
) -> RemoteSyncResult:
    client = client_from_profile(profile_id, backend=backend)
    remote_raw = client.get_governance(family_id)
    remote_state = validate_governance_envelope_bytes(remote_raw)
    if str(remote_state.get("family_id")) != str(family_id):
        raise RemoteRegistryError("Remote governance family ID mismatch.")

    try:
        local_state = get_governance_state(family_id)
    except ModelGovernanceError:
        local_state = None
    if local_state is not None:
        if governance_state_digest(local_state) == governance_state_digest(remote_state):
            return RemoteSyncResult(
                family_id=family_id,
                family_name=str(remote_state.get("family_name", "")),
                direction="pull",
                packages_transferred=0,
                governance_revision=int(remote_state.get("revision", 0)),
                champion_id=str(remote_state.get("champion_id", "") or ""),
                message="Local family is already synchronized.",
            )
        if not governance_is_fast_forward(local_state, remote_state):
            raise RemoteRegistryError(
                "Remote governance is not a safe fast-forward of local history. Synchronization stopped to prevent silent model-governance loss."
            )

    transferred = 0
    family_name = str(remote_state.get("family_name", ""))
    for package_id, row in (remote_state.get("models") or {}).items():
        if str((row or {}).get("status", "")) == "Removed":
            continue
        package_id = str(package_id)
        try:
            existing = load_registered_package_bytes(package_id)
            verified = load_signed_model_package(existing)
            if verified.package_id != package_id:
                raise RemoteRegistryError("Local package verification mismatch.")
            continue
        except Exception:
            pass
        raw = client.get_package(package_id)
        package = load_signed_model_package(raw)
        if package.package_id != package_id:
            raise RemoteRegistryError("Downloaded package ID does not match its signed payload.")
        register_signed_package(
            raw,
            label=str((row or {}).get("label", "")),
            model_family=family_name,
            initialize_governance=False,
        )
        transferred += 1

    try:
        imported = import_governance_envelope(remote_raw)
    except (ModelGovernanceError, ModelRegistryError) as exc:
        raise RemoteRegistryError(str(exc)) from exc
    return RemoteSyncResult(
        family_id=family_id,
        family_name=family_name,
        direction="pull",
        packages_transferred=transferred,
        governance_revision=int(imported.get("revision", 0)),
        champion_id=str(imported.get("champion_id", "") or ""),
        message="Family pulled and governance fast-forwarded safely.",
    )


# ── Background server process helpers ───────────────────────────────────────
def _pid_alive(pid: Optional[int]) -> bool:
    if not pid or int(pid) <= 0:
        return False
    pid = int(pid)
    if os.name == "nt":
        try:
            import ctypes
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            SYNCHRONIZE = 0x00100000
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE, False, pid)
            if not handle:
                return False
            try:
                WAIT_TIMEOUT = 0x00000102
                return kernel32.WaitForSingleObject(handle, 0) == WAIT_TIMEOUT
            finally:
                kernel32.CloseHandle(handle)
        except Exception:
            return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _write_process_status(config: Mapping[str, Any], *, pid: int, running: bool, last_error: str = "") -> None:
    payload = {
        "config_id": config.get("config_id", ""),
        "pid": int(pid),
        "running": bool(running),
        "scheme": config.get("scheme", "http"),
        "host": config.get("host", DEFAULT_HOST),
        "port": int(config.get("port", DEFAULT_PORT)),
        "started_at": _utc_now() if running else "",
        "last_error": _safe_text(last_error, limit=500),
    }
    _atomic_write(
        _status_path(str(config.get("config_id", ""))),
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8"),
    )


def read_remote_registry_process_status(config_id: str) -> RemoteRegistryProcessStatus:
    config = load_remote_registry_server_config(config_id)
    path = _status_path(config_id)
    payload: Dict[str, Any] = {}
    if path.is_file() and path.stat().st_size <= MAX_CONFIG_BYTES:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
    pid = int(payload.get("pid", 0) or 0)
    running = bool(payload.get("running")) and _pid_alive(pid)
    return RemoteRegistryProcessStatus(
        config_id=config_id,
        running=running,
        pid=pid if running else None,
        scheme=str(config.get("scheme", "http")),
        host=str(config.get("host", DEFAULT_HOST)),
        port=int(config.get("port", DEFAULT_PORT)),
        started_at=str(payload.get("started_at", "")) if running else "",
        last_error=str(payload.get("last_error", "")),
    )


def _server_command(config_id: str) -> list[str]:
    return [sys.executable, "-m", "modules.remote_model_registry", "--serve", config_id]


def start_remote_registry_process(config_id: str) -> RemoteRegistryProcessStatus:
    config = load_remote_registry_server_config(config_id)
    existing = read_remote_registry_process_status(config_id)
    if existing.running:
        return existing
    _load_remote_registry_token()
    creationflags = 0
    kwargs: Dict[str, Any] = {}
    if os.name == "nt":
        creationflags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
    else:
        kwargs["start_new_session"] = True
    process = subprocess.Popen(
        _server_command(config_id),
        cwd=str(Path(__file__).resolve().parents[1]),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
        **kwargs,
    )
    _write_process_status(config, pid=process.pid, running=True)
    time.sleep(0.35)
    if process.poll() is not None:
        _write_process_status(config, pid=process.pid, running=False, last_error="Registry server exited during startup.")
        raise RemoteRegistryError("Remote registry server exited during startup.")
    return read_remote_registry_process_status(config_id)


def stop_remote_registry_process(config_id: str) -> RemoteRegistryProcessStatus:
    config = load_remote_registry_server_config(config_id)
    status = read_remote_registry_process_status(config_id)
    if not status.running or not status.pid:
        _write_process_status(config, pid=0, running=False)
        return read_remote_registry_process_status(config_id)
    pid = int(status.pid)
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
                check=False,
            )
        else:
            os.kill(pid, 15)
    except Exception as exc:
        raise RemoteRegistryError(f"Could not stop registry server safely: {exc}") from exc
    for _ in range(30):
        if not _pid_alive(pid):
            break
        time.sleep(0.1)
    _write_process_status(config, pid=0, running=False)
    return read_remote_registry_process_status(config_id)


def run_remote_registry_server(config_id: str) -> None:
    config = load_remote_registry_server_config(config_id)
    token = _load_remote_registry_token()
    server = create_remote_registry_http_server(config, token)
    _write_process_status(config, pid=os.getpid(), running=True)
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()
        _write_process_status(config, pid=0, running=False)


def _main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="DataBridge AI Remote Model Registry")
    parser.add_argument("--serve", metavar="CONFIG_ID", default="")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.serve:
        run_remote_registry_server(args.serve)
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
