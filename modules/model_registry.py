# ════════════════════════════════════════════════════════
# DataBridge AI — Local Signed Model Registry
# Stage 14: verified package registry + monitoring report persistence
# ════════════════════════════════════════════════════════
from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping

from core.user_paths import model_registry_dir, monitoring_reports_dir
from modules.model_package import (
    MAX_PACKAGE_BYTES,
    LoadedModelPackage,
    ModelPackageError,
    load_signed_model_package,
)


class ModelRegistryError(ValueError):
    pass


def _safe_id(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._-")
    if not clean:
        raise ModelRegistryError("Invalid registry identifier.")
    return clean[:120]


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
        os.replace(temp_name, path)
    finally:
        try:
            Path(temp_name).unlink(missing_ok=True)
        except Exception:
            pass


def _metadata_from_package(
    package: LoadedModelPackage,
    *,
    label: str = "",
    model_family: str = "",
) -> Dict[str, Any]:
    manifest = package.manifest
    if not model_family:
        from modules.model_governance import default_family_name
        model_family = default_family_name(package.task, package.target)
    return {
        "package_id": package.package_id,
        "label": str(label or "")[:160],
        "model_family": str(model_family or "")[:120],
        "registered_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "created_at": str(manifest.get("created_at", "")),
        "application_version": str(manifest.get("application", {}).get("version", "")),
        "experiment_id": str(manifest.get("experiment", {}).get("experiment_id", "")),
        "task": package.task,
        "target": package.target,
        "selected_model": str(manifest.get("model", {}).get("selected_model", "")),
        "signer_key_id": package.signer_key_id,
        "has_monitoring_reference": bool(manifest.get("monitoring", {}).get("training_reference")),
    }


def register_signed_package(
    package_bytes: bytes,
    *,
    label: str = "",
    model_family: str = "",
    initialize_governance: bool = True,
) -> Dict[str, Any]:
    if not isinstance(package_bytes, (bytes, bytearray)) or not package_bytes:
        raise ModelRegistryError("Package bytes are required.")
    if len(package_bytes) > MAX_PACKAGE_BYTES:
        raise ModelRegistryError("Package exceeds the safe registry size limit.")
    try:
        package = load_signed_model_package(bytes(package_bytes))
    except ModelPackageError as exc:
        raise ModelRegistryError(str(exc)) from exc
    package_id = _safe_id(package.package_id)
    root = model_registry_dir()
    package_path = root / f"{package_id}.dbmlpkg"
    metadata_path = root / f"{package_id}.json"
    metadata = _metadata_from_package(package, label=label, model_family=model_family)
    _atomic_write(package_path, bytes(package_bytes))
    _atomic_write(
        metadata_path,
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8"),
    )
    # Stage 15: locally-created packages enter governance as Candidates. Stage 19
    # remote pulls can install verified artifacts first, then atomically import
    # the authenticated remote governance envelope without creating a divergent
    # local Candidate event.
    if initialize_governance:
        try:
            from modules.model_governance import ensure_registered_candidate
            governance = ensure_registered_candidate(package.package_id)
            metadata["governance_status"] = governance.get("status", "Candidate")
        except Exception as exc:
            package_path.unlink(missing_ok=True)
            metadata_path.unlink(missing_ok=True)
            raise ModelRegistryError(f"Could not initialize model governance safely: {exc}") from exc
    else:
        metadata["governance_status"] = "Remote sync pending"
    return metadata


def list_registered_packages() -> List[Dict[str, Any]]:
    root = model_registry_dir()
    rows: List[Dict[str, Any]] = []
    for path in root.glob("*.json"):
        try:
            if path.stat().st_size > 256 * 1024:
                continue
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict) or not value.get("package_id"):
                continue
            package_path = root / f"{_safe_id(value['package_id'])}.dbmlpkg"
            if not package_path.is_file():
                continue
            value["size_bytes"] = int(package_path.stat().st_size)
            rows.append(value)
        except Exception:
            continue
    rows.sort(key=lambda row: str(row.get("created_at", "")), reverse=True)
    return rows


def load_registered_package_bytes(package_id: str) -> bytes:
    package_id = _safe_id(package_id)
    path = model_registry_dir() / f"{package_id}.dbmlpkg"
    if not path.is_file():
        raise ModelRegistryError("Registered package not found.")
    size = path.stat().st_size
    if size <= 0 or size > MAX_PACKAGE_BYTES:
        raise ModelRegistryError("Registered package has an unsafe size.")
    raw = path.read_bytes()
    try:
        package = load_signed_model_package(raw)
    except ModelPackageError as exc:
        raise ModelRegistryError(f"Registered package failed verification: {exc}") from exc
    if package.package_id != package_id:
        raise ModelRegistryError("Registry metadata does not match the verified package ID.")
    return raw


def delete_registered_package(package_id: str) -> None:
    package_id = _safe_id(package_id)
    # Stage 15: the active Champion is protected from deletion. Non-Champion
    # removals remain in the authenticated governance event history.
    try:
        from modules.model_governance import (
            can_delete_registered_package,
            mark_registered_package_removed,
        )
        allowed, reason = can_delete_registered_package(package_id)
        if not allowed:
            raise ModelRegistryError(reason)
        mark_registered_package_removed(package_id)
    except ModelRegistryError:
        raise
    except Exception as exc:
        raise ModelRegistryError(f"Registry governance check failed safely: {exc}") from exc
    root = model_registry_dir()
    (root / f"{package_id}.dbmlpkg").unlink(missing_ok=True)
    (root / f"{package_id}.json").unlink(missing_ok=True)


def save_monitoring_report(report: Mapping[str, Any]) -> Path:
    package_id = _safe_id(str(report.get("package_id", "package")))
    created = str(report.get("created_at", "")).replace(":", "-").replace("+", "_")
    created = _safe_id(created or "monitor")
    path = monitoring_reports_dir() / f"{package_id}_{created}.json"
    payload = json.dumps(dict(report), ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    if len(payload) > 8 * 1024 * 1024:
        raise ModelRegistryError("Monitoring report is unexpectedly large.")
    _atomic_write(path, payload)
    return path
