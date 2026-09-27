# ════════════════════════════════════════════════════════
# DataBridge AI — Scheduled Monitoring Reports
# Stage 17: authenticated jobs + unattended Windows Task Scheduler runner
# ════════════════════════════════════════════════════════
from __future__ import annotations

import argparse
import hashlib
import hmac
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from config.constants import MAX_UPLOAD_SIZE_MB
from core.audit import write_persistent_audit_event
from core.security import redact_sensitive_text
from core.user_paths import scheduled_monitoring_dir, scheduled_monitoring_reports_dir
from modules.import_engine import smart_parse_file
from modules.model_governance import current_champion
from modules.model_monitoring import run_monitoring_analysis
from modules.model_package import (
    LoadedModelPackage,
    get_or_create_signing_key,
    load_signed_model_package,
    signing_key_id,
)
from modules.model_registry import load_registered_package_bytes

SCHEDULER_FORMAT = "DataBridgeAI Scheduled Monitoring Job"
SCHEDULER_VERSION = 1
SCHEDULER_CONTEXT = b"DataBridgeAI-Scheduled-Monitoring-v1\0"
TASK_PREFIX = "DataBridgeAI_Monitoring_"
MAX_JOB_BYTES = 256 * 1024
MAX_REPORT_BYTES = 8 * 1024 * 1024
MAX_JOBS = 200
LOCK_STALE_SECONDS = 6 * 60 * 60
ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".json", ".jsonl", ".ndjson", ".parquet"}
VALID_CADENCES = {"daily", "weekly", "monthly"}
VALID_WEEKDAYS = {"MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"}
RUN_COMPLETED = "Completed"
RUN_SKIPPED_UNCHANGED = "SkippedUnchanged"
RUN_SKIPPED_STALE = "SkippedStale"
RUN_DISABLED = "Disabled"
RUN_FAILED = "Failed"


class MonitoringSchedulerError(ValueError):
    pass


@dataclass(frozen=True)
class SourceSnapshot:
    path: Path
    frame: pd.DataFrame
    sha256: str
    size_bytes: int
    modified_at_utc: str


@dataclass(frozen=True)
class ScheduledRunResult:
    job_id: str
    status: str
    package_id: str = ""
    report_path: str = ""
    source_sha256: str = ""
    overall_status: str = ""
    drift_score: Optional[float] = None
    retraining_recommended: bool = False
    message: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_id(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "")).strip("._-")
    if not clean:
        raise MonitoringSchedulerError("Invalid scheduled monitoring identifier.")
    return clean[:96]


def _safe_name(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "").replace("\x00", " ")).strip()
    if not text:
        raise MonitoringSchedulerError("A job name is required.")
    return text[:120]


def _canonical(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(dict(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _job_hmac_key(signing_key: bytes) -> bytes:
    return hmac.new(signing_key, SCHEDULER_CONTEXT, hashlib.sha256).digest()


def _sign(payload: Mapping[str, Any], signing_key: bytes) -> str:
    return hmac.new(_job_hmac_key(signing_key), _canonical(payload), hashlib.sha256).hexdigest()


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
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    finally:
        Path(temp_name).unlink(missing_ok=True)


def _job_path(job_id: str) -> Path:
    return scheduled_monitoring_dir() / "jobs" / f"{_safe_id(job_id)}.json"


def _lock_path(job_id: str) -> Path:
    return scheduled_monitoring_dir() / "locks" / f"{_safe_id(job_id)}.lock"


def _task_name(job_id: str) -> str:
    return TASK_PREFIX + _safe_id(job_id)


def _normalise_time(value: str) -> str:
    match = re.fullmatch(r"(\d{1,2}):(\d{2})", str(value or "").strip())
    if not match:
        raise MonitoringSchedulerError("Schedule time must be HH:MM using local 24-hour time.")
    hour, minute = int(match.group(1)), int(match.group(2))
    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        raise MonitoringSchedulerError("Schedule time is outside 00:00–23:59.")
    return f"{hour:02d}:{minute:02d}"


def normalise_schedule(cadence: str, time_of_day: str, *, weekday: str = "MON", month_day: int = 1) -> Dict[str, Any]:
    cadence = str(cadence or "").strip().lower()
    if cadence not in VALID_CADENCES:
        raise MonitoringSchedulerError("Cadence must be daily, weekly, or monthly.")
    out: Dict[str, Any] = {"cadence": cadence, "time_of_day": _normalise_time(time_of_day)}
    if cadence == "weekly":
        day = str(weekday or "").strip().upper()[:3]
        if day not in VALID_WEEKDAYS:
            raise MonitoringSchedulerError("Weekly schedule requires a valid weekday.")
        out["weekday"] = day
    elif cadence == "monthly":
        day = int(month_day)
        if not 1 <= day <= 28:
            raise MonitoringSchedulerError("Monthly day must be 1–28 so it exists every month.")
        out["month_day"] = day
    return out


def _validate_source(mode: str, source_path: str, pattern: str = "") -> Dict[str, str]:
    mode = str(mode or "").strip().lower()
    if mode not in {"file", "latest_in_folder"}:
        raise MonitoringSchedulerError("Source mode must be file or latest_in_folder.")
    raw = str(source_path or "").strip().strip('"')
    if not raw:
        raise MonitoringSchedulerError("A monitoring source path is required.")
    if raw.startswith("\\\\") or raw.startswith("//"):
        raise MonitoringSchedulerError("Network/UNC paths are blocked for unattended monitoring.")
    path = Path(raw).expanduser()
    if mode == "file":
        if not path.is_file() or path.suffix.lower() not in ALLOWED_EXTENSIONS:
            raise MonitoringSchedulerError("Scheduled monitoring source file is missing or unsupported.")
        return {"mode": mode, "path": str(path.resolve()), "pattern": ""}
    if not path.is_dir():
        raise MonitoringSchedulerError("Scheduled monitoring source folder does not exist.")
    pattern = str(pattern or "").strip() or "*.csv"
    if len(pattern) > 120 or ".." in pattern or "/" in pattern or "\\" in pattern:
        raise MonitoringSchedulerError("Folder pattern must match file names only and cannot traverse folders.")
    return {"mode": mode, "path": str(path.resolve()), "pattern": pattern}


def _verified_family(family_id: str, signing_key: bytes) -> Dict[str, Any]:
    family_id = _safe_id(family_id)
    champion = current_champion(family_id, signing_key=signing_key)
    if not champion:
        raise MonitoringSchedulerError("Scheduled monitoring requires an active governed Champion.")
    package_id = str(champion.get("package_id", ""))
    package = load_signed_model_package(load_registered_package_bytes(package_id), signing_key=signing_key)
    if not package.manifest.get("monitoring", {}).get("training_reference"):
        raise MonitoringSchedulerError("The current Champion has no Stage 14+ monitoring reference.")
    return {
        "family_id": family_id,
        "family_name": str(champion.get("family_name", "")),
        "task": package.task,
        "target": package.target,
        "champion_at_save": package.package_id,
    }


def create_monitoring_job(
    *, name: str, family_id: str, source_mode: str, source_path: str, file_pattern: str = "",
    cadence: str = "daily", time_of_day: str = "08:00", weekday: str = "MON", month_day: int = 1,
    actual_target_column: str = "", require_actual_target: bool = False,
    retention_reports: int = 60, skip_unchanged: bool = True, max_source_age_hours: int = 0,
    enabled: bool = True, job_id: str = "", signing_key: Optional[bytes] = None,
) -> Dict[str, Any]:
    key = bytes(signing_key) if signing_key is not None else get_or_create_signing_key()
    if not job_id and len(list((scheduled_monitoring_dir() / "jobs").glob("*.json"))) >= MAX_JOBS:
        raise MonitoringSchedulerError(f"A maximum of {MAX_JOBS} scheduled jobs is supported.")
    source = _validate_source(source_mode, source_path, file_pattern)
    schedule = normalise_schedule(cadence, time_of_day, weekday=weekday, month_day=month_day)
    family = _verified_family(family_id, key)
    now = _utc_now()
    if not job_id:
        seed = f"{name}|{family_id}|{source['path']}|{time.time_ns()}"
        job_id = "mon-" + hashlib.sha256(seed.encode()).hexdigest()[:16]
    job_id = _safe_id(job_id)
    existing: Dict[str, Any] = {}
    if _job_path(job_id).is_file():
        existing = load_monitoring_job(job_id, signing_key=key)
    job = {
        "format": SCHEDULER_FORMAT,
        "version": SCHEDULER_VERSION,
        "job_id": job_id,
        "name": _safe_name(name),
        "enabled": bool(enabled),
        "created_at": str(existing.get("created_at") or now),
        "updated_at": now,
        "signer_key_id": signing_key_id(key),
        "family": family,
        "source": source,
        "schedule": schedule,
        "actual_target_column": str(actual_target_column or "").strip()[:200],
        "require_actual_target": bool(require_actual_target),
        "retention_reports": max(1, min(int(retention_reports), 365)),
        "skip_unchanged": bool(skip_unchanged),
        "max_source_age_hours": max(0, min(int(max_source_age_hours), 24 * 365)),
        "last_run": dict(existing.get("last_run") or {}),
    }
    save_monitoring_job(job, signing_key=key)
    return load_monitoring_job(job_id, signing_key=key)


def save_monitoring_job(job: Mapping[str, Any], *, signing_key: Optional[bytes] = None) -> Path:
    key = bytes(signing_key) if signing_key is not None else get_or_create_signing_key()
    payload = dict(job)
    payload.pop("signature", None)
    if payload.get("format") != SCHEDULER_FORMAT or int(payload.get("version", 0)) != SCHEDULER_VERSION:
        raise MonitoringSchedulerError("Scheduled job format is invalid.")
    payload["job_id"] = _safe_id(str(payload.get("job_id", "")))
    payload["signer_key_id"] = signing_key_id(key)
    wrapper = {**payload, "signature": _sign(payload, key)}
    data = json.dumps(wrapper, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    if len(data) > MAX_JOB_BYTES:
        raise MonitoringSchedulerError("Scheduled job is unexpectedly large.")
    path = _job_path(payload["job_id"])
    _atomic_write(path, data)
    return path


def load_monitoring_job(job_id: str, *, signing_key: Optional[bytes] = None) -> Dict[str, Any]:
    key = bytes(signing_key) if signing_key is not None else get_or_create_signing_key()
    path = _job_path(job_id)
    if not path.is_file() or not 0 < path.stat().st_size <= MAX_JOB_BYTES:
        raise MonitoringSchedulerError("Scheduled monitoring job was not found or has an unsafe size.")
    try:
        wrapper = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise MonitoringSchedulerError("Scheduled monitoring job is unreadable.") from exc
    if not isinstance(wrapper, dict):
        raise MonitoringSchedulerError("Scheduled monitoring job structure is invalid.")
    signature = str(wrapper.pop("signature", ""))
    if wrapper.get("format") != SCHEDULER_FORMAT or int(wrapper.get("version", 0)) != SCHEDULER_VERSION:
        raise MonitoringSchedulerError("Scheduled monitoring job format/version is unsupported.")
    if str(wrapper.get("signer_key_id", "")) != signing_key_id(key):
        raise MonitoringSchedulerError("Scheduled monitoring job belongs to a different local signing key.")
    if not signature or not hmac.compare_digest(signature, _sign(wrapper, key)):
        raise MonitoringSchedulerError("Scheduled monitoring job authentication failed; it may have been modified.")
    schedule = dict(wrapper.get("schedule") or {})
    normalise_schedule(str(schedule.get("cadence", "")), str(schedule.get("time_of_day", "")),
                       weekday=str(schedule.get("weekday", "MON")), month_day=int(schedule.get("month_day", 1) or 1))
    return wrapper


def list_monitoring_jobs(*, signing_key: Optional[bytes] = None) -> List[Dict[str, Any]]:
    key = bytes(signing_key) if signing_key is not None else get_or_create_signing_key()
    rows: List[Dict[str, Any]] = []
    for path in sorted((scheduled_monitoring_dir() / "jobs").glob("*.json")):
        try:
            rows.append(load_monitoring_job(path.stem, signing_key=key))
        except MonitoringSchedulerError as exc:
            rows.append({"job_id": path.stem, "name": path.stem, "enabled": False, "invalid": True,
                         "error": redact_sensitive_text(str(exc))})
    return sorted(rows, key=lambda row: str(row.get("name", "")).casefold())


def set_job_enabled(job_id: str, enabled: bool, *, signing_key: Optional[bytes] = None) -> Dict[str, Any]:
    key = bytes(signing_key) if signing_key is not None else get_or_create_signing_key()
    job = load_monitoring_job(job_id, signing_key=key)
    job["enabled"] = bool(enabled)
    job["updated_at"] = _utc_now()
    save_monitoring_job(job, signing_key=key)
    return load_monitoring_job(job_id, signing_key=key)


def delete_monitoring_job(job_id: str, *, remove_task: bool = True) -> None:
    job_id = _safe_id(job_id)
    if remove_task and os.name == "nt":
        try:
            remove_windows_scheduled_task(job_id)
        except MonitoringSchedulerError:
            pass
    _job_path(job_id).unlink(missing_ok=True)
    _lock_path(job_id).unlink(missing_ok=True)


def _resolve_source_path(job: Mapping[str, Any]) -> Path:
    source = dict(job.get("source") or {})
    base = Path(str(source.get("path", ""))).expanduser()
    if str(base).startswith("\\\\") or str(base).startswith("//"):
        raise MonitoringSchedulerError("Network sources remain blocked for unattended monitoring.")
    mode = str(source.get("mode", ""))
    if mode == "file":
        path = base
    elif mode == "latest_in_folder":
        if not base.is_dir():
            raise MonitoringSchedulerError("Scheduled monitoring source folder is unavailable.")
        pattern = str(source.get("pattern", "")) or "*.csv"
        if ".." in pattern or "/" in pattern or "\\" in pattern:
            raise MonitoringSchedulerError("Scheduled monitoring pattern is unsafe.")
        candidates = [p for p in base.glob(pattern) if p.is_file() and p.suffix.lower() in ALLOWED_EXTENSIONS]
        if not candidates:
            raise MonitoringSchedulerError("No supported monitoring file matches the configured folder pattern.")
        path = max(candidates, key=lambda p: (p.stat().st_mtime_ns, p.name.casefold()))
    else:
        raise MonitoringSchedulerError("Scheduled monitoring source mode is invalid.")
    if not path.is_file() or path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise MonitoringSchedulerError("Scheduled monitoring source file is unavailable or unsupported.")
    size = path.stat().st_size
    if size <= 0 or size > int(MAX_UPLOAD_SIZE_MB) * 1024 * 1024:
        raise MonitoringSchedulerError(f"Scheduled source must be between 1 byte and {MAX_UPLOAD_SIZE_MB} MB.")
    return path.resolve()


class _NamedBytesIO(io.BytesIO):
    def __init__(self, raw: bytes, name: str):
        super().__init__(raw)
        self.name = name
        self.size = len(raw)


def load_job_source(job: Mapping[str, Any]) -> SourceSnapshot:
    path = _resolve_source_path(job)
    stat = path.stat()
    max_age = int(job.get("max_source_age_hours", 0) or 0)
    if max_age and time.time() - stat.st_mtime > max_age * 3600:
        raise MonitoringSchedulerError(f"Scheduled source is older than the configured {max_age}-hour freshness limit.")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    try:
        frame, import_report = smart_parse_file(_NamedBytesIO(raw, path.name))
    except Exception as exc:
        raise MonitoringSchedulerError(f"Scheduled source parsing failed safely: {redact_sensitive_text(str(exc))}") from exc
    if len(import_report.get("json_array_candidates") or []) > 1:
        raise MonitoringSchedulerError("Scheduled JSON has multiple arrays and cannot be selected unattended.")
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise MonitoringSchedulerError("Scheduled monitoring source produced no rows.")
    return SourceSnapshot(
        path=path,
        frame=frame,
        sha256=digest,
        size_bytes=int(stat.st_size),
        modified_at_utc=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).replace(microsecond=0).isoformat(),
    )


def _report_dir(job_id: str) -> Path:
    path = scheduled_monitoring_reports_dir() / _safe_id(job_id)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _apply_retention(job_id: str, keep: int) -> None:
    root = _report_dir(job_id)
    files = sorted(root.glob("*.json"), key=lambda p: p.stat().st_mtime_ns, reverse=True)
    for path in files[max(1, min(int(keep), 365)):]:
        path.unlink(missing_ok=True)
        path.with_suffix(".csv").unlink(missing_ok=True)


def save_scheduled_report(job: Mapping[str, Any], report: Mapping[str, Any], feature_table: pd.DataFrame,
                          *, source: SourceSnapshot, package_id: str) -> Path:
    job_id = _safe_id(str(job.get("job_id", "")))
    payload = dict(report)
    payload["scheduled_run"] = {
        "job_id": job_id,
        "job_name": str(job.get("name", ""))[:120],
        "family_id": str((job.get("family") or {}).get("family_id", "")),
        "package_id": package_id,
        "source_name": source.path.name,
        "source_sha256": source.sha256,
        "source_size_bytes": source.size_bytes,
        "source_modified_at_utc": source.modified_at_utc,
        "source_path_persisted": False,
        "schedule": dict(job.get("schedule") or {}),
        "retraining_recommended": str(report.get("overall_status", "")) in {"Drifted", "Critical"},
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    base = _report_dir(job_id) / f"{stamp}_{_safe_id(package_id)[:40]}"
    json_path, csv_path = base.with_suffix(".json"), base.with_suffix(".csv")
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    if len(encoded) > MAX_REPORT_BYTES:
        raise MonitoringSchedulerError("Scheduled monitoring report is unexpectedly large.")
    _atomic_write(json_path, encoded)
    _atomic_write(csv_path, feature_table.to_csv(index=False).encode("utf-8-sig"))
    _apply_retention(job_id, int(job.get("retention_reports", 60) or 60))
    return json_path


def list_scheduled_reports(job_id: str, *, limit: int = 100) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    files = sorted(_report_dir(job_id).glob("*.json"), key=lambda p: p.stat().st_mtime_ns, reverse=True)
    for path in files[:max(1, min(int(limit), 365))]:
        try:
            if path.stat().st_size > MAX_REPORT_BYTES:
                continue
            report = json.loads(path.read_text(encoding="utf-8"))
            scheduled = dict(report.get("scheduled_run") or {})
            rows.append({
                "path": str(path), "created_at": report.get("created_at", ""), "package_id": report.get("package_id", ""),
                "overall_status": report.get("overall_status", ""), "drift_score": report.get("drift_score"),
                "rows": report.get("rows"), "source_name": scheduled.get("source_name", ""),
                "source_sha256": scheduled.get("source_sha256", ""),
                "retraining_recommended": bool(scheduled.get("retraining_recommended", False)),
            })
        except Exception:
            continue
    return rows


def load_scheduled_report(path: str | Path) -> Dict[str, Any]:
    candidate = Path(path).expanduser().resolve()
    root = scheduled_monitoring_reports_dir().resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise MonitoringSchedulerError("Scheduled report path is outside the application report directory.") from exc
    if not candidate.is_file() or candidate.suffix.lower() != ".json" or candidate.stat().st_size > MAX_REPORT_BYTES:
        raise MonitoringSchedulerError("Scheduled monitoring report is unavailable or unsafe.")
    try:
        report = json.loads(candidate.read_text(encoding="utf-8"))
    except Exception as exc:
        raise MonitoringSchedulerError("Scheduled monitoring report is unreadable.") from exc
    if not isinstance(report, dict) or not report.get("package_id"):
        raise MonitoringSchedulerError("Scheduled monitoring report structure is invalid.")
    return report


@contextmanager
def _job_lock(job_id: str):
    path = _lock_path(job_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            if time.time() - path.stat().st_mtime > LOCK_STALE_SECONDS:
                path.unlink(missing_ok=True)
        except OSError:
            pass
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise MonitoringSchedulerError("This scheduled monitoring job is already running.") from exc
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(f"pid={os.getpid()}\nstarted={_utc_now()}\n")
        yield
    finally:
        path.unlink(missing_ok=True)


def _update_last_run(job: Dict[str, Any], result: ScheduledRunResult, key: bytes) -> None:
    job["last_run"] = {
        "finished_at": _utc_now(), "status": result.status, "package_id": result.package_id,
        "report_path": Path(result.report_path).name if result.report_path else "",
        "source_sha256": result.source_sha256, "overall_status": result.overall_status,
        "drift_score": result.drift_score, "retraining_recommended": result.retraining_recommended,
        "message": redact_sensitive_text(result.message)[:500],
    }
    job["updated_at"] = _utc_now()
    save_monitoring_job(job, signing_key=key)


def run_scheduled_job(job_id: str, *, signing_key: Optional[bytes] = None) -> ScheduledRunResult:
    key = bytes(signing_key) if signing_key is not None else get_or_create_signing_key()
    job = load_monitoring_job(job_id, signing_key=key)
    if not bool(job.get("enabled", True)):
        result = ScheduledRunResult(_safe_id(job_id), RUN_DISABLED, message="Job is disabled.")
        _update_last_run(job, result, key)
        return result
    with _job_lock(job_id):
        try:
            family_id = _safe_id(str((job.get("family") or {}).get("family_id", "")))
            champion = current_champion(family_id, signing_key=key)
            if not champion:
                raise MonitoringSchedulerError("The governed family has no active Champion.")
            package_id = str(champion.get("package_id", ""))
            package = load_signed_model_package(load_registered_package_bytes(package_id), signing_key=key)
            if not isinstance(package, LoadedModelPackage) or not package.trusted:
                raise MonitoringSchedulerError("The current Champion failed signed-package verification.")
            if not package.manifest.get("monitoring", {}).get("training_reference"):
                raise MonitoringSchedulerError("The current Champion has no monitoring reference.")
            try:
                source = load_job_source(job)
            except MonitoringSchedulerError as exc:
                if "freshness limit" in str(exc):
                    result = ScheduledRunResult(_safe_id(job_id), RUN_SKIPPED_STALE, package_id=package_id, message=str(exc))
                    _update_last_run(job, result, key)
                    return result
                raise
            previous = str((job.get("last_run") or {}).get("source_sha256", ""))
            if bool(job.get("skip_unchanged", True)) and previous and hmac.compare_digest(previous, source.sha256):
                result = ScheduledRunResult(_safe_id(job_id), RUN_SKIPPED_UNCHANGED, package_id=package_id,
                                            source_sha256=source.sha256, message="Source is unchanged since the previous run.")
                _update_last_run(job, result, key)
                return result
            actual = str(job.get("actual_target_column", "") or "").strip() or None
            if actual and bool(job.get("require_actual_target", False)) and actual not in source.frame.columns:
                raise MonitoringSchedulerError("The required actual-outcome column is missing from the scheduled source.")
            if actual and actual not in source.frame.columns:
                actual = None
            monitoring = run_monitoring_analysis(package, source.frame, signing_key=key,
                                                 actual_target_column=actual, include_prediction_output=False)
            report_path = save_scheduled_report(job, monitoring.report, monitoring.feature_table,
                                                source=source, package_id=package_id)
            overall = str(monitoring.report.get("overall_status", ""))
            try:
                drift_score = float(monitoring.report.get("drift_score"))
            except (TypeError, ValueError):
                drift_score = None
            result = ScheduledRunResult(
                _safe_id(job_id), RUN_COMPLETED, package_id=package_id, report_path=str(report_path),
                source_sha256=source.sha256, overall_status=overall, drift_score=drift_score,
                retraining_recommended=overall in {"Drifted", "Critical"},
                message="Scheduled monitoring report completed successfully.",
            )
            _update_last_run(job, result, key)
            write_persistent_audit_event({
                "event": "scheduled_monitoring_completed", "action": "Scheduled model monitoring completed",
                "job_id": job_id, "package_id": package_id, "overall_status": overall,
                "drift_score": drift_score, "retraining_recommended": result.retraining_recommended,
                "rows": len(source.frame),
            })
            return result
        except Exception as exc:
            message = redact_sensitive_text(str(exc))[:500]
            result = ScheduledRunResult(_safe_id(job_id), RUN_FAILED, message=message)
            try:
                _update_last_run(job, result, key)
            except Exception:
                pass
            write_persistent_audit_event({"event": "scheduled_monitoring_failed",
                                          "action": "Scheduled model monitoring failed safely",
                                          "job_id": job_id, "error": message})
            if isinstance(exc, MonitoringSchedulerError):
                raise
            raise MonitoringSchedulerError(f"Scheduled monitoring failed safely: {message}") from exc


def _runner_python() -> Path:
    current = Path(sys.executable).resolve()
    if current.name.lower() == "pythonw.exe" and current.with_name("python.exe").is_file():
        return current.with_name("python.exe")
    return current


def windows_task_action(job_id: str) -> str:
    return f'"{_runner_python()}" "{Path(__file__).resolve()}" --run-job "{_safe_id(job_id)}"'


def windows_task_arguments(job: Mapping[str, Any]) -> List[str]:
    schedule = dict(job.get("schedule") or {})
    cadence = str(schedule.get("cadence", ""))
    args = ["schtasks", "/Create", "/TN", _task_name(str(job.get("job_id", ""))),
            "/TR", windows_task_action(str(job.get("job_id", ""))), "/ST", _normalise_time(str(schedule.get("time_of_day", ""))),
            "/RL", "LIMITED", "/F"]
    if cadence == "daily":
        args += ["/SC", "DAILY"]
    elif cadence == "weekly":
        day = str(schedule.get("weekday", "MON")).upper()
        if day not in VALID_WEEKDAYS:
            raise MonitoringSchedulerError("Weekly task has an invalid weekday.")
        args += ["/SC", "WEEKLY", "/D", day]
    elif cadence == "monthly":
        day = int(schedule.get("month_day", 1) or 1)
        if not 1 <= day <= 28:
            raise MonitoringSchedulerError("Monthly task has an invalid day.")
        args += ["/SC", "MONTHLY", "/D", str(day)]
    else:
        raise MonitoringSchedulerError("Scheduled job cadence is invalid.")
    return args


def _run_windows(args: Sequence[str]) -> subprocess.CompletedProcess[str]:
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        return subprocess.run(list(args), capture_output=True, text=True, timeout=30, creationflags=flags, check=False)
    except Exception as exc:
        raise MonitoringSchedulerError(f"Windows Task Scheduler command failed: {redact_sensitive_text(str(exc))}") from exc


def install_windows_scheduled_task(job_id: str) -> None:
    if os.name != "nt":
        raise MonitoringSchedulerError("Windows Task Scheduler integration is available only on Windows.")
    job = load_monitoring_job(job_id)
    result = _run_windows(windows_task_arguments(job))
    if result.returncode != 0:
        msg = redact_sensitive_text((result.stderr or result.stdout or "Task creation failed").strip())
        raise MonitoringSchedulerError(f"Could not install scheduled task: {msg[:500]}")


def remove_windows_scheduled_task(job_id: str) -> None:
    if os.name != "nt":
        raise MonitoringSchedulerError("Windows Task Scheduler integration is available only on Windows.")
    result = _run_windows(["schtasks", "/Delete", "/TN", _task_name(job_id), "/F"])
    if result.returncode != 0:
        text = (result.stderr or result.stdout or "").casefold()
        if "cannot find" not in text and "not exist" not in text:
            raise MonitoringSchedulerError("Could not remove the Windows scheduled task.")


def windows_task_installed(job_id: str) -> bool:
    if os.name != "nt":
        return False
    return _run_windows(["schtasks", "/Query", "/TN", _task_name(job_id), "/FO", "LIST"]).returncode == 0


def _cli(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="DataBridge AI scheduled monitoring runner")
    parser.add_argument("--run-job", default="")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.run_job:
        parser.error("--run-job is required")
    try:
        result = run_scheduled_job(args.run_job)
        print(json.dumps(result.as_dict(), ensure_ascii=False, sort_keys=True))
        return 0 if result.status != RUN_FAILED else 2
    except Exception as exc:
        print(redact_sensitive_text(str(exc)), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(_cli())
