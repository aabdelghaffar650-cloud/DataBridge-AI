"""Security-conscious audit logging and export utilities."""
from __future__ import annotations

import csv
import io
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from core.security import redact_sensitive_text
from core.user_paths import logs_dir

_SECRET_FIELD_TOKENS = (
    "password",
    "passwd",
    "api_key",
    "apikey",
    "token",
    "secret",
    "credential",
    "authorization",
    "connection_url",
    "database_url",
)
_MAX_TEXT_LENGTH = 500


def _is_secret_field(name: str) -> bool:
    folded = str(name or "").casefold()
    return any(token in folded for token in _SECRET_FIELD_TOKENS)


def _safe_scalar(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    text = redact_sensitive_text(str(value))
    if len(text) > _MAX_TEXT_LENGTH:
        text = text[:_MAX_TEXT_LENGTH] + "…"
    return text


def sanitise_audit_value(value: Any, *, field_name: str = "") -> Any:
    """Recursively remove secrets and normalise values for JSON export."""
    if _is_secret_field(field_name):
        return "***REDACTED***"
    if isinstance(value, Mapping):
        return {
            str(key): sanitise_audit_value(item, field_name=str(key))
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [sanitise_audit_value(item, field_name=field_name) for item in value]
    return _safe_scalar(value)


def sanitise_audit_event(event: Mapping[str, Any]) -> dict[str, Any]:
    item = sanitise_audit_value(dict(event))
    assert isinstance(item, dict)
    timestamp = item.get("timestamp", time.time())
    try:
        timestamp_value = float(timestamp)
    except Exception:
        timestamp_value = time.time()
    item["timestamp"] = timestamp_value
    item["timestamp_utc"] = datetime.fromtimestamp(
        timestamp_value, tz=timezone.utc
    ).isoformat(timespec="seconds")
    item.setdefault("schema_version", 1)
    return item


def _persistent_audit_path() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return logs_dir() / f"audit-{stamp}.jsonl"


def write_persistent_audit_event(event: Mapping[str, Any]) -> None:
    """Append a sanitized JSONL event. Fail closed without breaking the app."""
    if os.getenv("DATABRIDGE_DISABLE_PERSISTENT_AUDIT", "").strip() == "1":
        return
    try:
        item = sanitise_audit_event(event)
        path = _persistent_audit_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    except Exception:
        # Audit logging must never leak source data through a secondary exception.
        return


def audit_jsonl_bytes(events: Iterable[Mapping[str, Any]]) -> bytes:
    lines = [
        json.dumps(sanitise_audit_event(event), ensure_ascii=False, sort_keys=True)
        for event in events
    ]
    return (("\n".join(lines) + "\n") if lines else "").encode("utf-8")


def audit_csv_bytes(events: Iterable[Mapping[str, Any]]) -> bytes:
    rows = [sanitise_audit_event(event) for event in events]
    preferred = [
        "timestamp_utc",
        "event",
        "action",
        "revision",
        "before_shape",
        "after_shape",
        "quality_score",
        "task",
        "target",
        "selected_model",
        "package_id",
        "rows",
    ]
    extra = sorted({key for row in rows for key in row.keys()} - set(preferred))
    fieldnames = [key for key in preferred if any(key in row for row in rows)] + extra
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        flattened = {}
        for key in fieldnames:
            value = row.get(key, "")
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False, sort_keys=True)
            flattened[key] = value
        writer.writerow(flattened)
    return output.getvalue().encode("utf-8-sig")


def load_persistent_audit_events(*, limit: int = 5000) -> list[dict[str, Any]]:
    """Read recent sanitized JSONL audit events from the per-user log directory."""
    cap = max(1, min(int(limit), 50_000))
    events: list[dict[str, Any]] = []
    try:
        files = sorted(logs_dir().glob("audit-*.jsonl"), reverse=True)
        for path in files:
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except Exception:
                continue
            for line in reversed(lines):
                try:
                    item = json.loads(line)
                    if isinstance(item, dict):
                        events.append(sanitise_audit_event(item))
                except Exception:
                    continue
                if len(events) >= cap:
                    return list(reversed(events))
        return list(reversed(events))
    except Exception:
        return []
