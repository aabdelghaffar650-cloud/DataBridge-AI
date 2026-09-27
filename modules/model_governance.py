# ════════════════════════════════════════════════════════
# DataBridge AI — Local Model Governance
# Stage 15: Champion / Challenger lifecycle + human approval gate
# ════════════════════════════════════════════════════════
from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

from core.user_paths import model_registry_dir
from modules.model_package import (
    LoadedModelPackage,
    ModelPackageError,
    get_or_create_signing_key,
    load_signed_model_package,
    signing_key_id,
)


GOVERNANCE_FORMAT = "DataBridgeAI Local Model Governance"
GOVERNANCE_VERSION = 1
GOVERNANCE_CONTEXT = b"DataBridgeAI-Model-Governance-v1\0"
MAX_GOVERNANCE_BYTES = 2 * 1024 * 1024
MAX_EVENTS = 1000

STATUS_CANDIDATE = "Candidate"
STATUS_CHALLENGER = "Challenger"
STATUS_CHAMPION = "Champion"
STATUS_ARCHIVED = "Archived"
STATUS_REJECTED = "Rejected"
STATUS_REMOVED = "Removed"
VALID_STATUSES = {
    STATUS_CANDIDATE,
    STATUS_CHALLENGER,
    STATUS_CHAMPION,
    STATUS_ARCHIVED,
    STATUS_REJECTED,
    STATUS_REMOVED,
}


class ModelGovernanceError(ValueError):
    """Raised when a governance transition is unsafe or invalid."""


@dataclass(frozen=True)
class PromotionAssessment:
    family_id: str
    family_name: str
    challenger_id: str
    champion_id: str
    ready: bool
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    comparison: Dict[str, Any]
    approval_token: str
    state_revision: int

    def report(self) -> Dict[str, Any]:
        return {
            "family_id": self.family_id,
            "family_name": self.family_name,
            "challenger_id": self.challenger_id,
            "champion_id": self.champion_id,
            "ready": bool(self.ready),
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
            "comparison": dict(self.comparison),
            "approval_token": self.approval_token,
            "state_revision": int(self.state_revision),
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_text(value: Any, *, limit: int = 500) -> str:
    text = str(value or "").replace("\x00", " ").strip()
    return text[:limit]


def _safe_family_name(value: str) -> str:
    text = re.sub(r"\s+", " ", _safe_text(value, limit=120)).strip()
    if not text:
        raise ModelGovernanceError("Model family name is required.")
    return text


def default_family_name(task: str, target: str) -> str:
    task = _safe_text(task, limit=40).lower() or "model"
    target = _safe_text(target, limit=80) or "target"
    return f"{task} · {target}"


def model_family_id(family_name: str) -> str:
    name = _safe_family_name(family_name)
    digest = hashlib.sha256(name.casefold().encode("utf-8")).hexdigest()[:20]
    return f"family-{digest}"


def _governance_dir() -> Path:
    path = model_registry_dir() / "governance"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _state_path(family_id: str) -> Path:
    if not re.fullmatch(r"family-[0-9a-f]{20}", str(family_id)):
        raise ModelGovernanceError("Invalid model family identifier.")
    return _governance_dir() / f"{family_id}.json"


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _signature(payload: Mapping[str, Any], key: bytes) -> str:
    return hmac.new(key, GOVERNANCE_CONTEXT + _canonical_bytes(payload), hashlib.sha256).hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    if len(data) > MAX_GOVERNANCE_BYTES:
        raise ModelGovernanceError("Governance record exceeded the safe local size limit.")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
        os.replace(tmp_name, path)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    finally:
        try:
            Path(tmp_name).unlink(missing_ok=True)
        except Exception:
            pass


def _empty_state(family_name: str, task: str, target: str, key: bytes) -> Dict[str, Any]:
    family_name = _safe_family_name(family_name)
    return {
        "format": GOVERNANCE_FORMAT,
        "version": GOVERNANCE_VERSION,
        "family_id": model_family_id(family_name),
        "family_name": family_name,
        "task": _safe_text(task, limit=40).lower(),
        "target": _safe_text(target, limit=160),
        "signer_key_id": signing_key_id(key),
        "revision": 0,
        "champion_id": "",
        "models": {},
        "events": [],
        "updated_at": _utc_now(),
    }


def _validate_state_structure(state: Mapping[str, Any]) -> None:
    if state.get("format") != GOVERNANCE_FORMAT or int(state.get("version", 0)) != GOVERNANCE_VERSION:
        raise ModelGovernanceError("Unsupported or invalid governance record format.")
    family_id = str(state.get("family_id", ""))
    if model_family_id(str(state.get("family_name", ""))) != family_id:
        raise ModelGovernanceError("Governance family identifier does not match its family name.")
    models = state.get("models")
    events = state.get("events")
    if not isinstance(models, dict) or not isinstance(events, list):
        raise ModelGovernanceError("Governance record structure is invalid.")
    champion_id = str(state.get("champion_id", "") or "")
    champion_count = 0
    for package_id, row in models.items():
        if not isinstance(row, dict):
            raise ModelGovernanceError("Governance model record is invalid.")
        status = str(row.get("status", ""))
        if status not in VALID_STATUSES:
            raise ModelGovernanceError("Governance model status is invalid.")
        if status == STATUS_CHAMPION:
            champion_count += 1
            if champion_id != str(package_id):
                raise ModelGovernanceError("Champion pointer and model status are inconsistent.")
    if champion_id:
        if champion_id not in models or str(models[champion_id].get("status")) != STATUS_CHAMPION:
            raise ModelGovernanceError("Governance champion pointer is invalid.")
        if champion_count != 1:
            raise ModelGovernanceError("A model family must contain exactly one active Champion.")
    elif champion_count:
        raise ModelGovernanceError("Champion status exists without an active champion pointer.")


def _load_state(family_id: str, *, signing_key: Optional[bytes] = None, key_path: Optional[Path | str] = None) -> Optional[Dict[str, Any]]:
    path = _state_path(family_id)
    if not path.is_file():
        return None
    if path.stat().st_size <= 0 or path.stat().st_size > MAX_GOVERNANCE_BYTES:
        raise ModelGovernanceError("Governance record has an unsafe size.")
    key = bytes(signing_key) if signing_key is not None else get_or_create_signing_key(key_path)
    try:
        envelope = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ModelGovernanceError("Governance record is unreadable.") from exc
    if not isinstance(envelope, dict) or set(envelope) != {"payload", "signature"}:
        raise ModelGovernanceError("Governance envelope structure is invalid.")
    state = envelope.get("payload")
    sig = str(envelope.get("signature", ""))
    if not isinstance(state, dict):
        raise ModelGovernanceError("Governance payload is invalid.")
    if str(state.get("signer_key_id", "")) != signing_key_id(key):
        raise ModelGovernanceError("Governance record belongs to a different signing-key installation.")
    expected = _signature(state, key)
    if not hmac.compare_digest(sig, expected):
        raise ModelGovernanceError("Governance record authentication failed.")
    _validate_state_structure(state)
    return state


def _save_state(state: Mapping[str, Any], *, signing_key: Optional[bytes] = None, key_path: Optional[Path | str] = None) -> Dict[str, Any]:
    key = bytes(signing_key) if signing_key is not None else get_or_create_signing_key(key_path)
    payload = json.loads(json.dumps(dict(state), ensure_ascii=False, allow_nan=False))
    payload["signer_key_id"] = signing_key_id(key)
    payload["updated_at"] = _utc_now()
    _validate_state_structure(payload)
    envelope = {"payload": payload, "signature": _signature(payload, key)}
    data = json.dumps(envelope, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False).encode("utf-8")
    _atomic_write(_state_path(str(payload["family_id"])), data)
    return payload


def _event(state: Dict[str, Any], event: str, package_id: str, *, actor: str, note: str = "", details: Optional[Mapping[str, Any]] = None) -> None:
    previous_hash = str(state.get("events", [])[-1].get("event_hash", "")) if state.get("events") else ""
    base = {
        "event": _safe_text(event, limit=60),
        "package_id": _safe_text(package_id, limit=140),
        "timestamp": _utc_now(),
        "actor": _safe_text(actor or "local-user", limit=100),
        "note": _safe_text(note, limit=1000),
        "details": dict(details or {}),
        "previous_event_hash": previous_hash,
    }
    base["event_hash"] = hashlib.sha256(_canonical_bytes(base)).hexdigest()
    events = list(state.get("events", []))
    events.append(base)
    state["events"] = events[-MAX_EVENTS:]


def _registry_metadata(package_id: str) -> Dict[str, Any]:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(package_id)).strip("._-")
    path = model_registry_dir() / f"{safe}.json"
    if not path.is_file() or path.stat().st_size > 256 * 1024:
        raise ModelGovernanceError("Registered package metadata was not found.")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ModelGovernanceError("Registered package metadata is unreadable.") from exc
    if not isinstance(value, dict) or str(value.get("package_id", "")) != str(package_id):
        raise ModelGovernanceError("Registered package metadata is invalid.")
    return value


def _registered_package_bytes(package_id: str) -> bytes:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(package_id)).strip("._-")
    path = model_registry_dir() / f"{safe}.dbmlpkg"
    if not path.is_file():
        raise ModelGovernanceError("Registered package file was not found.")
    raw = path.read_bytes()
    if not raw:
        raise ModelGovernanceError("Registered package is empty.")
    return raw


def _verified_package(package_id: str, *, signing_key: Optional[bytes] = None, key_path: Optional[Path | str] = None) -> LoadedModelPackage:
    raw = _registered_package_bytes(package_id)
    try:
        return load_signed_model_package(raw, signing_key=signing_key, key_path=key_path)
    except ModelPackageError as exc:
        raise ModelGovernanceError(f"Registered package failed signature verification: {exc}") from exc




def _find_existing_state_for_package(
    package_id: str,
    *,
    signing_key: bytes,
    key_path: Optional[Path | str] = None,
) -> Optional[Dict[str, Any]]:
    """Return authenticated existing membership; unsigned registry metadata cannot move it."""
    matches: List[Dict[str, Any]] = []
    for path in _governance_dir().glob("family-*.json"):
        state = _load_state(path.stem, signing_key=signing_key, key_path=key_path)
        if state is not None and package_id in (state.get("models") or {}):
            matches.append(state)
    if len(matches) > 1:
        raise ModelGovernanceError(
            "Package appears in more than one authenticated governance family; manual repair is required."
        )
    return matches[0] if matches else None

def _family_for_package(package_id: str, package: LoadedModelPackage) -> tuple[str, str]:
    meta = _registry_metadata(package_id)
    name = str(meta.get("model_family") or default_family_name(package.task, package.target))
    name = _safe_family_name(name)
    return name, model_family_id(name)


def _ensure_member(
    package_id: str,
    *,
    signing_key: Optional[bytes] = None,
    key_path: Optional[Path | str] = None,
) -> tuple[Dict[str, Any], LoadedModelPackage]:
    key = bytes(signing_key) if signing_key is not None else get_or_create_signing_key(key_path)
    package = _verified_package(package_id, signing_key=key, key_path=key_path)
    state = _find_existing_state_for_package(
        package_id, signing_key=key, key_path=key_path
    )
    changed = False
    if state is None:
        family_name, family_id = _family_for_package(package_id, package)
        state = _load_state(family_id, signing_key=key, key_path=key_path)
        if state is None:
            state = _empty_state(family_name, package.task, package.target, key)
            changed = True
    if str(state.get("task", "")) != package.task or str(state.get("target", "")) != package.target:
        raise ModelGovernanceError("The registered package does not match its governance family task/target contract.")
    models = state.setdefault("models", {})
    if package_id not in models:
        meta = _registry_metadata(package_id)
        models[package_id] = {
            "status": STATUS_CANDIDATE,
            "label": _safe_text(meta.get("label", ""), limit=160),
            "registered_at": _safe_text(meta.get("registered_at") or meta.get("created_at") or _utc_now(), limit=80),
            "submitted_at": "",
            "promoted_at": "",
            "rejected_at": "",
            "archived_at": "",
            "last_decision_note": "",
        }
        state["revision"] = int(state.get("revision", 0)) + 1
        _event(state, "candidate_registered", package_id, actor="system", note="Registered package entered governance as Candidate.")
        changed = True
    if changed:
        state = _save_state(state, signing_key=key, key_path=key_path)
    return state, package


def ensure_registered_candidate(
    package_id: str,
    *,
    signing_key: Optional[bytes] = None,
    key_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    state, _ = _ensure_member(package_id, signing_key=signing_key, key_path=key_path)
    return dict(state["models"][package_id])


def get_governance_state(
    family_id: str,
    *,
    signing_key: Optional[bytes] = None,
    key_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    state = _load_state(family_id, signing_key=signing_key, key_path=key_path)
    if state is None:
        raise ModelGovernanceError("Model family governance state does not exist.")
    return state


def list_governance_families(
    *, signing_key: Optional[bytes] = None, key_path: Optional[Path | str] = None
) -> List[Dict[str, Any]]:
    key = bytes(signing_key) if signing_key is not None else get_or_create_signing_key(key_path)
    rows: List[Dict[str, Any]] = []
    for path in _governance_dir().glob("family-*.json"):
        try:
            state = _load_state(path.stem, signing_key=key, key_path=key_path)
            if state is None:
                continue
            counts = {status: 0 for status in VALID_STATUSES}
            for row in state.get("models", {}).values():
                status = str(row.get("status", ""))
                if status in counts:
                    counts[status] += 1
            rows.append(
                {
                    "family_id": state["family_id"],
                    "family_name": state["family_name"],
                    "task": state["task"],
                    "target": state["target"],
                    "champion_id": state.get("champion_id", ""),
                    "revision": int(state.get("revision", 0)),
                    "challengers": counts[STATUS_CHALLENGER],
                    "candidates": counts[STATUS_CANDIDATE],
                    "archived": counts[STATUS_ARCHIVED],
                    "rejected": counts[STATUS_REJECTED],
                    "updated_at": state.get("updated_at", ""),
                }
            )
        except ModelGovernanceError:
            continue
    rows.sort(key=lambda row: (str(row.get("family_name", "")).casefold(), str(row.get("family_id", ""))))
    return rows


def governance_status_for_package(
    package_id: str,
    *,
    signing_key: Optional[bytes] = None,
    key_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    state, package = _ensure_member(package_id, signing_key=signing_key, key_path=key_path)
    row = dict(state["models"][package_id])
    row.update(
        {
            "package_id": package_id,
            "family_id": state["family_id"],
            "family_name": state["family_name"],
            "champion_id": state.get("champion_id", ""),
            "task": package.task,
            "target": package.target,
        }
    )
    return row


def submit_as_challenger(
    package_id: str,
    *,
    actor: str = "local-user",
    note: str = "",
    signing_key: Optional[bytes] = None,
    key_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    key = bytes(signing_key) if signing_key is not None else get_or_create_signing_key(key_path)
    state, _ = _ensure_member(package_id, signing_key=key, key_path=key_path)
    row = state["models"][package_id]
    status = str(row.get("status"))
    if status == STATUS_CHAMPION:
        raise ModelGovernanceError("The active Champion cannot be submitted as its own Challenger.")
    if status == STATUS_REJECTED:
        raise ModelGovernanceError("A rejected package cannot be re-submitted unchanged. Register a newly trained package instead.")
    if status == STATUS_REMOVED:
        raise ModelGovernanceError("A removed package cannot be submitted.")
    if status == STATUS_CHALLENGER:
        return governance_status_for_package(package_id, signing_key=key, key_path=key_path)
    row["status"] = STATUS_CHALLENGER
    row["submitted_at"] = _utc_now()
    row["last_decision_note"] = _safe_text(note, limit=1000)
    state["revision"] = int(state.get("revision", 0)) + 1
    _event(state, "challenger_submitted", package_id, actor=actor, note=note)
    state = _save_state(state, signing_key=key, key_path=key_path)
    return governance_status_for_package(package_id, signing_key=key, key_path=key_path)


def _number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _primary_holdout_metric(package: LoadedModelPackage) -> tuple[str, str, Optional[float]]:
    evaluation = package.manifest.get("evaluation", {}) or {}
    metrics = evaluation.get("holdout_metrics", {}) or {}
    if package.task == "classification":
        for name in ("F1 Weighted", "F1 Macro", "Balanced Accuracy", "Accuracy"):
            value = _number(metrics.get(name))
            if value is not None:
                return name, "higher", value
        return "F1 Weighted", "higher", None
    for name in ("RMSE", "MAE"):
        value = _number(metrics.get(name))
        if value is not None:
            return name, "lower", value
    return "RMSE", "lower", None


def _package_summary(package: LoadedModelPackage) -> Dict[str, Any]:
    manifest = package.manifest
    evaluation = manifest.get("evaluation", {}) or {}
    experiment = manifest.get("experiment", {}) or {}
    feature = manifest.get("feature_contract", {}) or {}
    model = manifest.get("model", {}) or {}
    metric, direction, value = _primary_holdout_metric(package)
    return {
        "package_id": package.package_id,
        "task": package.task,
        "target": package.target,
        "model": str(model.get("selected_model", "")),
        "metric": metric,
        "metric_direction": direction,
        "holdout_value": value,
        "holdout_rows": int(experiment.get("holdout_rows", 0) or 0),
        "training_rows": int(experiment.get("training_rows", 0) or 0),
        "dataset_fingerprint": str(experiment.get("dataset_fingerprint", "")),
        "split_strategy": str(experiment.get("split_strategy", "")),
        "pipeline_fingerprint": str(feature.get("effective_pipeline_spec_fingerprint", "") or feature.get("pipeline_spec_fingerprint", "")),
        "required_columns": list(map(str, manifest.get("input_contract", {}).get("required_columns", []) or [])),
        "class_labels": list(map(str, model.get("class_labels", []) or [])),
        "improvement_vs_baseline": _number(evaluation.get("improvement_vs_baseline")),
        "warnings": list(map(str, evaluation.get("warnings", []) or [])),
        "created_at": str(manifest.get("created_at", "")),
    }


def _assessment_payload(state: Mapping[str, Any], challenger: LoadedModelPackage, champion: Optional[LoadedModelPackage]) -> Dict[str, Any]:
    return {
        "family_id": state["family_id"],
        "revision": int(state.get("revision", 0)),
        "challenger_id": challenger.package_id,
        "champion_id": champion.package_id if champion else "",
        "challenger_artifact": challenger.manifest.get("artifact", {}).get("sha256", ""),
        "champion_artifact": champion.manifest.get("artifact", {}).get("sha256", "") if champion else "",
    }


def _approval_token(state: Mapping[str, Any], challenger: LoadedModelPackage, champion: Optional[LoadedModelPackage], key: bytes) -> str:
    payload = _assessment_payload(state, challenger, champion)
    return hmac.new(key, b"promotion-approval\0" + _canonical_bytes(payload), hashlib.sha256).hexdigest()


def assess_promotion(
    challenger_id: str,
    *,
    signing_key: Optional[bytes] = None,
    key_path: Optional[Path | str] = None,
) -> PromotionAssessment:
    key = bytes(signing_key) if signing_key is not None else get_or_create_signing_key(key_path)
    state, challenger = _ensure_member(challenger_id, signing_key=key, key_path=key_path)
    row = state["models"][challenger_id]
    blockers: List[str] = []
    warnings: List[str] = []

    if str(row.get("status")) != STATUS_CHALLENGER:
        blockers.append("Package must be explicitly submitted as a Challenger before promotion.")

    champion_id = str(state.get("champion_id", "") or "")
    champion: Optional[LoadedModelPackage] = None
    if champion_id:
        if champion_id == challenger_id:
            blockers.append("The Challenger is already the active Champion.")
        else:
            champion = _verified_package(champion_id, signing_key=key, key_path=key_path)

    ch = _package_summary(challenger)
    cp = _package_summary(champion) if champion else None

    if not challenger.manifest.get("monitoring", {}).get("training_reference"):
        blockers.append("Challenger has no training-only monitoring reference. Rebuild it with Stage 14+.")
    baseline = ch.get("improvement_vs_baseline")
    if baseline is not None and baseline <= 0:
        blockers.append("Challenger did not beat the baseline during training cross-validation.")
    if int(ch.get("holdout_rows", 0)) < 20:
        warnings.append("Challenger holdout contains fewer than 20 rows; promotion evidence is weak.")
    if ch.get("holdout_value") is None:
        blockers.append("Challenger has no usable holdout primary metric.")

    comparison: Dict[str, Any] = {
        "challenger": ch,
        "champion": cp,
        "same_dataset": None,
        "same_split_strategy": None,
        "same_feature_contract": None,
        "same_input_schema": None,
        "metric_delta": None,
        "metric_delta_favourable": None,
        "metric_comparison_strength": "first_champion" if champion is None else "indicative",
    }

    if champion is not None and cp is not None:
        if challenger.task != champion.task or challenger.target != champion.target:
            blockers.append("Champion and Challenger task/target contracts do not match.")
        same_dataset = bool(ch["dataset_fingerprint"] and ch["dataset_fingerprint"] == cp["dataset_fingerprint"])
        same_split = ch["split_strategy"] == cp["split_strategy"]
        same_features = bool(ch["pipeline_fingerprint"] and ch["pipeline_fingerprint"] == cp["pipeline_fingerprint"])
        same_schema = ch["required_columns"] == cp["required_columns"]
        comparison.update(
            {
                "same_dataset": same_dataset,
                "same_split_strategy": same_split,
                "same_feature_contract": same_features,
                "same_input_schema": same_schema,
                "metric_comparison_strength": "strong" if same_dataset and same_split else "indicative",
            }
        )
        if challenger.task == "classification" and ch["class_labels"] != cp["class_labels"]:
            warnings.append("Class labels changed between Champion and Challenger; downstream consumers must be reviewed.")
        if not same_schema:
            warnings.append("Input schema changed. Promotion may require changes to scoring integrations.")
        if not same_features:
            warnings.append("Feature pipeline changed between Champion and Challenger.")
        if not same_dataset:
            warnings.append("Holdout metrics come from different dataset fingerprints; the score delta is indicative, not an apples-to-apples benchmark.")
        if not same_split:
            warnings.append("Champion and Challenger use different holdout strategies; direct metric comparison is weaker.")

        if ch["metric"] == cp["metric"] and ch["holdout_value"] is not None and cp["holdout_value"] is not None:
            if ch["metric_direction"] == "higher":
                delta = float(ch["holdout_value"] - cp["holdout_value"])
                favourable = delta >= 0
            else:
                delta = float(cp["holdout_value"] - ch["holdout_value"])
                favourable = delta >= 0
            comparison["metric_delta"] = delta
            comparison["metric_delta_favourable"] = favourable
            if same_dataset and same_split and not favourable:
                blockers.append("Challenger is worse than the current Champion on the comparable primary holdout metric.")
            elif not favourable:
                warnings.append("Challenger primary holdout metric is worse than Champion, but the evaluation populations are not fully comparable.")
        else:
            warnings.append("Champion and Challenger do not expose a directly comparable primary holdout metric.")

    token = _approval_token(state, challenger, champion, key)
    return PromotionAssessment(
        family_id=str(state["family_id"]),
        family_name=str(state["family_name"]),
        challenger_id=challenger.package_id,
        champion_id=champion_id,
        ready=not blockers,
        blockers=tuple(dict.fromkeys(blockers)),
        warnings=tuple(dict.fromkeys(warnings)),
        comparison=comparison,
        approval_token=token,
        state_revision=int(state.get("revision", 0)),
    )


def promote_challenger(
    challenger_id: str,
    *,
    approval_token: str,
    approval_note: str,
    actor: str = "local-user",
    signing_key: Optional[bytes] = None,
    key_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    note = _safe_text(approval_note, limit=1000)
    if len(note) < 3:
        raise ModelGovernanceError("A short approval note is required for an auditable promotion.")
    key = bytes(signing_key) if signing_key is not None else get_or_create_signing_key(key_path)
    assessment = assess_promotion(challenger_id, signing_key=key, key_path=key_path)
    if not assessment.ready:
        raise ModelGovernanceError("Promotion gate is blocked: " + "; ".join(assessment.blockers))
    if not approval_token or not hmac.compare_digest(str(approval_token), assessment.approval_token):
        raise ModelGovernanceError("Promotion assessment is stale or invalid. Re-run the promotion assessment before approving.")

    state = _load_state(assessment.family_id, signing_key=key, key_path=key_path)
    if state is None:
        raise ModelGovernanceError("Model family governance state disappeared before promotion.")
    if int(state.get("revision", -1)) != assessment.state_revision:
        raise ModelGovernanceError("Model family changed after assessment. Re-run the promotion assessment.")
    challenger_row = state["models"].get(challenger_id)
    if not challenger_row or challenger_row.get("status") != STATUS_CHALLENGER:
        raise ModelGovernanceError("Challenger status changed after assessment.")

    previous = str(state.get("champion_id", "") or "")
    now = _utc_now()
    if previous:
        previous_row = state["models"].get(previous)
        if not previous_row or previous_row.get("status") != STATUS_CHAMPION:
            raise ModelGovernanceError("Current Champion state is inconsistent; promotion was blocked.")
        previous_row["status"] = STATUS_ARCHIVED
        previous_row["archived_at"] = now
        previous_row["last_decision_note"] = f"Superseded by {challenger_id}. {note}"[:1000]
    challenger_row["status"] = STATUS_CHAMPION
    challenger_row["promoted_at"] = now
    challenger_row["last_decision_note"] = note
    state["champion_id"] = challenger_id
    state["revision"] = int(state.get("revision", 0)) + 1
    _event(
        state,
        "challenger_promoted",
        challenger_id,
        actor=actor,
        note=note,
        details={"previous_champion_id": previous, "assessment_revision": assessment.state_revision},
    )
    state = _save_state(state, signing_key=key, key_path=key_path)
    return {
        "family_id": state["family_id"],
        "family_name": state["family_name"],
        "champion_id": challenger_id,
        "previous_champion_id": previous,
        "revision": int(state["revision"]),
        "promoted_at": now,
    }


def reject_challenger(
    package_id: str,
    *,
    reason: str,
    actor: str = "local-user",
    signing_key: Optional[bytes] = None,
    key_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    reason = _safe_text(reason, limit=1000)
    if len(reason) < 3:
        raise ModelGovernanceError("A rejection reason is required.")
    key = bytes(signing_key) if signing_key is not None else get_or_create_signing_key(key_path)
    state, _ = _ensure_member(package_id, signing_key=key, key_path=key_path)
    row = state["models"][package_id]
    if row.get("status") != STATUS_CHALLENGER:
        raise ModelGovernanceError("Only a Challenger can be rejected.")
    row["status"] = STATUS_REJECTED
    row["rejected_at"] = _utc_now()
    row["last_decision_note"] = reason
    state["revision"] = int(state.get("revision", 0)) + 1
    _event(state, "challenger_rejected", package_id, actor=actor, note=reason)
    state = _save_state(state, signing_key=key, key_path=key_path)
    return governance_status_for_package(package_id, signing_key=key, key_path=key_path)


def resubmit_archived_as_challenger(
    package_id: str,
    *,
    reason: str,
    actor: str = "local-user",
    signing_key: Optional[bytes] = None,
    key_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    reason = _safe_text(reason, limit=1000)
    if len(reason) < 3:
        raise ModelGovernanceError("A rollback/resubmission reason is required.")
    key = bytes(signing_key) if signing_key is not None else get_or_create_signing_key(key_path)
    state, _ = _ensure_member(package_id, signing_key=key, key_path=key_path)
    row = state["models"][package_id]
    if row.get("status") != STATUS_ARCHIVED:
        raise ModelGovernanceError("Only an Archived former Champion can be re-submitted for rollback review.")
    row["status"] = STATUS_CHALLENGER
    row["submitted_at"] = _utc_now()
    row["last_decision_note"] = reason
    state["revision"] = int(state.get("revision", 0)) + 1
    _event(state, "archived_resubmitted", package_id, actor=actor, note=reason)
    state = _save_state(state, signing_key=key, key_path=key_path)
    return governance_status_for_package(package_id, signing_key=key, key_path=key_path)


def can_delete_registered_package(
    package_id: str,
    *,
    signing_key: Optional[bytes] = None,
    key_path: Optional[Path | str] = None,
) -> tuple[bool, str]:
    try:
        status = governance_status_for_package(package_id, signing_key=signing_key, key_path=key_path)
    except ModelGovernanceError:
        # If governance cannot be established, fail closed for a signed registry object.
        return False, "Governance status could not be verified; deletion is blocked."
    if status.get("status") == STATUS_CHAMPION:
        return False, "The active Champion cannot be deleted. Promote another Challenger first."
    return True, ""


def mark_registered_package_removed(
    package_id: str,
    *,
    actor: str = "local-user",
    note: str = "Removed from local registry",
    signing_key: Optional[bytes] = None,
    key_path: Optional[Path | str] = None,
) -> None:
    key = bytes(signing_key) if signing_key is not None else get_or_create_signing_key(key_path)
    state, _ = _ensure_member(package_id, signing_key=key, key_path=key_path)
    row = state["models"][package_id]
    if row.get("status") == STATUS_CHAMPION:
        raise ModelGovernanceError("The active Champion cannot be removed from the registry.")
    row["status"] = STATUS_REMOVED
    row["last_decision_note"] = _safe_text(note, limit=1000)
    state["revision"] = int(state.get("revision", 0)) + 1
    _event(state, "registry_package_removed", package_id, actor=actor, note=note)
    _save_state(state, signing_key=key, key_path=key_path)


def current_champion(
    family_id: str,
    *,
    signing_key: Optional[bytes] = None,
    key_path: Optional[Path | str] = None,
) -> Optional[Dict[str, Any]]:
    state = _load_state(family_id, signing_key=signing_key, key_path=key_path)
    if state is None or not state.get("champion_id"):
        return None
    package_id = str(state["champion_id"])
    package = _verified_package(package_id, signing_key=signing_key, key_path=key_path)
    summary = _package_summary(package)
    summary.update(
        {
            "family_id": family_id,
            "family_name": state["family_name"],
            "status": STATUS_CHAMPION,
            "governance_revision": int(state.get("revision", 0)),
        }
    )
    return summary


def list_family_models(
    family_id: str,
    *,
    signing_key: Optional[bytes] = None,
    key_path: Optional[Path | str] = None,
) -> List[Dict[str, Any]]:
    state = get_governance_state(family_id, signing_key=signing_key, key_path=key_path)
    rows: List[Dict[str, Any]] = []
    for package_id, governance in state.get("models", {}).items():
        if governance.get("status") == STATUS_REMOVED:
            continue
        try:
            package = _verified_package(package_id, signing_key=signing_key, key_path=key_path)
        except ModelGovernanceError:
            # Surface broken registry members without deserializing an unverified payload.
            rows.append(
                {
                    "package_id": package_id,
                    "status": governance.get("status", "Unknown"),
                    "model": "Verification failed",
                    "task": state.get("task", ""),
                    "target": state.get("target", ""),
                    "verification_error": True,
                    "holdout_value": None,
                    "metric": "",
                    "created_at": "",
                    "label": governance.get("label", ""),
                }
            )
            continue
        summary = _package_summary(package)
        summary.update(
            {
                "status": governance.get("status", STATUS_CANDIDATE),
                "label": governance.get("label", ""),
                "submitted_at": governance.get("submitted_at", ""),
                "promoted_at": governance.get("promoted_at", ""),
                "rejected_at": governance.get("rejected_at", ""),
                "archived_at": governance.get("archived_at", ""),
                "last_decision_note": governance.get("last_decision_note", ""),
                "verification_error": False,
            }
        )
        rows.append(summary)
    order = {STATUS_CHAMPION: 0, STATUS_CHALLENGER: 1, STATUS_CANDIDATE: 2, STATUS_ARCHIVED: 3, STATUS_REJECTED: 4}
    rows.sort(key=lambda row: (order.get(str(row.get("status")), 99), str(row.get("created_at", ""))), reverse=False)
    return rows


def governance_history(
    family_id: str,
    *,
    signing_key: Optional[bytes] = None,
    key_path: Optional[Path | str] = None,
) -> List[Dict[str, Any]]:
    state = get_governance_state(family_id, signing_key=signing_key, key_path=key_path)
    return [dict(item) for item in state.get("events", [])]

# ════════════════════════════════════════════════════════
# Stage 19 — authenticated governance envelope exchange
# ════════════════════════════════════════════════════════
def governance_state_digest(state: Mapping[str, Any]) -> str:
    """Stable SHA-256 digest of an already-authenticated governance payload."""
    _validate_state_structure(state)
    return hashlib.sha256(_canonical_bytes(state)).hexdigest()


def validate_governance_envelope_bytes(
    raw: bytes,
    *,
    signing_key: Optional[bytes] = None,
    key_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Verify a serialized governance envelope without writing it to disk."""
    if not isinstance(raw, (bytes, bytearray)) or not raw or len(raw) > MAX_GOVERNANCE_BYTES:
        raise ModelGovernanceError("Governance envelope has an unsafe size.")
    key = bytes(signing_key) if signing_key is not None else get_or_create_signing_key(key_path)
    try:
        envelope = json.loads(bytes(raw).decode("utf-8"))
    except Exception as exc:
        raise ModelGovernanceError("Governance envelope is unreadable.") from exc
    if not isinstance(envelope, dict) or set(envelope) != {"payload", "signature"}:
        raise ModelGovernanceError("Governance envelope structure is invalid.")
    state = envelope.get("payload")
    signature = str(envelope.get("signature", ""))
    if not isinstance(state, dict):
        raise ModelGovernanceError("Governance payload is invalid.")
    if str(state.get("signer_key_id", "")) != signing_key_id(key):
        raise ModelGovernanceError("Governance envelope belongs to a different model trust key.")
    if not hmac.compare_digest(signature, _signature(state, key)):
        raise ModelGovernanceError("Governance envelope authentication failed.")
    _validate_state_structure(state)
    return dict(state)


def export_governance_envelope(
    family_id: str,
    *,
    signing_key: Optional[bytes] = None,
    key_path: Optional[Path | str] = None,
) -> bytes:
    """Return the exact authenticated governance envelope used by the local registry."""
    path = _state_path(str(family_id))
    if not path.is_file():
        raise ModelGovernanceError("Model family governance state does not exist.")
    raw = path.read_bytes()
    state = validate_governance_envelope_bytes(raw, signing_key=signing_key, key_path=key_path)
    if str(state.get("family_id")) != str(family_id):
        raise ModelGovernanceError("Governance envelope family identifier mismatch.")
    return raw


def _governance_last_event_hash(state: Mapping[str, Any]) -> str:
    events = state.get("events") or []
    if not events:
        return ""
    last = events[-1]
    return str(last.get("event_hash", "")) if isinstance(last, Mapping) else ""


def governance_is_fast_forward(current: Mapping[str, Any], incoming: Mapping[str, Any]) -> bool:
    """True only when incoming is identical to, or an authenticated descendant of, current."""
    _validate_state_structure(current)
    _validate_state_structure(incoming)
    identity = ("family_id", "family_name", "task", "target", "signer_key_id")
    if any(str(current.get(key, "")) != str(incoming.get(key, "")) for key in identity):
        return False
    current_revision = int(current.get("revision", 0))
    incoming_revision = int(incoming.get("revision", 0))
    if incoming_revision < current_revision:
        return False
    if incoming_revision == current_revision:
        return governance_state_digest(current) == governance_state_digest(incoming)
    last_hash = _governance_last_event_hash(current)
    if not last_hash:
        return current_revision == 0
    incoming_hashes = {
        str(event.get("event_hash", ""))
        for event in (incoming.get("events") or [])
        if isinstance(event, Mapping)
    }
    return last_hash in incoming_hashes


def import_governance_envelope(
    raw: bytes,
    *,
    signing_key: Optional[bytes] = None,
    key_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Fast-forward local governance from a trusted remote registry.

    Divergent or stale histories are rejected. Every non-removed package referenced
    by the incoming state must already exist locally and pass signed-package
    verification before the governance pointer can move.
    """
    key = bytes(signing_key) if signing_key is not None else get_or_create_signing_key(key_path)
    incoming = validate_governance_envelope_bytes(raw, signing_key=key, key_path=key_path)
    family_id = str(incoming["family_id"])
    current = _load_state(family_id, signing_key=key, key_path=key_path)
    if current is not None:
        if governance_state_digest(current) == governance_state_digest(incoming):
            return incoming
        if not governance_is_fast_forward(current, incoming):
            raise ModelGovernanceError(
                "Remote governance is stale or diverged from local history. Pull/push conflict requires manual resolution."
            )

    for package_id, row in (incoming.get("models") or {}).items():
        if str((row or {}).get("status", "")) == STATUS_REMOVED:
            continue
        _verified_package(str(package_id), signing_key=key, key_path=key_path)

    _atomic_write(_state_path(family_id), bytes(raw))
    return incoming
