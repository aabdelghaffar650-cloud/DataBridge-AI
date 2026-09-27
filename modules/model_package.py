# ════════════════════════════════════════════════════════
#  DataBridge AI — Signed Model Package & Prediction Engine
#  Stage 9: authenticated package export, verify-before-deserialize loading,
#           schema validation, and batch prediction
# ════════════════════════════════════════════════════════
from __future__ import annotations

import base64
import hashlib
import hmac
import io
import json
import os
import pickle
import platform
import re
import secrets
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from config.constants import APP_NAME, APP_VERSION
from core.user_paths import signing_key_file
from core.feature_derivation import (
    DERIVATION_RECIPE_VERSION,
    FeatureDerivationError,
    effective_derivation_recipe,
    feature_derivation_recipe_fingerprint,
    normalise_feature_derivation_recipe,
    prepare_training_dataframe_for_contract,
    replay_feature_derivations,
    source_columns_for_pipeline,
    validate_recipe_against_dataframe,
)
from modules.feature_pipeline import (
    feature_pipeline_spec_fingerprint,
    normalise_feature_pipeline_spec,
)
from modules.ml_engine import CLASSIFICATION, REGRESSION, SupervisedExperimentResult
from modules.model_monitoring import (
    build_prediction_reference,
    protect_monitoring_reference,
)


MODEL_PACKAGE_FORMAT = "DataBridgeAI Signed Model Package"
MODEL_PACKAGE_VERSION = 1
MODEL_PAYLOAD_VERSION = 3
SUPPORTED_MODEL_PAYLOAD_VERSIONS = {1, 2, 3}
MODEL_PACKAGE_EXTENSION = ".dbmlpkg"
MODEL_MEMBER = "model.bin"
MANIFEST_MEMBER = "manifest.json"
SIGNATURE_MEMBER = "signature.txt"
README_MEMBER = "README.txt"
ALLOWED_MEMBERS = {MODEL_MEMBER, MANIFEST_MEMBER, SIGNATURE_MEMBER, README_MEMBER}
SIGNING_CONTEXT = b"DataBridgeAI-Signed-Model-Package-v1\0"
MAX_PACKAGE_BYTES = 300 * 1024 * 1024
MAX_MODEL_BYTES = 250 * 1024 * 1024
MAX_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_README_BYTES = 256 * 1024
MAX_ARCHIVE_MEMBERS = 8
MAX_TOTAL_UNCOMPRESSED_BYTES = 280 * 1024 * 1024
MAX_COMPRESSION_RATIO = 250.0
DEFAULT_MAX_PREDICTION_ROWS = 250_000
MAX_PROBABILITY_COLUMNS = 50
NUMERIC_SEMANTICS = {
    "Numeric Continuous",
    "Numeric Discrete",
    "Currency",
    "Percentage",
}


class ModelPackageError(ValueError):
    """Raised when a model package is invalid, untrusted, or unsafe."""


@dataclass(frozen=True)
class ModelPackageBuild:
    package_bytes: bytes
    manifest: Dict[str, Any]
    file_name: str


@dataclass(frozen=True)
class LoadedModelPackage:
    manifest: Dict[str, Any]
    fitted_pipeline: Pipeline
    target_encoder: Optional[LabelEncoder]
    package_id: str
    signer_key_id: str
    trusted: bool = True
    retraining_contract: Optional[Dict[str, Any]] = None
    feature_derivation_recipe: tuple[Dict[str, Any], ...] = ()

    @property
    def task(self) -> str:
        return str(self.manifest.get("model", {}).get("task", ""))

    @property
    def target(self) -> str:
        return str(self.manifest.get("model", {}).get("target", ""))

    @property
    def required_columns(self) -> list[str]:
        """Columns required from source/prediction data before derivation replay."""
        contract = self.manifest.get("input_contract", {}) or {}
        values = contract.get("source_required_columns") or contract.get("required_columns") or []
        return list(map(str, values))

    @property
    def pipeline_required_columns(self) -> list[str]:
        """Columns consumed by the fitted sklearn pipeline after derivation replay."""
        return list(
            map(
                str,
                self.manifest.get("input_contract", {}).get("required_columns", []) or [],
            )
        )

    @property
    def derived_columns(self) -> list[str]:
        return [str(step.get("output_column", "")) for step in self.feature_derivation_recipe]

    @property
    def classes(self) -> list[str]:
        return list(
            map(str, self.manifest.get("model", {}).get("class_labels", []) or [])
        )


@dataclass(frozen=True)
class PredictionRunResult:
    output: pd.DataFrame
    package_id: str
    task: str
    target: str
    rows: int
    prediction_column: str
    probability_columns: tuple[str, ...]
    warnings: tuple[str, ...]
    created_at: str


# ── Generic helpers ──────────────────────────────────────────────────────────
def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return str(value)


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        _json_safe(dict(payload)),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _version_pair(value: str) -> tuple[int, int]:
    numbers = re.findall(r"\d+", str(value))
    if len(numbers) < 2:
        return (0, 0)
    return (int(numbers[0]), int(numbers[1]))


def _normalised_user_data_dir() -> Path:
    base = (
        os.environ.get("DATABRIDGE_USER_DATA_DIR")
        or os.environ.get("LOCALAPPDATA")
        or os.environ.get("APPDATA")
        or str(Path.home())
    )
    path = Path(base).expanduser()
    if path.name.casefold() != "databridgeai":
        path = path / "DataBridgeAI"
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_signing_key_path() -> Path:
    return signing_key_file()


def get_or_create_signing_key(key_path: Optional[Path | str] = None) -> bytes:
    """Return the per-installation 256-bit signing key using atomic creation."""
    path = Path(key_path).expanduser() if key_path is not None else default_signing_key_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        data = path.read_bytes()
        if len(data) != 32:
            raise ModelPackageError(
                "The local model-package signing key is invalid. Restore it from backup or remove it only if old packages no longer need to be opened."
            )
        return data

    key = secrets.token_bytes(32)
    temp = path.with_name(path.name + f".{secrets.token_hex(6)}.tmp")
    try:
        with temp.open("xb") as handle:
            handle.write(key)
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
        try:
            os.chmod(temp, 0o600)
        except OSError:
            pass
        try:
            os.replace(temp, path)
        except OSError:
            if path.exists():
                temp.unlink(missing_ok=True)
                data = path.read_bytes()
                if len(data) != 32:
                    raise ModelPackageError("The local signing key is invalid.")
                return data
            raise
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
        return key
    finally:
        temp.unlink(missing_ok=True)


def signing_key_id(key: bytes) -> str:
    """Return the identifier of the shared model-package trust key.

    This identifies the trust domain, not a unique installation: Stage 19
    intentionally permits trusted installations to share the same signing key.
    """
    if not isinstance(key, (bytes, bytearray)) or len(key) != 32:
        raise ModelPackageError("A valid 256-bit signing key is required.")
    return _sha256(bytes(key))[:16]

MODEL_KEY_BACKUP_FORMAT = "DataBridgeAI Encrypted Signing Key Backup"
MODEL_KEY_BACKUP_VERSION = 1
MODEL_KEY_BACKUP_KDF_ITERATIONS = 600_000


def signing_key_info(key_path: Optional[Path | str] = None) -> Dict[str, Any]:
    path = Path(key_path).expanduser() if key_path is not None else default_signing_key_path()
    key = get_or_create_signing_key(path)
    return {
        "path": str(path),
        "key_id": signing_key_id(key),
        "exists": path.exists(),
        "size": len(key),
    }


def create_encrypted_signing_key_backup(
    passphrase: str,
    *,
    key_path: Optional[Path | str] = None,
) -> bytes:
    """Create an authenticated AES-GCM backup of the installation signing key."""
    password = str(passphrase or "")
    if len(password) < 12:
        raise ModelPackageError("Backup passphrase must be at least 12 characters.")
    try:
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    except Exception as exc:
        raise ModelPackageError(
            "Encrypted key backup requires the cryptography dependency."
        ) from exc

    key = get_or_create_signing_key(key_path)
    salt = secrets.token_bytes(16)
    nonce = secrets.token_bytes(12)
    derived = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=MODEL_KEY_BACKUP_KDF_ITERATIONS,
    ).derive(password.encode("utf-8"))
    metadata = {
        "format": MODEL_KEY_BACKUP_FORMAT,
        "version": MODEL_KEY_BACKUP_VERSION,
        "created_at": _utc_now(),
        "key_id": signing_key_id(key),
        "kdf": "PBKDF2-HMAC-SHA256",
        "iterations": MODEL_KEY_BACKUP_KDF_ITERATIONS,
    }
    aad = _canonical_json_bytes(metadata)
    ciphertext = AESGCM(derived).encrypt(nonce, key, aad)
    payload = dict(metadata)
    payload.update(
        {
            "salt": base64.b64encode(salt).decode("ascii"),
            "nonce": base64.b64encode(nonce).decode("ascii"),
            "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
        }
    )
    return _canonical_json_bytes(payload)


def _decrypt_signing_key_backup(backup_bytes: bytes, passphrase: str) -> tuple[bytes, Dict[str, Any]]:
    password = str(passphrase or "")
    if len(password) < 12:
        raise ModelPackageError("Backup passphrase must be at least 12 characters.")
    if not isinstance(backup_bytes, (bytes, bytearray)) or len(backup_bytes) > 256 * 1024:
        raise ModelPackageError("The signing-key backup is invalid or too large.")
    try:
        payload = json.loads(bytes(backup_bytes).decode("utf-8"))
        if not isinstance(payload, dict):
            raise ValueError
        if payload.get("format") != MODEL_KEY_BACKUP_FORMAT or int(payload.get("version", 0)) != MODEL_KEY_BACKUP_VERSION:
            raise ValueError
        salt = base64.b64decode(payload["salt"], validate=True)
        nonce = base64.b64decode(payload["nonce"], validate=True)
        ciphertext = base64.b64decode(payload["ciphertext"], validate=True)
        iterations = int(payload.get("iterations", 0))
        if len(salt) != 16 or len(nonce) != 12 or iterations < 300_000:
            raise ValueError
    except Exception as exc:
        raise ModelPackageError("The signing-key backup format is invalid.") from exc

    metadata = {
        "format": payload["format"],
        "version": int(payload["version"]),
        "created_at": payload.get("created_at", ""),
        "key_id": payload.get("key_id", ""),
        "kdf": payload.get("kdf", ""),
        "iterations": iterations,
    }
    try:
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

        derived = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=iterations,
        ).derive(password.encode("utf-8"))
        key = AESGCM(derived).decrypt(nonce, ciphertext, _canonical_json_bytes(metadata))
    except Exception as exc:
        raise ModelPackageError("The backup passphrase is incorrect or the backup was modified.") from exc
    if len(key) != 32 or signing_key_id(key) != str(metadata.get("key_id", "")):
        raise ModelPackageError("The decrypted signing key failed integrity validation.")
    return key, metadata


def restore_encrypted_signing_key_backup(
    backup_bytes: bytes,
    passphrase: str,
    *,
    overwrite: bool = False,
    key_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Restore a verified backup atomically. Existing keys require explicit overwrite."""
    key, metadata = _decrypt_signing_key_backup(backup_bytes, passphrase)
    path = Path(key_path).expanduser() if key_path is not None else default_signing_key_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        current = path.read_bytes()
        if current == key:
            return {"restored": False, "key_id": signing_key_id(key), "reason": "already_active"}
        raise ModelPackageError("A different signing key already exists. Explicit overwrite confirmation is required.")
    temp = path.with_name(path.name + f".{secrets.token_hex(6)}.tmp")
    try:
        with temp.open("xb") as handle:
            handle.write(key)
            handle.flush()
            try:
                os.fsync(handle.fileno())
            except OSError:
                pass
        try:
            os.chmod(temp, 0o600)
        except OSError:
            pass
        os.replace(temp, path)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    finally:
        temp.unlink(missing_ok=True)
    return {
        "restored": True,
        "key_id": signing_key_id(key),
        "backup_created_at": metadata.get("created_at", ""),
        "path": str(path),
    }


def rotate_signing_key(
    confirm_key_id: str,
    *,
    key_path: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Rotate the local signing key and preserve the prior key as a protected backup file."""
    path = Path(key_path).expanduser() if key_path is not None else default_signing_key_path()
    current = get_or_create_signing_key(path)
    current_id = signing_key_id(current)
    if str(confirm_key_id or "").strip() != current_id:
        raise ModelPackageError("Signing-key rotation confirmation does not match the active key ID.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archived = path.with_name(f"{path.name}.{current_id}.{stamp}.bak")
    if archived.exists():
        raise ModelPackageError("A rotation backup with the same timestamp already exists.")
    os.replace(path, archived)
    try:
        os.chmod(archived, 0o600)
    except OSError:
        pass
    try:
        new_key = get_or_create_signing_key(path)
    except Exception:
        os.replace(archived, path)
        raise
    return {
        "old_key_id": current_id,
        "new_key_id": signing_key_id(new_key),
        "archived_key_path": str(archived),
        "active_key_path": str(path),
    }


def _sign_manifest(manifest: Mapping[str, Any], key: bytes) -> str:
    return hmac.new(
        key,
        SIGNING_CONTEXT + _canonical_json_bytes(manifest),
        hashlib.sha256,
    ).hexdigest()


def _safe_filename_component(value: str, fallback: str = "model") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._-")
    return (cleaned or fallback)[:70]


def _required_input_columns(
    result: SupervisedExperimentResult,
    spec: Mapping[str, Any],
) -> list[str]:
    fitted_names = getattr(result.fitted_pipeline, "feature_names_in_", None)
    if fitted_names is not None and len(fitted_names):
        return list(dict.fromkeys(map(str, fitted_names)))
    clean = normalise_feature_pipeline_spec(spec)
    return list(dict.fromkeys(map(str, clean.get("feature_columns", []) or [])))


def _profile_semantic(profile: Mapping[str, Any]) -> str:
    return str(
        profile.get("effective_semantic_type")
        or profile.get("semantic_type")
        or "Unknown"
    )


def _input_schema(
    source_df: pd.DataFrame,
    columns: Sequence[str],
    semantic_profiles: Optional[Mapping[str, Mapping[str, Any]]],
) -> list[Dict[str, Any]]:
    profiles = semantic_profiles or {}
    rows: list[Dict[str, Any]] = []
    for column in columns:
        if column not in source_df.columns:
            raise ModelPackageError(
                f"The trained pipeline requires column '{column}', but it is absent from the source dataset."
            )
        series = source_df[column]
        non_null = int(series.notna().sum())
        rows.append(
            {
                "name": str(column),
                "pandas_dtype": str(series.dtype),
                "semantic_type": _profile_semantic(profiles.get(column, {}) or {}),
                "nullable": bool(series.isna().any()),
                "training_non_null_ratio": round(non_null / max(len(series), 1), 6),
                "training_unique": int(series.nunique(dropna=True)),
            }
        )
    return rows


def _feature_contract_summary(
    spec: Mapping[str, Any],
    required_columns: Sequence[str],
    result: SupervisedExperimentResult,
) -> Dict[str, Any]:
    clean = normalise_feature_pipeline_spec(spec)
    required = set(map(str, required_columns))
    groups = {
        str(name): [str(column) for column in columns if str(column) in required]
        for name, columns in (clean.get("groups", {}) or {}).items()
    }
    groups = {name: columns for name, columns in groups.items() if columns}
    settings = clean.get("settings", {}) or {}
    safe_settings = {
        key: value
        for key, value in settings.items()
        if key
        in {
            "numeric_imputer",
            "numeric_scaler",
            "categorical_imputer",
            "categorical_encoder",
            "rare_category_threshold",
            "date_parts",
            "date_cyclical",
            "date_elapsed",
            "text_max_features",
            "text_ngram_max",
        }
    }
    names = list(map(str, result.feature_names))
    names_json = _canonical_json_bytes({"feature_names": names})
    include_names = len(names_json) <= 1_000_000 and len(names) <= 10_000
    return {
        "pipeline_spec_fingerprint": result.pipeline_spec_fingerprint,
        "effective_pipeline_spec_fingerprint": result.effective_pipeline_spec_fingerprint,
        "configured_spec_fingerprint": feature_pipeline_spec_fingerprint(clean),
        "input_groups": groups,
        "settings": _json_safe(safe_settings),
        "transformed_feature_count": int(len(names)),
        "transformed_feature_names": names if include_names else [],
        "transformed_feature_names_truncated": not include_names,
        "transformed_feature_names_sha256": _sha256(names_json),
    }


def _runtime_manifest() -> Dict[str, str]:
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "platform": platform.system(),
    }


def _validate_result_for_export(result: SupervisedExperimentResult) -> None:
    if not isinstance(result, SupervisedExperimentResult):
        raise ModelPackageError("A completed ML Studio V2 experiment is required.")
    if not isinstance(result.fitted_pipeline, Pipeline):
        raise ModelPackageError("The experiment does not contain a fitted sklearn Pipeline.")
    if result.task not in {CLASSIFICATION, REGRESSION}:
        raise ModelPackageError("Only supervised classification/regression packages are supported.")
    if result.task == CLASSIFICATION and not isinstance(result.target_encoder, LabelEncoder):
        raise ModelPackageError("The classification target encoder is missing.")


# ── Package creation ─────────────────────────────────────────────────────────
def create_signed_model_package(
    result: SupervisedExperimentResult,
    source_df: pd.DataFrame,
    feature_pipeline_spec: Mapping[str, Any],
    *,
    semantic_profiles: Optional[Mapping[str, Mapping[str, Any]]] = None,
    feature_derivation_recipe: Any = None,
    signing_key: Optional[bytes] = None,
    key_path: Optional[Path | str] = None,
) -> ModelPackageBuild:
    """
    Build a locally authenticated package.

    The fitted Python object is serialized only for a package created by this
    installation. Import verifies HMAC and SHA-256 before pickle is touched.
    Raw datasets, holdout rows, and row-level predictions are never included.
    """
    _validate_result_for_export(result)
    if not isinstance(source_df, pd.DataFrame) or source_df.empty:
        raise ModelPackageError("The source dataset is unavailable.")
    key = bytes(signing_key) if signing_key is not None else get_or_create_signing_key(key_path)
    key_id = signing_key_id(key)
    required_columns = _required_input_columns(result, feature_pipeline_spec)
    if not required_columns:
        raise ModelPackageError("The fitted model has no input feature contract.")

    try:
        contract_source_df, effective_recipe = prepare_training_dataframe_for_contract(
            source_df,
            feature_derivation_recipe,
            required_columns,
        )
        source_required_columns = source_columns_for_pipeline(
            effective_recipe,
            required_columns,
        )
    except FeatureDerivationError as exc:
        raise ModelPackageError(f"Feature-derivation contract is invalid: {exc}") from exc

    if not result.monitoring_reference:
        raise ModelPackageError(
            "The experiment has no Stage 14 training monitoring reference. Retrain the experiment after installing Stage 14."
        )
    protected_monitoring_reference = protect_monitoring_reference(
        result.monitoring_reference, key
    )
    prediction_reference = build_prediction_reference(result.task, result.predictions)

    # Stage 16 replay metadata is kept inside the authenticated binary payload,
    # not the human-readable manifest. This avoids exposing ordinal/category
    # contract values while still allowing exact, verified retraining replay.
    from modules.retraining_workflow import build_retraining_contract

    retraining_contract = build_retraining_contract(result, feature_pipeline_spec)
    payload = {
        "payload_version": MODEL_PAYLOAD_VERSION,
        "fitted_pipeline": result.fitted_pipeline,
        "target_encoder": result.target_encoder,
        "retraining_contract": retraining_contract,
        "feature_derivation_recipe": effective_recipe,
    }
    try:
        model_bytes = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as exc:
        raise ModelPackageError(f"The fitted pipeline could not be serialized: {exc}") from exc
    if len(model_bytes) > MAX_MODEL_BYTES:
        raise ModelPackageError(
            f"The fitted model is {len(model_bytes) / (1024 * 1024):.1f} MB, above the {MAX_MODEL_BYTES / (1024 * 1024):.0f} MB package limit."
        )

    package_id = f"dbml-{secrets.token_hex(8)}"
    manifest: Dict[str, Any] = {
        "format": MODEL_PACKAGE_FORMAT,
        "package_version": MODEL_PACKAGE_VERSION,
        "package_id": package_id,
        "created_at": _utc_now(),
        "signer_key_id": key_id,
        "application": {"name": APP_NAME, "version": APP_VERSION},
        "runtime": _runtime_manifest(),
        "experiment": {
            "experiment_id": result.experiment_id,
            "created_at": result.created_at,
            "dataset_revision": int(result.dataset_revision),
            "dataset_fingerprint": str(result.dataset_fingerprint),
            "split_strategy": result.split_strategy,
            "split_column": result.split_column,
            "random_state": int(result.random_state),
            "training_rows": int(result.train_rows),
            "holdout_rows": int(result.holdout_rows),
            "cv_folds": int(result.actual_cv_folds),
        },
        "model": {
            "task": result.task,
            "target": result.target,
            "selected_model": result.selected_model,
            "tuned": bool(result.tuned),
            "best_params": _json_safe(result.best_params),
            "class_labels": list(map(str, result.classes)),
        },
        "input_contract": {
            # Legacy field remains the exact fitted-pipeline input schema.
            "required_columns": required_columns,
            "schema": _input_schema(contract_source_df, required_columns, semantic_profiles),
            # Stage 20.4 source contract is what prediction/monitoring callers must provide.
            "source_required_columns": source_required_columns,
            "source_schema": _input_schema(source_df, source_required_columns, semantic_profiles),
            "derived_columns": [str(step["output_column"]) for step in effective_recipe],
            "missing_columns_allowed": False,
            "extra_columns_allowed": True,
        },
        "derivation_contract": {
            "available": bool(effective_recipe),
            "recipe_version": DERIVATION_RECIPE_VERSION,
            "step_count": len(effective_recipe),
            "recipe_fingerprint": feature_derivation_recipe_fingerprint(effective_recipe),
            "recipe_location": "authenticated model payload",
        },
        "feature_contract": _feature_contract_summary(
            feature_pipeline_spec,
            required_columns,
            result,
        ),
        "monitoring": {
            "reference_version": int(protected_monitoring_reference.get("reference_version", 1)),
            "training_reference": protected_monitoring_reference,
            "prediction_reference": _json_safe(prediction_reference),
            "privacy": {
                "training_rows_persisted": False,
                "raw_category_labels_persisted": False,
                "category_tokens": "installation-bound HMAC-SHA256",
            },
        },
        "retraining": {
            "available": True,
            "contract_version": int(retraining_contract.get("contract_version", 1)),
            "feature_logic_fingerprint": str(retraining_contract.get("feature_logic_fingerprint", "")),
            "contract_location": "authenticated-model-payload",
            "auto_promotion_allowed": False,
            "target_governance_status": "Challenger",
        },
        "evaluation": {
            "primary_metric": result.primary_metric,
            "primary_direction": result.primary_direction,
            "holdout_metrics": _json_safe(result.holdout_metrics),
            "baseline_model": result.baseline_model,
            "baseline_cv_value": _json_safe(result.baseline_cv_value),
            "selected_cv_value": _json_safe(result.selected_cv_value),
            "improvement_vs_baseline": _json_safe(result.improvement_vs_baseline),
            "warnings": list(map(str, result.warnings)),
        },
        "artifact": {
            "member": MODEL_MEMBER,
            "serialization": "authenticated-python-pickle",
            "sha256": _sha256(model_bytes),
            "size_bytes": int(len(model_bytes)),
        },
        "security": {
            "signature_algorithm": "HMAC-SHA256",
            "integrity_algorithm": "SHA-256",
            "verification_order": "archive-checks -> manifest -> model-hash -> signature -> deserialize",
            "raw_pickle_import_allowed": False,
            "signed_for_this_installation": True,
            "contains_raw_dataset": False,
            "contains_row_level_predictions": False,
            "contains_raw_monitoring_category_labels": False,
            "monitoring_reference_scope": "training_rows_only",
        },
    }
    manifest_bytes = json.dumps(
        _json_safe(manifest),
        indent=2,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    if len(manifest_bytes) > MAX_MANIFEST_BYTES:
        raise ModelPackageError("The model manifest exceeds the safe size limit.")
    signature = _sign_manifest(manifest, key).encode("ascii")
    readme = (
        "DataBridge AI signed model package\n"
        "==================================\n"
        f"Package ID: {package_id}\n"
        f"Task: {result.task}\n"
        f"Target: {result.target}\n"
        f"Model: {result.selected_model}\n\n"
        "Security: DataBridge AI verifies archive structure, model SHA-256, and the configured model-trust-key HMAC before deserializing the fitted Python pipeline. Raw .pkl imports are not accepted.\n"
        "Portability: this package is authenticated by a DataBridge AI model trust key. Installations that intentionally share that trust key can verify the package; preserve the key if packages must remain loadable after migration.\n"
    ).encode("utf-8")

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        archive.writestr(MANIFEST_MEMBER, manifest_bytes)
        archive.writestr(MODEL_MEMBER, model_bytes)
        archive.writestr(SIGNATURE_MEMBER, signature)
        archive.writestr(README_MEMBER, readme)
    package_bytes = buffer.getvalue()
    if len(package_bytes) > MAX_PACKAGE_BYTES:
        raise ModelPackageError("The compressed package exceeds the safe package size limit.")

    model_name = _safe_filename_component(result.selected_model)
    file_name = f"databridge_{result.task}_{model_name}_{result.experiment_id}{MODEL_PACKAGE_EXTENSION}"
    return ModelPackageBuild(package_bytes=package_bytes, manifest=manifest, file_name=file_name)


# ── Safe package loading ─────────────────────────────────────────────────────
def _validate_archive_member(info: zipfile.ZipInfo) -> None:
    path = PurePosixPath(info.filename)
    if info.filename != path.name or len(path.parts) != 1 or path.name in {"", ".", ".."}:
        raise ModelPackageError("The package contains an unsafe archive path.")
    if info.is_dir():
        raise ModelPackageError("Directories are not allowed in a model package.")
    if info.flag_bits & 0x1:
        raise ModelPackageError("Encrypted archive entries are not supported.")
    unix_mode = (info.external_attr >> 16) & 0o170000
    if unix_mode == 0o120000:
        raise ModelPackageError("Symbolic links are not allowed in a model package.")
    if info.file_size < 0 or info.compress_size < 0:
        raise ModelPackageError("Invalid archive entry size.")
    if info.file_size and info.compress_size == 0:
        raise ModelPackageError("Invalid archive compression metadata.")
    if info.compress_size:
        ratio = info.file_size / max(info.compress_size, 1)
        if ratio > MAX_COMPRESSION_RATIO:
            raise ModelPackageError("The package has an unsafe compression ratio.")


def _read_member_bounded(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    maximum: int,
) -> bytes:
    if info.file_size > maximum:
        raise ModelPackageError(
            f"Package entry '{info.filename}' exceeds its safe size limit."
        )
    with archive.open(info, "r") as handle:
        data = handle.read(maximum + 1)
    if len(data) > maximum or len(data) != info.file_size:
        raise ModelPackageError(f"Package entry '{info.filename}' is invalid or oversized.")
    return data


def _validate_manifest(manifest: Mapping[str, Any]) -> None:
    if manifest.get("format") != MODEL_PACKAGE_FORMAT:
        raise ModelPackageError("This is not a DataBridge AI signed model package.")
    if int(manifest.get("package_version", -1)) != MODEL_PACKAGE_VERSION:
        raise ModelPackageError("Unsupported model package version.")
    package_id = str(manifest.get("package_id", ""))
    if not re.fullmatch(r"dbml-[0-9a-f]{16}", package_id):
        raise ModelPackageError("Invalid package identifier.")
    signer = str(manifest.get("signer_key_id", ""))
    if not re.fullmatch(r"[0-9a-f]{16}", signer):
        raise ModelPackageError("Invalid package signer identifier.")
    model = manifest.get("model")
    contract = manifest.get("input_contract")
    artifact = manifest.get("artifact")
    runtime = manifest.get("runtime")
    if not isinstance(model, Mapping) or not isinstance(contract, Mapping) or not isinstance(artifact, Mapping):
        raise ModelPackageError("The package manifest is incomplete.")
    if str(model.get("task")) not in {CLASSIFICATION, REGRESSION}:
        raise ModelPackageError("Unsupported model task in package.")
    required = contract.get("required_columns")
    schema = contract.get("schema")
    if not isinstance(required, list) or not required or not all(isinstance(c, str) and c for c in required):
        raise ModelPackageError("The input contract has no valid required columns.")
    if len(required) != len(set(required)):
        raise ModelPackageError("The input contract contains duplicate columns.")
    if not isinstance(schema, list) or len(schema) != len(required):
        raise ModelPackageError("The input schema does not match the required columns.")
    schema_names = [str(item.get("name", "")) for item in schema if isinstance(item, Mapping)]
    if schema_names != required:
        raise ModelPackageError("The input schema order does not match the required columns.")
    source_required = contract.get("source_required_columns")
    source_schema = contract.get("source_schema")
    if source_required is not None or source_schema is not None:
        if not isinstance(source_required, list) or not source_required or not all(isinstance(c, str) and c for c in source_required):
            raise ModelPackageError("The source input contract has no valid required columns.")
        if len(source_required) != len(set(source_required)):
            raise ModelPackageError("The source input contract contains duplicate columns.")
        if not isinstance(source_schema, list) or len(source_schema) != len(source_required):
            raise ModelPackageError("The source input schema does not match the source required columns.")
        source_schema_names = [str(item.get("name", "")) for item in source_schema if isinstance(item, Mapping)]
        if source_schema_names != source_required:
            raise ModelPackageError("The source input schema order does not match the source required columns.")
    if artifact.get("member") != MODEL_MEMBER:
        raise ModelPackageError("The model artifact reference is invalid.")
    model_hash = str(artifact.get("sha256", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", model_hash):
        raise ModelPackageError("The model artifact hash is invalid.")
    size = int(artifact.get("size_bytes", -1))
    if size < 1 or size > MAX_MODEL_BYTES:
        raise ModelPackageError("The model artifact size is invalid.")
    if not isinstance(runtime, Mapping):
        raise ModelPackageError("The runtime compatibility metadata is missing.")
    if _version_pair(str(runtime.get("python", ""))) != _version_pair(platform.python_version()):
        raise ModelPackageError("The package was built with an incompatible Python major/minor version.")
    if _version_pair(str(runtime.get("scikit_learn", ""))) != _version_pair(sklearn.__version__):
        raise ModelPackageError("The package was built with an incompatible scikit-learn major/minor version.")


def load_signed_model_package(
    package_bytes: bytes,
    *,
    signing_key: Optional[bytes] = None,
    key_path: Optional[Path | str] = None,
) -> LoadedModelPackage:
    """Verify every package boundary before deserializing the trusted payload."""
    if not isinstance(package_bytes, (bytes, bytearray)):
        raise ModelPackageError("Model package bytes are required.")
    raw_package = bytes(package_bytes)
    if not raw_package:
        raise ModelPackageError("The model package is empty.")
    if len(raw_package) > MAX_PACKAGE_BYTES:
        raise ModelPackageError("The model package exceeds the safe size limit.")
    if raw_package[:2] != b"PK":
        raise ModelPackageError(
            "Raw .pkl/joblib files are not accepted. Load a signed .dbmlpkg package created by DataBridge AI."
        )

    try:
        with zipfile.ZipFile(io.BytesIO(raw_package), mode="r") as archive:
            infos = archive.infolist()
            if not infos or len(infos) > MAX_ARCHIVE_MEMBERS:
                raise ModelPackageError("The package has an invalid number of archive entries.")
            names = [info.filename for info in infos]
            if len(names) != len(set(names)):
                raise ModelPackageError("Duplicate archive entries are not allowed.")
            if set(names) != ALLOWED_MEMBERS:
                unexpected = sorted(set(names) - ALLOWED_MEMBERS)
                missing = sorted(ALLOWED_MEMBERS - set(names))
                detail = []
                if unexpected:
                    detail.append("unexpected: " + ", ".join(unexpected))
                if missing:
                    detail.append("missing: " + ", ".join(missing))
                raise ModelPackageError("Invalid package contents (" + "; ".join(detail) + ").")
            total_uncompressed = 0
            by_name: Dict[str, zipfile.ZipInfo] = {}
            for info in infos:
                _validate_archive_member(info)
                total_uncompressed += int(info.file_size)
                by_name[info.filename] = info
            if total_uncompressed > MAX_TOTAL_UNCOMPRESSED_BYTES:
                raise ModelPackageError("The package expands beyond the safe size limit.")

            manifest_raw = _read_member_bounded(
                archive, by_name[MANIFEST_MEMBER], MAX_MANIFEST_BYTES
            )
            signature_raw = _read_member_bounded(archive, by_name[SIGNATURE_MEMBER], 256)
            model_raw = _read_member_bounded(archive, by_name[MODEL_MEMBER], MAX_MODEL_BYTES)
            _read_member_bounded(archive, by_name[README_MEMBER], MAX_README_BYTES)
    except zipfile.BadZipFile as exc:
        raise ModelPackageError("The model package ZIP structure is invalid.") from exc

    try:
        manifest_value = json.loads(manifest_raw.decode("utf-8"))
    except Exception as exc:
        raise ModelPackageError("The package manifest is not valid UTF-8 JSON.") from exc
    if not isinstance(manifest_value, dict):
        raise ModelPackageError("The package manifest must be a JSON object.")
    manifest = dict(manifest_value)
    _validate_manifest(manifest)

    artifact = manifest["artifact"]
    if int(artifact["size_bytes"]) != len(model_raw):
        raise ModelPackageError("The model artifact size does not match the signed manifest.")
    if not hmac.compare_digest(str(artifact["sha256"]), _sha256(model_raw)):
        raise ModelPackageError("The model artifact hash is invalid; the package may be corrupted or modified.")

    try:
        signature = signature_raw.decode("ascii").strip().lower()
    except UnicodeDecodeError as exc:
        raise ModelPackageError("The package signature is invalid.") from exc
    if not re.fullmatch(r"[0-9a-f]{64}", signature):
        raise ModelPackageError("The package signature is invalid.")
    key = bytes(signing_key) if signing_key is not None else get_or_create_signing_key(key_path)
    local_key_id = signing_key_id(key)
    if not hmac.compare_digest(str(manifest["signer_key_id"]), local_key_id):
        raise ModelPackageError(
            "This package was signed with a different DataBridge AI model trust key. It was not deserialized."
        )
    expected_signature = _sign_manifest(manifest, key)
    if not hmac.compare_digest(signature, expected_signature):
        raise ModelPackageError(
            "Package authentication failed. The model was not deserialized."
        )

    # Authentication by the model trust key is the deserialization trust boundary.
    # Pickle is never invoked before archive, artifact-hash, trust-key-id, and HMAC
    # verification succeed. If that signing key is compromised, a malicious but
    # correctly authenticated pickle can execute during pickle.loads(); the type
    # and schema checks below validate the trusted payload contract, not a sandbox.
    try:
        payload = pickle.loads(model_raw)
    except Exception as exc:
        raise ModelPackageError(f"The authenticated model payload could not be loaded: {exc}") from exc
    if not isinstance(payload, dict):
        raise ModelPackageError("The authenticated model payload structure is invalid.")
    payload_version = int(payload.get("payload_version", -1))
    if payload_version not in SUPPORTED_MODEL_PAYLOAD_VERSIONS:
        raise ModelPackageError("Unsupported model payload version.")
    expected_keys = {"payload_version", "fitted_pipeline", "target_encoder"}
    if payload_version >= 2:
        expected_keys.add("retraining_contract")
    if payload_version >= 3:
        expected_keys.add("feature_derivation_recipe")
    if set(payload) != expected_keys:
        raise ModelPackageError("The authenticated model payload structure is invalid.")
    pipeline = payload.get("fitted_pipeline")
    encoder = payload.get("target_encoder")
    retraining_contract = payload.get("retraining_contract") if payload_version >= 2 else None
    derivation_recipe_raw = payload.get("feature_derivation_recipe", []) if payload_version >= 3 else []
    try:
        derivation_recipe = normalise_feature_derivation_recipe(derivation_recipe_raw)
    except FeatureDerivationError as exc:
        raise ModelPackageError(f"The authenticated feature-derivation contract is invalid: {exc}") from exc
    if retraining_contract is not None and not isinstance(retraining_contract, dict):
        raise ModelPackageError("The authenticated retraining contract is invalid.")
    if not isinstance(pipeline, Pipeline):
        raise ModelPackageError("The authenticated payload does not contain an sklearn Pipeline.")
    if encoder is not None and not isinstance(encoder, LabelEncoder):
        raise ModelPackageError("The authenticated target encoder is invalid.")
    required = list(map(str, manifest["input_contract"]["required_columns"]))
    pipeline_input = list(map(str, getattr(pipeline, "feature_names_in_", required)))
    if pipeline_input != required:
        raise ModelPackageError("The fitted pipeline input schema does not match the signed manifest.")
    source_required = list(map(str, manifest["input_contract"].get("source_required_columns") or required))
    try:
        expected_source = source_columns_for_pipeline(derivation_recipe, required)
    except FeatureDerivationError as exc:
        raise ModelPackageError(f"The authenticated feature-derivation dependencies are invalid: {exc}") from exc
    if expected_source != source_required:
        raise ModelPackageError("The authenticated feature-derivation recipe does not match the signed source input contract.")
    derivation_manifest = manifest.get("derivation_contract", {}) or {}
    if derivation_manifest:
        expected_fp = str(derivation_manifest.get("recipe_fingerprint", ""))
        actual_fp = feature_derivation_recipe_fingerprint(derivation_recipe)
        if expected_fp and not hmac.compare_digest(expected_fp, actual_fp):
            raise ModelPackageError("The authenticated feature-derivation recipe fingerprint does not match the signed manifest.")
    if str(manifest["model"]["task"]) == CLASSIFICATION:
        if not isinstance(encoder, LabelEncoder):
            raise ModelPackageError("The classification package has no target decoder.")
        signed_classes = list(map(str, manifest["model"].get("class_labels", []) or []))
        if list(map(str, encoder.classes_)) != signed_classes:
            raise ModelPackageError("The target decoder classes do not match the signed manifest.")

    return LoadedModelPackage(
        manifest=manifest,
        fitted_pipeline=pipeline,
        target_encoder=encoder,
        package_id=str(manifest["package_id"]),
        signer_key_id=local_key_id,
        trusted=True,
        retraining_contract=retraining_contract,
        feature_derivation_recipe=tuple(dict(step) for step in derivation_recipe),
    )


# ── Prediction schema and execution ──────────────────────────────────────────
def _schema_by_name(package: LoadedModelPackage) -> Dict[str, Mapping[str, Any]]:
    contract = package.manifest.get("input_contract", {}) or {}
    schema = contract.get("source_schema") or contract.get("schema") or []
    return {
        str(item.get("name")): item
        for item in schema
        if isinstance(item, Mapping) and item.get("name")
    }


def _normalise_numeric_probe(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    text = text.str.replace(r"[,$£€¥₹]", "", regex=True)
    text = text.str.replace("%", "", regex=False)
    text = text.str.replace(r"\s+", "", regex=True)
    return pd.to_numeric(text, errors="coerce")


def _datetime_probe(series: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(series, errors="coerce", format="mixed", dayfirst=True)
    except (TypeError, ValueError):
        return pd.to_datetime(series, errors="coerce", dayfirst=True)


def validate_prediction_frame(
    df: pd.DataFrame,
    package: LoadedModelPackage,
    *,
    max_rows: int = DEFAULT_MAX_PREDICTION_ROWS,
) -> Dict[str, Any]:
    if not isinstance(package, LoadedModelPackage) or not package.trusted:
        raise ModelPackageError("A verified model package is required.")
    blockers: list[str] = []
    warnings: list[str] = []
    if not isinstance(df, pd.DataFrame):
        blockers.append("Prediction input is not a DataFrame.")
        return {"status": "Blocked", "valid": False, "blockers": blockers, "warnings": warnings}
    if df.empty:
        blockers.append("Prediction input has no rows.")
    if len(df) > int(max_rows):
        blockers.append(
            f"Prediction input has {len(df):,} rows, above the explicit safety cap of {int(max_rows):,}."
        )
    if df.columns.duplicated().any():
        duplicates = list(map(str, df.columns[df.columns.duplicated()].tolist()))
        blockers.append("Duplicate input column names: " + ", ".join(duplicates[:10]))

    required = package.required_columns
    missing = [column for column in required if column not in df.columns]
    extra = [str(column) for column in df.columns if str(column) not in set(required)]
    if missing:
        blockers.append("Missing required columns: " + ", ".join(missing))
    if extra:
        warnings.append(
            f"{len(extra):,} extra column(s) will be preserved in output but ignored by the model."
        )
    if package.feature_derivation_recipe:
        supplied_derived = [column for column in package.derived_columns if column in df.columns]
        if supplied_derived:
            warnings.append(
                "Supplied derived feature column(s) are not trusted as model inputs and will be recomputed from the signed recipe: "
                + ", ".join(supplied_derived)
            )
        warnings.append(
            f"{len(package.feature_derivation_recipe):,} signed feature derivation step(s) will be replayed automatically before scoring."
        )

    schema = _schema_by_name(package)
    checks: list[Dict[str, Any]] = []
    for column in required:
        expected = schema.get(column, {}) or {}
        semantic = str(expected.get("semantic_type", "Unknown"))
        row: Dict[str, Any] = {
            "Column": column,
            "Expected semantic": semantic,
            "Expected dtype": str(expected.get("pandas_dtype", "")),
            "Actual dtype": str(df[column].dtype) if column in df.columns else "Missing",
            "Non-null": int(df[column].notna().sum()) if column in df.columns else 0,
            "Parse success %": None,
            "Status": "Missing" if column not in df.columns else "Ready",
        }
        if column not in df.columns:
            checks.append(row)
            continue
        series = df[column]
        non_null = series.dropna()
        if non_null.empty:
            row["Status"] = "Warning"
            warnings.append(f"Required column '{column}' is entirely missing in prediction input.")
            checks.append(row)
            continue

        ratio: Optional[float] = None
        if semantic in NUMERIC_SEMANTICS:
            ratio = float(_normalise_numeric_probe(non_null).notna().mean())
        elif semantic == "Datetime":
            ratio = float(_datetime_probe(non_null).notna().mean())
        elif semantic == "Boolean":
            tokens = non_null.astype("string").str.strip().str.casefold()
            allowed = {
                "true", "false", "1", "0", "yes", "no", "y", "n",
                "نعم", "لا", "صح", "خطأ",
            }
            ratio = float(tokens.isin(allowed).mean())
        if ratio is not None:
            row["Parse success %"] = round(ratio * 100, 1)
            if ratio == 0:
                row["Status"] = "Blocked"
                blockers.append(
                    f"Column '{column}' cannot be parsed as expected semantic type {semantic}."
                )
            elif ratio < 0.80:
                row["Status"] = "Warning"
                warnings.append(
                    f"Only {ratio * 100:.1f}% of non-null values in '{column}' match expected type {semantic}; invalid values may be imputed by the fitted pipeline."
                )
        checks.append(row)

    return {
        "status": "Blocked" if blockers else "Ready",
        "valid": not blockers,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "required_columns": required,
        "missing_columns": missing,
        "extra_columns": extra,
        "blockers": list(dict.fromkeys(blockers)),
        "warnings": list(dict.fromkeys(warnings)),
        "column_checks": checks,
        "package_id": package.package_id,
    }


def _unique_column_name(existing: Sequence[Any], desired: str) -> str:
    used = {str(value) for value in existing}
    if desired not in used:
        return desired
    counter = 2
    while f"{desired}_{counter}" in used:
        counter += 1
    return f"{desired}_{counter}"


def _safe_probability_label(value: str) -> str:
    cleaned = re.sub(r"[^\w\-]+", "_", str(value), flags=re.UNICODE).strip("_")
    return (cleaned or "class")[:60]


def prepare_model_input_frame(
    package: LoadedModelPackage,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Recreate signed source-level derived features on an isolated input copy."""
    if not isinstance(package, LoadedModelPackage) or not package.trusted:
        raise ModelPackageError("A verified model package is required.")
    if not isinstance(df, pd.DataFrame):
        raise ModelPackageError("Prediction input is not a DataFrame.")
    try:
        prepared = replay_feature_derivations(
            df,
            package.feature_derivation_recipe,
            pipeline_required_columns=package.pipeline_required_columns,
        )
    except FeatureDerivationError as exc:
        raise ModelPackageError(f"Feature derivation replay failed safely: {exc}") from exc
    missing = [column for column in package.pipeline_required_columns if column not in prepared.columns]
    if missing:
        raise ModelPackageError(
            "Prepared model input is missing pipeline column(s): " + ", ".join(missing)
        )
    return prepared


def run_batch_prediction(
    package: LoadedModelPackage,
    df: pd.DataFrame,
    *,
    max_rows: int = DEFAULT_MAX_PREDICTION_ROWS,
    include_probabilities: bool = True,
) -> PredictionRunResult:
    report = validate_prediction_frame(df, package, max_rows=max_rows)
    if not report["valid"]:
        raise ModelPackageError("Prediction blocked: " + "; ".join(report["blockers"]))
    prepared = prepare_model_input_frame(package, df)
    required = package.pipeline_required_columns
    X = prepared.loc[:, required].copy(deep=True)
    try:
        raw_predictions = np.asarray(package.fitted_pipeline.predict(X))
    except Exception as exc:
        raise ModelPackageError(f"The verified model could not score this input safely: {exc}") from exc
    if len(raw_predictions) != len(df):
        raise ModelPackageError("The model returned an unexpected number of predictions.")

    output = df.copy(deep=True)
    prediction_column = _unique_column_name(output.columns, "Prediction")
    probability_columns: list[str] = []
    warnings = list(report.get("warnings", []))

    if package.task == CLASSIFICATION:
        encoder = package.target_encoder
        if not isinstance(encoder, LabelEncoder):
            raise ModelPackageError("The verified classification package has no target decoder.")
        try:
            numeric_predictions = raw_predictions.astype(int)
            decoded = encoder.inverse_transform(numeric_predictions)
        except Exception as exc:
            raise ModelPackageError("The model returned class values outside its signed target contract.") from exc
        output[prediction_column] = decoded

        if include_probabilities and hasattr(package.fitted_pipeline, "predict_proba"):
            try:
                probabilities = np.asarray(package.fitted_pipeline.predict_proba(X), dtype=float)
                classes = list(map(str, encoder.classes_))
                if probabilities.ndim != 2 or probabilities.shape != (len(df), len(classes)):
                    raise ValueError("unexpected probability matrix shape")
                confidence_name = _unique_column_name(output.columns, "Prediction_Confidence")
                output[confidence_name] = probabilities.max(axis=1)
                probability_columns.append(confidence_name)
                if len(classes) <= MAX_PROBABILITY_COLUMNS:
                    used_names = set(map(str, output.columns))
                    for index, class_label in enumerate(classes):
                        base = f"Probability_{_safe_probability_label(class_label)}"
                        name = base
                        counter = 2
                        while name in used_names:
                            name = f"{base}_{counter}"
                            counter += 1
                        used_names.add(name)
                        output[name] = probabilities[:, index]
                        probability_columns.append(name)
                else:
                    warnings.append(
                        f"Per-class probability columns were omitted because the model has {len(classes):,} classes; only prediction confidence was added."
                    )
            except Exception as exc:
                warnings.append(f"Probability output was unavailable: {exc}")
        elif include_probabilities:
            warnings.append(
                "This estimator does not expose calibrated probabilities; class predictions were produced without confidence columns."
            )
    elif package.task == REGRESSION:
        output[prediction_column] = pd.to_numeric(
            pd.Series(raw_predictions, index=output.index), errors="coerce"
        ).to_numpy()
    else:
        raise ModelPackageError("Unsupported model task.")

    return PredictionRunResult(
        output=output,
        package_id=package.package_id,
        task=package.task,
        target=package.target,
        rows=int(len(output)),
        prediction_column=prediction_column,
        probability_columns=tuple(probability_columns),
        warnings=tuple(dict.fromkeys(map(str, warnings))),
        created_at=_utc_now(),
    )
