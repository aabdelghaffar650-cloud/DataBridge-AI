# ════════════════════════════════════════════════════════
#  DataBridge AI — Model Monitoring & Drift Engine
#  Stage 14: privacy-preserving training references, schema/data/prediction
#            drift, observed performance, and monitoring summaries
# ════════════════════════════════════════════════════════
from __future__ import annotations

import hashlib
import hmac
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from modules.feature_pipeline import normalise_feature_pipeline_spec


MONITORING_REFERENCE_VERSION = 1
MONITORING_REPORT_VERSION = 1
MONITORING_TOKEN_CONTEXT = b"DataBridgeAI-Monitoring-Category-Token-v1\0"

STATUS_STABLE = "Stable"
STATUS_WATCH = "Watch"
STATUS_DRIFTED = "Drifted"
STATUS_CRITICAL = "Critical"
_STATUS_ORDER = {
    STATUS_STABLE: 0,
    STATUS_WATCH: 1,
    STATUS_DRIFTED: 2,
    STATUS_CRITICAL: 3,
}

MAX_REFERENCE_CATEGORIES = 60
MAX_FULL_VOCABULARY = 250
MAX_MONITORING_COLUMNS = 500
EPSILON = 1e-6


class ModelMonitoringError(ValueError):
    """Raised when a monitoring calculation cannot be completed safely."""


@dataclass(frozen=True)
class MonitoringResult:
    report: Dict[str, Any]
    feature_table: pd.DataFrame
    prediction_output: Optional[pd.DataFrame]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _safe_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return str(value)


def _status_max(*statuses: str) -> str:
    valid = [status for status in statuses if status in _STATUS_ORDER]
    if not valid:
        return STATUS_STABLE
    return max(valid, key=lambda status: _STATUS_ORDER[status])


def _status_score(status: str) -> int:
    return {STATUS_STABLE: 0, STATUS_WATCH: 33, STATUS_DRIFTED: 67, STATUS_CRITICAL: 100}.get(status, 100)


def _python_string_series(series: pd.Series) -> pd.Series:
    """Avoid Arrow regex edge cases and preserve nulls as pandas missing values."""
    result = series.astype("object")
    mask = result.notna()
    out = pd.Series(pd.NA, index=series.index, dtype="string[python]")
    if mask.any():
        out.loc[mask] = result.loc[mask].map(str).astype("string[python]")
    return out


def _parse_numeric(series: pd.Series, *, percentage: bool = False) -> pd.Series:
    text = _python_string_series(series).str.strip()
    text = text.str.replace(r"[,$£€¥₹]", "", regex=True)
    has_percent = text.str.contains("%", regex=False, na=False)
    text = text.str.replace("%", "", regex=False)
    text = text.str.replace(r"[\s\u00A0]", "", regex=True)
    values = pd.to_numeric(text, errors="coerce")
    if percentage:
        values = values.where(~has_percent, values / 100.0)
    return values.astype(float)


def _parse_datetime(series: pd.Series) -> pd.Series:
    try:
        parsed = pd.to_datetime(series, errors="coerce", format="mixed", dayfirst=True, utc=True)
    except (TypeError, ValueError):
        parsed = pd.to_datetime(series, errors="coerce", dayfirst=True, utc=True)
    return parsed


def _normalised_category_values(series: pd.Series) -> pd.Series:
    return _python_string_series(series).str.strip().fillna("__MISSING__")


def _numeric_profile(values: pd.Series) -> Dict[str, Any]:
    numeric = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan)
    non_null = numeric.dropna().astype(float)
    total = max(len(numeric), 1)
    missing_ratio = float(numeric.isna().sum() / total)
    if non_null.empty:
        return {
            "kind": "numeric",
            "count": 0,
            "missing_ratio": round(missing_ratio, 8),
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "bin_edges": [],
            "bin_shares": [],
        }

    quantiles = np.linspace(0.0, 1.0, 11)
    q_values = np.quantile(non_null.to_numpy(dtype=float), quantiles)
    internal = sorted({float(value) for value in q_values[1:-1] if math.isfinite(float(value))})
    # Keep only strict interior edges. Duplicate quantiles are intentionally collapsed.
    low = float(non_null.min())
    high = float(non_null.max())
    internal = [value for value in internal if low < value < high]
    bins = np.asarray([-np.inf, *internal, np.inf], dtype=float)
    counts, _ = np.histogram(non_null.to_numpy(dtype=float), bins=bins)
    shares = (counts / max(int(counts.sum()), 1)).astype(float)
    std = float(non_null.std(ddof=0)) if len(non_null) > 1 else 0.0
    return {
        "kind": "numeric",
        "count": int(len(non_null)),
        "missing_ratio": round(missing_ratio, 8),
        "mean": _safe_float(non_null.mean()),
        "std": _safe_float(std),
        "min": _safe_float(low),
        "max": _safe_float(high),
        "bin_edges": [_safe_float(value) for value in internal],
        "bin_shares": [_safe_float(value) or 0.0 for value in shares.tolist()],
    }


def _categorical_profile(series: pd.Series) -> Dict[str, Any]:
    values = _normalised_category_values(series)
    total = max(len(values), 1)
    missing_count = int((values == "__MISSING__").sum())
    observed = values[values != "__MISSING__"]
    counts = observed.value_counts(dropna=False)
    unique = int(len(counts))
    top = counts.head(MAX_REFERENCE_CATEGORIES)
    distribution = [
        {"value": str(value), "share": float(count / max(len(observed), 1))}
        for value, count in top.items()
    ]
    top_count = int(top.sum()) if not top.empty else 0
    other_share = float(max(len(observed) - top_count, 0) / max(len(observed), 1))
    full_vocabulary = list(map(str, counts.index.tolist())) if unique <= MAX_FULL_VOCABULARY else []
    return {
        "kind": "categorical",
        "count": int(len(observed)),
        "missing_ratio": round(missing_count / total, 8),
        "unique": unique,
        "top_distribution": distribution,
        "other_share": round(other_share, 8),
        "full_vocabulary": full_vocabulary,
        "vocabulary_complete": bool(full_vocabulary or unique == 0),
    }


def _text_profile(series: pd.Series) -> Dict[str, Any]:
    text = _python_string_series(series)
    lengths = text.str.len().astype(float)
    profile = _numeric_profile(lengths)
    profile["kind"] = "text_length"
    return profile


def _datetime_profile(series: pd.Series) -> Dict[str, Any]:
    parsed = _parse_datetime(series)
    numeric = pd.Series(np.nan, index=series.index, dtype=float)
    valid = parsed.notna()
    if valid.any():
        numeric.loc[valid] = parsed.loc[valid].astype("int64") / 86_400_000_000_000.0
    profile = _numeric_profile(numeric)
    profile["kind"] = "datetime"
    return profile


def build_monitoring_reference(
    training_frame: pd.DataFrame,
    feature_pipeline_spec: Mapping[str, Any],
) -> Dict[str, Any]:
    """
    Build an aggregate-only reference from training rows.

    Categorical values are kept only in the in-memory experiment artifact. Model
    package creation converts them to HMAC tokens before persistence.
    """
    if not isinstance(training_frame, pd.DataFrame) or training_frame.empty:
        raise ModelMonitoringError("Training reference requires non-empty training rows.")
    spec = normalise_feature_pipeline_spec(feature_pipeline_spec)
    feature_columns = list(map(str, spec.get("feature_columns", []) or []))
    if len(feature_columns) > MAX_MONITORING_COLUMNS:
        raise ModelMonitoringError("Too many model input columns for safe monitoring reference generation.")
    groups = spec.get("groups", {}) or {}
    group_for: Dict[str, str] = {}
    for group_name, columns in groups.items():
        for column in columns or []:
            group_for[str(column)] = str(group_name)
    semantics = {str(k): str(v) for k, v in (spec.get("semantic_types", {}) or {}).items()}

    columns: Dict[str, Any] = {}
    for column in feature_columns:
        if column not in training_frame.columns:
            continue
        group = group_for.get(column, "unknown")
        semantic = semantics.get(column, "Unknown")
        series = training_frame[column]
        if group == "numeric":
            profile = _numeric_profile(_parse_numeric(series, percentage=(semantic == "Percentage")))
        elif group == "datetime":
            profile = _datetime_profile(series)
        elif group == "text":
            profile = _text_profile(series)
        else:
            profile = _categorical_profile(series)
        profile["group"] = group
        profile["semantic_type"] = semantic
        columns[column] = profile

    return {
        "reference_version": MONITORING_REFERENCE_VERSION,
        "reference_scope": "training_rows_only",
        "rows": int(len(training_frame)),
        "columns": columns,
        "contains_row_level_data": False,
    }


def _monitoring_token_key(signing_key: bytes) -> bytes:
    if not isinstance(signing_key, (bytes, bytearray)) or len(signing_key) != 32:
        raise ModelMonitoringError("A valid 256-bit model-package signing key is required.")
    return hmac.new(bytes(signing_key), MONITORING_TOKEN_CONTEXT, hashlib.sha256).digest()


def _category_token(value: str, token_key: bytes) -> str:
    return hmac.new(token_key, str(value).encode("utf-8"), hashlib.sha256).hexdigest()


def protect_monitoring_reference(reference: Mapping[str, Any], signing_key: bytes) -> Dict[str, Any]:
    """Replace persisted category labels with installation-bound HMAC tokens."""
    token_key = _monitoring_token_key(signing_key)
    protected = {
        "reference_version": int(reference.get("reference_version", MONITORING_REFERENCE_VERSION)),
        "reference_scope": str(reference.get("reference_scope", "training_rows_only")),
        "rows": int(reference.get("rows", 0) or 0),
        "contains_row_level_data": False,
        "category_token_scheme": "HMAC-SHA256/local-signing-key-v1",
        "columns": {},
    }
    for column, raw_profile in (reference.get("columns", {}) or {}).items():
        profile = dict(raw_profile or {})
        if profile.get("kind") == "categorical":
            dist = []
            for item in profile.get("top_distribution", []) or []:
                dist.append(
                    {
                        "token": _category_token(str(item.get("value", "")), token_key),
                        "share": _safe_float(item.get("share")) or 0.0,
                    }
                )
            vocabulary = [
                _category_token(str(value), token_key)
                for value in (profile.get("full_vocabulary", []) or [])
            ]
            profile.pop("top_distribution", None)
            profile.pop("full_vocabulary", None)
            profile["top_distribution_tokens"] = dist
            profile["vocabulary_tokens"] = vocabulary
        protected["columns"][str(column)] = _json_safe(profile)
    return protected


def build_prediction_reference(task: str, predictions: pd.DataFrame) -> Dict[str, Any]:
    if not isinstance(predictions, pd.DataFrame) or "Predicted" not in predictions.columns:
        return {}
    if str(task) == "classification":
        counts = predictions["Predicted"].astype("string[python]").value_counts(dropna=False)
        total = max(int(counts.sum()), 1)
        return {
            "kind": "classification",
            "distribution": [
                {"label": str(label), "share": float(count / total)}
                for label, count in counts.items()
            ],
        }
    return {
        "kind": "regression",
        "distribution": _numeric_profile(pd.to_numeric(predictions["Predicted"], errors="coerce")),
    }


def _psi(expected: Sequence[float], actual: Sequence[float]) -> float:
    left = np.asarray(expected, dtype=float)
    right = np.asarray(actual, dtype=float)
    if left.shape != right.shape or left.size == 0:
        return 0.0
    left = np.clip(left, EPSILON, None)
    right = np.clip(right, EPSILON, None)
    left = left / left.sum()
    right = right / right.sum()
    return float(np.sum((right - left) * np.log(right / left)))


def _js_divergence(expected: Sequence[float], actual: Sequence[float]) -> float:
    left = np.asarray(expected, dtype=float)
    right = np.asarray(actual, dtype=float)
    if left.shape != right.shape or left.size == 0:
        return 0.0
    left = np.clip(left, 0.0, None)
    right = np.clip(right, 0.0, None)
    if left.sum() <= 0 or right.sum() <= 0:
        return 0.0
    left = left / left.sum()
    right = right / right.sum()
    middle = 0.5 * (left + right)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / np.clip(b[mask], EPSILON, None))))

    return 0.5 * _kl(left, middle) + 0.5 * _kl(right, middle)


def _missing_status(delta: float) -> str:
    if delta >= 0.20:
        return STATUS_CRITICAL
    if delta >= 0.10:
        return STATUS_DRIFTED
    if delta >= 0.05:
        return STATUS_WATCH
    return STATUS_STABLE


def _numeric_status(psi: float, out_of_range: float, std_shift: Optional[float], missing_delta: float) -> str:
    status = _missing_status(missing_delta)
    if psi >= 0.35 or out_of_range >= 0.25 or (std_shift is not None and std_shift >= 3.0):
        status = _status_max(status, STATUS_CRITICAL)
    elif psi >= 0.20 or out_of_range >= 0.10 or (std_shift is not None and std_shift >= 2.0):
        status = _status_max(status, STATUS_DRIFTED)
    elif psi >= 0.10 or out_of_range >= 0.03 or (std_shift is not None and std_shift >= 1.0):
        status = _status_max(status, STATUS_WATCH)
    return status


def _categorical_status(js: float, unseen: Optional[float], missing_delta: float) -> str:
    status = _missing_status(missing_delta)
    if js >= 0.35 or (unseen is not None and unseen >= 0.50):
        status = _status_max(status, STATUS_CRITICAL)
    elif js >= 0.20 or (unseen is not None and unseen >= 0.25):
        status = _status_max(status, STATUS_DRIFTED)
    elif js >= 0.10 or (unseen is not None and unseen >= 0.10):
        status = _status_max(status, STATUS_WATCH)
    return status


def _numeric_drift(current: pd.Series, reference: Mapping[str, Any]) -> Dict[str, Any]:
    values = pd.to_numeric(current, errors="coerce").replace([np.inf, -np.inf], np.nan)
    non_null = values.dropna().astype(float)
    missing_ratio = float(values.isna().sum() / max(len(values), 1))
    missing_delta = abs(missing_ratio - float(reference.get("missing_ratio", 0.0) or 0.0))
    edges = [float(value) for value in (reference.get("bin_edges", []) or []) if _safe_float(value) is not None]
    bins = np.asarray([-np.inf, *edges, np.inf], dtype=float)
    if non_null.empty:
        actual_shares = np.zeros(max(len(bins) - 1, 1), dtype=float)
    else:
        counts, _ = np.histogram(non_null.to_numpy(dtype=float), bins=bins)
        actual_shares = counts / max(int(counts.sum()), 1)
    expected_shares = [float(value or 0.0) for value in (reference.get("bin_shares", []) or [])]
    if len(expected_shares) != len(actual_shares):
        psi_value = 0.0
    else:
        psi_value = _psi(expected_shares, actual_shares)

    ref_min = _safe_float(reference.get("min"))
    ref_max = _safe_float(reference.get("max"))
    if non_null.empty or ref_min is None or ref_max is None:
        out_of_range = 0.0
    else:
        out_of_range = float(((non_null < ref_min) | (non_null > ref_max)).mean())
    ref_mean = _safe_float(reference.get("mean"))
    ref_std = _safe_float(reference.get("std"))
    current_mean = _safe_float(non_null.mean()) if not non_null.empty else None
    std_shift: Optional[float] = None
    if ref_mean is not None and current_mean is not None:
        scale = max(abs(ref_std or 0.0), EPSILON)
        std_shift = abs(current_mean - ref_mean) / scale
    status = _numeric_status(psi_value, out_of_range, std_shift, missing_delta)
    return {
        "status": status,
        "metric": "PSI",
        "drift_value": round(psi_value, 6),
        "missing_ratio": round(missing_ratio, 6),
        "missing_delta": round(missing_delta, 6),
        "out_of_range_ratio": round(out_of_range, 6),
        "standardized_mean_shift": round(std_shift, 6) if std_shift is not None and math.isfinite(std_shift) else None,
    }


def _categorical_drift(current: pd.Series, reference: Mapping[str, Any], token_key: bytes) -> Dict[str, Any]:
    values = _normalised_category_values(current)
    missing_ratio = float((values == "__MISSING__").mean()) if len(values) else 0.0
    missing_delta = abs(missing_ratio - float(reference.get("missing_ratio", 0.0) or 0.0))
    observed = values[values != "__MISSING__"]
    token_counts: Dict[str, int] = {}
    for value, count in observed.value_counts(dropna=False).items():
        token_counts[_category_token(str(value), token_key)] = int(count)
    total = max(int(len(observed)), 1)

    expected_items = reference.get("top_distribution_tokens", []) or []
    expected_tokens = [str(item.get("token", "")) for item in expected_items]
    expected = [float(item.get("share", 0.0) or 0.0) for item in expected_items]
    actual = [float(token_counts.get(token, 0) / total) for token in expected_tokens]
    expected_other = float(reference.get("other_share", 0.0) or 0.0)
    actual_other = max(0.0, 1.0 - sum(actual)) if len(observed) else 0.0
    expected.append(expected_other)
    actual.append(actual_other)
    js = _js_divergence(expected, actual)

    unseen: Optional[float] = None
    if bool(reference.get("vocabulary_complete", False)):
        vocabulary = set(map(str, reference.get("vocabulary_tokens", []) or []))
        if len(observed):
            unseen_count = sum(count for token, count in token_counts.items() if token not in vocabulary)
            unseen = float(unseen_count / len(observed))
        else:
            unseen = 0.0
    status = _categorical_status(js, unseen, missing_delta)
    return {
        "status": status,
        "metric": "Jensen-Shannon",
        "drift_value": round(js, 6),
        "missing_ratio": round(missing_ratio, 6),
        "missing_delta": round(missing_delta, 6),
        "unseen_category_ratio": round(unseen, 6) if unseen is not None else None,
        "vocabulary_complete": bool(reference.get("vocabulary_complete", False)),
    }


def _classification_prediction_drift(current: pd.Series, reference: Mapping[str, Any]) -> Dict[str, Any]:
    expected_items = reference.get("distribution", []) or []
    labels = [str(item.get("label", "")) for item in expected_items]
    expected = [float(item.get("share", 0.0) or 0.0) for item in expected_items]
    counts = current.astype("string[python]").value_counts(dropna=False)
    total = max(int(counts.sum()), 1)
    actual = [float(counts.get(label, 0) / total) for label in labels]
    extra_share = max(0.0, 1.0 - sum(actual))
    expected_extra = max(0.0, 1.0 - sum(expected))
    js = _js_divergence([*expected, expected_extra], [*actual, extra_share])
    status = _categorical_status(js, extra_share if extra_share > 0 else 0.0, 0.0)
    return {
        "status": status,
        "metric": "Jensen-Shannon",
        "value": round(js, 6),
        "new_prediction_label_share": round(extra_share, 6),
    }


def _performance_summary(task: str, actual: pd.Series, predicted: pd.Series) -> Dict[str, Any]:
    if task == "classification":
        mask = actual.notna() & predicted.notna()
        if int(mask.sum()) < 2:
            return {"available": False, "reason": "Not enough labelled rows."}
        y_true = actual.loc[mask].astype("string[python]").astype(str)
        y_pred = predicted.loc[mask].astype("string[python]").astype(str)
        return {
            "available": True,
            "labelled_rows": int(mask.sum()),
            "metrics": {
                "Accuracy": float(accuracy_score(y_true, y_pred)),
                "Balanced Accuracy": float(balanced_accuracy_score(y_true, y_pred)),
                "F1 Macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            },
        }

    y_true = pd.to_numeric(actual, errors="coerce")
    y_pred = pd.to_numeric(predicted, errors="coerce")
    mask = y_true.notna() & y_pred.notna()
    if int(mask.sum()) < 3:
        return {"available": False, "reason": "Not enough labelled numeric rows."}
    yt = y_true.loc[mask].astype(float)
    yp = y_pred.loc[mask].astype(float)
    return {
        "available": True,
        "labelled_rows": int(mask.sum()),
        "metrics": {
            "MAE": float(mean_absolute_error(yt, yp)),
            "RMSE": float(math.sqrt(mean_squared_error(yt, yp))),
            "R²": float(r2_score(yt, yp)),
        },
    }


def run_monitoring_analysis(
    package: Any,
    frame: pd.DataFrame,
    *,
    signing_key: bytes,
    actual_target_column: Optional[str] = None,
    include_prediction_output: bool = True,
) -> MonitoringResult:
    """Run schema, feature, prediction, and optional observed-performance monitoring."""
    # Local import avoids a model_package -> monitoring import cycle during package creation.
    from modules.model_package import (
        LoadedModelPackage,
        prepare_model_input_frame,
        run_batch_prediction,
        validate_prediction_frame,
    )

    if not isinstance(package, LoadedModelPackage) or not package.trusted:
        raise ModelMonitoringError("A verified signed model package is required.")
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ModelMonitoringError("Monitoring input must contain at least one row.")

    monitoring = package.manifest.get("monitoring", {}) or {}
    reference = monitoring.get("training_reference", {}) or {}
    if not reference or not reference.get("columns"):
        raise ModelMonitoringError(
            "This package has no Stage 14 training reference. Rebuild the signed package from ML Studio V2 after installing Stage 14."
        )
    token_key = _monitoring_token_key(signing_key)
    schema_report = validate_prediction_frame(frame, package)
    try:
        prepared_frame = prepare_model_input_frame(package, frame) if schema_report.get("valid") else frame
    except Exception as exc:
        prepared_frame = frame
        schema_report = dict(schema_report)
        schema_report["valid"] = False
        schema_report["status"] = "Blocked"
        blockers = list(schema_report.get("blockers", []) or [])
        blockers.append(f"Feature derivation replay failed: {exc}")
        schema_report["blockers"] = list(dict.fromkeys(blockers))
    feature_rows: list[Dict[str, Any]] = []
    statuses: list[str] = []

    for column in package.pipeline_required_columns:
        ref = dict((reference.get("columns", {}) or {}).get(column, {}) or {})
        if column not in prepared_frame.columns:
            row = {
                "Column": column,
                "Type": ref.get("kind", "unknown"),
                "Status": STATUS_CRITICAL,
                "Metric": "Schema",
                "Drift": None,
                "Missing Δ": None,
                "Out-of-range": None,
                "Unseen categories": None,
            }
            feature_rows.append(row)
            statuses.append(STATUS_CRITICAL)
            continue
        kind = str(ref.get("kind", ""))
        series = prepared_frame[column]
        if kind == "numeric":
            semantic = str(ref.get("semantic_type", ""))
            detail = _numeric_drift(_parse_numeric(series, percentage=(semantic == "Percentage")), ref)
        elif kind == "datetime":
            parsed = _parse_datetime(series)
            numeric = pd.Series(np.nan, index=series.index, dtype=float)
            valid = parsed.notna()
            if valid.any():
                numeric.loc[valid] = parsed.loc[valid].astype("int64") / 86_400_000_000_000.0
            detail = _numeric_drift(numeric, ref)
        elif kind == "text_length":
            detail = _numeric_drift(_python_string_series(series).str.len().astype(float), ref)
        else:
            detail = _categorical_drift(series, ref, token_key)
        status = str(detail.get("status", STATUS_CRITICAL))
        statuses.append(status)
        feature_rows.append(
            {
                "Column": column,
                "Type": kind or "categorical",
                "Status": status,
                "Metric": detail.get("metric"),
                "Drift": detail.get("drift_value"),
                "Missing Δ": detail.get("missing_delta"),
                "Out-of-range": detail.get("out_of_range_ratio"),
                "Unseen categories": detail.get("unseen_category_ratio"),
                "Mean shift (σ)": detail.get("standardized_mean_shift"),
            }
        )

    feature_table = pd.DataFrame(feature_rows)
    prediction_output: Optional[pd.DataFrame] = None
    prediction_report: Dict[str, Any] = {"available": False}
    performance: Dict[str, Any] = {"available": False}
    if schema_report.get("valid"):
        prediction_result = run_batch_prediction(package, frame, include_probabilities=False)
        prediction_output = prediction_result.output if include_prediction_output else None
        pred_col = prediction_result.prediction_column
        pred_series = prediction_result.output[pred_col]
        pred_reference = monitoring.get("prediction_reference", {}) or {}
        if pred_reference:
            if package.task == "classification":
                prediction_report = {
                    "available": True,
                    **_classification_prediction_drift(pred_series, pred_reference),
                }
            else:
                distribution = pred_reference.get("distribution", {}) or {}
                detail = _numeric_drift(pd.to_numeric(pred_series, errors="coerce"), distribution)
                prediction_report = {
                    "available": True,
                    "status": detail["status"],
                    "metric": detail["metric"],
                    "value": detail["drift_value"],
                    "out_of_range_ratio": detail.get("out_of_range_ratio"),
                }
            statuses.append(str(prediction_report.get("status", STATUS_STABLE)))

        if actual_target_column and actual_target_column in frame.columns:
            performance = _performance_summary(
                package.task,
                frame[actual_target_column],
                prediction_result.output[pred_col],
            )
            performance["actual_column"] = str(actual_target_column)
            performance["signed_holdout_metrics"] = _json_safe(
                package.manifest.get("evaluation", {}).get("holdout_metrics", {}) or {}
            )

    schema_status = STATUS_CRITICAL if not schema_report.get("valid") else STATUS_STABLE
    statuses.append(schema_status)
    overall = _status_max(*statuses)
    scores = [_status_score(status) for status in statuses]
    drift_score = round((0.55 * max(scores, default=0)) + (0.45 * (sum(scores) / max(len(scores), 1))), 1)

    report = {
        "report_version": MONITORING_REPORT_VERSION,
        "created_at": _utc_now(),
        "package_id": package.package_id,
        "task": package.task,
        "target": package.target,
        "rows": int(len(frame)),
        "overall_status": overall,
        "drift_score": drift_score,
        "schema": _json_safe(schema_report),
        "feature_drift": _json_safe(feature_rows),
        "prediction_drift": _json_safe(prediction_report),
        "observed_performance": _json_safe(performance),
        "thresholds": {
            "numeric_psi_watch": 0.10,
            "numeric_psi_drifted": 0.20,
            "numeric_psi_critical": 0.35,
            "categorical_js_watch": 0.10,
            "categorical_js_drifted": 0.20,
            "categorical_js_critical": 0.35,
            "missing_delta_watch": 0.05,
            "missing_delta_drifted": 0.10,
            "missing_delta_critical": 0.20,
        },
        "reference": {
            "scope": reference.get("reference_scope", "training_rows_only"),
            "training_rows": int(reference.get("rows", 0) or 0),
            "contains_row_level_data": False,
            "category_labels_persisted": False,
        },
    }
    return MonitoringResult(report=report, feature_table=feature_table, prediction_output=prediction_output)
