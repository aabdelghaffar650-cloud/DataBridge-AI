# ════════════════════════════════════════════════════════
#  DataBridge AI — Data Quality Engine V2
#  Stage 11: weighted, deduplicated, policy-aware quality scoring
# ════════════════════════════════════════════════════════
from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence

import numpy as np
import pandas as pd
import streamlit as st


QUALITY_ENGINE_VERSION = "11.0"
DEFAULT_DIMENSION_WEIGHTS: Dict[str, float] = {
    "completeness": 0.40,
    "validity": 0.40,
    "uniqueness": 0.20,
}
QUALITY_WEIGHT_PRESETS: Dict[str, Dict[str, float]] = {
    "Balanced": dict(DEFAULT_DIMENSION_WEIGHTS),
    "Strict completeness": {
        "completeness": 0.50,
        "validity": 0.35,
        "uniqueness": 0.15,
    },
    "Strict validity": {
        "completeness": 0.30,
        "validity": 0.55,
        "uniqueness": 0.15,
    },
    "Strict uniqueness": {
        "completeness": 0.30,
        "validity": 0.30,
        "uniqueness": 0.40,
    },
}

_NUMERIC_SEMANTIC_TYPES = {
    "Numeric Continuous",
    "Numeric Discrete",
    "Currency",
    "Percentage",
}
_DATE_NAME_RE = re.compile(
    r"(?:^|[_\-\s])(date|datetime|timestamp|time|تاريخ|وقت)(?:$|[_\-\s])",
    re.IGNORECASE,
)
_BOOLEAN_TRUE = {
    "true", "yes", "y", "1", "on", "نعم", "صح", "صحيح", "موافق",
}
_BOOLEAN_FALSE = {
    "false", "no", "n", "0", "off", "لا", "خطأ", "خاطئ", "غير موافق",
}
_BOOLEAN_VALUES = _BOOLEAN_TRUE | _BOOLEAN_FALSE


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, float(value)))


def _safe_string_series(series: pd.Series) -> pd.Series:
    """Use Python-backed strings to avoid pandas/PyArrow regex incompatibilities."""
    try:
        return series.astype("string[python]")
    except Exception:
        return series.map(lambda value: "" if pd.isna(value) else str(value)).astype(
            "string[python]"
        )


def _stable_value(value: Any) -> str:
    if isinstance(value, dict):
        try:
            return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
        except Exception:
            return repr(value)
    if isinstance(value, (list, tuple, set, np.ndarray)):
        try:
            normalised = (
                sorted(value, key=str) if isinstance(value, set) else list(value)
            )
            return json.dumps(normalised, ensure_ascii=False, default=str)
        except Exception:
            return repr(value)
    try:
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)) and bool(missing):
            return "<NA>"
    except Exception:
        pass
    return str(value)


def _duplicate_mask(df: pd.DataFrame) -> pd.Series:
    try:
        return df.duplicated()
    except (TypeError, ValueError):
        comparable = df.copy(deep=False)
        for column in comparable.select_dtypes(include="object").columns:
            comparable[column] = comparable[column].map(_stable_value)
        return comparable.duplicated()


def _normalise_weights(raw: Mapping[str, Any] | None) -> Dict[str, float]:
    source = dict(raw or DEFAULT_DIMENSION_WEIGHTS)
    weights: Dict[str, float] = {}
    for name in DEFAULT_DIMENSION_WEIGHTS:
        try:
            value = float(source.get(name, DEFAULT_DIMENSION_WEIGHTS[name]))
        except (TypeError, ValueError):
            value = DEFAULT_DIMENSION_WEIGHTS[name]
        weights[name] = max(value, 0.0)
    total = sum(weights.values())
    if total <= 0:
        return dict(DEFAULT_DIMENSION_WEIGHTS)
    return {name: value / total for name, value in weights.items()}


def infer_critical_columns(
    df: pd.DataFrame,
    semantic_profiles: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[str]:
    """Infer a conservative set of columns whose defects deserve more weight."""
    profiles = semantic_profiles or {}
    inferred: list[str] = []
    for raw_column in df.columns:
        column = str(raw_column)
        profile = profiles.get(column, {}) or {}
        semantic_type = str(
            profile.get("effective_semantic_type")
            or profile.get("semantic_type")
            or ""
        )
        business_role = str(profile.get("business_role", "") or "")
        target_score = float(profile.get("target_candidate_score", 0.0) or 0.0)
        name = column.strip().lower().replace("-", "_").replace(" ", "_")
        name_tokens = set(filter(None, name.split("_")))

        is_key = semantic_type == "Identifier" or bool(
            name_tokens & {"id", "key", "uuid", "guid", "code", "معرف", "رقم", "كود"}
        )
        is_target = target_score >= 0.65
        is_time_key = semantic_type == "Datetime" and business_role == "Date"
        if is_key or is_target or is_time_key:
            inferred.append(column)
    return list(dict.fromkeys(inferred))


def normalise_quality_policy(
    df: pd.DataFrame,
    policy: Mapping[str, Any] | None = None,
    semantic_profiles: Mapping[str, Mapping[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Return a validated policy containing only columns in the current schema."""
    source = dict(policy or {})
    columns = [str(column) for column in df.columns]
    column_set = set(columns)

    explicit = [
        str(column)
        for column in source.get("critical_columns", []) or []
        if str(column) in column_set
    ]
    auto_enabled = bool(source.get("auto_critical_columns", True))
    inferred = infer_critical_columns(df, semantic_profiles) if auto_enabled else []
    resolved = list(dict.fromkeys(explicit + inferred))

    future_allowed = [
        str(column)
        for column in source.get("future_dates_allowed", []) or []
        if str(column) in column_set
    ]
    try:
        multiplier = float(source.get("critical_multiplier", 2.0))
    except (TypeError, ValueError):
        multiplier = 2.0

    return {
        "critical_columns": explicit,
        "auto_critical_columns": auto_enabled,
        "inferred_critical_columns": inferred,
        "resolved_critical_columns": resolved,
        "critical_multiplier": round(_clamp(multiplier, 1.0, 5.0), 2),
        "future_dates_allowed": future_allowed,
        "dimension_weights": _normalise_weights(source.get("dimension_weights")),
        "policy_version": QUALITY_ENGINE_VERSION,
    }


def quality_policy_signature(policy: Mapping[str, Any] | None) -> str:
    """Fingerprint score-affecting policy fields for honest report comparison."""
    source = dict(policy or {})
    payload = {
        "critical_columns": sorted(map(str, source.get("resolved_critical_columns", []) or [])),
        "critical_multiplier": round(float(source.get("critical_multiplier", 2.0) or 2.0), 6),
        "future_dates_allowed": sorted(map(str, source.get("future_dates_allowed", []) or [])),
        "dimension_weights": {
            key: round(float(value), 8)
            for key, value in sorted((source.get("dimension_weights", {}) or {}).items())
        },
        "policy_version": str(source.get("policy_version", QUALITY_ENGINE_VERSION)),
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _column_weight(column: str, policy: Mapping[str, Any]) -> float:
    critical = set(policy.get("resolved_critical_columns", []) or [])
    return (
        float(policy.get("critical_multiplier", 2.0) or 2.0)
        if column in critical
        else 1.0
    )


def _numeric_parse(series: pd.Series) -> pd.Series:
    text = _safe_string_series(series).str.strip()
    text = text.str.replace("\u00a0", "", regex=False)
    text = text.str.replace(r"\s+", "", regex=True)
    text = text.str.replace(",", "", regex=False)
    text = text.str.replace("%", "", regex=False)
    text = text.str.replace(r"^[\$€£¥₹]|[\$€£¥₹]$", "", regex=True)
    return pd.to_numeric(text, errors="coerce")


def _date_parse(series: pd.Series) -> pd.Series:
    with pd.option_context("mode.chained_assignment", None):
        try:
            return pd.to_datetime(series, errors="coerce", utc=True, format="mixed")
        except (TypeError, ValueError):
            return pd.to_datetime(series, errors="coerce", utc=True)


def _severity(rate: float, *, critical: bool, count: int) -> str:
    if count <= 0:
        return "none"
    if rate >= 0.50 or (critical and rate >= 0.05):
        return "critical"
    if rate >= 0.20 or (critical and rate >= 0.01):
        return "high"
    if rate >= 0.05:
        return "medium"
    return "low"


def _issue(
    *,
    issue_type: str,
    dimension: str,
    column: str | None,
    count: int,
    denominator: int,
    critical: bool,
    weight: float,
    description: str,
    recommendation: str,
    sample_indices: Sequence[Any] | None = None,
) -> Dict[str, Any]:
    rate = float(count / max(denominator, 1))
    return {
        "issue_id": f"{issue_type}::{column or '__dataset__'}",
        "issue_type": issue_type,
        "dimension": dimension,
        "column": column,
        "count": int(count),
        "denominator": int(denominator),
        "rate": round(rate, 6),
        "percentage": round(rate * 100, 2),
        "critical_column": bool(critical),
        "column_weight": round(float(weight), 3),
        "weighted_count": round(float(count) * float(weight), 3),
        "severity": _severity(rate, critical=critical, count=count),
        "description": description,
        "recommended_action": recommendation,
        "sample_indices": [str(value) for value in (sample_indices or [])[:20]],
    }


def _profile_semantic_type(
    column: str,
    semantic_profiles: Mapping[str, Mapping[str, Any]],
) -> str:
    profile = semantic_profiles.get(column, {}) or {}
    return str(
        profile.get("effective_semantic_type")
        or profile.get("semantic_type")
        or ""
    )


def _is_date_candidate(
    column: str,
    series: pd.Series,
    semantic_profiles: Mapping[str, Mapping[str, Any]],
) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series.dtype):
        return True
    if _profile_semantic_type(column, semantic_profiles) == "Datetime":
        return True
    normalised_name = re.sub(r"[_\-]+", " ", column.strip())
    return bool(_DATE_NAME_RE.search(f" {normalised_name} "))


def _is_numeric_candidate(
    column: str,
    series: pd.Series,
    semantic_profiles: Mapping[str, Mapping[str, Any]],
) -> bool:
    if pd.api.types.is_numeric_dtype(series.dtype) and not pd.api.types.is_bool_dtype(series.dtype):
        return True
    return _profile_semantic_type(column, semantic_profiles) in _NUMERIC_SEMANTIC_TYPES


def _is_boolean_candidate(
    column: str,
    series: pd.Series,
    semantic_profiles: Mapping[str, Mapping[str, Any]],
) -> bool:
    if pd.api.types.is_bool_dtype(series.dtype):
        return True
    return _profile_semantic_type(column, semantic_profiles) == "Boolean"


def _score_status(score: float) -> str:
    if score >= 90:
        return "Excellent"
    if score >= 75:
        return "Good"
    if score >= 60:
        return "Needs Review"
    return "Critical"


def _dimension_payload(
    *,
    score: float,
    weight: float,
    defect_count: int,
    weighted_defect_count: float,
    denominator: int,
    weighted_denominator: float,
) -> Dict[str, Any]:
    defect_rate = float(weighted_defect_count / max(weighted_denominator, 1.0))
    return {
        "score": round(_clamp(score, 0.0, 100.0), 1),
        "weight": round(float(weight), 4),
        "defect_count": int(defect_count),
        "weighted_defect_count": round(float(weighted_defect_count), 3),
        "denominator": int(denominator),
        "weighted_denominator": round(float(weighted_denominator), 3),
        "defect_rate": round(defect_rate, 6),
        "defect_percentage": round(defect_rate * 100, 2),
    }


def _run_quality_engine_impl(
    df: pd.DataFrame,
    policy: Mapping[str, Any] | None = None,
    semantic_profiles: Mapping[str, Mapping[str, Any]] | None = None,
    dataset_revision: int = 0,
    dataset_fingerprint: str = "",
) -> Dict[str, Any]:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("run_quality_engine expects a pandas DataFrame.")

    profiles: Mapping[str, Mapping[str, Any]] = semantic_profiles or {}
    resolved_policy = normalise_quality_policy(df, policy, profiles)
    critical_columns = set(resolved_policy["resolved_critical_columns"])
    future_allowed = set(resolved_policy["future_dates_allowed"])
    weights = resolved_policy["dimension_weights"]

    row_count = int(len(df))
    column_count = int(len(df.columns))
    total_cells = int(row_count * column_count)
    column_weights = {
        str(column): _column_weight(str(column), resolved_policy)
        for column in df.columns
    }
    weighted_cells = float(row_count * sum(column_weights.values()))

    issues: list[Dict[str, Any]] = []
    affected_rows = np.zeros(row_count, dtype=bool)
    validity_union_by_column: Dict[str, np.ndarray] = {}
    raw_validity_events = 0

    # ── Completeness ────────────────────────────────────────────────
    null_series = df.isna().sum()
    null_by_col = {
        str(column): int(count)
        for column, count in null_series.items()
        if int(count) > 0
    }
    total_nulls = int(null_series.sum())
    weighted_missing = 0.0
    for raw_column in df.columns:
        column = str(raw_column)
        mask = df[raw_column].isna().to_numpy(dtype=bool)
        count = int(mask.sum())
        if not count:
            continue
        affected_rows |= mask
        weight = column_weights[column]
        weighted_missing += count * weight
        issues.append(
            _issue(
                issue_type="missing",
                dimension="completeness",
                column=column,
                count=count,
                denominator=row_count,
                critical=column in critical_columns,
                weight=weight,
                description="Missing values reduce completeness and may bias analysis or model training.",
                recommendation="Choose an explicit train-safe imputation, recovery, or exclusion policy.",
                sample_indices=df.index[mask].tolist(),
            )
        )

    completeness_score = (
        0.0
        if row_count == 0 or column_count == 0
        else 100.0 * (1.0 - weighted_missing / max(float(total_cells), 1.0))
    )

    # ── Uniqueness ─────────────────────────────────────────────────
    duplicate_mask = _duplicate_mask(df) if row_count else pd.Series([], dtype=bool)
    duplicate_count = int(duplicate_mask.sum()) if row_count else 0
    if duplicate_count:
        duplicate_array = duplicate_mask.to_numpy(dtype=bool)
        affected_rows |= duplicate_array
        issues.append(
            _issue(
                issue_type="duplicate_row",
                dimension="uniqueness",
                column=None,
                count=duplicate_count,
                denominator=row_count,
                critical=False,
                weight=1.0,
                description="Rows duplicate an earlier record across all columns.",
                recommendation="Review business keys before removing duplicates; repeated events may be valid.",
                sample_indices=df.index[duplicate_array].tolist(),
            )
        )
    uniqueness_score = (
        0.0 if row_count == 0 else 100.0 * (1.0 - duplicate_count / row_count)
    )

    # ── Validity: each cell is counted once even if multiple checks flag it ──
    type_errors: Dict[str, int] = {}
    type_error_details: Dict[str, Dict[str, Any]] = {}
    date_errors: Dict[str, Dict[str, int]] = {}
    non_finite_by_col: Dict[str, int] = {}
    boolean_errors: Dict[str, int] = {}
    now_utc = pd.Timestamp.now(tz="UTC")

    for raw_column in df.columns:
        column = str(raw_column)
        series = df[raw_column]
        non_null_mask = series.notna().to_numpy(dtype=bool)
        if not bool(non_null_mask.any()):
            validity_union_by_column[column] = np.zeros(row_count, dtype=bool)
            continue

        union_mask = np.zeros(row_count, dtype=bool)
        weight = column_weights[column]
        critical = column in critical_columns
        date_candidate = _is_date_candidate(column, series, profiles)
        numeric_candidate = _is_numeric_candidate(column, series, profiles)
        boolean_candidate = _is_boolean_candidate(column, series, profiles)

        if date_candidate:
            parsed = _date_parse(series)
            parsed_valid = parsed.notna().to_numpy(dtype=bool)
            invalid_mask = non_null_mask & ~parsed_valid
            invalid_count = int(invalid_mask.sum())
            if invalid_count:
                raw_validity_events += invalid_count
                union_mask |= invalid_mask
                date_errors.setdefault(column, {})["invalid_dates"] = invalid_count
                issues.append(
                    _issue(
                        issue_type="invalid_date",
                        dimension="validity",
                        column=column,
                        count=invalid_count,
                        denominator=int(non_null_mask.sum()),
                        critical=critical,
                        weight=weight,
                        description="Non-null values cannot be parsed as dates consistently.",
                        recommendation="Review formats and convert only after previewing values that would become missing.",
                        sample_indices=df.index[invalid_mask].tolist(),
                    )
                )

            if column not in future_allowed:
                future_mask = parsed_valid & (parsed > now_utc).to_numpy(dtype=bool)
                future_count = int(future_mask.sum())
                if future_count:
                    raw_validity_events += future_count
                    union_mask |= future_mask
                    date_errors.setdefault(column, {})["future_dates"] = future_count
                    issues.append(
                        _issue(
                            issue_type="future_date",
                            dimension="validity",
                            column=column,
                            count=future_count,
                            denominator=int(parsed_valid.sum()),
                            critical=critical,
                            weight=weight,
                            description="Dates occur after the current time under the active quality policy.",
                            recommendation="Validate the business meaning or allow future dates for this column in Quality Policy.",
                            sample_indices=df.index[future_mask].tolist(),
                        )
                    )

        elif boolean_candidate and not pd.api.types.is_bool_dtype(series.dtype):
            normalised = _safe_string_series(series).str.strip().str.lower()
            valid_bool = normalised.isin(_BOOLEAN_VALUES).to_numpy(dtype=bool)
            invalid_mask = non_null_mask & ~valid_bool
            invalid_count = int(invalid_mask.sum())
            if invalid_count:
                raw_validity_events += invalid_count
                union_mask |= invalid_mask
                boolean_errors[column] = invalid_count
                issues.append(
                    _issue(
                        issue_type="invalid_boolean",
                        dimension="validity",
                        column=column,
                        count=invalid_count,
                        denominator=int(non_null_mask.sum()),
                        critical=critical,
                        weight=weight,
                        description="Values do not match the supported boolean vocabulary.",
                        recommendation="Map reviewed true/false tokens explicitly instead of coercing unknown values.",
                        sample_indices=df.index[invalid_mask].tolist(),
                    )
                )

        elif numeric_candidate or (
            pd.api.types.is_object_dtype(series.dtype)
            or pd.api.types.is_string_dtype(series.dtype)
        ):
            if pd.api.types.is_numeric_dtype(series.dtype) and not pd.api.types.is_bool_dtype(series.dtype):
                numeric = pd.to_numeric(series, errors="coerce")
                parse_ratio = 1.0
            else:
                numeric = _numeric_parse(series)
                parse_ratio = float(numeric[series.notna()].notna().mean())

            should_validate_numeric = numeric_candidate or (
                int(numeric.notna().sum()) >= 2 and 0.50 < parse_ratio < 1.0
            )
            if should_validate_numeric:
                numeric_valid = numeric.notna().to_numpy(dtype=bool)
                invalid_mask = non_null_mask & ~numeric_valid
                invalid_count = int(invalid_mask.sum())
                if invalid_count:
                    raw_validity_events += invalid_count
                    union_mask |= invalid_mask
                    type_errors[column] = invalid_count
                    type_error_details[column] = {
                        "invalid_numeric_values": invalid_count,
                        "numeric_parse_ratio": round(parse_ratio, 4),
                    }
                    issues.append(
                        _issue(
                            issue_type="mixed_numeric_text",
                            dimension="validity",
                            column=column,
                            count=invalid_count,
                            denominator=int(non_null_mask.sum()),
                            critical=critical,
                            weight=weight,
                            description="A mostly numeric column contains values that cannot be parsed as numbers.",
                            recommendation="Review invalid tokens before numeric conversion; coercion may create new missing values.",
                            sample_indices=df.index[invalid_mask].tolist(),
                        )
                    )

                numeric_array = pd.to_numeric(numeric, errors="coerce").to_numpy(dtype=float, na_value=np.nan)
                non_finite_mask = non_null_mask & ~np.isnan(numeric_array) & ~np.isfinite(numeric_array)
                non_finite_count = int(non_finite_mask.sum())
                if non_finite_count:
                    raw_validity_events += non_finite_count
                    union_mask |= non_finite_mask
                    non_finite_by_col[column] = non_finite_count
                    issues.append(
                        _issue(
                            issue_type="non_finite_numeric",
                            dimension="validity",
                            column=column,
                            count=non_finite_count,
                            denominator=int(non_null_mask.sum()),
                            critical=critical,
                            weight=weight,
                            description="Numeric values include positive or negative infinity.",
                            recommendation="Replace infinities only after validating their source and intended meaning.",
                            sample_indices=df.index[non_finite_mask].tolist(),
                        )
                    )

        validity_union_by_column[column] = union_mask
        affected_rows |= union_mask

    unique_invalid_cells = int(sum(mask.sum() for mask in validity_union_by_column.values()))
    weighted_invalid = float(
        sum(
            int(validity_union_by_column[str(column)].sum())
            * column_weights[str(column)]
            for column in df.columns
        )
    )
    weighted_non_null = float(
        sum(
            int(df[raw_column].notna().sum()) * column_weights[str(raw_column)]
            for raw_column in df.columns
        )
    )
    base_non_null = float(max(total_cells - total_nulls, 0))
    validity_score = (
        0.0
        if row_count == 0 or column_count == 0
        else 100.0 * (1.0 - weighted_invalid / max(base_non_null, 1.0))
    )

    dimensions = {
        "completeness": _dimension_payload(
            score=completeness_score,
            weight=weights["completeness"],
            defect_count=total_nulls,
            weighted_defect_count=weighted_missing,
            denominator=total_cells,
            weighted_denominator=float(total_cells),
        ),
        "validity": _dimension_payload(
            score=validity_score,
            weight=weights["validity"],
            defect_count=unique_invalid_cells,
            weighted_defect_count=weighted_invalid,
            denominator=max(total_cells - total_nulls, 0),
            weighted_denominator=base_non_null,
        ),
        "uniqueness": _dimension_payload(
            score=uniqueness_score,
            weight=weights["uniqueness"],
            defect_count=duplicate_count,
            weighted_defect_count=float(duplicate_count),
            denominator=row_count,
            weighted_denominator=float(row_count),
        ),
    }

    if row_count == 0 or column_count == 0:
        quality_score = 0.0
    else:
        quality_score = sum(
            dimensions[name]["score"] * weights[name]
            for name in weights
        )
    quality_score = round(_clamp(quality_score, 0.0, 100.0), 1)

    severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for item in issues:
        severity = str(item.get("severity", ""))
        if severity in severity_counts:
            severity_counts[severity] += 1
    issues.sort(
        key=lambda item: (
            {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(
                str(item.get("severity")), 4
            ),
            -float(item.get("weighted_count", 0.0)),
            str(item.get("column") or ""),
        )
    )

    duplicate_indices = (
        df.index[duplicate_mask.to_numpy(dtype=bool)].tolist()[:500]
        if duplicate_count
        else []
    )
    duplicate_sample = (
        df.loc[duplicate_mask].head(20).copy()
        if duplicate_count
        else pd.DataFrame()
    )

    unique_defect_cells = int(total_nulls + unique_invalid_cells)
    total_errors_legacy = int(unique_defect_cells + duplicate_count)
    report: Dict[str, Any] = {
        "engine_version": QUALITY_ENGINE_VERSION,
        "analysis_version": QUALITY_ENGINE_VERSION,
        "quality_score": quality_score,
        "status": _score_status(quality_score),
        "dimensions": dimensions,
        "dimension_scores": {
            name: payload["score"] for name, payload in dimensions.items()
        },
        "dimension_weights": dict(weights),
        "policy": resolved_policy,
        "policy_signature": quality_policy_signature(resolved_policy),
        "critical_columns": resolved_policy["resolved_critical_columns"],
        "issues": issues,
        "severity_counts": severity_counts,
        "total_cells": total_cells,
        "weighted_cells": round(weighted_cells, 3),
        "total_errors": total_errors_legacy,
        "unique_defect_cells": unique_defect_cells,
        "unique_invalid_cells": unique_invalid_cells,
        "raw_validity_events": int(raw_validity_events),
        "overlap_avoided": int(max(raw_validity_events - unique_invalid_cells, 0)),
        "affected_rows": int(affected_rows.sum()),
        "affected_row_pct": round(float(affected_rows.mean() * 100), 2) if row_count else 0.0,
        "duplicate_count": duplicate_count,
        "duplicate_indices": duplicate_indices,
        "duplicate_indices_truncated": duplicate_count > 500,
        "duplicate_sample": duplicate_sample,
        "null_by_col": null_by_col,
        "total_nulls": total_nulls,
        "type_errors": type_errors,
        "type_error_details": type_error_details,
        "boolean_errors": boolean_errors,
        "non_finite_by_col": non_finite_by_col,
        "date_errors": date_errors,
        "row_count": row_count,
        "column_count": column_count,
        "dataset_revision": int(dataset_revision or 0),
        "dataset_fingerprint": str(dataset_fingerprint or ""),
        "scoring_note": (
            "Completeness, validity, and uniqueness are scored independently. "
            "Each invalid cell is counted once inside validity; critical defects receive the configured multiplier without diluting other columns."
        ),
    }
    return report


@st.cache_data(show_spinner=False)
def run_quality_engine(
    df: pd.DataFrame,
    policy: Mapping[str, Any] | None = None,
    semantic_profiles: Mapping[str, Mapping[str, Any]] | None = None,
    dataset_revision: int = 0,
    dataset_fingerprint: str = "",
) -> Dict[str, Any]:
    """Run the deterministic Quality Engine V2 scanner."""
    return _run_quality_engine_impl(
        df,
        policy=policy,
        semantic_profiles=semantic_profiles,
        dataset_revision=dataset_revision,
        dataset_fingerprint=dataset_fingerprint,
    )


def compare_quality_reports(
    reference: Mapping[str, Any] | None,
    current: Mapping[str, Any] | None,
) -> Dict[str, Any]:
    """Compare two reports without inventing missing historical values."""
    if not reference or not current:
        return {
            "available": False,
            "score_delta": 0.0,
            "dimension_deltas": {},
            "defect_delta": 0,
            "duplicate_delta": 0,
            "null_delta": 0,
            "validity_delta": 0,
        }

    reference_policy = str(reference.get("policy_signature", "") or "")
    current_policy = str(current.get("policy_signature", "") or "")
    if reference_policy and current_policy and reference_policy != current_policy:
        return {
            "available": False,
            "reason": "policy_mismatch",
            "score_delta": 0.0,
            "dimension_deltas": {},
            "defect_delta": 0,
            "duplicate_delta": 0,
            "null_delta": 0,
            "validity_delta": 0,
        }

    reference_dimensions = reference.get("dimension_scores", {}) or {}
    current_dimensions = current.get("dimension_scores", {}) or {}
    dimension_deltas = {
        name: round(
            float(current_dimensions.get(name, 0.0) or 0.0)
            - float(reference_dimensions.get(name, 0.0) or 0.0),
            1,
        )
        for name in DEFAULT_DIMENSION_WEIGHTS
    }
    return {
        "available": True,
        "reference_score": float(reference.get("quality_score", 0.0) or 0.0),
        "current_score": float(current.get("quality_score", 0.0) or 0.0),
        "score_delta": round(
            float(current.get("quality_score", 0.0) or 0.0)
            - float(reference.get("quality_score", 0.0) or 0.0),
            1,
        ),
        "dimension_deltas": dimension_deltas,
        "defect_delta": int(current.get("unique_defect_cells", 0) or 0)
        - int(reference.get("unique_defect_cells", 0) or 0),
        "duplicate_delta": int(current.get("duplicate_count", 0) or 0)
        - int(reference.get("duplicate_count", 0) or 0),
        "null_delta": int(current.get("total_nulls", 0) or 0)
        - int(reference.get("total_nulls", 0) or 0),
        "validity_delta": int(current.get("unique_invalid_cells", 0) or 0)
        - int(reference.get("unique_invalid_cells", 0) or 0),
        "reference_revision": int(reference.get("dataset_revision", 0) or 0),
        "current_revision": int(current.get("dataset_revision", 0) or 0),
    }
