# ════════════════════════════════════════════════════════
#  DataBridge AI — Leakage-Safe Feature Pipeline Engine
#  Stage 7: declarative feature engineering for ML
# ════════════════════════════════════════════════════════
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)


PIPELINE_SPEC_VERSION = 1
MAX_DEFAULT_TEXT_FEATURES = 1000
MAX_ALLOWED_TEXT_FEATURES = 10000
MAX_SAFE_PROJECTED_FEATURES = 20000

_NUMERIC_TYPES = {
    "Numeric Continuous",
    "Numeric Discrete",
    "Currency",
    "Percentage",
}
_CATEGORICAL_TYPES = {"Categorical"}
_ORDINAL_TYPES = {"Ordinal"}
_BOOLEAN_TYPES = {"Boolean"}
_DATETIME_TYPES = {"Datetime"}
_TEXT_TYPES = {"Free Text"}
_ALWAYS_EXCLUDED_TYPES = {"Identifier", "Email", "Phone", "Unknown"}

_TRUE_VALUES = {
    "true", "yes", "y", "1", "on", "نعم", "صح", "صحيح", "موافق",
}
_FALSE_VALUES = {
    "false", "no", "n", "0", "off", "لا", "خطأ", "خاطئ", "غير موافق",
}


class FeaturePipelineError(ValueError):
    """Raised when a feature pipeline is invalid or cannot be fitted safely."""


def _as_frame(X: Any, feature_names: Optional[Sequence[str]] = None) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X.copy(deep=False)
    if isinstance(X, pd.Series):
        name = str(feature_names[0]) if feature_names else str(X.name or "feature")
        return X.to_frame(name=name)
    arr = np.asarray(X, dtype=object)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise FeaturePipelineError("Transformer input must be one- or two-dimensional.")
    if feature_names and len(feature_names) == arr.shape[1]:
        columns = list(map(str, feature_names))
    else:
        columns = [f"feature_{index}" for index in range(arr.shape[1])]
    return pd.DataFrame(arr, columns=columns)


def _safe_mixed_datetime(series: pd.Series, *, dayfirst: bool = True) -> pd.Series:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return pd.to_datetime(
                series,
                errors="coerce",
                dayfirst=dayfirst,
                format="mixed",
                utc=True,
            )
        except (TypeError, ValueError):
            return pd.to_datetime(series, errors="coerce", dayfirst=dayfirst, utc=True)


def _is_missing_scalar(value: Any) -> bool:
    try:
        result = pd.isna(value)
        if isinstance(result, (bool, np.bool_)):
            return bool(result)
    except Exception:
        pass
    return False


def _normalise_string_series(series: pd.Series) -> pd.Series:
    values = series.astype("object")
    missing = values.map(_is_missing_scalar)
    result = values.map(
        lambda value: np.nan if _is_missing_scalar(value) else str(value).strip()
    )
    result.loc[missing] = np.nan
    return result.astype("object")


def _python_string_series(series: pd.Series) -> pd.Series:
    """Return a Python-backed string Series regardless of pandas' global backend.

    Pandas can use PyArrow-backed strings by default.  PyArrow delegates regular
    expressions to RE2, which intentionally rejects Python-style ``\\uXXXX``
    escapes.  Feature transformers must be deterministic across pandas storage
    backends, so parsing code explicitly uses the Python string implementation.
    """
    try:
        return series.astype(pd.StringDtype(storage="python"))
    except (TypeError, ValueError):
        # Compatibility fallback for older pandas versions. Object-backed string
        # methods use Python's regex engine and preserve missing values safely.
        values = series.astype("object")
        return values.map(
            lambda value: pd.NA if _is_missing_scalar(value) else str(value)
        )


def _parse_numeric_series(
    series: pd.Series,
    *,
    percentage: bool = False,
    percent_as_fraction: bool = True,
) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series.dtype):
        return pd.to_numeric(series, errors="coerce").astype(float)

    text = _python_string_series(series).str.strip()
    explicit_percent = text.str.contains("%", regex=False, na=False)
    negative_parentheses = text.str.match(r"^\(.*\)$", na=False)
    cleaned = (
        text.str.replace("\u00A0", "", regex=False)
        .str.replace(r"\s+", "", regex=True)
        .str.replace(",", "", regex=False)
        .str.replace(r"^[\$€£¥₹]|[\$€£¥₹]$", "", regex=True)
        .str.replace("%", "", regex=False)
        .str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    )
    numeric = pd.to_numeric(cleaned, errors="coerce").astype(float)
    if percentage and percent_as_fraction:
        numeric.loc[explicit_percent & numeric.notna()] = numeric.loc[
            explicit_percent & numeric.notna()
        ] / 100.0
    numeric.loc[negative_parentheses & numeric.gt(0)] *= -1.0
    return numeric


class NumericCoercer(BaseEstimator, TransformerMixin):
    """Safely coerce numeric/currency/percentage columns without touching source data."""

    def __init__(
        self,
        semantic_types: Optional[Mapping[str, str]] = None,
        percent_as_fraction: bool = True,
    ) -> None:
        self.semantic_types = semantic_types
        self.percent_as_fraction = percent_as_fraction

    def fit(self, X: Any, y: Any = None):
        frame = _as_frame(X)
        self.feature_names_in_ = np.asarray(list(map(str, frame.columns)), dtype=object)
        return self

    def transform(self, X: Any) -> np.ndarray:
        frame = _as_frame(X, getattr(self, "feature_names_in_", None))
        result = pd.DataFrame(index=frame.index)
        for column in frame.columns:
            semantic = dict(self.semantic_types or {}).get(str(column), "")
            result[str(column)] = _parse_numeric_series(
                frame[column],
                percentage=semantic == "Percentage",
                percent_as_fraction=self.percent_as_fraction,
            )
        return result.to_numpy(dtype=float)

    def get_feature_names_out(self, input_features: Optional[Sequence[str]] = None) -> np.ndarray:
        if input_features is not None:
            return np.asarray(list(map(str, input_features)), dtype=object)
        return np.asarray(getattr(self, "feature_names_in_", []), dtype=object)


class CategoricalSanitizer(BaseEstimator, TransformerMixin):
    """Convert mixed category values to consistent strings while preserving missingness."""

    def fit(self, X: Any, y: Any = None):
        frame = _as_frame(X)
        self.feature_names_in_ = np.asarray(list(map(str, frame.columns)), dtype=object)
        return self

    def transform(self, X: Any) -> np.ndarray:
        frame = _as_frame(X, getattr(self, "feature_names_in_", None))
        result = pd.DataFrame(index=frame.index)
        for column in frame.columns:
            result[str(column)] = _normalise_string_series(frame[column])
        return result.to_numpy(dtype=object)

    def get_feature_names_out(self, input_features: Optional[Sequence[str]] = None) -> np.ndarray:
        if input_features is not None:
            return np.asarray(list(map(str, input_features)), dtype=object)
        return np.asarray(getattr(self, "feature_names_in_", []), dtype=object)


class BooleanCoercer(BaseEstimator, TransformerMixin):
    """Convert common boolean representations to 0/1 and keep unknown values missing."""

    def fit(self, X: Any, y: Any = None):
        frame = _as_frame(X)
        self.feature_names_in_ = np.asarray(list(map(str, frame.columns)), dtype=object)
        return self

    def transform(self, X: Any) -> np.ndarray:
        frame = _as_frame(X, getattr(self, "feature_names_in_", None))
        output = np.full((len(frame), len(frame.columns)), np.nan, dtype=float)
        for idx, column in enumerate(frame.columns):
            series = frame[column]
            if pd.api.types.is_bool_dtype(series.dtype):
                output[:, idx] = series.astype("float64").to_numpy()
                continue
            normalised = _python_string_series(series).str.strip().str.lower()
            output[normalised.isin(_TRUE_VALUES).fillna(False).to_numpy(), idx] = 1.0
            output[normalised.isin(_FALSE_VALUES).fillna(False).to_numpy(), idx] = 0.0
        return output

    def get_feature_names_out(self, input_features: Optional[Sequence[str]] = None) -> np.ndarray:
        names = input_features if input_features is not None else getattr(self, "feature_names_in_", [])
        return np.asarray([f"{name}_boolean" for name in map(str, names)], dtype=object)


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Train-only frequency encoding with deterministic handling of unseen categories."""

    def __init__(self, missing_token: str = "__MISSING__") -> None:
        self.missing_token = missing_token

    def fit(self, X: Any, y: Any = None):
        frame = _as_frame(X)
        self.feature_names_in_ = np.asarray(list(map(str, frame.columns)), dtype=object)
        self.frequency_maps_: Dict[str, Dict[str, float]] = {}
        denominator = max(len(frame), 1)
        for column in frame.columns:
            series = _normalise_string_series(frame[column]).fillna(self.missing_token)
            counts = series.value_counts(dropna=False)
            self.frequency_maps_[str(column)] = {
                str(key): float(value / denominator) for key, value in counts.items()
            }
        return self

    def transform(self, X: Any) -> np.ndarray:
        if not hasattr(self, "frequency_maps_"):
            raise FeaturePipelineError("FrequencyEncoder must be fitted before transform.")
        frame = _as_frame(X, getattr(self, "feature_names_in_", None))
        output = np.zeros((len(frame), len(frame.columns)), dtype=float)
        for idx, column in enumerate(frame.columns):
            mapping = self.frequency_maps_.get(str(column), {})
            values = _normalise_string_series(frame[column]).fillna(self.missing_token)
            output[:, idx] = values.map(lambda value: mapping.get(str(value), 0.0)).to_numpy(dtype=float)
        return output

    def get_feature_names_out(self, input_features: Optional[Sequence[str]] = None) -> np.ndarray:
        names = input_features if input_features is not None else getattr(self, "feature_names_in_", [])
        return np.asarray([f"{name}_frequency" for name in map(str, names)], dtype=object)


class DateTimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract date parts using train-only reference dates for elapsed features."""

    _VALID_PARTS = {
        "year",
        "quarter",
        "month",
        "day",
        "dayofweek",
        "dayofyear",
        "weekofyear",
        "is_weekend",
        "elapsed_days",
    }

    def __init__(
        self,
        parts: Sequence[str] = (
            "year",
            "quarter",
            "month",
            "dayofweek",
            "is_weekend",
            "elapsed_days",
        ),
        cyclical: bool = True,
        dayfirst: bool = True,
    ) -> None:
        self.parts = parts
        self.cyclical = cyclical
        self.dayfirst = dayfirst

    def fit(self, X: Any, y: Any = None):
        invalid = set(self.parts) - self._VALID_PARTS
        if invalid:
            raise FeaturePipelineError(
                "Unsupported datetime part(s): " + ", ".join(sorted(invalid))
            )
        frame = _as_frame(X)
        self.feature_names_in_ = np.asarray(list(map(str, frame.columns)), dtype=object)
        self.reference_dates_: Dict[str, Optional[pd.Timestamp]] = {}
        for column in frame.columns:
            parsed = _safe_mixed_datetime(frame[column], dayfirst=self.dayfirst)
            valid = parsed.dropna()
            self.reference_dates_[str(column)] = valid.min() if not valid.empty else None
        return self

    def _column_features(self, column: str, series: pd.Series) -> Dict[str, np.ndarray]:
        parsed = _safe_mixed_datetime(series, dayfirst=self.dayfirst)
        output: Dict[str, np.ndarray] = {}
        iso = parsed.dt.isocalendar()
        for part in self.parts:
            if part == "year":
                values = parsed.dt.year
            elif part == "quarter":
                values = parsed.dt.quarter
            elif part == "month":
                values = parsed.dt.month
            elif part == "day":
                values = parsed.dt.day
            elif part == "dayofweek":
                values = parsed.dt.dayofweek
            elif part == "dayofyear":
                values = parsed.dt.dayofyear
            elif part == "weekofyear":
                values = iso.week
            elif part == "is_weekend":
                values = parsed.dt.dayofweek.ge(5).where(parsed.notna())
            elif part == "elapsed_days":
                reference = self.reference_dates_.get(column)
                values = (
                    (parsed - reference).dt.total_seconds() / 86400.0
                    if reference is not None
                    else pd.Series(np.nan, index=series.index)
                )
            else:  # guarded in fit
                continue
            output[f"{column}_{part}"] = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)

        if self.cyclical:
            cycles = {
                "month": (parsed.dt.month, 12.0),
                "dayofweek": (parsed.dt.dayofweek, 7.0),
                "dayofyear": (parsed.dt.dayofyear, 365.25),
            }
            for name, (values, period) in cycles.items():
                numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
                radians = 2.0 * math.pi * numeric / period
                output[f"{column}_{name}_sin"] = np.sin(radians)
                output[f"{column}_{name}_cos"] = np.cos(radians)
        return output

    def transform(self, X: Any) -> np.ndarray:
        if not hasattr(self, "reference_dates_"):
            raise FeaturePipelineError("DateTimeFeatureExtractor must be fitted before transform.")
        frame = _as_frame(X, getattr(self, "feature_names_in_", None))
        blocks: list[np.ndarray] = []
        for column in frame.columns:
            features = self._column_features(str(column), frame[column])
            blocks.extend(features.values())
        if not blocks:
            return np.empty((len(frame), 0), dtype=float)
        return np.column_stack(blocks).astype(float)

    def get_feature_names_out(self, input_features: Optional[Sequence[str]] = None) -> np.ndarray:
        names = list(map(str, input_features or getattr(self, "feature_names_in_", [])))
        result: list[str] = []
        for column in names:
            result.extend([f"{column}_{part}" for part in self.parts])
            if self.cyclical:
                result.extend(
                    [
                        f"{column}_month_sin",
                        f"{column}_month_cos",
                        f"{column}_dayofweek_sin",
                        f"{column}_dayofweek_cos",
                        f"{column}_dayofyear_sin",
                        f"{column}_dayofyear_cos",
                    ]
                )
        return np.asarray(result, dtype=object)


class TextSanitizer(BaseEstimator, TransformerMixin):
    """Return one clean text vector accepted by TfidfVectorizer."""

    def __init__(self, lowercase: bool = True) -> None:
        self.lowercase = lowercase

    def fit(self, X: Any, y: Any = None):
        frame = _as_frame(X)
        if frame.shape[1] != 1:
            raise FeaturePipelineError("Each text transformer must receive exactly one column.")
        self.feature_name_in_ = str(frame.columns[0])
        return self

    def transform(self, X: Any) -> np.ndarray:
        frame = _as_frame(X, [getattr(self, "feature_name_in_", "text")])
        series = frame.iloc[:, 0].fillna("").astype(str)
        series = series.str.replace(r"\s+", " ", regex=True).str.strip()
        if self.lowercase:
            series = series.str.lower()
        return series.to_numpy(dtype=object)

    def get_feature_names_out(self, input_features: Optional[Sequence[str]] = None) -> np.ndarray:
        if input_features is not None:
            if isinstance(input_features, str):
                return np.asarray([input_features], dtype=object)
            return np.asarray(list(map(str, input_features)), dtype=object)
        return np.asarray([getattr(self, "feature_name_in_", "text")], dtype=object)


@dataclass
class FittedFeaturePipeline:
    preprocessor: ColumnTransformer
    spec: Dict[str, Any]
    feature_names: list[str]
    fitted_rows: int
    transformed_features: int
    spec_fingerprint: str

    def transform(self, data: pd.DataFrame):
        report = validate_feature_pipeline_spec(data, self.spec, require_target=False)
        if not report["valid"]:
            raise FeaturePipelineError("; ".join(report["blockers"]))
        features = data[list(self.spec["feature_columns"])]
        return self.preprocessor.transform(features)


def _semantic_type(profile: Mapping[str, Any]) -> str:
    return str(
        profile.get("effective_semantic_type")
        or profile.get("semantic_type")
        or "Unknown"
    )


def _infer_task(series: pd.Series) -> str:
    clean = series.dropna()
    if clean.empty:
        return "classification"
    unique = int(clean.nunique(dropna=True))
    if not pd.api.types.is_numeric_dtype(clean.dtype):
        return "classification"
    threshold = max(20, int(len(clean) * 0.05))
    return "classification" if unique <= threshold else "regression"


def _default_feature_groups(
    df: pd.DataFrame,
    semantic_profiles: Mapping[str, Mapping[str, Any]],
    target: str,
) -> tuple[Dict[str, list[str]], list[str], list[Dict[str, str]]]:
    groups: Dict[str, list[str]] = {
        "numeric": [],
        "categorical": [],
        "frequency": [],
        "ordinal": [],
        "boolean": [],
        "datetime": [],
        "text": [],
    }
    excluded: list[str] = []
    reasons: list[Dict[str, str]] = []

    for column in map(str, df.columns):
        if column == target:
            continue
        profile = dict(semantic_profiles.get(column, {}))
        semantic = _semantic_type(profile)
        is_constant = bool(profile.get("is_constant", False))
        all_missing = bool(profile.get("all_missing", False) or profile.get("non_null_count", 1) == 0)
        leakage_reasons = [str(item) for item in profile.get("leakage_reasons", []) or []]
        hard_leakage = any(
            "outcome" in item.lower()
            or "post-event" in item.lower()
            or "prediction" in item.lower()
            for item in leakage_reasons
        )
        high_cardinality = bool(profile.get("high_cardinality", False))

        reason = ""
        if is_constant:
            reason = "constant feature"
        elif all_missing:
            reason = "all values are missing"
        elif semantic in _ALWAYS_EXCLUDED_TYPES:
            reason = f"semantic type {semantic} is excluded by default"
        elif hard_leakage:
            reason = "potential post-outcome target leakage"

        if reason:
            excluded.append(column)
            reasons.append({"column": column, "reason": reason})
            continue

        if semantic in _NUMERIC_TYPES:
            groups["numeric"].append(column)
        elif semantic in _CATEGORICAL_TYPES:
            if high_cardinality:
                groups["frequency"].append(column)
            else:
                groups["categorical"].append(column)
        elif semantic in _ORDINAL_TYPES:
            groups["ordinal"].append(column)
        elif semantic in _BOOLEAN_TYPES:
            groups["boolean"].append(column)
        elif semantic in _DATETIME_TYPES:
            groups["datetime"].append(column)
        elif semantic in _TEXT_TYPES:
            excluded.append(column)
            reasons.append({"column": column, "reason": "free text requires explicit TF-IDF enablement"})
        elif pd.api.types.is_numeric_dtype(df[column]):
            groups["numeric"].append(column)
        else:
            excluded.append(column)
            reasons.append({"column": column, "reason": "unresolved semantic type"})

    return groups, excluded, reasons


def create_default_feature_pipeline_spec(
    df: pd.DataFrame,
    semantic_profiles: Mapping[str, Mapping[str, Any]],
    target: str,
    *,
    task: str = "auto",
    configured_revision: int = 0,
    configured_fingerprint: str = "",
) -> Dict[str, Any]:
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise FeaturePipelineError("A non-empty DataFrame is required.")
    target = str(target)
    if target not in df.columns:
        raise FeaturePipelineError(f"Target column '{target}' does not exist.")
    resolved_task = _infer_task(df[target]) if task == "auto" else str(task)
    if resolved_task not in {"classification", "regression"}:
        raise FeaturePipelineError("Task must be classification, regression, or auto.")

    groups, excluded, exclusion_reasons = _default_feature_groups(
        df, semantic_profiles, target
    )
    feature_columns = [
        column
        for group_name in (
            "numeric",
            "categorical",
            "frequency",
            "ordinal",
            "boolean",
            "datetime",
            "text",
        )
        for column in groups[group_name]
    ]

    semantic_types = {
        column: _semantic_type(semantic_profiles.get(column, {}))
        for column in feature_columns
    }

    spec: Dict[str, Any] = {
        "spec_version": PIPELINE_SPEC_VERSION,
        "name": "Safe Feature Pipeline",
        "target": target,
        "task": resolved_task,
        "feature_columns": feature_columns,
        "excluded_columns": sorted(set(excluded)),
        "exclusion_reasons": exclusion_reasons,
        "groups": groups,
        "semantic_types": semantic_types,
        "numeric": {
            "imputer": "median",
            "scaler": "standard",
            "add_missing_indicator": True,
            "percent_as_fraction": True,
        },
        "categorical": {
            "imputer": "most_frequent",
            "encoding": "one_hot",
            "min_frequency": 2,
            "max_categories": 100,
            "add_missing_indicator": False,
        },
        "frequency": {
            "enabled": True,
        },
        "ordinal": {
            "imputer": "most_frequent",
            "orders": {},
        },
        "boolean": {
            "imputer": "most_frequent",
        },
        "datetime": {
            "parts": [
                "year",
                "quarter",
                "month",
                "dayofweek",
                "is_weekend",
                "elapsed_days",
            ],
            "cyclical": True,
            "imputer": "median",
            "scaler": "standard",
            "dayfirst": True,
        },
        "text": {
            "enabled": False,
            "columns": list(groups["text"]),
            "max_features": MAX_DEFAULT_TEXT_FEATURES,
            "ngram_min": 1,
            "ngram_max": 2,
            "min_df": 1,
            "max_df": 1.0,
            "lowercase": True,
        },
        "output": {
            "sparse": True,
        },
        "configured_revision": int(configured_revision),
        "configured_fingerprint": str(configured_fingerprint or ""),
    }
    return normalise_feature_pipeline_spec(spec)


def normalise_feature_pipeline_spec(spec: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a stable, JSON-serialisable and de-duplicated pipeline specification."""
    result = copy.deepcopy(dict(spec or {}))
    result.setdefault("spec_version", PIPELINE_SPEC_VERSION)
    result.setdefault("name", "Safe Feature Pipeline")
    result.setdefault("target", "")
    result.setdefault("task", "classification")
    result.setdefault("groups", {})
    result.setdefault("semantic_types", {})
    result.setdefault("excluded_columns", [])
    result.setdefault("exclusion_reasons", [])
    result.setdefault("numeric", {})
    result.setdefault("categorical", {})
    result.setdefault("frequency", {})
    result.setdefault("ordinal", {})
    result.setdefault("boolean", {})
    result.setdefault("datetime", {})
    result.setdefault("text", {})
    result.setdefault("output", {})
    result.setdefault("configured_revision", 0)
    result.setdefault("configured_fingerprint", "")

    group_names = (
        "numeric",
        "categorical",
        "frequency",
        "ordinal",
        "boolean",
        "datetime",
        "text",
    )
    groups: Dict[str, list[str]] = {}
    seen: set[str] = set()
    for group_name in group_names:
        values = result["groups"].get(group_name, []) or []
        clean: list[str] = []
        for value in values:
            column = str(value)
            if column and column not in seen:
                clean.append(column)
                seen.add(column)
        groups[group_name] = clean
    result["groups"] = groups
    result["feature_columns"] = [
        column for group_name in group_names for column in groups[group_name]
    ]
    result["excluded_columns"] = sorted(
        {str(column) for column in result.get("excluded_columns", []) if str(column)}
    )
    result["semantic_types"] = {
        str(column): str(value)
        for column, value in dict(result.get("semantic_types", {})).items()
    }

    result["numeric"] = {
        "imputer": str(result["numeric"].get("imputer", "median")),
        "scaler": str(result["numeric"].get("scaler", "standard")),
        "add_missing_indicator": bool(result["numeric"].get("add_missing_indicator", True)),
        "percent_as_fraction": bool(result["numeric"].get("percent_as_fraction", True)),
    }
    min_frequency = result["categorical"].get("min_frequency", 2)
    if isinstance(min_frequency, float) and 0 < min_frequency < 1:
        clean_min_frequency: int | float = float(min_frequency)
    else:
        clean_min_frequency = max(1, int(min_frequency or 1))
    max_categories = result["categorical"].get("max_categories", 100)
    result["categorical"] = {
        "imputer": str(result["categorical"].get("imputer", "most_frequent")),
        "encoding": "one_hot",
        "min_frequency": clean_min_frequency,
        "max_categories": max(2, int(max_categories or 100)),
        "add_missing_indicator": bool(result["categorical"].get("add_missing_indicator", False)),
    }
    result["frequency"] = {
        "enabled": bool(result["frequency"].get("enabled", True)),
    }
    orders = {
        str(column): list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))
        for column, values in dict(result["ordinal"].get("orders", {})).items()
        if values
    }
    result["ordinal"] = {
        "imputer": str(result["ordinal"].get("imputer", "most_frequent")),
        "orders": orders,
    }
    result["boolean"] = {
        "imputer": str(result["boolean"].get("imputer", "most_frequent")),
    }
    valid_parts = [
        str(part)
        for part in result["datetime"].get("parts", [])
        if str(part) in DateTimeFeatureExtractor._VALID_PARTS
    ]
    if not valid_parts:
        valid_parts = ["year", "month", "dayofweek", "elapsed_days"]
    result["datetime"] = {
        "parts": list(dict.fromkeys(valid_parts)),
        "cyclical": bool(result["datetime"].get("cyclical", True)),
        "imputer": str(result["datetime"].get("imputer", "median")),
        "scaler": str(result["datetime"].get("scaler", "standard")),
        "dayfirst": bool(result["datetime"].get("dayfirst", True)),
    }
    text_max_features = min(
        MAX_ALLOWED_TEXT_FEATURES,
        max(10, int(result["text"].get("max_features", MAX_DEFAULT_TEXT_FEATURES))),
    )
    ngram_min = max(1, int(result["text"].get("ngram_min", 1)))
    ngram_max = max(ngram_min, min(3, int(result["text"].get("ngram_max", 2))))
    min_df_raw = result["text"].get("min_df", 1)
    if isinstance(min_df_raw, float) and 0 < min_df_raw <= 1:
        min_df: int | float = float(min_df_raw)
    else:
        min_df = max(1, int(min_df_raw or 1))
    max_df_raw = result["text"].get("max_df", 1.0)
    max_df = float(max_df_raw)
    max_df = min(1.0, max(0.01, max_df))
    result["text"] = {
        "enabled": bool(result["text"].get("enabled", False)),
        "columns": [str(column) for column in result["groups"]["text"]],
        "max_features": text_max_features,
        "ngram_min": ngram_min,
        "ngram_max": ngram_max,
        "min_df": min_df,
        "max_df": max_df,
        "lowercase": bool(result["text"].get("lowercase", True)),
    }
    result["output"] = {
        "sparse": bool(result["output"].get("sparse", True)),
    }
    result["configured_revision"] = int(result.get("configured_revision", 0) or 0)
    result["configured_fingerprint"] = str(result.get("configured_fingerprint", "") or "")
    return result


def feature_pipeline_spec_fingerprint(spec: Mapping[str, Any]) -> str:
    clean = normalise_feature_pipeline_spec(spec)
    payload = json.dumps(clean, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _estimate_output_features(df: pd.DataFrame, spec: Mapping[str, Any]) -> Dict[str, int]:
    groups = spec["groups"]
    numeric = len(groups["numeric"])
    boolean = len(groups["boolean"])
    frequency = len(groups["frequency"])
    ordinal = 0
    max_categories = int(spec["categorical"]["max_categories"])
    for column in groups["ordinal"]:
        if spec["ordinal"]["orders"].get(column):
            ordinal += 1
        elif column in df.columns:
            ordinal += min(max_categories, int(df[column].nunique(dropna=True)) + 1)
    date_per_column = len(spec["datetime"]["parts"]) + (6 if spec["datetime"]["cyclical"] else 0)
    datetime = len(groups["datetime"]) * date_per_column

    categorical = 0
    for column in groups["categorical"]:
        if column in df.columns:
            categorical += min(max_categories, int(df[column].nunique(dropna=True)) + 1)

    text = 0
    if spec["text"]["enabled"]:
        text = len(groups["text"]) * int(spec["text"]["max_features"])

    components = {
        "numeric": numeric,
        "categorical": categorical,
        "frequency": frequency,
        "ordinal": ordinal,
        "boolean": boolean,
        "datetime": datetime,
        "text": text,
    }
    components["total"] = int(sum(components.values()))
    return components


def validate_feature_pipeline_spec(
    df: pd.DataFrame,
    spec: Mapping[str, Any],
    *,
    require_target: bool = True,
    current_revision: Optional[int] = None,
    current_fingerprint: Optional[str] = None,
) -> Dict[str, Any]:
    blockers: list[str] = []
    warnings_list: list[str] = []
    clean = normalise_feature_pipeline_spec(spec)

    if not isinstance(df, pd.DataFrame) or len(df.columns) == 0:
        blockers.append("No usable dataset is loaded.")
        return {
            "valid": False,
            "status": "Invalid",
            "blockers": blockers,
            "warnings": warnings_list,
            "spec": clean,
            "estimated_features": {"total": 0},
            "stale": True,
        }

    target = str(clean.get("target", ""))
    if require_target and not target:
        blockers.append("A target column must be selected.")
    if require_target and target and target not in df.columns:
        blockers.append(f"Target column '{target}' is missing from the current dataset.")

    feature_columns = list(clean.get("feature_columns", []))
    missing_features = [column for column in feature_columns if column not in df.columns]
    if missing_features:
        blockers.append(
            "Configured feature columns are missing: " + ", ".join(missing_features[:10])
        )
    if target and target in feature_columns:
        blockers.append("The target column cannot also be a feature.")
    if not feature_columns:
        blockers.append("Select at least one feature column.")

    membership: Dict[str, list[str]] = {}
    for group_name, columns in clean["groups"].items():
        for column in columns:
            membership.setdefault(column, []).append(group_name)
    overlaps = {column: groups for column, groups in membership.items() if len(groups) > 1}
    if overlaps:
        blockers.append("A feature cannot belong to more than one transformation group.")

    if clean["task"] not in {"classification", "regression"}:
        blockers.append("Task must be classification or regression.")

    usable_rows = len(df)
    target_missing = 0
    if require_target and target and target in df.columns:
        target_series = df[target]
        target_missing = int(target_series.isna().sum())
        usable_rows = int(target_series.notna().sum())
        if target_missing:
            warnings_list.append(
                f"{target_missing:,} rows with missing target values will be excluded from training."
            )
        if clean["task"] == "classification":
            class_counts = target_series.dropna().value_counts(dropna=False)
            if len(class_counts) < 2:
                blockers.append("Classification requires at least two target classes.")
            elif int(class_counts.min()) < 2:
                blockers.append("Every target class needs at least two rows for a safe split.")
            elif int(class_counts.min()) < 5:
                warnings_list.append("At least one target class has fewer than five rows.")
        else:
            numeric_target = pd.to_numeric(target_series, errors="coerce")
            invalid = int(target_series.notna().sum() - numeric_target.notna().sum())
            if invalid:
                blockers.append(
                    f"Regression target contains {invalid:,} non-numeric values."
                )
            if int(numeric_target.notna().sum()) < 10:
                blockers.append("Regression requires at least ten valid target rows.")

    for column in clean["groups"]["numeric"]:
        if column in df.columns:
            semantic = clean["semantic_types"].get(column, "")
            parsed = _parse_numeric_series(
                df[column],
                percentage=semantic == "Percentage",
                percent_as_fraction=clean["numeric"]["percent_as_fraction"],
            )
            ratio = float(parsed.notna().sum() / max(df[column].notna().sum(), 1))
            if ratio < 0.80:
                blockers.append(
                    f"Numeric feature '{column}' has only {ratio:.0%} parseable non-null values."
                )
            elif ratio < 0.95:
                warnings_list.append(
                    f"Numeric feature '{column}' has {ratio:.0%} parseable non-null values."
                )

    for column in clean["groups"]["ordinal"]:
        order = clean["ordinal"]["orders"].get(column, [])
        if not order:
            warnings_list.append(
                f"Ordinal feature '{column}' has no explicit order and will use one-hot encoding safely."
            )
        elif column in df.columns:
            observed = set(
                _normalise_string_series(df[column]).dropna().astype(str).tolist()
            )
            missing_from_order = sorted(observed - set(map(str, order)))
            if missing_from_order:
                blockers.append(
                    f"Ordinal order for '{column}' does not include observed values: "
                    + ", ".join(missing_from_order[:10])
                )

    if clean["groups"]["text"] and not clean["text"]["enabled"]:
        warnings_list.append("Text features are selected but TF-IDF is disabled; they will be excluded.")
    if clean["text"]["enabled"] and clean["groups"]["text"]:
        if clean["text"]["max_features"] > 5000:
            warnings_list.append("High TF-IDF feature limit may increase memory and training time.")
        if require_target and usable_rows < 20:
            blockers.append("TF-IDF requires more training rows for stable vocabulary fitting.")

    estimated = _estimate_output_features(df, clean)
    if estimated["total"] > MAX_SAFE_PROJECTED_FEATURES:
        blockers.append(
            f"Projected feature count ({estimated['total']:,}) exceeds the safe limit of {MAX_SAFE_PROJECTED_FEATURES:,}."
        )
    elif estimated["total"] > 5000:
        warnings_list.append(
            f"Projected feature count is high ({estimated['total']:,}); use frequency encoding or lower TF-IDF limits."
        )

    stale = False
    configured_fp = str(clean.get("configured_fingerprint", ""))
    if current_fingerprint is not None and configured_fp:
        stale = configured_fp != str(current_fingerprint)
    elif current_revision is not None:
        stale = int(clean.get("configured_revision", 0)) != int(current_revision)
    if stale:
        warnings_list.append("The dataset changed after this pipeline was configured; review and save it again.")

    valid = not blockers
    status = "Invalid" if blockers else "Stale" if stale else "Configured"
    return {
        "valid": valid,
        "status": status,
        "blockers": blockers,
        "warnings": list(dict.fromkeys(warnings_list)),
        "spec": clean,
        "estimated_features": estimated,
        "input_feature_count": len(feature_columns),
        "usable_target_rows": usable_rows,
        "missing_target_rows": target_missing,
        "stale": stale,
        "spec_fingerprint": feature_pipeline_spec_fingerprint(clean),
    }


def reconcile_feature_pipeline_context(
    df: pd.DataFrame,
    spec: Mapping[str, Any] | None,
    *,
    current_revision: int,
    current_fingerprint: str,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    if not spec:
        return {}, {
            "valid": False,
            "status": "Pending",
            "blockers": ["No feature pipeline has been configured."],
            "warnings": [],
            "estimated_features": {"total": 0},
            "stale": False,
        }
    clean = normalise_feature_pipeline_spec(spec)
    report = validate_feature_pipeline_spec(
        df,
        clean,
        require_target=True,
        current_revision=current_revision,
        current_fingerprint=current_fingerprint,
    )
    return clean, report


def _imputer(strategy: str, *, add_indicator: bool = False) -> SimpleImputer:
    strategy = str(strategy)
    if strategy not in {"mean", "median", "most_frequent", "constant"}:
        raise FeaturePipelineError(f"Unsupported imputation strategy: {strategy}")
    kwargs: Dict[str, Any] = {
        "strategy": strategy,
        "add_indicator": bool(add_indicator),
        "keep_empty_features": True,
    }
    if strategy == "constant":
        kwargs["fill_value"] = 0
    return SimpleImputer(**kwargs)


def _categorical_imputer(strategy: str, *, add_indicator: bool = False) -> SimpleImputer:
    strategy = str(strategy)
    if strategy not in {"most_frequent", "constant"}:
        raise FeaturePipelineError(f"Unsupported categorical imputation strategy: {strategy}")
    return SimpleImputer(
        strategy=strategy,
        fill_value="__MISSING__" if strategy == "constant" else None,
        add_indicator=bool(add_indicator),
        keep_empty_features=True,
    )


def _scaler(name: str, *, sparse_safe: bool = False):
    name = str(name).lower()
    if name in {"none", "passthrough"}:
        return "passthrough"
    if name == "standard":
        return StandardScaler(with_mean=not sparse_safe)
    if name == "minmax":
        return MinMaxScaler()
    if name == "robust":
        return RobustScaler(with_centering=not sparse_safe)
    if name == "maxabs":
        return MaxAbsScaler()
    raise FeaturePipelineError(f"Unsupported scaler: {name}")


def _ordinal_transformer(column: str, order: Sequence[str], imputer_strategy: str) -> Pipeline:
    categories = [list(map(str, order))]
    return Pipeline(
        steps=[
            ("sanitize", CategoricalSanitizer()),
            ("impute", _categorical_imputer(imputer_strategy)),
            (
                "encode",
                OrdinalEncoder(
                    categories=categories,
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                    dtype=np.float32,
                ),
            ),
        ]
    )


def build_feature_preprocessor(spec: Mapping[str, Any]) -> ColumnTransformer:
    clean = normalise_feature_pipeline_spec(spec)
    groups = clean["groups"]
    transformers: list[tuple[str, Any, Any]] = []

    if groups["numeric"]:
        numeric_steps: list[tuple[str, Any]] = [
            (
                "coerce",
                NumericCoercer(
                    semantic_types=clean["semantic_types"],
                    percent_as_fraction=clean["numeric"]["percent_as_fraction"],
                ),
            ),
            (
                "impute",
                _imputer(
                    clean["numeric"]["imputer"],
                    add_indicator=clean["numeric"]["add_missing_indicator"],
                ),
            ),
        ]
        scaler = _scaler(clean["numeric"]["scaler"])
        if scaler != "passthrough":
            numeric_steps.append(("scale", scaler))
        transformers.append(("numeric", Pipeline(numeric_steps), groups["numeric"]))

    if groups["categorical"]:
        encoder = OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=clean["categorical"]["min_frequency"],
            max_categories=clean["categorical"]["max_categories"],
            sparse_output=clean["output"]["sparse"],
            dtype=np.float32,
            feature_name_combiner="concat",
        )
        categorical_pipeline = Pipeline(
            steps=[
                ("sanitize", CategoricalSanitizer()),
                (
                    "impute",
                    _categorical_imputer(
                        clean["categorical"]["imputer"],
                        add_indicator=clean["categorical"]["add_missing_indicator"],
                    ),
                ),
                ("encode", encoder),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, groups["categorical"]))

    if groups["frequency"] and clean["frequency"]["enabled"]:
        transformers.append(("frequency", FrequencyEncoder(), groups["frequency"]))

    ordinal_without_order: list[str] = []
    for column in groups["ordinal"]:
        order = clean["ordinal"]["orders"].get(column, [])
        if order:
            transformers.append(
                (
                    f"ordinal_{len(transformers)}",
                    _ordinal_transformer(column, order, clean["ordinal"]["imputer"]),
                    [column],
                )
            )
        else:
            ordinal_without_order.append(column)
    if ordinal_without_order:
        safe_ordinal = Pipeline(
            steps=[
                ("sanitize", CategoricalSanitizer()),
                ("impute", _categorical_imputer(clean["ordinal"]["imputer"])),
                (
                    "encode",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=clean["output"]["sparse"],
                        dtype=np.float32,
                    ),
                ),
            ]
        )
        transformers.append(("ordinal_unordered", safe_ordinal, ordinal_without_order))

    if groups["boolean"]:
        transformers.append(
            (
                "boolean",
                Pipeline(
                    steps=[
                        ("coerce", BooleanCoercer()),
                        ("impute", _imputer(clean["boolean"]["imputer"])),
                    ]
                ),
                groups["boolean"],
            )
        )

    if groups["datetime"]:
        datetime_steps: list[tuple[str, Any]] = [
            (
                "extract",
                DateTimeFeatureExtractor(
                    parts=clean["datetime"]["parts"],
                    cyclical=clean["datetime"]["cyclical"],
                    dayfirst=clean["datetime"]["dayfirst"],
                ),
            ),
            ("impute", _imputer(clean["datetime"]["imputer"])),
        ]
        datetime_scaler = _scaler(clean["datetime"]["scaler"])
        if datetime_scaler != "passthrough":
            datetime_steps.append(("scale", datetime_scaler))
        transformers.append(("datetime", Pipeline(datetime_steps), groups["datetime"]))

    if clean["text"]["enabled"]:
        for index, column in enumerate(groups["text"]):
            text_pipeline = Pipeline(
                steps=[
                    ("sanitize", TextSanitizer(lowercase=clean["text"]["lowercase"])),
                    (
                        "tfidf",
                        TfidfVectorizer(
                            max_features=clean["text"]["max_features"],
                            ngram_range=(
                                clean["text"]["ngram_min"],
                                clean["text"]["ngram_max"],
                            ),
                            min_df=clean["text"]["min_df"],
                            max_df=clean["text"]["max_df"],
                            lowercase=False,
                            strip_accents=None,
                            token_pattern=r"(?u)\b\w\w+\b",
                            dtype=np.float32,
                        ),
                    ),
                ]
            )
            # A scalar selector makes ColumnTransformer pass a one-dimensional series.
            transformers.append((f"text_{index}_{column}", text_pipeline, column))

    if not transformers:
        raise FeaturePipelineError("The pipeline has no active feature transformers.")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=1.0 if clean["output"]["sparse"] else 0.0,
        verbose_feature_names_out=True,
    )


def _feature_names(preprocessor: ColumnTransformer, width: int) -> list[str]:
    try:
        names = [str(name) for name in preprocessor.get_feature_names_out()]
        if len(names) == width:
            return names
    except Exception:
        pass
    return [f"feature_{index}" for index in range(width)]


def fit_feature_preprocessor(
    train_df: pd.DataFrame,
    spec: Mapping[str, Any],
) -> FittedFeaturePipeline:
    clean = normalise_feature_pipeline_spec(spec)
    report = validate_feature_pipeline_spec(train_df, clean, require_target=False)
    if not report["valid"]:
        raise FeaturePipelineError("; ".join(report["blockers"]))
    features = train_df[list(clean["feature_columns"])]
    preprocessor = build_feature_preprocessor(clean)
    transformed = preprocessor.fit_transform(features)
    width = int(transformed.shape[1])
    if width <= 0:
        raise FeaturePipelineError("The fitted pipeline produced zero features.")
    if width > MAX_SAFE_PROJECTED_FEATURES:
        raise FeaturePipelineError(
            f"The fitted pipeline produced {width:,} features, above the safe limit."
        )
    return FittedFeaturePipeline(
        preprocessor=preprocessor,
        spec=clean,
        feature_names=_feature_names(preprocessor, width),
        fitted_rows=len(train_df),
        transformed_features=width,
        spec_fingerprint=feature_pipeline_spec_fingerprint(clean),
    )


def build_model_pipeline(spec: Mapping[str, Any], estimator: Any) -> Pipeline:
    """Create one sklearn Pipeline so preprocessing is fitted inside CV/train folds."""
    return Pipeline(
        steps=[
            ("features", build_feature_preprocessor(spec)),
            ("model", estimator),
        ]
    )


def safe_training_preview(
    df: pd.DataFrame,
    spec: Mapping[str, Any],
    *,
    test_size: float = 0.20,
    random_state: int = 42,
    max_preview_rows: int = 25,
    max_fit_rows: int = 50000,
) -> Dict[str, Any]:
    """
    Fit preprocessing on the training split only and transform the holdout.

    This is a feature preview, not model training. The source DataFrame is never
    modified and holdout statistics never participate in fitting.
    """
    clean = normalise_feature_pipeline_spec(spec)
    report = validate_feature_pipeline_spec(df, clean, require_target=True)
    if not report["valid"]:
        raise FeaturePipelineError("; ".join(report["blockers"]))

    target = clean["target"]
    modelling = df[list(clean["feature_columns"]) + [target]].copy(deep=True)
    modelling = modelling.loc[modelling[target].notna()].copy()
    if len(modelling) < 10:
        raise FeaturePipelineError("At least ten rows with a valid target are required.")

    source_rows = int(len(modelling))
    preview_sampled = False
    max_fit_rows = max(1000, int(max_fit_rows))
    if len(modelling) > max_fit_rows:
        if clean["task"] == "classification":
            counts = modelling[target].value_counts()
            can_stratify_sample = (
                len(counts) >= 2
                and int(counts.min()) >= 2
                and max_fit_rows >= len(counts) * 2
            )
            if can_stratify_sample:
                modelling, _ = train_test_split(
                    modelling,
                    train_size=max_fit_rows,
                    random_state=int(random_state),
                    stratify=modelling[target],
                )
            else:
                modelling = modelling.sample(n=max_fit_rows, random_state=int(random_state))
        else:
            modelling = modelling.sample(n=max_fit_rows, random_state=int(random_state))
        preview_sampled = True

    stratify = None
    if clean["task"] == "classification":
        counts = modelling[target].value_counts()
        if len(counts) >= 2 and int(counts.min()) >= 2:
            requested_test_rows = int(math.ceil(len(modelling) * float(test_size)))
            requested_train_rows = len(modelling) - requested_test_rows
            if requested_test_rows >= len(counts) and requested_train_rows >= len(counts):
                stratify = modelling[target]

    train_df, holdout_df = train_test_split(
        modelling,
        test_size=float(test_size),
        random_state=int(random_state),
        stratify=stratify,
    )
    fitted = fit_feature_preprocessor(train_df, clean)
    train_matrix = fitted.preprocessor.transform(train_df[clean["feature_columns"]])
    holdout_matrix = fitted.preprocessor.transform(holdout_df[clean["feature_columns"]])

    preview_rows = min(max_preview_rows, holdout_matrix.shape[0])
    preview_cols = min(30, holdout_matrix.shape[1])
    preview_block = holdout_matrix[:preview_rows, :preview_cols]
    if sparse.issparse(preview_block):
        preview_values = preview_block.toarray()
    else:
        preview_values = np.asarray(preview_block)
    preview = pd.DataFrame(
        preview_values,
        columns=fitted.feature_names[:preview_cols],
        index=holdout_df.index[:preview_rows],
    )

    return {
        "fitted": fitted,
        "source_rows": source_rows,
        "sampled_rows": int(len(modelling)),
        "preview_sampled": preview_sampled,
        "train_rows": int(len(train_df)),
        "holdout_rows": int(len(holdout_df)),
        "output_features": int(fitted.transformed_features),
        "sparse_output": bool(sparse.issparse(train_matrix)),
        "train_matrix_shape": tuple(map(int, train_matrix.shape)),
        "holdout_matrix_shape": tuple(map(int, holdout_matrix.shape)),
        "feature_names": list(fitted.feature_names),
        "preview": preview,
        "train_indices": list(train_df.index),
        "holdout_indices": list(holdout_df.index),
        "target": target,
        "task": clean["task"],
        "spec_fingerprint": fitted.spec_fingerprint,
    }
