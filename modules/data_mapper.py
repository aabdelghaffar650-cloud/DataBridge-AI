# ════════════════════════════════════════════════════════
#  DataBridge AI — Content-Aware Semantic Mapper
#  Stage 6: semantic typing + ML readiness assessment
# ════════════════════════════════════════════════════════
from __future__ import annotations

import copy
import json
import math
import re
import unicodedata
import warnings
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from config.constants import (
    BUSINESS_ROLE_GROUPS,
    ML_SEMANTIC_TYPES,
    SEMANTIC_GROUPS,
)


_EMAIL_RE = re.compile(r"^[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}$", re.IGNORECASE)
_PHONE_RE = re.compile(r"^\+?[0-9][0-9\s().\-]{5,}[0-9]$")
_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)

_BOOLEAN_TRUE = {
    "true", "yes", "y", "1", "on", "نعم", "صح", "صحيح", "موافق",
}
_BOOLEAN_FALSE = {
    "false", "no", "n", "0", "off", "لا", "خطأ", "خاطئ", "غير موافق",
}
_BOOLEAN_VALUES = _BOOLEAN_TRUE | _BOOLEAN_FALSE

_ORDINAL_VALUE_SETS = [
    {"low", "medium", "high"},
    {"very low", "low", "medium", "high", "very high"},
    {"small", "medium", "large"},
    {"poor", "fair", "good", "very good", "excellent"},
    {"junior", "mid", "senior", "lead"},
    {"bronze", "silver", "gold", "platinum"},
    {"منخفض", "متوسط", "مرتفع"},
    {"ضعيف", "مقبول", "جيد", "جيد جدا", "ممتاز"},
]

_TARGET_NAME_TOKENS = {
    "target", "label", "outcome", "response", "class", "result", "churn",
    "fraud", "default", "approved", "approval", "converted", "conversion",
    "y", "هدف", "نتيجة", "تصنيف", "مخرج", "استجابة", "نجاح", "فشل",
}
_POST_OUTCOME_TOKENS = {
    "prediction", "predicted", "probability", "final", "approved", "approval",
    "outcome", "result", "decision", "status after", "post", "forecast",
    "تنبؤ", "متوقع", "نهائي", "قرار", "بعد", "نتيجة",
}
_IDENTIFIER_TOKENS = {
    "id", "identifier", "uuid", "guid", "key", "code", "ref", "reference",
    "serial", "number", "no", "num", "رقم", "معرف", "كود", "مرجع", "مسلسل",
}
_DATE_TOKENS = {
    "date", "datetime", "timestamp", "time", "year", "month", "day", "period",
    "تاريخ", "وقت", "سنة", "شهر", "يوم", "فترة",
}
_PHONE_TOKENS = {"phone", "mobile", "tel", "telephone", "whatsapp", "fax", "هاتف", "جوال", "واتساب", "فاكس"}
_EMAIL_TOKENS = {"email", "mail", "e mail", "بريد", "ايميل", "إيميل"}
_PERCENT_TOKENS = {"percent", "percentage", "pct", "rate", "ratio", "share", "نسبة", "معدل", "حصة"}
_CURRENCY_TOKENS = {
    "amount", "price", "cost", "revenue", "sales", "salary", "budget", "fee",
    "payment", "income", "expense", "value", "total", "مبلغ", "سعر", "تكلفة",
    "ايراد", "إيراد", "مبيعات", "راتب", "ميزانية", "رسوم", "قيمة", "اجمالي", "إجمالي",
}
_ORDINAL_NAME_TOKENS = {
    "rank", "level", "priority", "grade", "rating", "stage", "tier", "severity",
    "order", "رتبة", "مستوى", "اولوية", "أولوية", "درجة", "مرحلة", "فئة ترتيبية",
}
_TEXT_NAME_TOKENS = {
    "description", "notes", "comment", "remarks", "details", "text", "message",
    "وصف", "ملاحظات", "تعليق", "تفاصيل", "نص", "رسالة",
}


def _normalise_name(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).strip().lower()
    text = re.sub(r"[_\-./\\]+", " ", text)
    return re.sub(r"\s+", " ", text)


def _name_tokens(value: Any) -> set[str]:
    normalised = _normalise_name(value)
    tokens = set(_TOKEN_RE.findall(normalised))
    tokens.add(normalised)
    return {token for token in tokens if token}


def _keyword_match(name: Any, keywords: Iterable[str]) -> tuple[bool, float, str]:
    normalised = _normalise_name(name)
    tokens = _name_tokens(name)
    best_keyword = ""
    best_score = 0.0
    for keyword in keywords:
        key = _normalise_name(keyword)
        if not key:
            continue
        if normalised == key:
            score = 1.0
        elif key in tokens:
            score = 0.92
        elif " " in key and key in normalised:
            score = 0.82
        else:
            score = 0.0
        if score > best_score:
            best_keyword = str(keyword)
            best_score = score
    return best_score > 0, best_score, best_keyword




def _stable_value_text(value: Any) -> str:
    if isinstance(value, dict):
        try:
            return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
        except Exception:
            return repr(value)
    if isinstance(value, (list, tuple, set, np.ndarray)):
        try:
            normalised = list(value) if not isinstance(value, set) else sorted(value, key=str)
            return json.dumps(normalised, ensure_ascii=False, default=str)
        except Exception:
            return repr(value)
    try:
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)) and bool(missing):
            return ""
    except Exception:
        pass
    return str(value)


def _safe_nunique(series: pd.Series) -> int:
    try:
        return int(series.nunique(dropna=True))
    except (TypeError, ValueError):
        clean = series.dropna().map(_stable_value_text)
        return int(clean.nunique(dropna=True))


def _sample_non_null(series: pd.Series, sample_size: int) -> pd.Series:
    clean = series.dropna()
    if len(clean) <= sample_size:
        return clean
    positions = np.linspace(0, len(clean) - 1, num=sample_size, dtype=int)
    return clean.iloc[positions]


def _safe_string_series(series: pd.Series) -> pd.Series:
    try:
        return series.astype("string").str.strip()
    except Exception:
        return series.map(lambda value: _stable_value_text(value).strip()).astype("string")


def _safe_ratio(mask: pd.Series | np.ndarray | Sequence[bool], denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    try:
        return float(np.asarray(mask, dtype=bool).sum() / denominator)
    except Exception:
        return 0.0


def _name_business_role(column: str) -> tuple[str, float]:
    best_role = "Unknown"
    best_score = 0.0
    best_keyword_length = 0
    for role, keywords in BUSINESS_ROLE_GROUPS.items():
        if role == "Unknown":
            continue
        matched, score, keyword = _keyword_match(column, keywords)
        if matched and (score > best_score or (score == best_score and len(keyword) > best_keyword_length)):
            best_role = role
            best_score = score
            best_keyword_length = len(keyword)
    return best_role, round(best_score, 2)


def _name_semantic_hint(column: str) -> tuple[str, float, str]:
    best_type = "Unknown"
    best_score = 0.0
    best_keyword = ""
    for semantic_type, keywords in SEMANTIC_GROUPS.items():
        if semantic_type == "Unknown":
            continue
        matched, score, keyword = _keyword_match(column, keywords)
        if matched and score > best_score:
            best_type = semantic_type
            best_score = score
            best_keyword = keyword
    return best_type, round(best_score, 2), best_keyword


def _is_integer_like(values: pd.Series) -> bool:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return False
    arr = numeric.astype(float).to_numpy()
    return bool(np.isfinite(arr).all() and np.isclose(arr, np.round(arr), atol=1e-9).mean() >= 0.98)


def _numeric_parse_ratio(strings: pd.Series) -> float:
    if strings.empty:
        return 0.0
    cleaned = (
        strings.str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .str.replace(r"^[\$€£¥₹]|[\$€£¥₹]$", "", regex=True)
        .str.strip()
    )
    parsed = pd.to_numeric(cleaned, errors="coerce")
    return round(float(parsed.notna().mean()), 4)


def _date_parse_ratio(strings: pd.Series, *, name_has_date_hint: bool) -> float:
    if strings.empty:
        return 0.0
    non_empty = strings[strings.ne("")]
    if non_empty.empty:
        return 0.0

    # Do not treat generic integer codes as dates. Four-digit years are accepted
    # only when the column name explicitly indicates time/date semantics.
    digit_only = non_empty.str.fullmatch(r"[+-]?\d+(?:\.0+)?", na=False)
    if bool(digit_only.mean() >= 0.95):
        four_digit_years = non_empty.str.fullmatch(r"(?:19|20|21)\d{2}", na=False)
        if not (name_has_date_hint and bool(four_digit_years.mean() >= 0.85)):
            return 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            parsed = pd.to_datetime(non_empty, errors="coerce", dayfirst=True, format="mixed")
        except (TypeError, ValueError):
            parsed = pd.to_datetime(non_empty, errors="coerce", dayfirst=True)
    return round(float(parsed.notna().mean()), 4)


def _profile_column(column: str, series: pd.Series, row_count: int, sample_size: int) -> Dict[str, Any]:
    non_null_count = int(series.notna().sum())
    missing_count = int(row_count - non_null_count)
    unique_count = _safe_nunique(series) if non_null_count else 0
    unique_ratio = float(unique_count / non_null_count) if non_null_count else 0.0
    sample = _sample_non_null(series, sample_size)
    sample_strings = _safe_string_series(sample)
    sample_len = len(sample_strings)
    name_tokens = _name_tokens(column)

    is_bool_dtype = bool(pd.api.types.is_bool_dtype(series.dtype))
    is_datetime_dtype = bool(pd.api.types.is_datetime64_any_dtype(series.dtype))
    is_numeric_dtype = bool(pd.api.types.is_numeric_dtype(series.dtype) and not is_bool_dtype)
    is_category_dtype = bool(isinstance(series.dtype, pd.CategoricalDtype))

    normalised_values = sample_strings.str.lower().str.replace(r"\s+", " ", regex=True)
    boolean_ratio = _safe_ratio(normalised_values.isin(_BOOLEAN_VALUES), sample_len)
    email_ratio = _safe_ratio(normalised_values.map(lambda value: bool(_EMAIL_RE.fullmatch(str(value)))), sample_len)
    phone_mask = normalised_values.map(lambda value: bool(_PHONE_RE.fullmatch(str(value))))
    phone_ratio = _safe_ratio(phone_mask, sample_len)
    numeric_ratio = 1.0 if is_numeric_dtype else _numeric_parse_ratio(sample_strings)

    date_name_hint = bool(name_tokens & _DATE_TOKENS)
    date_ratio = 1.0 if is_datetime_dtype else (
        0.0 if is_numeric_dtype else _date_parse_ratio(sample_strings, name_has_date_hint=date_name_hint)
    )

    lengths = sample_strings.str.len() if sample_len else pd.Series(dtype="float64")
    word_counts = sample_strings.str.split().str.len() if sample_len else pd.Series(dtype="float64")
    avg_length = float(lengths.mean()) if sample_len else 0.0
    avg_words = float(word_counts.mean()) if sample_len else 0.0
    contains_phone_formatting = _safe_ratio(
        sample_strings.str.contains(r"[+()\-\s]", regex=True, na=False), sample_len
    )
    contains_currency_symbol = _safe_ratio(
        sample_strings.str.contains(r"[\$€£¥₹]", regex=True, na=False), sample_len
    )
    contains_percent_symbol = _safe_ratio(
        sample_strings.str.contains("%", regex=False, na=False), sample_len
    )

    id_name = bool(name_tokens & _IDENTIFIER_TOKENS)
    email_name = bool(name_tokens & _EMAIL_TOKENS)
    phone_name = bool(name_tokens & _PHONE_TOKENS)
    percent_name = bool(name_tokens & _PERCENT_TOKENS)
    currency_name = bool(name_tokens & _CURRENCY_TOKENS)
    ordinal_name = bool(name_tokens & _ORDINAL_NAME_TOKENS)
    text_name = bool(name_tokens & _TEXT_NAME_TOKENS)

    reasons: list[str] = []
    warnings_list: list[str] = []
    semantic_type = "Unknown"
    confidence = 0.0

    if non_null_count == 0:
        semantic_type, confidence = "Unknown", 0.0
        reasons.append("Column contains no non-null values.")
    elif is_bool_dtype or (unique_count <= 3 and boolean_ratio >= 0.95):
        semantic_type = "Boolean"
        confidence = 0.99 if is_bool_dtype else 0.94
        reasons.append("Values consistently match boolean tokens.")
    elif email_ratio >= 0.90 or (email_name and email_ratio >= 0.60):
        semantic_type = "Email"
        confidence = min(0.99, 0.82 + 0.17 * email_ratio + (0.04 if email_name else 0.0))
        reasons.append(f"{email_ratio:.0%} of sampled values match an email pattern.")
    elif is_datetime_dtype or date_ratio >= 0.88:
        semantic_type = "Datetime"
        confidence = 0.99 if is_datetime_dtype else min(0.96, 0.72 + 0.22 * date_ratio + (0.03 if date_name_hint else 0.0))
        reasons.append("Values have strong date/time parse consistency.")
    elif phone_ratio >= 0.88 and (phone_name or contains_phone_formatting >= 0.25):
        semantic_type = "Phone"
        confidence = min(0.98, 0.78 + 0.16 * phone_ratio + (0.05 if phone_name else 0.0))
        reasons.append(f"{phone_ratio:.0%} of sampled values match a phone pattern.")
    elif (percent_name and numeric_ratio >= 0.85) or contains_percent_symbol >= 0.60:
        semantic_type = "Percentage"
        confidence = min(0.96, 0.72 + 0.18 * numeric_ratio + (0.05 if percent_name else 0.0))
        reasons.append("Column name or values indicate percentage/rate semantics.")
    elif (currency_name and numeric_ratio >= 0.85) or contains_currency_symbol >= 0.60:
        semantic_type = "Currency"
        confidence = min(0.96, 0.72 + 0.18 * numeric_ratio + (0.05 if currency_name else 0.0))
        reasons.append("Column name or values indicate monetary semantics.")
    elif id_name and unique_ratio >= 0.75:
        semantic_type = "Identifier"
        confidence = min(0.97, 0.72 + 0.18 * unique_ratio + (0.05 if non_null_count == row_count else 0.0))
        reasons.append("Identifier name signal is supported by high uniqueness.")
    elif is_numeric_dtype or numeric_ratio >= 0.95:
        integer_like = _is_integer_like(sample if is_numeric_dtype else sample_strings)
        discrete_limit = max(12, min(50, int(math.sqrt(max(non_null_count, 1)) * 2)))
        if ordinal_name and integer_like and unique_count <= discrete_limit:
            semantic_type = "Ordinal"
            confidence = 0.88
            reasons.append("Ordered name signal and low-cardinality integer values indicate an ordinal feature.")
        elif integer_like and unique_count <= discrete_limit:
            semantic_type = "Numeric Discrete"
            confidence = 0.91 if is_numeric_dtype else 0.84
            reasons.append("Numeric values are integer-like with limited cardinality.")
        elif integer_like and unique_ratio >= 0.995 and non_null_count >= 20 and series.dropna().is_monotonic_increasing:
            semantic_type = "Identifier"
            confidence = 0.72
            reasons.append("Near-unique monotonic numbers behave like a row identifier.")
        else:
            semantic_type = "Numeric Continuous"
            confidence = 0.95 if is_numeric_dtype else 0.86
            reasons.append("Values are consistently numeric with continuous variation.")
    else:
        value_set = set(normalised_values.dropna().tolist())
        is_known_ordinal = any(value_set and value_set.issubset(order_set) for order_set in _ORDINAL_VALUE_SETS)
        category_limit = max(20, min(100, int(math.sqrt(max(non_null_count, 1)) * 4)))
        if ordinal_name or is_known_ordinal:
            semantic_type = "Ordinal"
            confidence = 0.92 if is_known_ordinal else 0.78
            reasons.append("Values or column name indicate an ordered category.")
        elif avg_length >= 40 or avg_words >= 5 or text_name:
            semantic_type = "Free Text"
            confidence = min(0.94, 0.75 + min(avg_length / 400, 0.14) + (0.04 if text_name else 0.0))
            reasons.append("Long or multi-word values behave like free text.")
        elif is_category_dtype or unique_count <= category_limit or unique_ratio <= 0.20:
            semantic_type = "Categorical"
            confidence = 0.92 if is_category_dtype else 0.84
            reasons.append("Non-numeric values have category-like cardinality.")
        else:
            semantic_type = "Categorical"
            confidence = 0.66
            reasons.append("Short text values are treated as high-cardinality categories pending review.")

    name_hint, name_hint_score, name_keyword = _name_semantic_hint(column)
    if name_hint != "Unknown":
        if name_hint == semantic_type:
            confidence = min(0.99, confidence + 0.04 * name_hint_score)
            reasons.append(f"Column name supports this type ({name_keyword}).")
        elif confidence < 0.80 and name_hint_score >= 0.90:
            warnings_list.append(
                f"Name suggests {name_hint}, while content suggests {semantic_type}."
            )
            confidence = max(0.45, confidence - 0.10)
        elif name_hint_score >= 0.90:
            warnings_list.append(
                f"Name/content conflict: name suggests {name_hint}; content supports {semantic_type}."
            )
            confidence = max(0.55, confidence - 0.06)

    is_constant = bool(non_null_count > 0 and unique_count <= 1)
    high_cardinality = bool(
        semantic_type in {"Categorical", "Ordinal"}
        and unique_count > max(50, int(non_null_count * 0.35))
    )
    near_unique = bool(non_null_count >= 20 and unique_ratio >= 0.98)

    leakage_reasons: list[str] = []
    if semantic_type in {"Identifier", "Email", "Phone"}:
        leakage_reasons.append("Identity/contact field can memorize records and should normally be excluded.")
    if near_unique and semantic_type not in {"Identifier", "Email", "Phone", "Free Text"}:
        leakage_reasons.append("Near-unique values may identify individual rows.")
    if name_tokens & _POST_OUTCOME_TOKENS:
        leakage_reasons.append("Name may describe an outcome, post-event value, or prediction.")

    if missing_count:
        warnings_list.append(f"{missing_count:,} missing values require an explicit preprocessing policy.")
    if high_cardinality:
        warnings_list.append("High-cardinality categorical feature needs controlled encoding.")
    if is_constant:
        warnings_list.append("Constant column carries no predictive information.")

    business_role, business_confidence = _name_business_role(column)
    profile: Dict[str, Any] = {
        "column": str(column),
        "semantic_type": semantic_type,
        "effective_semantic_type": semantic_type,
        "confidence": round(float(max(0.0, min(confidence, 0.99))), 2),
        "confidence_label": confidence_label(confidence),
        "business_role": business_role,
        "business_confidence": business_confidence,
        "dtype": str(series.dtype),
        "row_count": int(row_count),
        "non_null_count": non_null_count,
        "missing_count": missing_count,
        "missing_pct": round((missing_count / max(row_count, 1)) * 100, 2),
        "unique_count": unique_count,
        "unique_ratio": round(unique_ratio, 4),
        "sample_values": [str(value)[:80] for value in sample.head(5).tolist()],
        "signals": {
            "numeric_ratio": round(float(numeric_ratio), 4),
            "date_ratio": round(float(date_ratio), 4),
            "boolean_ratio": round(float(boolean_ratio), 4),
            "email_ratio": round(float(email_ratio), 4),
            "phone_ratio": round(float(phone_ratio), 4),
            "average_text_length": round(avg_length, 2),
            "average_word_count": round(avg_words, 2),
        },
        "all_missing": non_null_count == 0,
        "is_constant": is_constant,
        "near_unique": near_unique,
        "high_cardinality": high_cardinality,
        "leakage_risk": bool(leakage_reasons),
        "leakage_reasons": leakage_reasons,
        "feature_eligible": False,
        "target_candidate_score": 0.0,
        "target_task": None,
        "manual_override": False,
        "reasons": reasons,
        "warnings": list(dict.fromkeys(warnings_list)),
        "recommended_transformations": [],
    }
    _finalise_profile(profile, name_tokens)
    return profile


def _target_score_from_profile(profile: Mapping[str, Any], name_tokens: set[str]) -> tuple[float, str | None, list[str]]:
    semantic_type = str(profile.get("effective_semantic_type") or profile.get("semantic_type") or "Unknown")
    non_null = int(profile.get("non_null_count", 0) or 0)
    row_count = int(profile.get("row_count", 0) or 0)
    unique_count = int(profile.get("unique_count", 0) or 0)
    missing_pct = float(profile.get("missing_pct", 0.0) or 0.0)
    unique_ratio = float(profile.get("unique_ratio", 0.0) or 0.0)

    if non_null == 0 or unique_count <= 1 or semantic_type in {"Identifier", "Email", "Phone", "Datetime", "Free Text", "Unknown"}:
        return 0.0, None, []

    reasons: list[str] = []
    score = 0.0
    task: str | None = None
    if semantic_type in {"Boolean", "Categorical", "Ordinal"}:
        max_classes = max(20, min(100, int(math.sqrt(max(non_null, 1)) * 4)))
        if 2 <= unique_count <= max_classes:
            task = "classification"
            score = 0.62
            reasons.append(f"{unique_count} classes are suitable for classification.")
            if unique_count == 2:
                score += 0.08
    elif semantic_type in {"Numeric Continuous", "Currency", "Percentage"}:
        if unique_count >= min(10, max(3, int(non_null * 0.02))):
            task = "regression"
            score = 0.58
            reasons.append("Numeric variation is suitable for regression.")
    elif semantic_type == "Numeric Discrete":
        if unique_count <= max(20, int(math.sqrt(max(non_null, 1)) * 2)):
            task = "classification"
            score = 0.48
            reasons.append("Limited integer values may represent classes.")
        else:
            task = "regression"
            score = 0.48
            reasons.append("Discrete numeric variation may support regression.")

    if task is None:
        return 0.0, None, []
    if name_tokens & _TARGET_NAME_TOKENS:
        score += 0.25
        reasons.append("Column name strongly suggests an outcome/target.")
    if missing_pct > 30:
        score -= 0.20
        reasons.append("High missingness weakens target suitability.")
    elif missing_pct > 10:
        score -= 0.08
    if unique_ratio >= 0.98:
        score -= 0.30
    if row_count < 50:
        score -= 0.10
    return round(max(0.0, min(score, 0.99)), 2), task, reasons


def _recommended_transformations(semantic_type: str, profile: Mapping[str, Any]) -> list[str]:
    actions: list[str] = []
    if int(profile.get("missing_count", 0) or 0) > 0:
        if semantic_type in {"Numeric Continuous", "Numeric Discrete", "Currency", "Percentage"}:
            actions.append("numeric imputation fitted on training data")
        else:
            actions.append("categorical/text imputation fitted on training data")
    if semantic_type in {"Numeric Continuous", "Currency", "Percentage"}:
        actions.append("optional robust or standard scaling")
    elif semantic_type == "Numeric Discrete":
        actions.append("keep numeric or encode as category based on meaning")
    elif semantic_type == "Categorical":
        actions.append("one-hot encoding with unknown-category handling")
    elif semantic_type == "Ordinal":
        actions.append("explicit ordinal encoding with reviewed order")
    elif semantic_type == "Datetime":
        actions.append("derive calendar/cyclical features after train split")
    elif semantic_type == "Free Text":
        actions.append("text vectorization or approved text features")
    elif semantic_type in {"Identifier", "Email", "Phone"}:
        actions.append("exclude from model features by default")
    elif semantic_type == "Boolean":
        actions.append("stable boolean encoding")
    if bool(profile.get("high_cardinality")):
        actions.append("rare-category grouping or controlled high-cardinality encoder")
    return list(dict.fromkeys(actions))


def _finalise_profile(profile: Dict[str, Any], name_tokens: set[str] | None = None) -> None:
    semantic_type = str(profile.get("effective_semantic_type") or profile.get("semantic_type") or "Unknown")
    name_tokens = name_tokens or _name_tokens(profile.get("column", ""))
    score, task, target_reasons = _target_score_from_profile(profile, name_tokens)
    profile["target_candidate_score"] = score
    profile["target_task"] = task
    profile["target_reasons"] = target_reasons
    profile["feature_eligible"] = bool(
        not profile.get("all_missing")
        and not profile.get("is_constant")
        and semantic_type not in {"Identifier", "Email", "Phone", "Unknown"}
    )
    profile["recommended_transformations"] = _recommended_transformations(semantic_type, profile)


def _fallback_name_only_mapping(columns: Iterable[str]) -> Dict[str, Tuple[str, float]]:
    result: Dict[str, Tuple[str, float]] = {}
    for raw_column in columns:
        column = str(raw_column)
        semantic_type, score, _ = _name_semantic_hint(column)
        # Name-only evidence must never be presented as high-confidence content analysis.
        result[column] = (semantic_type, round(min(score * 0.65, 0.65), 2))
    return result


def auto_map_columns(data: pd.DataFrame | Iterable[str]) -> Dict[str, Tuple[str, float]]:
    """
    Return ``{column: (semantic_type, confidence)}``.

    Passing a DataFrame activates content-aware analysis. Passing only names is
    supported for compatibility, but confidence is intentionally capped because
    values were not inspected.
    """
    if isinstance(data, pd.DataFrame):
        return analyze_dataframe(data)["mappings"]
    return _fallback_name_only_mapping(tuple(str(column) for column in data))


def analyze_dataframe(df: pd.DataFrame, *, sample_size: int = 2000) -> Dict[str, Any]:
    """Profile every column and build a deterministic ML-readiness report."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("analyze_dataframe expects a pandas DataFrame.")
    if sample_size < 50:
        raise ValueError("sample_size must be at least 50.")

    profiles: Dict[str, Dict[str, Any]] = {}
    mappings: Dict[str, Tuple[str, float]] = {}
    row_count = int(len(df))
    for column in df.columns:
        name = str(column)
        profile = _profile_column(name, df[column], row_count, sample_size)
        profiles[name] = profile
        mappings[name] = (profile["semantic_type"], float(profile["confidence"]))

    effective_mappings = {column: mapping[0] for column, mapping in mappings.items()}
    readiness = build_ml_readiness_report(df, profiles, effective_mappings)
    return {
        "mappings": mappings,
        "profiles": profiles,
        "readiness": readiness,
        "analysis_version": "6.0",
        "sample_size": int(min(sample_size, max(row_count, 0))),
    }


def apply_manual_mappings(
    df: pd.DataFrame,
    profiles: Mapping[str, Mapping[str, Any]],
    mappings: Mapping[str, str],
) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Apply reviewed semantic types without modifying the dataset."""
    columns = [str(column) for column in df.columns]
    if set(mappings) != set(columns):
        raise ValueError("Manual mappings must include every dataset column exactly once.")
    invalid = sorted({str(value) for value in mappings.values()} - set(ML_SEMANTIC_TYPES))
    if invalid:
        raise ValueError("Unsupported semantic type(s): " + ", ".join(invalid))

    updated: Dict[str, Dict[str, Any]] = {}
    for column in columns:
        base = copy.deepcopy(dict(profiles.get(column, {})))
        if not base:
            base = _profile_column(column, df[column], len(df), 2000)
        selected = str(mappings[column])
        base["effective_semantic_type"] = selected
        base["manual_override"] = selected != base.get("semantic_type")
        if base["manual_override"]:
            base.setdefault("warnings", []).append(
                f"Human review changed the automatic type from {base.get('semantic_type')} to {selected}."
            )
        _finalise_profile(base, _name_tokens(column))
        updated[column] = base

    readiness = build_ml_readiness_report(df, updated, mappings)
    readiness["mappings_reviewed"] = True
    return updated, readiness


def build_ml_readiness_report(
    df: pd.DataFrame,
    profiles: Mapping[str, Mapping[str, Any]],
    effective_mappings: Mapping[str, str] | None = None,
) -> Dict[str, Any]:
    """Build a model-agnostic readiness report; no preprocessing is fitted here."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("build_ml_readiness_report expects a pandas DataFrame.")

    mappings = {
        str(column): str(
            (effective_mappings or {}).get(
                str(column),
                profiles.get(str(column), {}).get("effective_semantic_type")
                or profiles.get(str(column), {}).get("semantic_type")
                or "Unknown",
            )
        )
        for column in df.columns
    }

    blockers: list[str] = []
    warnings_list: list[str] = []
    actions: list[str] = []
    score = 100.0
    row_count, column_count = int(df.shape[0]), int(df.shape[1])

    if row_count < 10:
        blockers.append("Fewer than 10 rows are available; a reliable train/test workflow cannot be built.")
        score -= 45
    elif row_count < 50:
        warnings_list.append("Small dataset: validation metrics may be unstable.")
        score -= 15
    if column_count < 2:
        blockers.append("At least one target and one feature column are required.")
        score -= 40

    all_missing = [column for column, p in profiles.items() if bool(p.get("all_missing"))]
    constants = [column for column, p in profiles.items() if bool(p.get("is_constant"))]
    low_confidence = [column for column, p in profiles.items() if float(p.get("confidence", 0.0) or 0.0) < 0.75]
    high_cardinality = [column for column, p in profiles.items() if bool(p.get("high_cardinality"))]
    leakage_risks = [column for column, p in profiles.items() if bool(p.get("leakage_risk"))]
    missing_columns = [column for column, p in profiles.items() if int(p.get("missing_count", 0) or 0) > 0]
    identifiers = [column for column, semantic_type in mappings.items() if semantic_type in {"Identifier", "Email", "Phone"}]
    free_text = [column for column, semantic_type in mappings.items() if semantic_type == "Free Text"]
    datetimes = [column for column, semantic_type in mappings.items() if semantic_type == "Datetime"]
    categoricals = [column for column, semantic_type in mappings.items() if semantic_type in {"Categorical", "Ordinal", "Boolean"}]
    numeric = [column for column, semantic_type in mappings.items() if semantic_type in {"Numeric Continuous", "Numeric Discrete", "Currency", "Percentage"}]

    if all_missing:
        warnings_list.append(f"{len(all_missing)} all-missing column(s) must be removed or recovered.")
        actions.append("Remove all-missing columns before model training.")
        score -= min(20, len(all_missing) * 5)
    if constants:
        warnings_list.append(f"{len(constants)} constant column(s) provide no predictive signal.")
        actions.append("Drop constant and near-constant features inside the ML pipeline.")
        score -= min(12, len(constants) * 3)
    if missing_columns:
        warnings_list.append(f"{len(missing_columns)} column(s) contain missing values.")
        actions.append("Fit imputation strategies on training folds only.")
        score -= min(12, 4 + len(missing_columns))
    if low_confidence:
        warnings_list.append(f"{len(low_confidence)} semantic mapping(s) need human review.")
        actions.append("Review low-confidence or conflicting semantic types in Data Mapper.")
        score -= min(15, len(low_confidence) * 3)
    if high_cardinality:
        warnings_list.append(f"{len(high_cardinality)} high-cardinality categorical feature(s) need controlled encoding.")
        actions.append("Group rare categories or use a leakage-safe high-cardinality encoder.")
        score -= min(12, len(high_cardinality) * 4)
    if leakage_risks:
        warnings_list.append(f"{len(leakage_risks)} feature(s) have identity or potential leakage risk.")
        actions.append("Exclude identifiers and verify post-outcome fields before training.")
        score -= min(18, len(leakage_risks) * 4)
    if free_text:
        warnings_list.append(f"{len(free_text)} free-text feature(s) require an NLP/text pipeline.")
        actions.append("Choose explicit text features or train-only text vectorization.")
        score -= min(8, len(free_text) * 2)
    if datetimes:
        actions.append("Derive date and cyclical features after the data split; do not fit on the full dataset.")
    if categoricals:
        actions.append("Encode categories with unknown-category handling inside the training pipeline.")
    if numeric:
        actions.append("Fit scaling only where the selected model requires it.")

    target_candidates: list[Dict[str, Any]] = []
    feature_candidates: list[str] = []
    excluded_features: Dict[str, str] = {}
    for column in map(str, df.columns):
        profile = dict(profiles.get(column, {}))
        semantic_type = mappings[column]
        profile["effective_semantic_type"] = semantic_type
        target_score, target_task, target_reasons = _target_score_from_profile(profile, _name_tokens(column))
        if target_score >= 0.40:
            target_candidates.append(
                {
                    "column": column,
                    "task": target_task,
                    "score": target_score,
                    "unique_count": int(profile.get("unique_count", 0) or 0),
                    "missing_pct": float(profile.get("missing_pct", 0.0) or 0.0),
                    "reasons": target_reasons,
                }
            )

        if bool(profile.get("all_missing")):
            excluded_features[column] = "all missing"
        elif bool(profile.get("is_constant")):
            excluded_features[column] = "constant"
        elif semantic_type in {"Identifier", "Email", "Phone", "Unknown"}:
            excluded_features[column] = f"{semantic_type.lower()} excluded by default"
        else:
            feature_candidates.append(column)

    target_candidates.sort(key=lambda item: (-float(item["score"]), item["column"]))
    if not target_candidates:
        warnings_list.append("No strong target candidate was detected automatically.")
        actions.append("Select and confirm the prediction target manually in ML Studio.")
        score -= 15
    if not feature_candidates:
        blockers.append("No usable feature candidate remains after safety exclusions.")
        score -= 35
    elif len(feature_candidates) == 1:
        warnings_list.append("Only one usable feature candidate was detected.")
        score -= 8

    score = round(max(0.0, min(score, 100.0)), 1)
    if blockers:
        status = "Blocked"
    elif score >= 85 and not low_confidence and not leakage_risks:
        status = "Ready"
    else:
        status = "Needs Review"

    preprocessing_plan = {
        "numeric": numeric,
        "categorical": [column for column in categoricals if mappings[column] == "Categorical"],
        "ordinal": [column for column in categoricals if mappings[column] == "Ordinal"],
        "boolean": [column for column in categoricals if mappings[column] == "Boolean"],
        "datetime": datetimes,
        "text": free_text,
        "excluded_by_default": identifiers + [column for column, semantic_type in mappings.items() if semantic_type == "Unknown"],
    }

    return {
        "score": score,
        "status": status,
        "training_blocked": bool(blockers),
        "blockers": list(dict.fromkeys(blockers)),
        "warnings": list(dict.fromkeys(warnings_list)),
        "recommended_actions": list(dict.fromkeys(actions)),
        "target_candidates": target_candidates,
        "feature_candidates": feature_candidates,
        "excluded_features": excluded_features,
        "preprocessing_plan": preprocessing_plan,
        "counts": {
            "rows": row_count,
            "columns": column_count,
            "usable_features": len(feature_candidates),
            "target_candidates": len(target_candidates),
            "low_confidence": len(low_confidence),
            "missing_columns": len(missing_columns),
            "high_cardinality": len(high_cardinality),
            "leakage_risks": len(leakage_risks),
            "all_missing": len(all_missing),
            "constant": len(constants),
        },
        "mappings_reviewed": False,
        "analysis_version": "6.0",
    }


def confidence_label(score: float) -> str:
    """Convert numeric confidence to a review label."""
    if score >= 0.90:
        return "auto_accepted"
    if score >= 0.75:
        return "verify"
    if score >= 0.50:
        return "suspicious"
    return "unknown"


def build_confidence_summary(mappings: Mapping[str, Tuple[str, float]]) -> Dict[str, float]:
    """Summarise content-aware mapping confidence without inflating weak guesses."""
    total = len(mappings)
    if total == 0:
        return {
            "auto_accepted_pct": 0,
            "verify_pct": 0,
            "suspicious_pct": 0,
            "unknown_pct": 0,
            "overall_pct": 0.0,
            "auto_accepted": 0,
            "verify": 0,
            "suspicious": 0,
            "unknown": 0,
            "total": 0,
        }

    counts = {"auto_accepted": 0, "verify": 0, "suspicious": 0, "unknown": 0}
    confidence_total = 0.0
    for _, (_, raw_score) in mappings.items():
        score = float(raw_score)
        counts[confidence_label(score)] += 1
        confidence_total += max(0.0, min(score, 1.0))

    return {
        "auto_accepted_pct": round(counts["auto_accepted"] / total * 100),
        "verify_pct": round(counts["verify"] / total * 100),
        "suspicious_pct": round(counts["suspicious"] / total * 100),
        "unknown_pct": round(counts["unknown"] / total * 100),
        "overall_pct": round(confidence_total / total * 100, 1),
        **counts,
        "total": total,
    }
