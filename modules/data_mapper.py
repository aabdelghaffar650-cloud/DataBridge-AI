# ════════════════════════════════════════════════════════
#  DataBridge AI — Auto Data Mapper
# ════════════════════════════════════════════════════════
from __future__ import annotations

from typing import Dict, Iterable, Tuple

import streamlit as st

from config.constants import SEMANTIC_GROUPS


@st.cache_data(show_spinner=False)
def _auto_map_columns_cached(columns_tuple: tuple[str, ...]) -> Dict[str, Tuple[str, float]]:
    """Cached implementation keyed by an immutable tuple of column names."""
    result: Dict[str, Tuple[str, float]] = {}
    for col in columns_tuple:
        col_lower = col.lower().replace("_", " ").replace("-", " ").strip()
        best_group = "Unknown"
        best_score = 0.0
        best_kw_len = 0

        for group, keywords in SEMANTIC_GROUPS.items():
            if group == "Unknown":
                continue
            for kw in keywords:
                kw_l = kw.lower()
                if col_lower == kw_l:
                    score = 1.0
                elif col_lower.startswith(kw_l) or col_lower.endswith(kw_l):
                    score = 0.85
                elif kw_l in col_lower or col_lower in kw_l:
                    score = 0.65
                else:
                    score = 0.0

                if score > best_score or (score == best_score and len(kw_l) > best_kw_len):
                    best_score, best_group, best_kw_len = score, group, len(kw_l)

        result[col] = (best_group, round(best_score, 2))
    return result


def auto_map_columns(columns: Iterable[str]) -> Dict[str, Tuple[str, float]]:
    """
    Semantic rule-based mapping.
    Returns {raw_col: (semantic_group, confidence_score)}.
    """
    return _auto_map_columns_cached(tuple(str(c) for c in columns))


def confidence_label(score: float) -> str:
    """Convert numeric score to categorical label."""
    if score >= 0.85:
        return "auto_accepted"
    if score >= 0.60:
        return "verify"
    if score >= 0.30:
        return "suspicious"
    return "unknown"


def build_confidence_summary(mappings: Dict[str, Tuple[str, float]]) -> Dict[str, float]:
    """
    Returns summary dict:
    {auto_accepted_pct, verify_pct, suspicious_pct, unknown_pct, overall_pct}
    """
    total = len(mappings)
    if total == 0:
        return {}

    counts = {"auto_accepted": 0, "verify": 0, "suspicious": 0, "unknown": 0}
    for _, (_, score) in mappings.items():
        counts[confidence_label(score)] += 1

    overall = round(
        (counts["auto_accepted"] * 1.0 + counts["verify"] * 0.6 + counts["suspicious"] * 0.3)
        / total * 100,
        1,
    )
    return {
        "auto_accepted_pct": round(counts["auto_accepted"] / total * 100),
        "verify_pct": round(counts["verify"] / total * 100),
        "suspicious_pct": round(counts["suspicious"] / total * 100),
        "unknown_pct": round(counts["unknown"] / total * 100),
        "overall_pct": overall,
        **counts,
        "total": total,
    }
