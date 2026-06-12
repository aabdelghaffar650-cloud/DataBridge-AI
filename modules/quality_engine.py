# ════════════════════════════════════════════════════════
#  DataBridge AI — Data Quality Engine
# ════════════════════════════════════════════════════════
from __future__ import annotations

from typing import Dict, Any

import pandas as pd
import streamlit as st


@st.cache_data(show_spinner=False)
def run_quality_engine(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generic cached data health scanner.
    Checks: duplicates · nulls · type errors · date anomalies.
    """
    report: Dict[str, Any] = {}
    total_cells = df.shape[0] * df.shape[1]
    errors = 0

    # 1. Duplicates
    # Store indices only (not a DataFrame copy) to avoid doubling RAM usage on
    # large datasets. Pages that need to display duplicate rows should call
    # df.iloc[report["duplicate_indices"]] at render time.
    dup_mask = df.duplicated()
    dup_count = int(dup_mask.sum())
    report["duplicate_count"] = dup_count
    report["duplicate_indices"] = df.index[dup_mask].tolist()[:500]   # cap preview at 500
    report["duplicate_indices_truncated"] = dup_count > 500
    # Keep a tiny head (max 20 rows) for the quality summary card only
    report["duplicate_sample"] = df.loc[dup_mask].head(20).copy() if dup_count > 0 else pd.DataFrame()
    errors += dup_count

    # 2. Missing values
    null_series = df.isnull().sum()
    report["null_by_col"] = null_series[null_series > 0].to_dict()
    report["total_nulls"] = int(null_series.sum())
    errors += report["total_nulls"]

    # 3. Type errors — columns with mixed numeric/text content
    type_errors: Dict[str, int] = {}
    for col in df.select_dtypes(include="object").columns:
        sample = df[col].dropna()
        numeric_attempt = pd.to_numeric(sample, errors="coerce")
        n_numeric = numeric_attempt.notna().sum()
        n_total = len(sample)
        n_failed = n_total - n_numeric
        if n_total > 0 and n_numeric / n_total > 0.5 and n_failed > 0:
            type_errors[col] = int(n_failed)
    report["type_errors"] = type_errors
    errors += sum(type_errors.values())

    # 4. Date anomalies (future dates)
    date_cols = df.select_dtypes(include=["datetime64", "datetimetz"]).columns.tolist()
    date_errors: Dict[str, Dict[str, int]] = {}
    now = pd.Timestamp.now(tz=None)
    for col in date_cols:
        try:
            s = pd.to_datetime(df[col], errors="coerce")
            future = (s > now).sum()
            if future > 0:
                date_errors[str(col)] = {"future_dates": int(future)}
        except Exception:
            pass
    report["date_errors"] = date_errors

    # 5. Quality score
    clean_cells = total_cells - errors
    report["quality_score"] = round(max(clean_cells / max(total_cells, 1) * 100, 0), 1)
    report["total_cells"] = total_cells
    report["total_errors"] = errors

    return report
