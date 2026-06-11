# ════════════════════════════════════════════════════════
#  DataBridge AI — Shared Vectorized Utilities
# ════════════════════════════════════════════════════════
from typing import Dict, Any, List, Tuple

import numpy  as np
import pandas as pd


# ── Secure Filtering ─────────────────────────────────────────────────────────
def secure_filter_dataframe(
    df: pd.DataFrame,
    conditions: Dict[str, Any],
) -> pd.DataFrame:
    mask = pd.Series([True] * len(df), index=df.index)
    for col, condition in conditions.items():
        if col not in df.columns:
            continue
        if isinstance(condition, dict):
            op  = condition.get("op", "eq")
            val = condition.get("value")
            if op == "eq":       mask &= df[col] == val
            elif op == "ne":     mask &= df[col] != val
            elif op == "gt":     mask &= df[col] > val
            elif op == "lt":     mask &= df[col] < val
            elif op == "ge":     mask &= df[col] >= val
            elif op == "le":     mask &= df[col] <= val
            elif op == "contains" and isinstance(val, str):
                mask &= df[col].astype(str).str.contains(val, case=False, na=False, regex=False)
            elif op == "between":
                mask &= df[col].between(val[0], val[1])
        else:
            mask &= df[col] == condition
    return df[mask].copy()


def secure_multi_condition_filter(
    df: pd.DataFrame,
    conditions: List[Tuple[str, str, Any]],
) -> pd.DataFrame:
    mask = pd.Series([True] * len(df), index=df.index)
    for col, op, val in conditions:
        if col not in df.columns:
            continue
        s = df[col]
        if op == "==":      mask &= s == val
        elif op == "!=":    mask &= s != val
        elif op == ">":     mask &= s > val
        elif op == "<":     mask &= s < val
        elif op == ">=":    mask &= s >= val
        elif op == "<=":    mask &= s <= val
        elif op == "contains" and isinstance(val, str):
            mask &= s.astype(str).str.contains(val, case=False, na=False, regex=False)
        elif op == "in" and isinstance(val, (list, tuple, set)):
            mask &= s.isin(val)
    return df[mask].copy()


# ── Outlier Detection ─────────────────────────────────────────────────────────
def vectorized_outlier_detection(
    df: pd.DataFrame,
    col: str,
    method: str = "iqr",
    threshold: float = 3.0,
) -> pd.Series:
    s = df[col].dropna()
    if method == "iqr":
        Q1, Q3  = s.quantile(0.25), s.quantile(0.75)
        IQR     = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        return (df[col] < lower) | (df[col] > upper)
    else:  # z-score
        mean, std = s.mean(), s.std()
        if std == 0:
            return pd.Series([False] * len(df), index=df.index)
        z = np.abs((df[col] - mean) / std)
        return (z > threshold) & pd.notna(df[col])


# ── Fill Nulls ────────────────────────────────────────────────────────────────
def vectorized_fill_nulls(
    df: pd.DataFrame,
    strategy: str,
    columns: List[str] = None,
) -> pd.DataFrame:
    df_copy     = df.copy()
    target_cols = columns if columns else df.select_dtypes(include="number").columns.tolist()
    for col in target_cols:
        if col not in df_copy.columns:
            continue
        s = df_copy[col]
        if strategy == "mean":
            df_copy[col] = s.fillna(s.mean())
        elif strategy == "median":
            df_copy[col] = s.fillna(s.median())
        elif strategy == "mode":
            mode_val = s.mode()
            if not mode_val.empty:
                df_copy[col] = s.fillna(mode_val[0])
        elif strategy == "forward":
            df_copy[col] = s.ffill()
        elif strategy == "backward":
            df_copy[col] = s.bfill()
    return df_copy
