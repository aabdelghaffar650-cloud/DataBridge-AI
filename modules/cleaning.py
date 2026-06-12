# ════════════════════════════════════════════════════════
#  DataBridge AI — Cleaning Module
# ════════════════════════════════════════════════════════
from typing import List, Optional
import pandas as pd
from core.utils import vectorized_fill_nulls


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise column names: strip, lower, replace spaces."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def strip_all_strings(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip().replace("nan", pd.NA)
    return df


def fill_nulls(
    df: pd.DataFrame,
    strategy: str,
    columns: Optional[List[str]] = None,
    custom_value=None,
) -> pd.DataFrame:
    """
    strategy: mean | median | mode | forward | backward | custom
    """
    if strategy == "custom" and custom_value is not None:
        df = df.copy()
        target = columns or df.columns.tolist()
        for col in target:
            if col in df.columns:
                df[col] = df[col].fillna(custom_value)
        return df
    return vectorized_fill_nulls(df, strategy, columns)


def drop_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    return df.drop_duplicates(subset=subset).reset_index(drop=True)


def coerce_to_numeric(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df
