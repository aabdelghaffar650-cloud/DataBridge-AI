"""Streamlit display compatibility helpers for DataBridge AI.

This module is deliberately display-only. It never mutates the protected raw
or working dataset. Mixed Python object columns are normalized only in a
shallow display copy before Streamlit/PyArrow serialization.
"""
from __future__ import annotations

import json
from datetime import date, datetime, time
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st


def _is_scalar_missing(value: Any) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    try:
        result = pd.isna(value)
    except Exception:
        return False
    return isinstance(result, (bool, np.bool_)) and bool(result)


def _value_family(value: Any) -> str:
    if _is_scalar_missing(value):
        return "missing"
    if isinstance(value, (bool, np.bool_)):
        return "boolean"
    if isinstance(value, (complex, np.complexfloating)):
        return "other"
    if isinstance(value, (int, float, np.integer, np.floating)):
        return "number"
    if isinstance(value, str):
        return "text"
    if isinstance(value, (bytes, bytearray, memoryview)):
        return "bytes"
    if isinstance(value, (datetime, pd.Timestamp, np.datetime64)):
        return "datetime"
    if isinstance(value, date):
        return "date"
    if isinstance(value, time):
        return "time"
    if isinstance(value, (dict, list, tuple, set, frozenset)):
        return "nested"
    return "other"


def _display_text(value: Any) -> Any:
    if _is_scalar_missing(value):
        return pd.NA
    if isinstance(value, (dict, list, tuple, set, frozenset)):
        try:
            normalized = list(value) if isinstance(value, (set, frozenset, tuple)) else value
            return json.dumps(normalized, ensure_ascii=False, sort_keys=isinstance(normalized, dict), default=str)
        except Exception:
            return str(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        try:
            return bytes(value).decode("utf-8", errors="replace")
        except Exception:
            return str(value)
    return str(value)


def _object_column_needs_text_normalization(series: pd.Series) -> bool:
    """Return True only when an object column is unsafe/ambiguous for Arrow."""
    if not pd.api.types.is_object_dtype(series.dtype):
        return False

    families: set[str] = set()
    # Scan until the first incompatible family appears; this prevents a late mixed value
    # from reaching Streamlit's automatic Arrow-repair path.
    for value in series.array:
        family = _value_family(value)
        if family == "missing":
            continue
        families.add(family)
        if family in {"nested", "other"} or len(families) > 1:
            return True

    # A homogeneous object column consisting of nested/unsupported values is unsafe too.
    return bool(families & {"nested", "other"})


def prepare_dataframe_for_display(data: pd.DataFrame) -> pd.DataFrame:
    """Return an Arrow-friendly display view without changing ``data``.

    Homogeneous object columns are left untouched. Mixed or nested object
    columns are replaced in a shallow copy with Python-backed StringDtype.
    This avoids Streamlit's automatic Arrow repair path and its warning while
    preserving the real DataFrame for cleaning, ML, export, and prediction.
    """
    if not isinstance(data, pd.DataFrame) or data.empty:
        return data

    unsafe_columns = [
        column
        for column in data.columns
        if _object_column_needs_text_normalization(data[column])
    ]
    if not unsafe_columns:
        return data

    display = data.copy(deep=False)
    for column in unsafe_columns:
        converted = data[column].map(_display_text, na_action=None)
        display[column] = pd.Series(
            pd.array(converted.tolist(), dtype="string[python]"),
            index=data.index,
            name=column,
        )
    return display


def safe_dataframe(data: Any, *args: Any, **kwargs: Any):
    """Display a DataFrame through Streamlit without mutating source values."""
    prepared = prepare_dataframe_for_display(data) if isinstance(data, pd.DataFrame) else data
    return st.dataframe(prepared, *args, **kwargs)
