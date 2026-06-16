# ════════════════════════════════════════════════════════
#  DataBridge AI — Dataset activation pipeline
# ════════════════════════════════════════════════════════
from __future__ import annotations

import hashlib

import pandas as pd
import streamlit as st

from core.session import reset_file_state
from modules.data_mapper import auto_map_columns
from modules.quality_engine import run_quality_engine


def _df_fingerprint(df: pd.DataFrame) -> str:
    """
    Fast stable fingerprint of a DataFrame used as the @st.cache_data hash key.
    pd.util.hash_pandas_object is vectorised and much faster than hashing the
    full serialised bytes, but still unique enough for cache invalidation.
    """
    row_hashes = pd.util.hash_pandas_object(df, index=True).values
    return hashlib.md5(row_hashes.tobytes()).hexdigest()


def activate_dataset(parsed_df: pd.DataFrame, clean_report: dict, display_name: str | None = None) -> None:
    """
    Set a parsed dataframe as the active dataset and run the standard pipeline.

    Memory notes vs the previous build:
    - Only ONE working copy is kept (hdf and df point to the same object).
    - standard_df is set to None and populated lazily only if the user uses it.
    - A fingerprint hash is stored so run_quality_engine (cached with
      @st.cache_data) can skip re-computation when the data hasn't changed.
    """
    if parsed_df is None or parsed_df.empty:
        raise ValueError("The imported dataset is empty.")

    reset_file_state()

    working_df = parsed_df.copy()

    # Single copy — both df and hdf point to the same object.
    # Pages that mutate hdf must call .copy() themselves before modifying.
    st.session_state.df          = working_df
    st.session_state.hdf         = working_df      # shared reference, not a second copy
    st.session_state.standard_df = None            # populated lazily on first use
    st.session_state.data_clean_report = clean_report or {}

    st.session_state.file_name = (
        display_name
        or (clean_report or {}).get("source_name")
        or (clean_report or {}).get("table_selected")
        or "Imported dataset"
    )

    # Store fingerprint so cached quality engine can detect data changes cheaply
    st.session_state["df_fingerprint"] = _df_fingerprint(working_df)

    mappings = auto_map_columns(tuple(working_df.columns))
    st.session_state.mapping_confidence = mappings
    st.session_state.column_mappings    = {col: grp for col, (grp, _) in mappings.items()}

    # run_quality_engine is decorated with @st.cache_data — the fingerprint
    # in session state is informational; Streamlit hashes the df arg itself.
    st.session_state.quality_report  = run_quality_engine(working_df)
    st.session_state.show_import_report = True
