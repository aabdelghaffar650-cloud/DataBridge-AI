# ════════════════════════════════════════════════════════
#  DataBridge AI — Session State Initialiser
#  Single source of truth — call once at app startup
# ════════════════════════════════════════════════════════
import streamlit as st
from core.history import SmartHistoryManager
from config.settings import DEFAULT_LANGUAGE


def init_session_state() -> None:
    """Initialise every session-state key exactly once."""
    defaults = {
        # ── Auth / Language ───────────────────────────
        "is_authenticated":  False,
        "current_user":      "",
        "language":          DEFAULT_LANGUAGE,

        # ── Core DataFrame ──────────────────────────────
        "df":                  None,
        "hdf":                 None,       # Active working DataFrame
        "standard_df":         None,       # Post-clean standardised DF
        "file_name":           None,

        # ── History ─────────────────────────────────────
        "history_manager":     SmartHistoryManager(),
        "df_history":          [],

        # ── Import / Cleaning ────────────────────────────
        "data_clean_report":   {},

        # ── Data Mapper ─────────────────────────────────
        "mapper_approved":     False,
        "column_mappings":     {},         # {raw_col: semantic_group}
        "mapping_confidence":  {},         # {raw_col: (group, score)}

        # ── Quality Engine ───────────────────────────────
        "quality_report":      {},

        # ── KPI Tracker ──────────────────────────────────
        "kpi_targets":         {},         # {col: {annual, quarterly, agg}}

        # ── AI ───────────────────────────────────────────
        "ai_mode":             "demo",
        "ai_engine":           None,
        "ai_messages":         [],
        "show_import_report":  False,
        "gemini_api_key":      "",
        "gemini_allow_data":   False,
        "gemini_mask_pii":     True,
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default


def save_history() -> None:
    """Push current df onto the history stack."""
    if st.session_state.df is not None:
        st.session_state.history_manager.push(st.session_state.df)
        st.session_state.df_history = list(st.session_state.history_manager.history)


def reset_file_state() -> None:
    """Clear all file-related state when a new file is loaded."""
    st.session_state.history_manager  = SmartHistoryManager()
    st.session_state.df_history        = []
    st.session_state.mapper_approved   = False
    st.session_state.column_mappings   = {}
    st.session_state.mapping_confidence = {}
    st.session_state.quality_report    = {}
    st.session_state.data_clean_report = {}
    st.session_state.kpi_targets       = {}
    st.session_state.ai_messages       = []
