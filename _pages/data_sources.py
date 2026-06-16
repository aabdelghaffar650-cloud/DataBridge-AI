# ════════════════════════════════════════════════════════
#  DataBridge AI — Data Sources Page
# ════════════════════════════════════════════════════════
import streamlit as st
import pandas as pd

from core.dataset import activate_dataset
from core.security import safe_html
from modules.import_engine import (
    SUPPORTED_FILE_TYPES,
    MAX_SQL_ROWS,
    smart_parse_file,
    smart_parse_json,
    read_sqlalchemy_query,
)


def _source_badge(label: str, detail: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card" style="margin-bottom:.75rem;">
            <div class="label">{safe_html(label)}</div>
            <div style="font-size:.85rem;color:#c9c9df;line-height:1.55;">{safe_html(detail)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _handle_json_array_selection(uploaded_file) -> None:
    """
    Two-step JSON import: parse first, then let the user pick the array
    if the file contains multiple top-level arrays.
    State is stored under 'ds_json_candidates' and 'ds_json_file_name'.
    """
    # ── Step 1: initial parse (no array chosen yet) ──────────────────
    if st.session_state.get("ds_json_candidates") is None:
        try:
            parsed_df, clean_report = smart_parse_file(uploaded_file)
            candidates = clean_report.get("json_array_candidates")

            if candidates and len(candidates) > 1:
                # Store candidates and raw file bytes so we can re-parse after selection
                uploaded_file.seek(0)
                st.session_state["ds_json_candidates"] = candidates
                st.session_state["ds_json_raw"] = uploaded_file.read()
                st.session_state["ds_json_file_name"] = uploaded_file.name
                st.rerun()
            else:
                # Single array or root array — load directly
                activate_dataset(parsed_df, clean_report, display_name=uploaded_file.name)
                st.success(f"Loaded {uploaded_file.name} successfully.")
                st.rerun()
        except Exception as exc:
            st.error(f"Import error: {exc}")
        return

    # ── Step 2: present selection UI ────────────────────────────────
    candidates = st.session_state["ds_json_candidates"]
    file_name  = st.session_state.get("ds_json_file_name", "file.json")

    st.markdown(
        f"<div class='info-box'>The file <b>{safe_html(file_name)}</b> contains "
        f"<b>{len(candidates)}</b> arrays. Select the one you want to import:</div>",
        unsafe_allow_html=True,
    )

    options = [f"{c['key']}  ({c['length']:,} records)" for c in candidates]
    chosen_label = st.radio("Choose array to import:", options, index=0, key="ds_json_array_radio")
    chosen_key = candidates[options.index(chosen_label)]["key"]

    col_load, col_cancel = st.columns([1, 1])
    with col_load:
        if st.button("Load selected array", type="primary", use_container_width=True, key="ds_json_confirm"):
            try:
                import io
                raw = st.session_state["ds_json_raw"]
                fake_file = io.BytesIO(raw)
                fake_file.name = file_name  # type: ignore[attr-defined]
                fake_file.size = len(raw)   # type: ignore[attr-defined]
                # Re-parse and pick the chosen key
                parsed_df, clean_report = smart_parse_json(fake_file, selected_key=chosen_key)
                activate_dataset(parsed_df, clean_report, display_name=file_name)
                _clear_json_state()
                st.success(f"Loaded array '{chosen_key}' from {file_name}.")
                st.rerun()
            except Exception as exc:
                st.error(f"Import error: {exc}")
    with col_cancel:
        if st.button("Cancel", use_container_width=True, key="ds_json_cancel"):
            _clear_json_state()
            st.rerun()


def _clear_json_state() -> None:
    for key in ("ds_json_candidates", "ds_json_raw", "ds_json_file_name"):
        st.session_state.pop(key, None)


def render(df: pd.DataFrame) -> None:
    st.markdown(
        "<div class='section-header'><div class='icon'>🔌</div><h2>Data Sources</h2><div class='count'>Files · SQLite · SQL</div></div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='info-box'>Import a new source and send it through the same DataBridge pipeline: mapping, quality checks, cleaning, analysis, and export.</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        _source_badge("FILE SOURCES", "CSV, Excel, JSON, JSON Lines, Parquet, SQLite database files")
    with c2:
        _source_badge("SQL CONNECTOR", "Run read-only SELECT queries using SQLAlchemy URLs")
    with c3:
        _source_badge("SAFE IMPORT", "Raw .sql dumps are blocked. Only SELECT queries are allowed.")

    tab_file, tab_sql = st.tabs(["File Upload", "Database Connector"])

    # ── FILE UPLOAD TAB ──────────────────────────────────────────────
    with tab_file:
        # If we're mid JSON selection, show the picker and skip the uploader
        if st.session_state.get("ds_json_candidates") is not None:
            _handle_json_array_selection(None)
            return

        st.markdown("#### Upload a data file")
        uploaded = st.file_uploader(
            "Supported formats",
            type=SUPPORTED_FILE_TYPES,
            help="CSV, XLSX, XLS, JSON, JSONL, NDJSON, Parquet, DB, SQLite, SQLite3",
            key="data_sources_file_upload",
        )
        if uploaded:
            st.markdown(
                f"<div class='info-box'>Selected source: <b>{safe_html(uploaded.name)}</b></div>",
                unsafe_allow_html=True,
            )
            if st.button("Load this source", type="primary", use_container_width=True, key="data_sources_load_file"):
                # JSON files may need the two-step array selector
                if uploaded.name.lower().endswith(".json"):
                    _handle_json_array_selection(uploaded)
                else:
                    try:
                        parsed_df, clean_report = smart_parse_file(uploaded)
                        activate_dataset(parsed_df, clean_report, display_name=uploaded.name)
                        st.success(f"Loaded {uploaded.name} successfully.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Import error: {exc}")

    # ── DATABASE CONNECTOR TAB ───────────────────────────────────────
    with tab_sql:
        st.markdown("#### Connect using SQLAlchemy")
        st.markdown(
            f"<div class='info-box'>Use a read-only <b>SELECT</b> query. "
            f"Examples: <code>sqlite:///C:/data/app.db</code>, "
            f"<code>postgresql+psycopg2://user:pass@host:5432/db</code>, "
            f"<code>mysql+pymysql://user:pass@host:3306/db</code>.<br>"
            f"<b>Note:</b> Results are capped at <b>{MAX_SQL_ROWS:,} rows</b>. "
            f"Add a <code>LIMIT</code> clause to control result size.</div>",
            unsafe_allow_html=True,
        )
        connection_url = st.text_input(
            "Connection URL",
            type="password",
            placeholder="sqlite:///C:/data/my_database.db",
            key="data_sources_db_url",
        )
        query = st.text_area(
            "SELECT query",
            value="SELECT * FROM your_table LIMIT 1000",
            height=160,
            key="data_sources_db_query",
        )
        if st.button("Run query & load dataset", type="primary", use_container_width=True, key="data_sources_load_sql"):
            try:
                parsed_df, clean_report = read_sqlalchemy_query(connection_url, query)
                activate_dataset(parsed_df, clean_report, display_name="Database query")
                # Surface truncation warning prominently if it happened
                if clean_report.get("sql_rows_truncated"):
                    st.warning(
                        f"⚠️ Result truncated at {MAX_SQL_ROWS:,} rows. "
                        "Add a LIMIT clause to your query to avoid this."
                    )
                else:
                    st.success("Database query loaded successfully.")
                st.rerun()
            except Exception as exc:
                st.error(f"Database import error: {exc}")

    st.markdown("---")
    st.markdown("#### Current active dataset")
    if st.session_state.df is not None:
        st.markdown(
            f"""
            <div class='sidebar-status' style='margin-top:.5rem;'>
                <div class='status-label'>ACTIVE DATASET</div>
                <div class='status-file'>{safe_html(st.session_state.get('file_name') or 'Untitled')}</div>
                <div class='status-shape'>{st.session_state.df.shape[0]:,} rows × {st.session_state.df.shape[1]} cols</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
