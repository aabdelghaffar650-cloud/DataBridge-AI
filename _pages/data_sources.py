# ════════════════════════════════════════════════════════
#  DataBridge AI — Data Sources Page
# ════════════════════════════════════════════════════════
import streamlit as st
import pandas as pd

from core.dataset import activate_dataset, deactivate_dataset
from core.security import safe_html
from ui.import_review import render_import_safety_review
from modules.import_engine import (
    SUPPORTED_FILE_TYPES,
    MAX_SQL_ROWS,
    DEFAULT_SQL_ROWS,
    DEFAULT_SQL_QUERY_TIMEOUT_SECONDS,
    MAX_SQL_QUERY_TIMEOUT_SECONDS,
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
        if st.button("Load selected array", type="primary", width="stretch", key="ds_json_confirm"):
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
        if st.button("Cancel", width="stretch", key="ds_json_cancel"):
            _clear_json_state()
            st.rerun()


def _clear_json_state() -> None:
    for key in ("ds_json_candidates", "ds_json_raw", "ds_json_file_name"):
        st.session_state.pop(key, None)


def render(df: pd.DataFrame) -> None:
    # File-uploader values are widget-owned. Defer removal by one rerun so this
    # code can clear the uploader state before the widget is recreated, then
    # release all dataset-bound state through the authoritative reset path.
    if st.session_state.pop("_data_sources_remove_pending", False):
        st.session_state.pop("data_sources_file_upload", None)
        _clear_json_state()
        st.session_state.pop("data_sources_remove_confirm", None)
        deactivate_dataset()
        st.rerun()

    st.markdown(
        "<div class='section-header'><div class='icon'>🔌</div><h2>Data Sources</h2><div class='count'>Files · SQLite · SQL</div></div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='info-box'><b>Safe import policy:</b> the source is protected and no value conversion, type coercion, or deletion is applied without approval.</div>",
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        _source_badge("FILE SOURCES", "CSV, Excel, JSON, JSON Lines, Parquet, SQLite database files")
    with c2:
        _source_badge("SQL CONNECTOR", "Run read-only SELECT queries using SQLAlchemy URLs")
    with c3:
        _source_badge("SAFE IMPORT", "Source values stay unchanged until you approve review actions. Raw .sql dumps remain blocked.")

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
            if st.button("Load this source", type="primary", width="stretch", key="data_sources_load_file"):
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
        st.markdown("#### Secure read-only database connector")
        st.markdown(
            f"<div class='info-box'><b>Stage 4 safety:</b> DataBridge accepts one "
            f"validated SELECT/CTE statement, verifies a read-only database session, "
            f"applies a query timeout, and caps rows <b>before</b> pandas receives them.<br>"
            f"Supported secure dialects: <code>PostgreSQL</code>, <code>MySQL/MariaDB</code>, "
            f"and <code>SQLite</code>. Hard maximum: <b>{MAX_SQL_ROWS:,} rows</b>.<br>"
            f"Use a dedicated database account that has SELECT permissions only.</div>",
            unsafe_allow_html=True,
        )

        connection_url = st.text_input(
            "Connection URL",
            type="password",
            placeholder="postgresql+psycopg2://readonly_user:password@host:5432/db",
            key="data_sources_db_url",
            help="The password is masked in the interface and redacted from reports and connection errors.",
        )
        query = st.text_area(
            "SELECT query",
            value="SELECT * FROM your_table",
            height=160,
            key="data_sources_db_query",
            help="Comments and quoted text are handled safely. Multiple statements, writes, locking reads, and dangerous functions are blocked.",
        )

        sql_opt1, sql_opt2 = st.columns(2)
        with sql_opt1:
            row_limit = st.number_input(
                "Maximum rows to load",
                min_value=1,
                max_value=MAX_SQL_ROWS,
                value=DEFAULT_SQL_ROWS,
                step=1_000,
                key="data_sources_db_row_limit",
                help="The connector asks the database for at most this number plus one row to detect truncation.",
            )
        with sql_opt2:
            query_timeout = st.slider(
                "Query timeout (seconds)",
                min_value=5,
                max_value=MAX_SQL_QUERY_TIMEOUT_SECONDS,
                value=DEFAULT_SQL_QUERY_TIMEOUT_SECONDS,
                step=5,
                key="data_sources_db_timeout",
            )

        readonly_confirmed = st.checkbox(
            "I am using a dedicated read-only database account and understand that database permissions remain the final security boundary.",
            value=False,
            key="data_sources_db_readonly_confirm",
        )

        if st.button(
            "Run verified read-only query",
            type="primary",
            width="stretch",
            key="data_sources_load_sql",
        ):
            if not readonly_confirmed:
                st.error("Confirm that the database account is dedicated to read-only access before connecting.")
            else:
                try:
                    parsed_df, clean_report = read_sqlalchemy_query(
                        connection_url,
                        query,
                        row_limit=int(row_limit),
                        query_timeout_seconds=int(query_timeout),
                    )
                    dialect = clean_report.get("database_dialect", "database")
                    activate_dataset(
                        parsed_df,
                        clean_report,
                        display_name=f"Database query ({dialect})",
                    )
                    if clean_report.get("sql_rows_truncated"):
                        st.warning(
                            f"Result safely capped at {int(row_limit):,} rows before entering application memory."
                        )
                    else:
                        st.success(
                            f"Loaded {len(parsed_df):,} row(s) through a verified read-only session."
                        )
                    st.rerun()
                except Exception as exc:
                    st.error(f"Database import blocked or failed safely: {exc}")

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
        st.caption(
            "Remove releases the current Raw/Working dataset and its session history only. "
            "It never deletes the uploaded source file or changes an external database."
        )

        if not st.session_state.get("data_sources_remove_confirm", False):
            if st.button(
                "🗑️ Remove active dataset",
                width="stretch",
                key="data_sources_remove_active",
                help="Clear this dataset from the current DataBridge AI session.",
            ):
                st.session_state["data_sources_remove_confirm"] = True
                st.rerun()
        else:
            st.warning(
                "Remove the active dataset from this session? Working changes and Undo/Redo "
                "history for this dataset will be cleared. The original source file is not deleted."
            )
            remove_yes, remove_no = st.columns(2)
            with remove_yes:
                if st.button(
                    "Confirm remove",
                    type="primary",
                    width="stretch",
                    key="data_sources_remove_confirm_yes",
                ):
                    st.session_state["_data_sources_remove_pending"] = True
                    st.rerun()
            with remove_no:
                if st.button(
                    "Cancel",
                    width="stretch",
                    key="data_sources_remove_confirm_no",
                ):
                    st.session_state.pop("data_sources_remove_confirm", None)
                    st.rerun()

        st.markdown("---")
        render_import_safety_review(
            st.session_state.df,
            key_prefix="data_sources_import_review",
            heading=True,
        )
