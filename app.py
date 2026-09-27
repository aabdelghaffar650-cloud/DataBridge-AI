# ════════════════════════════════════════════════════════
#  DataBridge AI — Entry Point
#  Run: streamlit run app.py
# ════════════════════════════════════════════════════════
import sys
import os
import importlib

# Ensure project root is on the path (needed when packaged as exe)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

from config.settings  import PAGE_CONFIG
from core.session     import init_session_state
from core.auth        import render_login_gate, logout, render_change_password_form
from modules.import_engine  import smart_parse_file, SUPPORTED_FILE_TYPES, read_sqlalchemy_query
from core.dataset     import (
    activate_dataset,
    get_working_dataframe,
    reconcile_working_state,
    update_dataset_context,
)
from core.security    import safe_html
from ui.import_review import render_import_safety_review
from ui.styles  import inject_css
from ui.header  import render_header
from ui.sidebar import render_sidebar
from core.runtime_guard import ensure_user_data_writable
from core.team_access import (
    PERM_DATA_IMPORT,
    PERM_TEAM_MANAGE,
    can_access_page,
    has_permission,
)
ensure_user_data_writable()

# ── Page config (must be first Streamlit call) ──
st.set_page_config(**PAGE_CONFIG)

# ── CSS + Session + Login ──
inject_css()
init_session_state()

if not render_login_gate():

    st.stop()

# ── Header ──
render_header()

# ════════════════════════════════════════════════════════
#  IMPORT SUMMARY REPORT
# ════════════════════════════════════════════════════════
def _render_import_report(report: dict, df) -> None:
    """Show the structural import log and explicit safe-review recommendations."""
    steps = report.get("cleaning_steps", []) or []
    proposals = report.get("proposed_import_actions", []) or []

    sev_style = {
        "info": ("🟢", "#6bff8e", "#0d1a12", "#1e3a2a"),
        "warning": ("🟡", "#ffb86b", "#1a130d", "#3a2a1e"),
        "removed": ("🔴", "#ff6b6b", "#1a0d0d", "#3a1e1e"),
    }

    sheets = report.get("sheets_found", [])
    tables = report.get("tables_found", [])
    source_type = report.get("source_type") or "File"
    if tables:
        source_info = (
            f"{len(tables)} tables/views — loaded "
            f"{report.get('table_selected') or 'selected source'}"
        )
    elif len(sheets) > 1:
        source_info = (
            f"{len(sheets)} sheets — loaded "
            f"{report.get('sheet_selected') or 'selected sheet'}"
        )
    else:
        source_info = str(report.get("sheet_selected") or source_type)

    raw_shape = report.get("raw_shape") or st.session_state.get("raw_shape")
    raw_shape_text = (
        f"{raw_shape[0]:,} rows × {raw_shape[1]} columns"
        if isinstance(raw_shape, (tuple, list)) and len(raw_shape) == 2
        else "Protected source available"
    )
    hrow = int(report.get("header_row", 0) or 0)

    st.markdown(
        f"""
        <div style="background:#0d0d1f;border:1px solid #2a2a4e;border-left:4px solid #7c6aff;
             border-radius:12px;padding:1rem 1.4rem;margin:1rem 0;">
          <div style="font-size:.75rem;color:#7c6aff;text-transform:uppercase;
               letter-spacing:.1em;margin-bottom:.5rem;">🛡️ Safe Import Report</div>
          <div style="font-size:.78rem;color:#9a9ab0;margin-bottom:.8rem;">
            No cell-value conversion, type coercion, or row/column deletion was applied automatically.
          </div>
          <div style="display:flex;gap:2rem;flex-wrap:wrap;">
            <div><span style="color:#555;font-size:.78rem;">Protected source</span><br>
              <b style="color:#e0e0f0">{safe_html(raw_shape_text)}</b></div>
            <div><span style="color:#555;font-size:.78rem;">Working copy</span><br>
              <b style="color:#e0e0f0">{df.shape[0]:,} rows × {df.shape[1]} columns</b></div>
            <div><span style="color:#555;font-size:.78rem;">Automatic value changes</span><br>
              <b style="color:#6bff8e">0</b></div>
            <div><span style="color:#555;font-size:.78rem;">Pending review actions</span><br>
              <b style="color:#ffb86b">{len(proposals)}</b></div>
            <div><span style="color:#555;font-size:.78rem;">Source</span><br>
              <b style="color:#e0e0f0">{safe_html(source_info)}</b></div>
            {f"<div><span style='color:#555;font-size:.78rem;'>Header row</span><br><b style='color:#e0e0f0'>{hrow}</b></div>" if hrow > 0 else ""}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if steps:
        st.markdown("**Applied structural/source steps**")
        for step in steps:
            if isinstance(step, str):
                action, detail, count, severity = "Structural import step", step, 0, "info"
            else:
                action = str(step.get("action", "Import step"))
                detail = str(step.get("detail", ""))
                count = int(step.get("count", 0) or 0)
                severity = str(step.get("severity", "info"))
            emoji, color, bg, border = sev_style.get(
                severity,
                sev_style["info"],
            )
            st.markdown(
                f"""
                <div style="background:{bg};border:1px solid {border};border-left:3px solid {color};
                     border-radius:8px;padding:.7rem 1rem;margin:.35rem 0;
                     display:flex;align-items:flex-start;gap:1rem;">
                  <div style="font-size:1.1rem;min-width:1.5rem">{emoji}</div>
                  <div style="flex:1">
                    <div style="font-size:.82rem;font-weight:600;color:{color};margin-bottom:.2rem;">
                      {safe_html(action)}
                      <span style="font-size:.72rem;color:#555;margin-left:.5rem;">({count:,} affected)</span>
                    </div>
                    <div style="font-size:.78rem;color:#aaa;line-height:1.5;">{safe_html(detail)}</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    render_import_safety_review(
        df,
        key_prefix="main_import_review",
        heading=True,
    )


# ════════════════════════════════════════════════════════
#  FILE UPLOAD SCREEN (shown when no file loaded)
# ════════════════════════════════════════════════════════
if st.session_state.df is None:
    with st.sidebar:
        with st.expander("Account settings", expanded=False):
            render_change_password_form(prefix="no_file")
        control_pages = [
            ("model_governance", "Model Governance"),
            ("monitoring_scheduler", "Scheduled Monitoring"),
            ("deployment_api", "Deployment API"),
            ("remote_model_registry", "Remote Model Registry"),
            ("team_admin", "Team Administration"),
            ("settings", "Settings"),
        ]
        for page_key, label in control_pages:
            if can_access_page(page_key, session_state=st.session_state):
                if st.button(label, width="stretch", key=f"no_file_control_{page_key}"):
                    st.session_state["pre_dataset_page"] = page_key
                    st.rerun()
        if has_permission(PERM_DATA_IMPORT, session_state=st.session_state):
            if st.button("Upload / Connect Data", width="stretch", key="no_file_upload_home"):
                st.session_state["pre_dataset_page"] = "upload"
                st.rerun()
        if st.button("Logout", width="stretch", key="logout_no_file"):
            logout()
            st.rerun()
        st.markdown("---")
        st.markdown("<div style='font-size:.68rem;color:#333;'>DataBridge AI © 2026</div>", unsafe_allow_html=True)

    pre_page = str(st.session_state.get("pre_dataset_page") or "upload")
    pre_dataset_modules = {
        "team_admin": "_pages.team_admin",
        "settings": "_pages.settings",
        "model_governance": "_pages.model_governance",
        "monitoring_scheduler": "_pages.monitoring_scheduler",
        "deployment_api": "_pages.deployment_api",
        "remote_model_registry": "_pages.remote_model_registry",
    }
    if pre_page in pre_dataset_modules:
        if not can_access_page(pre_page, session_state=st.session_state):
            st.error("Your team role is not authorized to open this control page.")
        else:
            module = importlib.import_module(pre_dataset_modules[pre_page])
            module.render(None)
        st.stop()

    st.markdown(f"""
    <div style="text-align:center;padding:4rem 2rem;border:1px dashed #2a2a4e;
         border-radius:16px;background:#0d0d1a;margin:2rem 0;">
      <div style="font-size:3rem;margin-bottom:1rem;">🌉</div>
      <h3 style="color:#e0e0f0;margin-bottom:.5rem;">Upload your dataset</h3>
      <p style="color:#555;font-size:.85rem;">CSV and Excel files · Multi-sheet detection · Arabic text supported</p>
      <p style="color:#7c6aff;font-size:.78rem;">
        🛡️ Safe parser protects the source and waits for approval before value changes
      </p>
    </div>
    """, unsafe_allow_html=True)

    if not has_permission(PERM_DATA_IMPORT, session_state=st.session_state):
        st.info("Your team role has read-only access. Ask an Analyst, Data Scientist, or Admin to load a dataset for an editing session.")
        st.stop()

    uploaded = st.file_uploader("Upload dataset file", type=SUPPORTED_FILE_TYPES, label_visibility="collapsed")

    if uploaded:
        try:
            parsed_df, clean_report = smart_parse_file(uploaded)

            activate_dataset(parsed_df, clean_report, display_name=uploaded.name)
            st.rerun()

        except Exception as exc:
            st.error(f"Error: {exc}")


    with st.expander("🔌 Database Connector (SQL)", expanded=False):
        st.markdown(
            "<div class='info-box'>Run a read-only <b>SELECT</b> query from SQLite/PostgreSQL/MySQL using a SQLAlchemy URL.</div>",
            unsafe_allow_html=True,
        )
        db_url = st.text_input(
            "Connection URL",
            placeholder="sqlite:///C:/data/my_database.db  |  postgresql+psycopg2://user:pass@host:5432/db",
            type="password",
            key="initial_db_url",
        )
        db_query = st.text_area(
            "SELECT query",
            value="SELECT * FROM your_table LIMIT 1000",
            height=120,
            key="initial_db_query",
        )
        if st.button("Connect & Load Query", width="stretch", key="initial_db_load"):
            try:
                parsed_df, clean_report = read_sqlalchemy_query(db_url, db_query)
                activate_dataset(parsed_df, clean_report, display_name="Database query")
                st.rerun()
            except Exception as exc:
                st.error(f"Database import error: {exc}")

    st.stop()


# ════════════════════════════════════════════════════════
#  SIDEBAR + NAVIGATION
# ════════════════════════════════════════════════════════
# Finalize any page mutation from the previous Streamlit run before rendering
# status cards, Undo/Redo controls, quality results, or another page.
reconcile_working_state()
nav = render_sidebar()
df  = get_working_dataframe()

# ── Lazy page modules ─────────────────────────────────
# Heavy pages such as ML Studio and Feature Engineering are imported only when opened.
PAGE_MODULES = {
    "overview":            "_pages.overview",
    "data_sources":        "_pages.data_sources",
    "data_mapper":         "_pages.data_mapper",
    "quality_engine":      "_pages.quality_engine",
    "kpi_tracker":         "_pages.kpi_tracker",
    "filter_search":       "_pages.filter_search",
    "cleaning":            "_pages.cleaning",
    "data_types":          "_pages.data_types",
    "replace_values":      "_pages.replace_values",
    "feature_engineering": "_pages.feature_engineering",
    "visualization":       "_pages.visualization",
    "outlier_detection":   "_pages.outlier_detection",
    "ml_studio":           "_pages.ml_studio",
    "explainability_studio":"_pages.explainability_studio",
    "prediction_studio":   "_pages.prediction_studio",
    "model_monitoring":    "_pages.model_monitoring",
    "monitoring_scheduler": "_pages.monitoring_scheduler",
    "deployment_api":      "_pages.deployment_api",
    "remote_model_registry": "_pages.remote_model_registry",
    "retraining_workflow": "_pages.retraining_workflow",
    "model_governance":    "_pages.model_governance",
    "delete_dedupe":       "_pages.delete_dedupe",
    "export":              "_pages.export_engine",
    "settings":            "_pages.settings",
    "team_admin":          "_pages.team_admin",
    "ai_assistant":        "_pages.ai_assistant",
}


def _render_page(page_key: str, current_df) -> None:
    if not can_access_page(page_key, session_state=st.session_state):
        st.error("Your team role is not authorized to open this page.")
        return
    module_path = PAGE_MODULES.get(page_key)
    if not module_path:
        st.error(f"Page not found: {page_key}")
        return
    module = importlib.import_module(module_path)
    render_fn = getattr(module, "render", None)
    if not callable(render_fn):
        st.error(f"Page has no render() function: {module_path}")
        return
    render_fn(current_df)

# ── Show import report — persists until user dismisses it ──
if st.session_state.get("show_import_report"):
    _render_import_report(st.session_state.data_clean_report, df)
    if st.button("✕ Dismiss report", key="dismiss_report"):
        try:
            update_dataset_context(show_import_report=False)
            st.rerun()
        except Exception as exc:
            st.error(f"Could not update the import report state: {exc}")
    st.markdown("---")

_render_page(nav, df)
