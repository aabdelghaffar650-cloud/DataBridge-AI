<<<<<<< HEAD

=======
>>>>>>> c8e725118d9c65808b1a67b5349827ad4e22458d
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
<<<<<<< HEAD
=======
from core.i18n        import t, LANGUAGE_OPTIONS
>>>>>>> c8e725118d9c65808b1a67b5349827ad4e22458d
from modules.import_engine  import smart_parse_file, SUPPORTED_FILE_TYPES, read_sqlalchemy_query
from core.dataset     import activate_dataset
from ui.styles  import inject_css
from ui.header  import render_header
from ui.sidebar import render_sidebar
from core.runtime_guard import ensure_user_data_writable
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
    """Show a detailed, human-readable summary of what the import engine did."""
    steps = report.get("cleaning_steps", [])

    SEV_STYLE = {
        "info":    ("🟢", "#6bff8e", "#0d1a12", "#1e3a2a"),
        "warning": ("🟡", "#ffb86b", "#1a130d", "#3a2a1e"),
        "removed": ("🔴", "#ff6b6b", "#1a0d0d", "#3a1e1e"),
    }

    # ── Header banner ──
    sheets = report.get("sheets_found", [])
    tables = report.get("tables_found", [])
    source_type = report.get("source_type") or "File"
    sheet_info = (f"📑 {len(sheets)} sheets — loaded **{report.get('sheet_selected')}**"
                  if len(sheets) > 1 else "")
    table_info = (f"🗄️ {len(tables)} tables/views — loaded **{report.get('table_selected')}**"
                  if tables else "")
    source_info = table_info or sheet_info or str(report.get("sheet_selected") or source_type)
    hrow = report.get("header_row", 0)
    hrow_info = f"🎯 Header auto-detected at row {hrow}" if hrow > 0 else ""

    st.markdown(f"""
    <div style="background:#0d0d1f;border:1px solid #2a2a4e;border-left:4px solid #7c6aff;
         border-radius:12px;padding:1rem 1.4rem;margin:1rem 0;">
      <div style="font-size:.75rem;color:#7c6aff;text-transform:uppercase;
           letter-spacing:.1em;margin-bottom:.5rem;">📋 Import & Auto-Clean Report</div>
      <div style="display:flex;gap:2rem;flex-wrap:wrap;">
        <div><span style="color:#555;font-size:.78rem;">Final shape</span><br>
          <b style="color:#e0e0f0">{df.shape[0]:,} rows × {df.shape[1]} columns</b></div>
        <div><span style="color:#555;font-size:.78rem;">Remaining nulls</span><br>
          <b style="color:#ffb86b">{int(df.isnull().sum().sum()):,} cells</b></div>
        <div><span style="color:#555;font-size:.78rem;">Cleaning actions</span><br>
          <b style="color:#e0e0f0">{len(steps)}</b></div>
        <div><span style='color:#555;font-size:.78rem;'>Source</span><br><b style='color:#e0e0f0'>{source_info}</b></div>
        {"<div><span style='color:#555;font-size:.78rem;'>Header row</span><br><b style='color:#e0e0f0'>" + hrow_info + "</b></div>" if hrow_info else ""}
      </div>
    </div>
    """, unsafe_allow_html=True)

    if not steps:
        st.markdown(
            '<div class="success-box">✅ No cleaning was needed — data was already clean.</div>',
            unsafe_allow_html=True,
        )
        return

    # ── Step-by-step breakdown ──
    for step in steps:
        # Handle both old format (str) and new format (dict) gracefully
        if isinstance(step, str):
            action, detail, count, sev = "🔧 Auto-clean", step, 0, "info"
        else:
            action = step.get("action", "")
            detail = step.get("detail", "")
            count  = step.get("count", 0)
            sev    = step.get("severity", "info")
        emoji, color, bg, border = SEV_STYLE.get(sev, SEV_STYLE["info"])

        st.markdown(f"""
        <div style="background:{bg};border:1px solid {border};border-left:3px solid {color};
             border-radius:8px;padding:.7rem 1rem;margin:.35rem 0;
             display:flex;align-items:flex-start;gap:1rem;">
          <div style="font-size:1.1rem;min-width:1.5rem">{emoji}</div>
          <div style="flex:1">
            <div style="font-size:.82rem;font-weight:600;color:{color};margin-bottom:.2rem;">
              {action}
              <span style="font-size:.72rem;color:#555;margin-left:.5rem;">({count:,} affected)</span>
            </div>
            <div style="font-size:.78rem;color:#aaa;line-height:1.5;">{detail}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
#  FILE UPLOAD SCREEN (shown when no file loaded)
# ════════════════════════════════════════════════════════
if st.session_state.df is None:
    with st.sidebar:
<<<<<<< HEAD
        with st.expander("Account settings", expanded=False):
            render_change_password_form(prefix="no_file")
        if st.button("Logout", use_container_width=True, key="logout_no_file"):
=======
        st.selectbox(
            "🌐 Language / اللغة",
            list(LANGUAGE_OPTIONS.keys()),
            format_func=lambda code: LANGUAGE_OPTIONS[code],
            key="language",
            label_visibility="visible",
        )
        with st.expander("Account settings", expanded=False):
            render_change_password_form(prefix="no_file")
        if st.button(t("logout"), use_container_width=True, key="logout_no_file"):
>>>>>>> c8e725118d9c65808b1a67b5349827ad4e22458d
            logout()
            st.rerun()
        st.markdown("---")
        st.markdown("<div style='font-size:.68rem;color:#333;'>DataBridge AI © 2026</div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="text-align:center;padding:4rem 2rem;border:1px dashed #2a2a4e;
         border-radius:16px;background:#0d0d1a;margin:2rem 0;">
      <div style="font-size:3rem;margin-bottom:1rem;">🌉</div>
<<<<<<< HEAD
      <h3 style="color:#e0e0f0;margin-bottom:.5rem;">Upload your dataset</h3>
      <p style="color:#555;font-size:.85rem;">CSV and Excel files · Multi-sheet detection · Arabic text supported</p>
      <p style="color:#7c6aff;font-size:.78rem;">
        🧠 Smart parser auto-detects headers, cleans noise, and standardises data
=======
      <h3 style="color:#e0e0f0;margin-bottom:.5rem;">{t('upload_title')}</h3>
      <p style="color:#555;font-size:.85rem;">{t('upload_desc')}</p>
      <p style="color:#7c6aff;font-size:.78rem;">
        🧠 {t('upload_smart')}
>>>>>>> c8e725118d9c65808b1a67b5349827ad4e22458d
      </p>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("", type=SUPPORTED_FILE_TYPES, label_visibility="collapsed")

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
        if st.button("Connect & Load Query", use_container_width=True, key="initial_db_load"):
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
nav = render_sidebar()
df  = st.session_state.df

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
    "delete_dedupe":       "_pages.delete_dedupe",
    "export":              "_pages.export_engine",
    "settings":            "_pages.settings",
    "ai_assistant":        "_pages.ai_assistant",
}


def _render_page(page_key: str, current_df) -> None:
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
        st.session_state.show_import_report = False
        st.rerun()
    st.markdown("---")

_render_page(nav, df)
