# ════════════════════════════════════════════════════════
#  DataBridge AI — Professional Grouped Sidebar
# ════════════════════════════════════════════════════════
import streamlit as st

from config.settings import NAV_PAGE_KEYS
from core.auth       import logout
<<<<<<< HEAD
=======
from core.i18n       import LANGUAGE_OPTIONS, t
>>>>>>> c8e725118d9c65808b1a67b5349827ad4e22458d
from core.security   import safe_html
from modules.import_engine  import smart_parse_file, SUPPORTED_FILE_TYPES
from core.dataset    import activate_dataset
from ai import (
    DataBridgeAIEngine,
    GeminiCloudEngine,
    AnthropicCloudEngine,
    OllamaLocalEngine,
    DemoEngine,
)


NAV_GROUPS = [
    (
        "DATA",
<<<<<<< HEAD
        [
            ("overview", "Overview"),
            ("data_sources", "Data Sources"),
            ("data_mapper", "Data Mapper"),
            ("quality_engine", "Quality Engine"),
=======
        {
            "en": "DATA",
            "ar": "البيانات",
        },
        [
            ("overview", "Overview", "النظرة العامة"),
            ("data_sources", "Data Sources", "مصادر البيانات"),
            ("data_mapper", "Data Mapper", "ربط الأعمدة"),
            ("quality_engine", "Quality Engine", "جودة البيانات"),
>>>>>>> c8e725118d9c65808b1a67b5349827ad4e22458d
        ],
    ),
    (
        "TRANSFORM",
<<<<<<< HEAD
        [
            ("filter_search", "Filter & Search"),
            ("cleaning", "Clean Nulls"),
            ("data_types", "Data Types"),
            ("replace_values", "Replace Values"),
            ("feature_engineering", "Feature Engineering"),
=======
        {
            "en": "TRANSFORM",
            "ar": "المعالجة",
        },
        [
            ("filter_search", "Filter & Search", "فلترة وبحث"),
            ("cleaning", "Clean Nulls", "معالجة الفراغات"),
            ("data_types", "Data Types", "أنواع البيانات"),
            ("replace_values", "Replace Values", "استبدال القيم"),
            ("feature_engineering", "Feature Engineering", "إنشاء أعمدة"),
>>>>>>> c8e725118d9c65808b1a67b5349827ad4e22458d
        ],
    ),
    (
        "ANALYZE",
<<<<<<< HEAD
        [
            ("visualization", "Visualize"),
            ("outlier_detection", "Outlier Detection"),
            ("kpi_tracker", "KPI Tracker"),
            ("ml_studio", "ML Studio"),
            ("ai_assistant", "AI Assistant"),
=======
        {
            "en": "ANALYZE",
            "ar": "التحليل",
        },
        [
            ("visualization", "Visualize", "الرسوم البيانية"),
            ("outlier_detection", "Outlier Detection", "القيم الشاذة"),
            ("kpi_tracker", "KPI Tracker", "متابعة المؤشرات"),
            ("ml_studio", "ML Studio", "التعلم الآلي"),
            ("ai_assistant", "AI Assistant", "مساعد الذكاء الاصطناعي"),
>>>>>>> c8e725118d9c65808b1a67b5349827ad4e22458d
        ],
    ),
    (
        "MANAGE",
<<<<<<< HEAD
        [
            ("export", "Export"),
            ("delete_dedupe", "Delete & Dedupe"),
            ("settings", "Settings"),
=======
        {
            "en": "MANAGE",
            "ar": "الإدارة",
        },
        [
            ("export", "Export", "تصدير"),
            ("delete_dedupe", "Delete & Dedupe", "حذف وإزالة التكرار"),
            ("settings", "Settings", "الإعدادات"),
>>>>>>> c8e725118d9c65808b1a67b5349827ad4e22458d
        ],
    ),
]


<<<<<<< HEAD
def _render_nav_group(group_label: str, pages: list[tuple[str, str]], current_page: str) -> None:
    st.markdown(
        f"<div class='nav-group-label'>{safe_html(group_label)}</div>",
        unsafe_allow_html=True,
    )

    for page_key, label in pages:
=======
def _page_label(page_key: str, en: str, ar: str) -> str:
    language = st.session_state.get("language", "en")
    return ar if language == "ar" else en


def _group_label(labels: dict) -> str:
    language = st.session_state.get("language", "en")
    return labels.get(language, labels.get("en", ""))


def _render_nav_group(group_labels: dict, pages: list[tuple[str, str, str]], current_page: str) -> None:
    st.markdown(
        f"<div class='nav-group-label'>{safe_html(_group_label(group_labels))}</div>",
        unsafe_allow_html=True,
    )

    for page_key, en_label, ar_label in pages:
        label = _page_label(page_key, en_label, ar_label)
>>>>>>> c8e725118d9c65808b1a67b5349827ad4e22458d
        if page_key == current_page:
            st.markdown(
                f"<div class='nav-item active'>{safe_html(label)}</div>",
                unsafe_allow_html=True,
            )
        else:
            if st.button(label, key=f"nav_{page_key}", use_container_width=True):
                st.session_state["current_page"] = page_key
                st.rerun()


def _render_status_block() -> None:
    if st.session_state.df is None:
        return

    df = st.session_state.df
    file_name = safe_html(st.session_state.get("file_name") or "Untitled dataset")
    quality = st.session_state.get("quality_report", {}).get("quality_score")
    mapper_status = "Approved" if st.session_state.get("mapper_approved") else "Pending"
    mapper_class = "ok" if st.session_state.get("mapper_approved") else "warn"

    quality_html = ""
    if quality is not None:
        q_class = "ok" if quality >= 85 else "warn" if quality >= 60 else "danger"
        quality_html = f"<span class='mini-chip {q_class}'>Quality {quality}%</span>"

    st.markdown(
        f"""
        <div class='sidebar-status'>
            <div class='status-label'>OPEN DATASET</div>
            <div class='status-file' title='{file_name}'>{file_name}</div>
            <div class='status-shape'>{df.shape[0]:,} rows × {df.shape[1]} cols</div>
            <div class='status-chips'>
                {quality_html}
                <span class='mini-chip {mapper_class}'>Mapping {mapper_status}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> str:
    """Render full sidebar. Returns selected internal page key."""
    with st.sidebar:
        st.markdown(
            """
            <div class='sidebar-brand'>
                <div class='sidebar-logo'>DataBridge AI</div>
                <div class='sidebar-subtitle'>Data Intelligence Platform</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        current_page = st.session_state.get("current_page", NAV_PAGE_KEYS[0])
        if current_page not in NAV_PAGE_KEYS:
            current_page = NAV_PAGE_KEYS[0]
            st.session_state["current_page"] = current_page

        with st.container(key="sidebar_nav"):
<<<<<<< HEAD
            for group_label, pages in NAV_GROUPS:
                _render_nav_group(group_label, pages, current_page)

        st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)

        # ── Account ──────────────────────────────────────
=======
            for _, group_labels, pages in NAV_GROUPS:
                _render_nav_group(group_labels, pages, current_page)

        st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)

        # ── Language / Account ──────────────────────────
        st.selectbox(
            t("language"),
            list(LANGUAGE_OPTIONS.keys()),
            format_func=lambda code: LANGUAGE_OPTIONS[code],
            key="language",
            label_visibility="visible",
        )

>>>>>>> c8e725118d9c65808b1a67b5349827ad4e22458d
        user_name = st.session_state.get("current_user", "") or "admin"
        st.markdown(
            f"<div class='account-pill'>{safe_html(user_name)}</div>",
            unsafe_allow_html=True,
        )
<<<<<<< HEAD
        if st.button("Logout", use_container_width=True, key="logout_btn"):
=======
        if st.button(t("logout"), use_container_width=True, key="logout_btn"):
>>>>>>> c8e725118d9c65808b1a67b5349827ad4e22458d
            logout()
            st.rerun()

        # ── File Info ────────────────────────────────────
        _render_status_block()

        if st.session_state.df is not None:
            df = st.session_state.df
            c1, c2 = st.columns(2)
            with c1:
                if st.button(
                    "Undo",
                    disabled=not st.session_state.history_manager.can_undo,
                    use_container_width=True,
                ):
                    prev = st.session_state.history_manager.undo(df)
                    if prev is not None:
                        st.session_state.df = prev
                        st.rerun()
            with c2:
                if st.button(
                    "Redo",
                    disabled=not st.session_state.history_manager.can_redo,
                    use_container_width=True,
                ):
                    nxt = st.session_state.history_manager.redo()
                    if nxt is not None:
                        st.session_state.df = nxt
                        st.rerun()

        # ── Load New File ────────────────────────────────
        st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
        new_file = st.file_uploader(
<<<<<<< HEAD
            "Load new data source",
=======
            "Load new data source" if st.session_state.get("language") == "en" else "تحميل مصدر بيانات جديد",
>>>>>>> c8e725118d9c65808b1a67b5349827ad4e22458d
            type=SUPPORTED_FILE_TYPES,
            label_visibility="visible",
            help="CSV, Excel, JSON, JSONL, Parquet, SQLite DB",
        )
        if new_file and new_file.name != st.session_state.file_name:
            try:
                parsed_df, clean_report = smart_parse_file(new_file)

                activate_dataset(parsed_df, clean_report, display_name=new_file.name)
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

        # ── AI Settings ──────────────────────────────────
        st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
        st.markdown("<div class='sidebar-section-title'>AI ENGINE</div>", unsafe_allow_html=True)
        engine_choice = st.selectbox(
            "Engine",
            ["Demo", "Claude", "Ollama", "Gemini"],
            label_visibility="collapsed",
            key="ai_engine_choice",
        )

<<<<<<< HEAD
        active_strategy = None

=======
>>>>>>> c8e725118d9c65808b1a67b5349827ad4e22458d
        if engine_choice == "Claude":
            key_in = st.text_input("Anthropic API Key", type="password", placeholder="sk-ant-...")
            allow  = st.checkbox("Allow cloud data", value=False)
            if key_in:
<<<<<<< HEAD
                active_strategy = AnthropicCloudEngine(key_in)
                st.session_state.ai_engine = DataBridgeAIEngine(active_strategy, allow_cloud_data=allow)
                st.session_state.ai_mode = "anthropic"
            else:
                st.session_state.ai_mode   = "demo"
                st.session_state.ai_engine = DataBridgeAIEngine(DemoEngine())
=======
                st.session_state.ai_engine = DataBridgeAIEngine(
                    AnthropicCloudEngine(key_in), allow_cloud_data=allow
                )
                st.session_state.ai_mode = "anthropic"
            else:
                st.session_state.ai_mode = "demo"
>>>>>>> c8e725118d9c65808b1a67b5349827ad4e22458d

        elif engine_choice == "Ollama":
            host  = st.text_input("Host", value="http://localhost:11434")
            model = st.text_input("Model", value="llama3")
<<<<<<< HEAD
            active_strategy = OllamaLocalEngine(host, model)
            st.session_state.ai_engine = DataBridgeAIEngine(active_strategy)
=======
            st.session_state.ai_engine = DataBridgeAIEngine(OllamaLocalEngine(host, model))
>>>>>>> c8e725118d9c65808b1a67b5349827ad4e22458d
            st.session_state.ai_mode   = "ollama"

        elif engine_choice == "Gemini":
            gkey  = st.text_input("Gemini API Key", type="password", placeholder="AIza...")
            allow = st.checkbox("Allow cloud data", value=False)
            mask  = st.checkbox("Auto-mask PII", value=True)
            if gkey:
                st.session_state.gemini_api_key    = gkey
                st.session_state.gemini_allow_data = allow
                st.session_state.gemini_mask_pii   = mask
<<<<<<< HEAD
                active_strategy = GeminiCloudEngine(gkey, mask_pii=mask)
                st.session_state.ai_engine = DataBridgeAIEngine(active_strategy, allow_cloud_data=allow)
                st.session_state.ai_mode = "gemini"
            else:
                st.session_state.ai_mode   = "demo"
                st.session_state.ai_engine = DataBridgeAIEngine(DemoEngine())

        else:
            active_strategy = DemoEngine()
            st.session_state.ai_mode   = "demo"
            st.session_state.ai_engine = DataBridgeAIEngine(active_strategy)

        if active_strategy is not None:
            if st.button("Test Connection", use_container_width=True, key="test_ai_connection"):
                with st.spinner("Testing connection..."):
                    ok, message = active_strategy.test_connection()
                if ok:
                    st.success(message)
                else:
                    st.error(message)
=======
                st.session_state.ai_engine = DataBridgeAIEngine(
                    GeminiCloudEngine(gkey, mask_pii=mask), allow_cloud_data=allow
                )
                st.session_state.ai_mode = "gemini"
            else:
                st.session_state.ai_mode = "demo"

        else:
            st.session_state.ai_mode   = "demo"
            st.session_state.ai_engine = DataBridgeAIEngine(DemoEngine())
>>>>>>> c8e725118d9c65808b1a67b5349827ad4e22458d

        st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='sidebar-footer'>DataBridge AI © 2026</div>",
            unsafe_allow_html=True,
        )

    return st.session_state.get("current_page", NAV_PAGE_KEYS[0])
