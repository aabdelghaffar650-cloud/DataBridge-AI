# ════════════════════════════════════════════════════════
#  DataBridge AI — App Settings
# ════════════════════════════════════════════════════════
PAGE_CONFIG = {
    "page_title": "DataBridge AI",
    "page_icon":  "🌉",
    "layout":     "wide",
    "initial_sidebar_state": "expanded",
}

PLOTLY_THEME       = "plotly_dark"
PLOTLY_BG          = "#080810"
PLOTLY_PAPER_BG    = "#0c0c1e"
ACCENT_COLOR       = "#7c6aff"
ACCENT_COLORS_LIST = ["#7c6aff", "#ff6b9d", "#6bffb8", "#ffb86b", "#6bb8ff"]

<<<<<<< HEAD
# ── Login ────────────────────────────────────────────────
LOGIN_REQUIRED = True
=======
# ── Login / Language ───────────────────────────────────
LOGIN_REQUIRED = True
DEFAULT_LANGUAGE = "en"  # en / ar
>>>>>>> c8e725118d9c65808b1a67b5349827ad4e22458d
DEFAULT_LOGIN_USERNAME = "admin"  # username suggestion only; password is created at first run

# ── Navigation keys are internal; labels are translated in the sidebar ──
NAV_PAGE_KEYS = [
    "overview",
    "data_sources",
    "data_mapper",
    "quality_engine",
    "kpi_tracker",
    "filter_search",
    "cleaning",
    "data_types",
    "replace_values",
    "feature_engineering",
    "visualization",
    "outlier_detection",
    "ml_studio",
    "delete_dedupe",
    "export",
    "settings",
    "ai_assistant",
]

<<<<<<< HEAD
=======
NAV_LABELS = {
    "overview":            {"en": "🏠  Overview & Profile",  "ar": "🏠  النظرة العامة والملف"},
    "data_sources":        {"en": "Data Sources",           "ar": "مصادر البيانات"},
    "data_mapper":         {"en": "🗺️  Data Mapper",         "ar": "🗺️  ربط الأعمدة الذكي"},
    "quality_engine":      {"en": "🛡️  Quality Engine",      "ar": "🛡️  فحص جودة البيانات"},
    "kpi_tracker":         {"en": "🎯  KPI Tracker",         "ar": "🎯  متابعة المؤشرات"},
    "filter_search":       {"en": "🔍  Filter & Search",     "ar": "🔍  فلترة وبحث"},
    "cleaning":            {"en": "🧹  Clean & Fix Nulls",   "ar": "🧹  تنظيف ومعالجة الفراغات"},
    "data_types":          {"en": "🔄  Data Types",          "ar": "🔄  أنواع البيانات"},
    "replace_values":      {"en": "✏️  Replace Values",      "ar": "✏️  استبدال القيم"},
    "feature_engineering": {"en": "🔧  Feature Engineering", "ar": "🔧  إنشاء أعمدة ذكية"},
    "visualization":       {"en": "📊  Visualize",           "ar": "📊  الرسوم البيانية"},
    "outlier_detection":   {"en": "🎯  Outlier Detection",   "ar": "🎯  القيم الشاذة"},
    "ml_studio":           {"en": "🧠  ML Studio",           "ar": "🧠  استوديو التعلم الآلي"},
    "delete_dedupe":       {"en": "🗑️  Delete & Dedupe",     "ar": "🗑️  حذف وإزالة التكرار"},
    "export":              {"en": "💾  Export",              "ar": "💾  تصدير"},
    "settings":            {"en": "Settings",                "ar": "الإعدادات"},
    "ai_assistant":        {"en": "🤖  AI Assistant",        "ar": "🤖  مساعد الذكاء الاصطناعي"},
}

# Backward-compatible English labels if any old code imports NAV_PAGES.
NAV_PAGES = [NAV_LABELS[k]["en"] for k in NAV_PAGE_KEYS]


def get_nav_label(page_key: str, language: str = "en") -> str:
    labels = NAV_LABELS.get(page_key, {})
    return labels.get(language, labels.get("en", page_key))

>>>>>>> c8e725118d9c65808b1a67b5349827ad4e22458d
# ── AI Model Defaults ───────────────────────────────────
DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"
DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-20250514"
# Writable user-data locations. Do not store mutable files inside Program Files.
from pathlib import Path as _SettingsPath
import os as _settings_os
USER_DATA_DIR = _SettingsPath(
    _settings_os.environ.get("DATABRIDGE_USER_DATA_DIR")
    or _settings_os.environ.get("LOCALAPPDATA")
    or _settings_os.environ.get("APPDATA")
    or str(_SettingsPath.home())
) / "DataBridgeAI"
USER_DATA_DIR.mkdir(parents=True, exist_ok=True)
USER_SETTINGS_FILE = USER_DATA_DIR / "settings.json"
USER_AUTH_FILE = USER_DATA_DIR / "auth.json"
USER_LOG_DIR = USER_DATA_DIR / "logs"
USER_LOG_DIR.mkdir(parents=True, exist_ok=True)
