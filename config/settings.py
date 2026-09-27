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

# ── Login ────────────────────────────────────────────────
LOGIN_REQUIRED = True
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
    "explainability_studio",
    "prediction_studio",
    "model_monitoring",
    "monitoring_scheduler",
    "deployment_api",
    "remote_model_registry",
    "retraining_workflow",
    "model_governance",
    "delete_dedupe",
    "export",
    "settings",
    "team_admin",
    "ai_assistant",
]

# ── AI Model Defaults ───────────────────────────────────
DEFAULT_GEMINI_MODEL = "gemini-2.0-flash"
DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-20250514"
# Writable user-data locations. Do not store mutable files inside Program Files.
from core.user_paths import user_data_dir as _user_data_dir
USER_DATA_DIR = _user_data_dir()
USER_SETTINGS_FILE = USER_DATA_DIR / "settings.json"
USER_AUTH_FILE = USER_DATA_DIR / "auth.json"
USER_LOG_DIR = USER_DATA_DIR / "logs"
USER_LOG_DIR.mkdir(parents=True, exist_ok=True)
