# ════════════════════════════════════════════════════════
#  DataBridge AI — Lightweight i18n helpers
# ════════════════════════════════════════════════════════
import streamlit as st

LANGUAGE_OPTIONS = {
    "en": "English",
    "ar": "العربية",
}

STRINGS = {
    "subtitle": {
        "en": "Universal Data Intelligence & Analysis Platform",
        "ar": "منصة ذكية لتنظيف وتحليل وفهم البيانات",
    },
    "login_title": {"en": "Sign in to DataBridge AI", "ar": "تسجيل الدخول إلى DataBridge AI"},
    "login_subtitle": {
        "en": "Local secure access before opening datasets and AI tools.",
        "ar": "دخول محلي آمن قبل فتح ملفات البيانات وأدوات الذكاء الاصطناعي.",
    },
    "username": {"en": "Username", "ar": "اسم المستخدم"},
    "password": {"en": "Password", "ar": "كلمة المرور"},
    "sign_in": {"en": "Sign in", "ar": "دخول"},
    "logout": {"en": "Logout", "ar": "تسجيل الخروج"},
    "wrong_login": {"en": "Invalid username or password.", "ar": "اسم المستخدم أو كلمة المرور غير صحيحة."},
    "language": {"en": "Language", "ar": "اللغة"},
    "upload_title": {"en": "Upload your dataset", "ar": "ارفع ملف البيانات"},
    "upload_desc": {
        "en": "CSV and Excel files · Multi-sheet detection · Arabic text supported",
        "ar": "يدعم CSV و Excel · اكتشاف الشيت المناسب · دعم اللغة العربية",
    },
    "upload_smart": {
        "en": "Smart parser auto-detects headers, cleans noise, and standardises data",
        "ar": "المعالج الذكي يكتشف العناوين وينظف الضوضاء ويوحد القيم تلقائيًا",
    },
    "ai_title": {"en": "AI Data Assistant", "ar": "مساعد تحليل البيانات الذكي"},
    "ai_subtitle": {"en": "Dataset-aware intelligence", "ar": "أسئلة ذكية حسب الملف الحالي"},
    "smart_suggestions": {"en": "Smart suggestions based on this dataset", "ar": "أسئلة ذكية مبنية على الملف الحالي"},
    "ask_placeholder": {"en": "Ask anything about your data...", "ar": "اسأل أي سؤال عن البيانات..."},
    "send": {"en": "Send →", "ar": "إرسال ←"},
    "clear_chat": {"en": "🗑️ Clear Chat", "ar": "🗑️ مسح المحادثة"},
    "demo_mode": {"en": "Demo Mode", "ar": "وضع التجربة"},
}


def lang() -> str:
    return st.session_state.get("language", "en")


def is_ar() -> bool:
    return lang() == "ar"


def t(key: str) -> str:
    value = STRINGS.get(key, {})
    if isinstance(value, dict):
        return value.get(lang(), value.get("en", key))
    return str(value)
