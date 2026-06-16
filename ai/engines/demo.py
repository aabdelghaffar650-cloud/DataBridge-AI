# ════════════════════════════════════════════════════════
#  DataBridge AI — Demo Engine (no API key required)
# ════════════════════════════════════════════════════════
import pandas as pd
from ai.base import AIEngineStrategy


class DemoEngine(AIEngineStrategy):

    def generate_insights(self, context: str, prompt: str, history: list) -> str:
        p = prompt.lower()
<<<<<<< HEAD
        if any(w in p for w in ["null", "missing"]):
            return "🔍 Demo: Open the Quality Engine page to see the full missing-values report."
        if any(w in p for w in ["duplicate"]):
            return "🔍 Demo: Use the Quality Engine page → Duplicates tab to detect and remove them."
        if any(w in p for w in ["kpi", "target"]):
            return "🔍 Demo: Open the KPI Tracker page to set up your metrics and targets."
        if any(w in p for w in ["column", "mapper"]):
            return "🔍 Demo: Open the Data Mapper page to review and approve the column classification."
        return (
            "🤖 Demo Mode — no API key connected.\n\n"
            "Choose an AI engine from the sidebar:\n"
            "- 🔴 Google Gemini\n"
            "- 🔵 Claude (Anthropic)\n"
            "- 🟢 Ollama (local)"
=======
        if any(w in p for w in ["null", "missing", "فاقد", "مفقود"]):
            return "🔍 Demo: افتح صفحة Quality Engine لرؤية تقرير القيم المفقودة الكامل."
        if any(w in p for w in ["duplicate", "مكرر", "تكرار"]):
            return "🔍 Demo: استخدم صفحة Quality Engine ← تبويب Duplicates للكشف والحذف."
        if any(w in p for w in ["kpi", "target", "هدف", "مؤشر"]):
            return "🔍 Demo: افتح صفحة KPI Tracker لإعداد مؤشراتك وأهدافك."
        if any(w in p for w in ["column", "عمود", "خريطة", "mapper"]):
            return "🔍 Demo: افتح صفحة Data Mapper لمراجعة واعتماد تصنيف الأعمدة."
        return (
            "🤖 Demo Mode — لا يوجد API key متصل.\n\n"
            "اختر محرك AI من الـ sidebar:\n"
            "- 🔴 Google Gemini\n"
            "- 🔵 Claude (Anthropic)\n"
            "- 🟢 Ollama (محلي)"
>>>>>>> c8e725118d9c65808b1a67b5349827ad4e22458d
        )

    def get_engine_type(self) -> str:
        return "demo"
<<<<<<< HEAD

    def test_connection(self) -> tuple[bool, str]:
        return True, "Demo mode is active — no API key required."
=======
>>>>>>> c8e725118d9c65808b1a67b5349827ad4e22458d
