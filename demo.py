# ════════════════════════════════════════════════════════
#  DataBridge AI — Demo Engine (no API key required)
# ════════════════════════════════════════════════════════
import pandas as pd
from ai.base import AIEngineStrategy


class DemoEngine(AIEngineStrategy):

    def generate_insights(self, context: str, prompt: str, history: list) -> str:
        p = prompt.lower()
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
        )

    def get_engine_type(self) -> str:
        return "demo"
