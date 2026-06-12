# ════════════════════════════════════════════════════════
#  DataBridge AI — Google Gemini Engine
# ════════════════════════════════════════════════════════
import requests
from ai.base import AIEngineStrategy
from config.settings import DEFAULT_GEMINI_MODEL


class GeminiCloudEngine(AIEngineStrategy):

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, api_key: str, mask_pii: bool = True, model: str | None = None):
        self.api_key  = api_key
        self.mask_pii = mask_pii
        self.model    = model or DEFAULT_GEMINI_MODEL
        self.API_URL  = f"{self.BASE_URL}/{self.model}:generateContent"

    def generate_insights(self, context: str, prompt: str, history: list) -> str:
        history_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}"
            for m in history[-10:]
        )
        full_prompt = (
            f"أنت محلل بيانات خبير داخل DataBridge AI.\n"
            f"سياق البيانات:\n{context}\n\n"
            f"المحادثة السابقة:\n{history_text}\n\n"
            f"المستخدم: {prompt}\n\n"
            "أجب بنفس لغة المستخدم (عربي أو إنجليزي). كن دقيقاً ومختصراً وعملياً."
        )
        payload = {"contents": [{"parts": [{"text": full_prompt}]}]}
        resp = requests.post(
            self.API_URL,
            params={"key": self.api_key},
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"Gemini response parse error: {data}") from exc

    def get_engine_type(self) -> str:
        return "cloud"
