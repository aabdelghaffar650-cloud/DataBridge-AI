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
            f"You are an expert data analyst inside DataBridge AI.\n"
            f"Data context:\n{context}\n\n"
            f"Previous conversation:\n{history_text}\n\n"
            f"User: {prompt}\n\n"
            "Answer in English. Be precise, concise, and practical."
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

    def test_connection(self) -> tuple[bool, str]:
        try:
            resp = requests.get(
                f"{self.BASE_URL}/{self.model}",
                params={"key": self.api_key},
                timeout=15,
            )
        except requests.RequestException as exc:
            return False, f"Connection failed: {exc}"

        if resp.status_code == 200:
            return True, f"Connected — model '{self.model}' is available."

        try:
            message = resp.json().get("error", {}).get("message", resp.text)
        except Exception:
            message = resp.text
        return False, f"Gemini API error ({resp.status_code}): {message}"
