# ════════════════════════════════════════════════════════
#  DataBridge AI — Google Gemini Engine
#  Stage 12: orchestrator-enforced privacy and redacted errors
# ════════════════════════════════════════════════════════
from __future__ import annotations

import requests

from ai.base import AIEngineStrategy
from config.settings import DEFAULT_GEMINI_MODEL
from core.security import safe_error_message


class GeminiCloudEngine(AIEngineStrategy):
    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"

    def __init__(self, api_key: str, mask_pii: bool = True, model: str | None = None):
        self.api_key = str(api_key or "").strip()
        self.mask_pii = bool(mask_pii)  # compatibility; enforcement is in the orchestrator
        self.model = model or DEFAULT_GEMINI_MODEL
        self.API_URL = f"{self.BASE_URL}/{self.model}:generateContent"

    def generate_insights(self, context: str, prompt: str, history: list) -> str:
        history_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in history[-10:]
        )
        full_prompt = (
            "You are an expert data analyst inside DataBridge AI. "
            "Use only the supplied privacy-controlled context and do not infer hidden PII.\n"
            f"Data context:\n{context}\n\n"
            f"Previous conversation:\n{history_text}\n\n"
            f"User: {prompt}\n\n"
            "Be precise, practical, and explicit about uncertainty."
        )
        payload = {"contents": [{"parts": [{"text": full_prompt}]}]}
        try:
            response = requests.post(
                self.API_URL,
                params={"key": self.api_key},
                json=payload,
                timeout=60,
            )
            response.raise_for_status()
            data = response.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
            raise RuntimeError(
                f"Gemini request failed safely: {safe_error_message(exc, [self.api_key])}"
            ) from exc

    def get_engine_type(self) -> str:
        return "cloud"

    def test_connection(self) -> tuple[bool, str]:
        if not self.api_key:
            return False, "Gemini API key is not configured."
        try:
            response = requests.get(
                f"{self.BASE_URL}/{self.model}",
                params={"key": self.api_key},
                timeout=15,
            )
        except requests.RequestException as exc:
            return False, f"Connection failed: {safe_error_message(exc, [self.api_key])}"
        if response.status_code == 200:
            return True, f"Connected — model '{self.model}' is available."
        return False, f"Gemini API rejected the request (HTTP {response.status_code})."
