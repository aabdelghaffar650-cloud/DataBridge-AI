# ════════════════════════════════════════════════════════
#  DataBridge AI — Anthropic Claude Engine
#  Stage 12: verified connection checks and redacted errors
# ════════════════════════════════════════════════════════
from __future__ import annotations

import requests

from ai.base import AIEngineStrategy
from config.settings import DEFAULT_CLAUDE_MODEL
from core.security import safe_error_message


class AnthropicCloudEngine(AIEngineStrategy):
    MODELS_URL = "https://api.anthropic.com/v1/models"

    def __init__(self, api_key: str, model: str | None = None):
        self.api_key = str(api_key or "").strip()
        self.model = model or DEFAULT_CLAUDE_MODEL

    def generate_insights(self, context: str, prompt: str, history: list) -> str:
        try:
            import anthropic as _anthropic

            client = _anthropic.Anthropic(api_key=self.api_key)
            system = (
                "You are an expert data analyst inside DataBridge AI. "
                "Use only the supplied privacy-controlled dataset context. "
                "Do not infer hidden personal information.\n\nDataset context:\n" + context
            )
            messages = history[-20:] + [{"role": "user", "content": prompt}]
            response = client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=system,
                messages=messages,
            )
            return response.content[0].text
        except Exception as exc:
            raise RuntimeError(
                f"Claude request failed safely: {safe_error_message(exc, [self.api_key])}"
            ) from exc

    def get_engine_type(self) -> str:
        return "cloud"

    def test_connection(self) -> tuple[bool, str]:
        if not self.api_key:
            return False, "Anthropic API key is not configured."
        try:
            response = requests.get(
                self.MODELS_URL,
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                },
                params={"limit": 1},
                timeout=15,
            )
            if response.status_code == 200:
                return True, "Connected to Anthropic securely."
            return False, f"Anthropic API rejected the credentials (HTTP {response.status_code})."
        except requests.RequestException as exc:
            return False, f"Connection failed: {safe_error_message(exc, [self.api_key])}"
