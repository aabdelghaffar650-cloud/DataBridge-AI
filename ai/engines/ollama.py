# ════════════════════════════════════════════════════════
#  DataBridge AI — Ollama Engine
#  Stage 12: loopback trust classification and safe endpoints
# ════════════════════════════════════════════════════════
from __future__ import annotations

import requests

from ai.base import AIEngineStrategy
from core.security import is_loopback_url, safe_error_message, validate_http_endpoint


class OllamaLocalEngine(AIEngineStrategy):
    def __init__(self, host: str = "http://localhost:11434", model: str = "llama3"):
        self.host = validate_http_endpoint(host)
        self.model = str(model or "llama3").strip()
        if not self.model:
            raise ValueError("Ollama model name is required.")
        self.is_loopback = is_loopback_url(self.host)

    def generate_insights(self, context: str, prompt: str, history: list) -> str:
        history_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in history[-10:]
        )
        full_prompt = (
            "أنت محلل بيانات خبير داخل DataBridge AI. استخدم فقط سياق البيانات المسموح به.\n"
            f"سياق البيانات:\n{context}\n\n"
            f"المحادثة السابقة:\n{history_text}\n\n"
            f"المستخدم: {prompt}"
        )
        try:
            response = requests.post(
                f"{self.host}/api/generate",
                json={"model": self.model, "prompt": full_prompt, "stream": False},
                timeout=120,
            )
            response.raise_for_status()
            return response.json().get("response", "No response from Ollama.")
        except (requests.RequestException, ValueError) as exc:
            raise RuntimeError(f"Ollama request failed safely: {safe_error_message(exc)}") from exc

    def get_engine_type(self) -> str:
        return "local" if self.is_loopback else "remote"

    def test_connection(self) -> tuple[bool, str]:
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=10)
            response.raise_for_status()
            models = [m.get("name", "") for m in response.json().get("models", [])]
        except (requests.RequestException, ValueError) as exc:
            return False, f"Connection failed: {safe_error_message(exc)}"
        if any(m == self.model or m.startswith(f"{self.model}:") for m in models):
            trust = "loopback" if self.is_loopback else "remote"
            return True, f"Connected to {trust} Ollama — model '{self.model}' is available."
        return False, f"Connected, but model '{self.model}' was not found."
