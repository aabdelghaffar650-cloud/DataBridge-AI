# ════════════════════════════════════════════════════════
#  DataBridge AI — Ollama Local Engine
# ════════════════════════════════════════════════════════
import requests
from ai.base import AIEngineStrategy


class OllamaLocalEngine(AIEngineStrategy):

    def __init__(self, host: str = "http://localhost:11434", model: str = "llama3"):
        self.host  = host.rstrip("/")
        self.model = model

    def generate_insights(self, context: str, prompt: str, history: list) -> str:
        history_text = "\n".join(
            f"{m['role'].upper()}: {m['content']}"
            for m in history[-10:]
        )
        full_prompt = (
            f"أنت محلل بيانات خبير داخل DataBridge AI.\n"
            f"سياق البيانات:\n{context}\n\n"
            f"المحادثة السابقة:\n{history_text}\n\n"
            f"المستخدم: {prompt}"
        )
        resp = requests.post(
            f"{self.host}/api/generate",
            json={"model": self.model, "prompt": full_prompt, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json().get("response", "No response from Ollama.")

    def get_engine_type(self) -> str:
        return "local"
