# ════════════════════════════════════════════════════════
#  DataBridge AI — Anthropic Claude Engine
# ════════════════════════════════════════════════════════
from ai.base import AIEngineStrategy
from config.settings import DEFAULT_CLAUDE_MODEL


class AnthropicCloudEngine(AIEngineStrategy):

    def __init__(self, api_key: str, model: str | None = None):
        self.api_key = api_key
        self.model   = model or DEFAULT_CLAUDE_MODEL

    def generate_insights(self, context: str, prompt: str, history: list) -> str:
        import anthropic as _anthropic
        client = _anthropic.Anthropic(api_key=self.api_key)
        system = (
            "أنت محلل بيانات خبير داخل DataBridge AI. "
            "إليك ملف البيانات المحمل حالياً:\n" + context +
            "\nأجب بنفس لغة المستخدم (عربي أو إنجليزي). كن دقيقاً ومختصراً."
        )
        messages = history[-20:] + [{"role": "user", "content": prompt}]
        response = client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system,
            messages=messages,
        )
        return response.content[0].text

    def get_engine_type(self) -> str:
        return "cloud"
