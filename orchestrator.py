# ════════════════════════════════════════════════════════
#  DataBridge AI — AI Orchestrator (Strategy Pattern)
# ════════════════════════════════════════════════════════
import pandas as pd

from ai.base    import AIEngineStrategy
from ai.context import AIContextManager
from core.security import anonymise_df_for_ai


class DataBridgeAIEngine:
    """
    Orchestrates any AIEngineStrategy.
    Handles context building, PII masking, and error wrapping.
    """

    def __init__(self, strategy: AIEngineStrategy, allow_cloud_data: bool = False):
        self._strategy       = strategy
        self.allow_cloud_data = allow_cloud_data

    def set_strategy(self, strategy: AIEngineStrategy) -> None:
        self._strategy = strategy

    def process_task(
        self,
        df: pd.DataFrame,
        prompt: str,
        history: list,
    ) -> str:
        engine_type = self._strategy.get_engine_type()

        # Mask PII for cloud engines unless explicitly allowed
        if engine_type == "cloud" and not self.allow_cloud_data:
            df = anonymise_df_for_ai(df)

        context = AIContextManager.prepare_context(
            df,
            engine_type=engine_type,
            allow_sensitive=self.allow_cloud_data,
        )
        return self._strategy.generate_insights(context, prompt, history)

    @property
    def engine_type(self) -> str:
        return self._strategy.get_engine_type()
