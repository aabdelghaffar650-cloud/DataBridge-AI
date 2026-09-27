# ════════════════════════════════════════════════════════
#  DataBridge AI — AI Orchestrator
#  Stage 12: enforced privacy and endpoint trust
# ════════════════════════════════════════════════════════
from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from ai.base import AIEngineStrategy
from ai.context import (
    AIContextManager,
    PRIVACY_MASKED,
    PRIVACY_METADATA,
    PRIVACY_RAW,
    normalise_privacy_mode,
)
from core.security import PIIMaskReport, anonymise_df_for_ai


class AIPrivacyError(RuntimeError):
    pass


class DataBridgeAIEngine:
    """Orchestrate an AI strategy while enforcing data-transfer policy."""

    def __init__(
        self,
        strategy: AIEngineStrategy,
        allow_cloud_data: bool | None = None,
        *,
        privacy_mode: str | None = None,
        raw_data_confirmed: bool = False,
        allow_remote_endpoint: bool = False,
        semantic_profiles: Mapping[str, Mapping[str, Any]] | None = None,
    ):
        self._strategy = strategy
        if privacy_mode is None:
            privacy_mode = PRIVACY_RAW if allow_cloud_data else PRIVACY_METADATA
        self.privacy_mode = normalise_privacy_mode(privacy_mode)
        self.raw_data_confirmed = bool(raw_data_confirmed)
        self.allow_remote_endpoint = bool(allow_remote_endpoint)
        self.semantic_profiles = dict(semantic_profiles or {})
        self.last_privacy_report: dict[str, Any] = {}

    @property
    def allow_cloud_data(self) -> bool:
        """Backward-compatible indicator used by older page code."""
        return self.privacy_mode == PRIVACY_RAW and self.raw_data_confirmed

    def set_strategy(self, strategy: AIEngineStrategy) -> None:
        self._strategy = strategy

    def _prepare_dataframe(self, df: pd.DataFrame, engine_type: str) -> pd.DataFrame:
        mode = self.privacy_mode
        if engine_type == "remote" and not self.allow_remote_endpoint:
            raise AIPrivacyError(
                "The configured Ollama endpoint is not loopback. Explicit remote-host approval is required."
            )
        if mode == PRIVACY_RAW and engine_type in {"cloud", "remote"} and not self.raw_data_confirmed:
            raise AIPrivacyError(
                "Raw sample transfer is blocked until explicit confirmation is provided."
            )
        if mode == PRIVACY_MASKED:
            masked, report = anonymise_df_for_ai(
                df,
                self.semantic_profiles,
                return_report=True,
            )
            assert isinstance(report, PIIMaskReport)
            self.last_privacy_report = {
                "masked_columns": list(report.masked_columns),
                "value_redactions": report.value_redactions,
                "sample_rows": report.sample_rows,
            }
            return masked
        self.last_privacy_report = {
            "masked_columns": [],
            "value_redactions": 0,
            "sample_rows": int(len(df)),
        }
        return df

    def process_task(self, df: pd.DataFrame, prompt: str, history: list) -> str:
        engine_type = self._strategy.get_engine_type()
        prepared = self._prepare_dataframe(df, engine_type)
        context = AIContextManager.prepare_context(
            prepared,
            engine_type=engine_type,
            privacy_mode=self.privacy_mode,
            privacy_report=self.last_privacy_report,
        )
        return self._strategy.generate_insights(context, prompt, history)

    @property
    def engine_type(self) -> str:
        return self._strategy.get_engine_type()
