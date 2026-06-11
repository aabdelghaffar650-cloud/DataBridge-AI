# ════════════════════════════════════════════════════════
#  DataBridge AI — Smart History Manager (Undo/Redo)
# ════════════════════════════════════════════════════════
import logging
from collections import deque
from typing      import Optional

import pandas as pd

from config.constants import MAX_HISTORY, MAX_HISTORY_MEM_MB

logger = logging.getLogger(__name__)


class SmartHistoryManager:
    """Memory-efficient undo/redo using a bounded deque."""

    def __init__(
        self,
        max_history: int = MAX_HISTORY,
        max_memory_mb: int = MAX_HISTORY_MEM_MB,
    ):
        self.history:     deque        = deque(maxlen=max_history)
        self.redo_stack:  list         = []
        self.max_memory_mb             = max_memory_mb

    # ── internal ──────────────────────────────────────
    def _estimate_memory(self, df: pd.DataFrame) -> float:
        return df.memory_usage(deep=True).sum() / (1024 * 1024)

    # ── public ────────────────────────────────────────
    def push(self, df: pd.DataFrame) -> bool:
        mem = self._estimate_memory(df)
        if mem > self.max_memory_mb:
            logger.warning(f"DataFrame too large ({mem:.1f} MB) — storing first 1 000 rows.")
            self.history.append(df.head(1000).copy())
            return False
        self.history.append(df.copy(deep=True))
        self.redo_stack.clear()
        return True

    def undo(self, current_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        if not self.history:
            return None
        self.redo_stack.append(current_df.copy(deep=True))
        return self.history.pop().copy()

    def redo(self) -> Optional[pd.DataFrame]:
        if not self.redo_stack:
            return None
        return self.redo_stack.pop().copy()

    @property
    def can_undo(self) -> bool:
        return bool(self.history)

    @property
    def can_redo(self) -> bool:
        return bool(self.redo_stack)
