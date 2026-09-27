# ════════════════════════════════════════════════════════
#  DataBridge AI — AI Context Manager
#  Stage 12: explicit privacy modes
# ════════════════════════════════════════════════════════
from __future__ import annotations

import json
from typing import Any, Mapping

import pandas as pd

PRIVACY_METADATA = "metadata"
PRIVACY_MASKED = "masked"
PRIVACY_RAW = "raw"
PRIVACY_MODES = (PRIVACY_METADATA, PRIVACY_MASKED, PRIVACY_RAW)


def normalise_privacy_mode(value: str | None) -> str:
    mode = str(value or PRIVACY_METADATA).strip().casefold()
    if mode not in PRIVACY_MODES:
        raise ValueError("Unsupported AI privacy mode.")
    return mode


class AIContextManager:
    @staticmethod
    def prepare_context(
        df: pd.DataFrame,
        engine_type: str,
        allow_sensitive: bool | None = None,
        *,
        privacy_mode: str | None = None,
        sample_rows: int = 3,
        privacy_report: Mapping[str, Any] | None = None,
    ) -> str:
        """Build a bounded JSON context without mutating the source frame.

        ``allow_sensitive`` remains for compatibility with pre-Stage-12 callers.
        New callers should use ``privacy_mode`` explicitly.
        """
        if privacy_mode is None:
            privacy_mode = PRIVACY_RAW if allow_sensitive else PRIVACY_METADATA
        mode = normalise_privacy_mode(privacy_mode)
        safe_rows = max(0, min(int(sample_rows), 10))

        schema = {str(key): str(value) for key, value in df.dtypes.items()}
        null_summary = {str(key): int(value) for key, value in df.isnull().sum().items()}
        context_dict: dict[str, Any] = {
            "privacy_mode": mode,
            "engine_type": str(engine_type),
            "columns": list(map(str, df.columns)),
            "data_types": schema,
            "row_count": int(len(df)),
            "column_count": int(df.shape[1]),
            "null_summary": null_summary,
            "duplicate_count": int(df.duplicated().sum()),
            "contains_dates": any(
                pd.api.types.is_datetime64_any_dtype(dtype)
                for dtype in df.dtypes
            ),
        }

        if mode == PRIVACY_METADATA:
            context_dict["sample_data"] = "Not included in metadata-only mode."
        else:
            context_dict["sample_data"] = (
                df.head(safe_rows).to_dict(orient="records") if safe_rows else []
            )

        if privacy_report:
            context_dict["privacy_controls"] = {
                "masked_columns": list(privacy_report.get("masked_columns", [])),
                "value_redactions": int(privacy_report.get("value_redactions", 0) or 0),
            }

        return json.dumps(context_dict, ensure_ascii=False, indent=2, default=str)
