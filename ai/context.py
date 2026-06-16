# ════════════════════════════════════════════════════════
#  DataBridge AI — AI Context Manager
# ════════════════════════════════════════════════════════
import json
import pandas as pd


class AIContextManager:

    @staticmethod
    def prepare_context(
        df: pd.DataFrame,
        engine_type: str,
        allow_sensitive: bool = False,
    ) -> str:
        schema       = df.dtypes.astype(str).to_dict()
        null_summary = df.isnull().sum().to_dict()
        context_dict = {
            "columns":         list(df.columns),
            "data_types":      schema,
            "row_count":       len(df),
            "null_summary":    null_summary,
            "duplicate_count": int(df.duplicated().sum()),
            "contains_dates":  any(
                "datetime" in str(d) or "date" in str(d).lower()
                for d in df.dtypes
            ),
        }
        if engine_type == "cloud":
            context_dict["sample_data"] = (
                df.head(3).to_dict(orient="records")
                if allow_sensitive
                else "Redacted for privacy."
            )
        else:
            context_dict["sample_data"] = df.head(10).to_dict(orient="records")

        return json.dumps(context_dict, ensure_ascii=False, indent=2, default=str)
