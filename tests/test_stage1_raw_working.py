"""DataBridge AI Stage 1 + Stage 3 source/working safety verification.

Run from the project root:
    python tests/test_stage1_raw_working.py
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class AttrDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


fake_streamlit = types.ModuleType("streamlit")
fake_streamlit.session_state = AttrDict()


def _cache_data(*args, **kwargs):
    if args and callable(args[0]) and len(args) == 1 and not kwargs:
        return args[0]
    return lambda func: func


fake_streamlit.cache_data = _cache_data
sys.modules.setdefault("streamlit", fake_streamlit)

import core.dataset as dataset_module
import core.session as session_module
from modules.import_engine import (
    PROPOSED_IMPORT_ACTIONS_KEY,
    RAW_DATAFRAME_REPORT_KEY,
    _finalize_import_dataframe,
)


def main() -> None:
    source = pd.DataFrame(
        {
            " event_date ": ["01/01/2024", "02/01/2024", "bad-date"],
            "empty_column": [None, None, None],
            "amount": [10.0, 20.0, 30.0],
        }
    )

    report = {"source_type": "test", "source_name": "stage1-test"}
    working, _steps = _finalize_import_dataframe(source, report)

    raw_from_import = report[RAW_DATAFRAME_REPORT_KEY]
    assert raw_from_import.shape == (3, 3)
    assert tuple(working.shape) == (3, 3)
    assert " event_date " in raw_from_import.columns
    assert "event_date" in working.columns
    assert "empty_column" in working.columns
    # pandas 3.x uses StringDtype for text columns by default, while
    # pandas 2.x commonly uses object. Both are valid and non-destructive.
    event_dtype = working["event_date"].dtype
    assert not pd.api.types.is_datetime64_any_dtype(event_dtype)
    assert (
        pd.api.types.is_object_dtype(event_dtype)
        or pd.api.types.is_string_dtype(event_dtype)
    )
    assert working.loc[2, "event_date"] == "bad-date"
    assert report["automatic_value_changes"] == 0
    assert report["automatic_rows_removed"] == 0
    assert report["automatic_columns_removed"] == 0
    assert any(
        action["operation"] == "drop_empty_columns"
        for action in report[PROPOSED_IMPORT_ACTIONS_KEY]
    )

    fake_streamlit.session_state.clear()
    session_module.st = fake_streamlit
    dataset_module.st = fake_streamlit

    dataset_module.auto_map_columns = lambda columns: {
        col: ("Unknown", 0.0) for col in columns
    }
    dataset_module.run_quality_engine = lambda df: {
        "quality_score": 100.0,
        "total_cells": int(df.shape[0] * df.shape[1]),
    }

    session_module.init_session_state()
    dataset_module.activate_dataset(working, report, display_name="stage1-test.csv")

    state = fake_streamlit.session_state
    assert RAW_DATAFRAME_REPORT_KEY not in state.data_clean_report
    assert state.raw_df is not state.df
    assert tuple(state.raw_df.shape) == (3, 3)
    assert tuple(state.df.shape) == (3, 3)
    assert " event_date " in state.raw_df.columns
    assert "event_date" in state.df.columns

    original_amount = state.raw_df.loc[0, "amount"]
    state.df.loc[0, "amount"] = 999.0
    assert state.raw_df.loc[0, "amount"] == original_amount

    nested_source = pd.DataFrame({"payload": [{"items": [1, 2]}]})
    nested_copy = dataset_module.clone_dataframe(nested_source)
    nested_copy.loc[0, "payload"]["items"].append(999)
    assert 999 not in nested_source.loc[0, "payload"]["items"]
    assert dataset_module.raw_dataset_is_intact()

    dataset_module.restore_raw_dataset()

    assert tuple(state.df.shape) == tuple(state.raw_df.shape)
    assert "event_date" in state.df.columns
    assert " event_date " in state.raw_df.columns
    assert state.df["event_date"].tolist() == state.raw_df[" event_date "].tolist()
    assert state.history_manager.can_undo is False
    assert state.history_manager.can_redo is False
    assert dataset_module.raw_dataset_is_intact()

    print(
        "PASS: protected raw data and an independent non-destructive working copy are functioning."
    )


if __name__ == "__main__":
    main()
