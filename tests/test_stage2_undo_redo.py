"""DataBridge AI Stage 2 undo/redo verification.

Run from project root:
    python tests/test_stage2_undo_redo.py
"""
from __future__ import annotations

import sys
import tempfile
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

from core.history import SmartHistoryManager
import core.dataset as dataset_module
import core.session as session_module


def _assert_df_equal(left: pd.DataFrame, right: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(left, right, check_dtype=True, check_like=False)


def test_full_disk_snapshots_and_redo_chain() -> None:
    with tempfile.TemporaryDirectory() as temp_root:
        manager = SmartHistoryManager(
            max_history=10,
            memory_snapshot_mb=0.0001,
            max_disk_mb=128,
            temp_root=temp_root,
        )
        df0 = pd.DataFrame({
            "id": range(25_000),
            "value": [float(i % 97) for i in range(25_000)],
            "label": [f"row-{i}" for i in range(25_000)],
        })
        df1 = df0.copy(deep=True)
        df1["value"] = df1["value"] * 2
        df2 = df1.copy(deep=True)
        df2["flag"] = df2["id"] % 2 == 0

        manager.push(df0, action="Double values", fingerprint="fp0", context={"version": 0})
        manager.push(df1, action="Add flag", fingerprint="fp1", context={"version": 1})
        assert manager.storage_summary()["disk_bytes"] > 0

        restored = manager.undo(df2, current_context={"version": 2}, current_fingerprint="fp2")
        assert restored is not None
        _assert_df_equal(restored.dataframe, df1)
        assert restored.context["version"] == 1

        restored = manager.undo(df1, current_context={"version": 1}, current_fingerprint="fp1")
        assert restored is not None
        _assert_df_equal(restored.dataframe, df0)

        restored = manager.redo(df0, current_context={"version": 0}, current_fingerprint="fp0")
        assert restored is not None
        _assert_df_equal(restored.dataframe, df1)

        restored = manager.redo(df1, current_context={"version": 1}, current_fingerprint="fp1")
        assert restored is not None
        _assert_df_equal(restored.dataframe, df2)

        restored = manager.undo(df2, current_context={"version": 2}, current_fingerprint="fp2")
        assert restored is not None
        _assert_df_equal(restored.dataframe, df1)
        manager.close()


def test_session_integration_and_noop() -> None:
    fake_streamlit.session_state.clear()
    session_module.st = fake_streamlit
    dataset_module.st = fake_streamlit

    dataset_module.auto_map_columns = lambda columns: {
        col: ("Unknown", 0.0) for col in columns
    }
    dataset_module.run_quality_engine = lambda df: {
        "quality_score": 100.0,
        "total_cells": int(df.shape[0] * df.shape[1]),
        "columns": tuple(df.columns),
    }

    session_module.init_session_state()
    base = pd.DataFrame({"id": [1, 2, 3], "amount": [10.0, 20.0, 30.0]})
    report = {"source_type": "test", "source_name": "stage2"}
    dataset_module.activate_dataset(base, report, display_name="stage2.csv")

    state = fake_streamlit.session_state
    raw_fp = state.raw_fingerprint

    session_module.save_history(
        "Add calculated column",
        details={"column": "tax"},
    )
    state.df["tax"] = state.df["amount"] * 0.14
    changed = dataset_module.reconcile_working_state()
    assert changed is True
    assert "tax" in state.df.columns
    assert state.hdf is state.df
    assert state.working_fingerprint == dataset_module.dataframe_fingerprint(state.df)
    assert state.history_manager.undo_count == 1
    assert state.history_manager.redo_count == 0
    assert state.quality_report["columns"] == tuple(state.df.columns)
    assert state.dataset_audit_log[-1]["action"] == "Add calculated column"
    assert dataset_module.raw_dataset_is_intact()
    assert state.raw_fingerprint == raw_fp

    assert dataset_module.perform_undo() is True
    assert "tax" not in state.df.columns
    assert state.history_manager.can_redo is True
    assert state.hdf is state.df

    assert dataset_module.perform_redo() is True
    assert "tax" in state.df.columns
    assert dataset_module.perform_undo() is True
    assert "tax" not in state.df.columns

    undo_before = state.history_manager.undo_count
    redo_before = state.history_manager.redo_count
    session_module.save_history("No-op test")
    changed = dataset_module.reconcile_working_state()
    assert changed is False
    assert state.history_manager.undo_count == undo_before
    assert state.history_manager.redo_count == redo_before

    # Nested mutable objects must also round-trip independently.
    session_module.save_history("Add nested payload")
    state.df["payload"] = [{"items": [1]}, {"items": [2]}, {"items": [3]}]
    dataset_module.reconcile_working_state()
    assert dataset_module.perform_undo() is True
    assert "payload" not in state.df.columns
    assert dataset_module.raw_dataset_is_intact()


def main() -> None:
    test_full_disk_snapshots_and_redo_chain()
    test_session_integration_and_noop()
    print("PASS: complete disk-backed Undo/Redo, synchronization, no-op handling, and raw protection are functioning.")


if __name__ == "__main__":
    main()
