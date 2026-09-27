"""DataBridge AI Stage 5 unified state and atomic mutation verification.

Run from project root:
    python tests/test_stage5_unified_state.py
"""
from __future__ import annotations

import re
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
from core.dataset_state import (
    DatasetMutationError,
    DatasetState,
    StaleDatasetRevisionError,
)


def _configure_fake_state() -> None:
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
    base = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "amount": [10.0, 20.0, 30.0],
            "group": ["a", "b", "a"],
        }
    )
    dataset_module.activate_dataset(
        base,
        {"source_type": "test", "source_name": "stage5"},
        display_name="stage5.csv",
    )


def test_unified_authority_and_atomic_success() -> None:
    _configure_fake_state()
    session = fake_streamlit.session_state
    state = session.dataset_state

    assert isinstance(state, DatasetState)
    assert state.working_df is session.df
    assert state.working_df is session.hdf
    assert state.raw_df is session.raw_df
    assert state.quality_report is session.quality_report
    assert state.column_mappings is session.column_mappings
    assert dataset_module.dataset_state_health(deep=True)["ok"] is True

    raw_before = state.raw_df.copy(deep=True)
    revision_before = state.revision
    history_before = session.history_manager.undo_count

    result = dataset_module.apply_dataset_change(
        "Add tax atomically",
        lambda working: working.assign(tax=working["amount"] * 0.14),
        details={"column": "tax"},
        expected_revision=revision_before,
    )

    assert result.changed is True
    assert result.revision == revision_before + 1
    assert "tax" in session.df.columns
    assert session.dataset_state.working_df is session.df
    assert session.hdf is session.df
    assert session.history_manager.undo_count == history_before + 1
    assert session.quality_report["columns"] == tuple(session.df.columns)
    assert session.dataset_audit_log[-1]["event"] == "atomic_change"
    pd.testing.assert_frame_equal(session.raw_df, raw_before)
    assert dataset_module.dataset_state_health(deep=True)["ok"] is True

    assert dataset_module.perform_undo() is True
    assert "tax" not in session.df.columns
    assert dataset_module.perform_redo() is True
    assert "tax" in session.df.columns
    assert dataset_module.dataset_state_health(deep=True)["ok"] is True


def test_noop_failure_and_stale_revision_are_safe() -> None:
    _configure_fake_state()
    session = fake_streamlit.session_state

    revision = session.dataset_state.revision
    history = session.history_manager.undo_count
    audit = len(session.dataset_audit_log)
    fingerprint = session.working_fingerprint

    no_change = dataset_module.apply_dataset_change(
        "No-op",
        lambda working: working,
        expected_revision=revision,
    )
    assert no_change.changed is False
    assert session.dataset_state.revision == revision
    assert session.history_manager.undo_count == history
    assert len(session.dataset_audit_log) == audit

    try:
        dataset_module.apply_dataset_change(
            "Broken transform",
            lambda _working: (_ for _ in ()).throw(ValueError("boom")),
            expected_revision=revision,
        )
    except DatasetMutationError:
        pass
    else:
        raise AssertionError("A failing transform was not blocked.")

    assert session.dataset_state.revision == revision
    assert session.working_fingerprint == fingerprint
    assert session.history_manager.undo_count == history
    assert len(session.dataset_audit_log) == audit

    dataset_module.apply_dataset_change(
        "Real change",
        lambda working: working.assign(flag=True),
        expected_revision=revision,
    )
    try:
        dataset_module.apply_dataset_change(
            "Stale change",
            lambda working: working.assign(stale=True),
            expected_revision=revision,
        )
    except StaleDatasetRevisionError:
        pass
    else:
        raise AssertionError("A stale page revision was allowed to change data.")
    assert "stale" not in session.df.columns


def test_refresh_failure_rolls_back_everything() -> None:
    _configure_fake_state()
    session = fake_streamlit.session_state
    before = session.df.copy(deep=True)
    revision = session.dataset_state.revision
    history = session.history_manager.undo_count
    fingerprint = session.working_fingerprint

    original_quality = dataset_module.run_quality_engine
    dataset_module.run_quality_engine = lambda _df: (_ for _ in ()).throw(RuntimeError("quality failed"))
    try:
        try:
            dataset_module.apply_dataset_change(
                "Change with failed dependent refresh",
                lambda working: working.assign(new_col=1),
                expected_revision=revision,
            )
        except DatasetMutationError:
            pass
        else:
            raise AssertionError("Dependent-state failure did not roll back the change.")
    finally:
        dataset_module.run_quality_engine = original_quality

    pd.testing.assert_frame_equal(session.df, before)
    assert session.dataset_state.revision == revision
    assert session.working_fingerprint == fingerprint
    assert session.history_manager.undo_count == history
    assert dataset_module.dataset_state_health(deep=True)["ok"] is True


def test_context_uses_authoritative_state() -> None:
    _configure_fake_state()
    session = fake_streamlit.session_state
    revision = session.dataset_state.revision

    dataset_module.update_dataset_context(
        expected_revision=revision,
        mapper_approved=True,
        column_mappings={"id": "ID", "amount": "Value", "group": "Category"},
        kpi_targets={"amount": {"annual": 100.0}},
        show_import_report=False,
    )

    state = session.dataset_state
    assert state.mapper_approved is True
    assert session.mapper_approved is True
    assert state.column_mappings is session.column_mappings
    assert state.kpi_targets is session.kpi_targets
    assert state.show_import_report is False
    assert session.show_import_report is False
    assert dataset_module.dataset_state_health(deep=True)["ok"] is True


def test_internal_pages_do_not_write_legacy_dataframe_mirrors() -> None:
    targets = list((PROJECT_ROOT / "_pages").glob("*.py"))
    targets.extend((PROJECT_ROOT / "ui").glob("*.py"))
    targets.append(PROJECT_ROOT / "app.py")

    direct_patterns = [
        re.compile(r"st\.session_state\.df\s*="),
        re.compile(r"st\.session_state\.df\s*\["),
        re.compile(r"st\.session_state\.df\.loc"),
        re.compile(r"st\.session_state\[['\"]df['\"]\]\s*="),
    ]
    violations = []
    legacy_history_imports = []
    for path in targets:
        text = path.read_text(encoding="utf-8")
        for pattern in direct_patterns:
            if pattern.search(text):
                violations.append(f"{path.relative_to(PROJECT_ROOT)}: {pattern.pattern}")
        if "save_history" in text:
            legacy_history_imports.append(str(path.relative_to(PROJECT_ROOT)))

    assert not violations, "Direct legacy DataFrame writes found: " + "; ".join(violations)
    assert not legacy_history_imports, "Legacy save_history use found: " + "; ".join(legacy_history_imports)


def main() -> None:
    test_unified_authority_and_atomic_success()
    test_noop_failure_and_stale_revision_are_safe()
    test_refresh_failure_rolls_back_everything()
    test_context_uses_authoritative_state()
    test_internal_pages_do_not_write_legacy_dataframe_mirrors()
    print(
        "PASS: Stage 5 unified dataset state, atomic mutations, rollback, stale-revision protection, and mirror synchronization are functioning."
    )


if __name__ == "__main__":
    main()
