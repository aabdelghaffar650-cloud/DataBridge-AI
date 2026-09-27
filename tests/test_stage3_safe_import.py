"""DataBridge AI Stage 3 non-destructive import verification.

Run from project root:
    python tests/test_stage3_safe_import.py
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
    apply_import_actions,
    build_import_action_proposals,
)


def _action_map(proposals):
    return {
        (action["operation"], action.get("column", "")): action
        for action in proposals
    }


def test_no_automatic_mutation_and_reviewed_derivations() -> None:
    dates = [
        "01/01/2024",
        "02/01/2024",
        "03/01/2024",
        "04/01/2024",
        "05/01/2024",
        "06/01/2024",
        "07/01/2024",
        "08/01/2024",
        "09/01/2024",
        "bad-date",
        None,
    ]
    flags = [" yes ", "no", "true", "false", "1", "0", "نعم", "لا", "y", "n", None]
    notes = [" alpha ", "N/A", "ok", "ok", "ok", "ok", "ok", "ok", "ok", "ok", None]
    amounts = list(range(10)) + [None]

    source = pd.DataFrame(
        {
            " event_date ": dates,
            "flag": flags,
            "note": notes,
            "amount": amounts,
            "empty_column": [None] * 11,
        }
    )
    report = {"source_type": "test", "source_name": "stage3"}
    working, _steps = _finalize_import_dataframe(source, report)

    # Initial working data is structurally addressable but value/row/column safe.
    assert tuple(working.shape) == (11, 5)
    assert "event_date" in working.columns
    assert " event_date " in report[RAW_DATAFRAME_REPORT_KEY].columns
    assert working.loc[0, "flag"] == " yes "
    assert working.loc[1, "note"] == "N/A"
    assert working.loc[9, "event_date"] == "bad-date"
    # pandas 3.x uses StringDtype for text columns by default, while
    # pandas 2.x commonly uses object. Both are valid and non-destructive.
    event_dtype = working["event_date"].dtype
    assert not pd.api.types.is_datetime64_any_dtype(event_dtype)
    assert (
        pd.api.types.is_object_dtype(event_dtype)
        or pd.api.types.is_string_dtype(event_dtype)
    )
    assert "empty_column" in working.columns
    assert report["automatic_value_changes"] == 0
    assert report["automatic_rows_removed"] == 0
    assert report["automatic_columns_removed"] == 0

    proposals = report[PROPOSED_IMPORT_ACTIONS_KEY]
    actions = _action_map(proposals)
    required = {
        ("strip_whitespace", "flag"),
        ("strip_whitespace", "note"),
        ("convert_fake_nulls", "note"),
        ("derive_boolean", "flag"),
        ("derive_datetime", "event_date"),
        ("drop_empty_rows", ""),
        ("drop_empty_columns", ""),
    }
    assert required.issubset(set(actions))
    assert all(action["default_selected"] is False for action in proposals)

    selected_ids = [
        actions[("strip_whitespace", "flag")]["id"],
        actions[("strip_whitespace", "note")]["id"],
        actions[("convert_fake_nulls", "note")]["id"],
        actions[("derive_boolean", "flag")]["id"],
        actions[("derive_datetime", "event_date")]["id"],
    ]
    reviewed, applied_steps, signatures = apply_import_actions(
        working,
        proposals,
        selected_ids,
    )

    assert tuple(reviewed.shape) == (11, 7)
    assert reviewed.loc[0, "flag"] == "yes"
    assert pd.isna(reviewed.loc[1, "note"])
    assert reviewed.loc[9, "event_date"] == "bad-date"
    assert "flag__boolean" in reviewed.columns
    assert "event_date__date" in reviewed.columns
    assert reviewed.loc[0, "flag__boolean"] == "Yes"
    assert pd.isna(reviewed.loc[9, "event_date__date"])
    assert "empty_column" in reviewed.columns
    assert len(applied_steps) == 5
    assert len(signatures) == 5

    rescanned = build_import_action_proposals(
        reviewed,
        excluded_signatures=signatures,
    )
    rescanned_signatures = {action["signature"] for action in rescanned}
    assert not set(signatures).intersection(rescanned_signatures)

    delete_ids = [
        actions[("drop_empty_rows", "")]["id"],
        actions[("drop_empty_columns", "")]["id"],
    ]
    deleted, _, _ = apply_import_actions(working, proposals, delete_ids)
    assert tuple(deleted.shape) == (10, 4)


def test_pandas_string_dtype_compatibility() -> None:
    """Text must remain text under both pandas 2.x and pandas 3.x defaults."""
    source = pd.DataFrame(
        {
            " event_date ": pd.Series(
                ["01/01/2024", "bad-date", None],
                dtype="string",
            ),
            "flag": pd.Series(["yes", "no", None], dtype="string"),
        }
    )
    report = {}
    working, _ = _finalize_import_dataframe(source, report)

    dtype = working["event_date"].dtype
    assert not pd.api.types.is_datetime64_any_dtype(dtype)
    assert pd.api.types.is_string_dtype(dtype)
    assert working.loc[1, "event_date"] == "bad-date"
    assert working.loc[0, "flag"] == "yes"
    assert report["automatic_value_changes"] == 0
    assert report["automatic_rows_removed"] == 0
    assert report["automatic_columns_removed"] == 0


def test_deletion_scope_cannot_expand_after_fake_null_conversion() -> None:
    source = pd.DataFrame(
        {
            "value": [None, "N/A", "real"],
            "other": [None, None, "x"],
        }
    )
    report = {}
    working, _ = _finalize_import_dataframe(source, report)
    proposals = report[PROPOSED_IMPORT_ACTIONS_KEY]
    actions = _action_map(proposals)

    assert actions[("drop_empty_rows", "")]["count"] == 1
    selected = [
        actions[("drop_empty_rows", "")]["id"],
        actions[("convert_fake_nulls", "value")]["id"],
    ]
    result, _, _ = apply_import_actions(working, proposals, selected)

    # Only the row reviewed as empty is deleted. The N/A row becomes empty after
    # conversion but remains present for a future review.
    assert len(result) == 2
    assert result.iloc[0].isna().all()
    assert result.iloc[1]["value"] == "real"


def test_unknown_action_is_blocked() -> None:
    source = pd.DataFrame({"value": [" x "]})
    report = {}
    working, _ = _finalize_import_dataframe(source, report)
    try:
        apply_import_actions(
            working,
            report[PROPOSED_IMPORT_ACTIONS_KEY],
            ["forged-action"],
        )
    except ValueError:
        return
    raise AssertionError("Unknown import-review action was not blocked.")



def test_import_review_context_follows_undo_redo() -> None:
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
    source = pd.DataFrame({"name": [" a ", "b"]})
    report = {}
    working, _ = _finalize_import_dataframe(source, report)
    dataset_module.activate_dataset(working, report, display_name="stage3.csv")

    state = fake_streamlit.session_state
    before_report = dict(state.data_clean_report)
    proposal = next(
        action
        for action in state.data_clean_report[PROPOSED_IMPORT_ACTIONS_KEY]
        if action["operation"] == "strip_whitespace"
    )
    updated, applied, completed = apply_import_actions(
        state.df,
        state.data_clean_report[PROPOSED_IMPORT_ACTIONS_KEY],
        [proposal["id"]],
    )

    session_module.save_history("Apply approved import actions")
    state.data_clean_report = dict(state.data_clean_report)
    state.data_clean_report["cleaning_steps"] = list(
        state.data_clean_report.get("cleaning_steps", [])
    ) + applied
    state.data_clean_report["completed_import_action_signatures"] = completed
    state.data_clean_report["import_review_fingerprint"] = dataset_module.dataframe_fingerprint(updated)
    state.df = updated
    dataset_module.reconcile_working_state()

    after_report = dict(state.data_clean_report)
    assert state.df.loc[0, "name"] == "a"
    assert dataset_module.perform_undo() is True
    assert state.df.loc[0, "name"] == " a "
    assert state.data_clean_report == before_report

    assert dataset_module.perform_redo() is True
    assert state.df.loc[0, "name"] == "a"
    assert state.data_clean_report == after_report


def main() -> None:
    test_no_automatic_mutation_and_reviewed_derivations()
    test_pandas_string_dtype_compatibility()
    test_deletion_scope_cannot_expand_after_fake_null_conversion()
    test_unknown_action_is_blocked()
    test_import_review_context_follows_undo_redo()
    print(
        "PASS: Stage 3 safe import review prevents automatic mutation and scopes approved actions correctly."
    )


if __name__ == "__main__":
    main()
