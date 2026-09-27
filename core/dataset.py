# ════════════════════════════════════════════════════════
#  DataBridge AI — Dataset activation and safety layer
#  Stage 11: unified state + atomic mutations + quality/ML context
# ════════════════════════════════════════════════════════
from __future__ import annotations

import copy
import hashlib
import inspect
import time
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import pandas as pd
import streamlit as st

from core.dataset_state import (
    DatasetChangeResult,
    DatasetMutationError,
    DatasetState,
    DatasetStateError,
    StaleDatasetRevisionError,
    ensure_dataset_state,
    sync_state_to_session,
    update_state_context,
)
from core.history import HistoryError, HistoryRestore, SmartHistoryManager
from core.session import (
    _history_context,
    append_audit_event,
    reset_file_state,
    sync_history_mirror,
)
from modules.data_mapper import analyze_dataframe, apply_manual_mappings, auto_map_columns
from modules.quality_engine import normalise_quality_policy, run_quality_engine
from modules.import_engine import (
    COMPLETED_IMPORT_SIGNATURES_KEY,
    PROPOSED_IMPORT_ACTIONS_KEY,
    RAW_DATAFRAME_REPORT_KEY,
    prepare_safe_working_dataframe,
)


_MUTABLE_OBJECT_TYPES = (dict, list, set, bytearray)


class _NoDataset(RuntimeError):
    pass


def _state() -> DatasetState:
    return ensure_dataset_state(st.session_state)


def _sync_state(state: Optional[DatasetState] = None) -> DatasetState:
    return sync_state_to_session(st.session_state, state or _state())


def clone_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return an independent DataFrame copy, including mutable object values."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    cloned = df.copy(deep=True)
    for col in cloned.select_dtypes(include="object").columns:
        series = cloned[col]
        try:
            has_mutable_values = bool(
                series.map(lambda value: isinstance(value, _MUTABLE_OBJECT_TYPES)).any()
            )
        except Exception:
            has_mutable_values = False

        if has_mutable_values:
            cloned[col] = series.map(
                lambda value: copy.deepcopy(value)
                if isinstance(value, _MUTABLE_OBJECT_TYPES)
                else value
            )
    return cloned


def dataframe_fingerprint(df: pd.DataFrame) -> str:
    """Create a stable fingerprint including values, index, columns and dtypes."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Expected a pandas DataFrame.")

    digest = hashlib.sha256()
    digest.update(str(tuple(df.shape)).encode("utf-8"))
    digest.update(repr(tuple(map(str, df.columns))).encode("utf-8"))
    digest.update(repr(tuple(map(str, df.dtypes))).encode("utf-8"))

    try:
        row_hashes = pd.util.hash_pandas_object(df, index=True).values
        digest.update(row_hashes.tobytes())
    except Exception:
        digest.update(
            df.to_json(orient="split", date_format="iso", default_handler=str).encode("utf-8")
        )
    return digest.hexdigest()


def _validate_working_dataframe(df: pd.DataFrame) -> None:
    if not isinstance(df, pd.DataFrame):
        raise DatasetMutationError("A dataset operation must return a pandas DataFrame.")
    if len(df.columns) == 0:
        raise DatasetMutationError("The operation would leave the dataset with no columns.")
    if not df.columns.is_unique:
        duplicates = list(dict.fromkeys(map(str, df.columns[df.columns.duplicated()])))
        raise DatasetMutationError(
            "Duplicate column names are not allowed: " + ", ".join(duplicates[:10])
        )


def _extract_raw_and_report(
    parsed_df: pd.DataFrame,
    clean_report: Dict[str, Any] | None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    report = dict(clean_report or {})
    raw_candidate = report.pop(RAW_DATAFRAME_REPORT_KEY, None)
    if raw_candidate is None:
        raw_candidate = parsed_df
    if not isinstance(raw_candidate, pd.DataFrame):
        raise TypeError("The import engine returned an invalid raw dataset snapshot.")
    return clone_dataframe(raw_candidate), report


def _schema_signature(df: pd.DataFrame) -> Dict[str, tuple[str, ...]]:
    return {
        "columns": tuple(map(str, df.columns)),
        "dtypes": tuple(map(str, df.dtypes)),
    }


def _context_matches_columns(context: Dict[str, Any], df: pd.DataFrame) -> bool:
    mappings = context.get("column_mappings") or {}
    confidence = context.get("mapping_confidence") or {}
    columns = set(df.columns)
    return set(mappings.keys()) == columns and set(confidence.keys()) == columns


def _filter_kpi_targets(targets: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    numeric = set(df.select_dtypes(include="number").columns)
    return {
        key: copy.deepcopy(value)
        for key, value in (targets or {}).items()
        if key in numeric
    }


def _run_quality_scan(
    df: pd.DataFrame,
    *,
    policy: Mapping[str, Any],
    semantic_profiles: Mapping[str, Any],
    revision: int,
    fingerprint: str,
) -> Dict[str, Any]:
    """Call Quality V2 while preserving compatibility with legacy test doubles."""
    try:
        parameters = inspect.signature(run_quality_engine).parameters
    except (TypeError, ValueError):
        parameters = {}
    if parameters:
        accepts_var_keywords = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        positional = [
            parameter
            for parameter in parameters.values()
            if parameter.kind in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            }
        ]
        keyword_names = set(parameters)
        supports_v2 = accepts_var_keywords or {
            "policy", "semantic_profiles", "dataset_revision", "dataset_fingerprint"
        }.issubset(keyword_names)
        if len(positional) == 1 and not supports_v2:
            return run_quality_engine(df)
    return run_quality_engine(
        df,
        policy=policy,
        semantic_profiles=semantic_profiles,
        dataset_revision=revision,
        dataset_fingerprint=fingerprint,
    )


def _refresh_dependent_state(
    df: pd.DataFrame,
    *,
    schema_changed: bool,
    restored_context: Dict[str, Any] | None = None,
) -> None:
    """Refresh quality, semantic typing, ML readiness, and dependent context."""
    state = _state()
    prior_quality_report = copy.deepcopy(state.quality_report)
    context = dict(restored_context or {})
    columns = set(map(str, df.columns))
    if restored_context is not None:
        state.feature_derivation_recipe = copy.deepcopy(context.get("feature_derivation_recipe", []))

    if restored_context is not None and _context_matches_columns(context, df):
        state.mapping_confidence = copy.deepcopy(context.get("mapping_confidence", {}))
        state.column_mappings = copy.deepcopy(context.get("column_mappings", {}))
        state.mapper_approved = bool(context.get("mapper_approved", False))

        restored_profiles = context.get("semantic_profiles") or {}
        restored_readiness = context.get("ml_readiness_report") or {}
        if set(restored_profiles.keys()) == columns and restored_readiness:
            state.semantic_profiles = copy.deepcopy(restored_profiles)
            state.ml_readiness_report = copy.deepcopy(restored_readiness)
        else:
            analysis = analyze_dataframe(df)
            if state.mapper_approved:
                try:
                    profiles, readiness = apply_manual_mappings(
                        df, analysis["profiles"], state.column_mappings
                    )
                    state.semantic_profiles = profiles
                    state.ml_readiness_report = readiness
                except ValueError:
                    # Upgrade compatibility: old business-role mappings are not
                    # valid Stage 6 ML semantic types and must be reviewed again.
                    state.column_mappings = {
                        col: semantic_type
                        for col, (semantic_type, _) in analysis["mappings"].items()
                    }
                    state.mapper_approved = False
                    state.mapping_confidence = analysis["mappings"]
                    state.semantic_profiles = analysis["profiles"]
                    state.ml_readiness_report = analysis["readiness"]
            else:
                state.semantic_profiles = analysis["profiles"]
                state.ml_readiness_report = analysis["readiness"]
    else:
        analysis = analyze_dataframe(df)
        state.mapping_confidence = analysis["mappings"]
        mappings_invalid = set(state.column_mappings.keys()) != columns

        if schema_changed or mappings_invalid:
            state.column_mappings = {
                col: semantic_type
                for col, (semantic_type, _) in analysis["mappings"].items()
            }
            state.mapper_approved = False
            state.semantic_profiles = analysis["profiles"]
            state.ml_readiness_report = analysis["readiness"]
        elif state.mapper_approved:
            try:
                profiles, readiness = apply_manual_mappings(
                    df, analysis["profiles"], state.column_mappings
                )
                state.semantic_profiles = profiles
                state.ml_readiness_report = readiness
            except ValueError:
                state.column_mappings = {
                    col: semantic_type
                    for col, (semantic_type, _) in analysis["mappings"].items()
                }
                state.mapper_approved = False
                state.semantic_profiles = analysis["profiles"]
                state.ml_readiness_report = analysis["readiness"]
        else:
            state.column_mappings = {
                col: semantic_type
                for col, (semantic_type, _) in analysis["mappings"].items()
            }
            state.semantic_profiles = analysis["profiles"]
            state.ml_readiness_report = analysis["readiness"]

    source_quality_policy = (
        context.get("quality_policy", {})
        if restored_context is not None
        else state.quality_policy
    )
    state.quality_policy = normalise_quality_policy(
        df, source_quality_policy, state.semantic_profiles
    )
    current_quality_report = _run_quality_scan(
        df,
        policy=state.quality_policy,
        semantic_profiles=state.semantic_profiles,
        revision=int(state.revision),
        fingerprint=str(state.working_fingerprint or ""),
    )
    if restored_context is not None:
        restored_baseline = context.get("quality_baseline_report", {}) or {}
        restored_previous = context.get("quality_previous_report", {}) or {}
        state.quality_baseline_report = copy.deepcopy(
            restored_baseline or state.quality_baseline_report
        )
        state.quality_previous_report = copy.deepcopy(restored_previous)
    else:
        state.quality_previous_report = prior_quality_report
    state.quality_report = current_quality_report
    if not state.quality_baseline_report:
        state.quality_baseline_report = copy.deepcopy(current_quality_report)

    source_targets = (
        context.get("kpi_targets", {})
        if restored_context is not None
        else state.kpi_targets
    )
    state.kpi_targets = _filter_kpi_targets(source_targets, df)

    from modules.feature_pipeline import reconcile_feature_pipeline_context

    source_pipeline_spec = (
        context.get("feature_pipeline_spec", {})
        if restored_context is not None
        else state.feature_pipeline_spec
    )
    state.feature_pipeline_spec, state.feature_pipeline_report = (
        reconcile_feature_pipeline_context(
            df,
            source_pipeline_spec,
            current_revision=int(state.revision),
            current_fingerprint=str(state.working_fingerprint or ""),
        )
    )

    from modules.feature_pipeline import feature_pipeline_spec_fingerprint
    from modules.ml_engine import reconcile_ml_experiment_report

    source_experiment_report = (
        context.get("ml_experiment_report", {})
        if restored_context is not None
        else state.ml_experiment_report
    )
    pipeline_fp = (
        feature_pipeline_spec_fingerprint(state.feature_pipeline_spec)
        if state.feature_pipeline_spec
        else ""
    )
    state.ml_experiment_report = reconcile_ml_experiment_report(
        source_experiment_report,
        current_revision=int(state.revision),
        current_fingerprint=str(state.working_fingerprint or ""),
        current_pipeline_spec_fingerprint=pipeline_fp,
        artifact_available=False,
    )

    from modules.model_explainability import reconcile_explainability_report

    source_explainability_report = (
        context.get("explainability_report", {})
        if restored_context is not None
        else state.explainability_report
    )
    state.explainability_report = reconcile_explainability_report(
        source_explainability_report,
        current_revision=int(state.revision),
        current_fingerprint=str(state.working_fingerprint or ""),
        current_experiment_id=str(state.ml_experiment_report.get("experiment_id", "") or ""),
        current_pipeline_spec_fingerprint=pipeline_fp,
        artifact_available=False,
    )

    if restored_context is not None and "data_clean_report" in context:
        state.data_clean_report = copy.deepcopy(context.get("data_clean_report", {}))
        state.show_import_report = bool(context.get("show_import_report", False))

    # Previous AI messages, feature previews, and fitted ML artifacts describe an older revision.
    st.session_state.ai_messages = []
    st.session_state.feature_pipeline_preview = None
    st.session_state.ml_experiment_artifact = None
    st.session_state.ml_clustering_result = None
    st.session_state.model_explainability_artifact = None
    st.session_state.local_explanation_result = None
    st.session_state.shap_explanation_result = None
    _sync_state(state)


def _install_working_dataframe(
    working_df: pd.DataFrame,
    *,
    reset_analysis: bool,
    action_label: str,
    restored_context: Dict[str, Any] | None = None,
    explicit_revision: int | None = None,
) -> None:
    _validate_working_dataframe(working_df)
    state = _state()
    working_copy = clone_dataframe(working_df)
    state.working_df = working_copy
    state.working_fingerprint = dataframe_fingerprint(working_copy)
    state.revision = (
        int(explicit_revision)
        if explicit_revision is not None
        else int(state.revision) + 1
    )
    state.last_action = action_label
    st.session_state.pending_dataset_change = None
    _sync_state(state)

    if reset_analysis:
        _refresh_dependent_state(
            working_copy,
            schema_changed=True,
            restored_context=restored_context,
        )


def activate_dataset(
    parsed_df: pd.DataFrame,
    clean_report: Dict[str, Any] | None,
    display_name: str | None = None,
) -> None:
    """Activate a protected raw snapshot and an independent working dataset."""
    if parsed_df is None or not isinstance(parsed_df, pd.DataFrame):
        raise ValueError("The imported dataset is invalid.")
    if parsed_df.empty:
        raise ValueError("The imported dataset is empty.")
    _validate_working_dataframe(parsed_df)

    raw_df, public_report = _extract_raw_and_report(parsed_df, clean_report)
    reset_file_state()
    state = _state()

    state.raw_df = raw_df
    state.raw_fingerprint = dataframe_fingerprint(raw_df)
    state.raw_shape = tuple(raw_df.shape)
    state.data_clean_report = public_report
    state.file_name = (
        display_name
        or public_report.get("source_name")
        or public_report.get("table_selected")
        or "Imported dataset"
    )
    _sync_state(state)

    _install_working_dataframe(
        parsed_df,
        reset_analysis=True,
        action_label="Dataset imported safely",
    )
    state = _state()
    state.initial_working_shape = tuple(state.working_df.shape) if state.working_df is not None else None
    public_report["import_review_fingerprint"] = state.working_fingerprint
    public_report["import_review_shape"] = state.shape
    public_report["import_review_status"] = (
        "pending" if public_report.get(PROPOSED_IMPORT_ACTIONS_KEY) else "clean"
    )
    state.data_clean_report = public_report
    state.show_import_report = True
    _sync_state(state)

    append_audit_event(
        {
            "event": "activate",
            "action": "Dataset imported",
            "after_shape": state.shape,
            "after_fingerprint": state.working_fingerprint,
            "revision": state.revision,
        }
    )


def deactivate_dataset() -> None:
    """Release the active dataset from the current session without touching its source.

    This clears protected Raw/Working in-memory state, Undo/Redo history, derived ML
    context, prediction/monitoring inputs, and related dataset artifacts through the
    same reset path used before activating a new source. No uploaded source file or
    external database object is modified or deleted.
    """
    reset_file_state()
    st.session_state["pre_dataset_page"] = "upload"


def get_working_dataframe(*, copy_frame: bool = False) -> pd.DataFrame:
    state = _state()
    if state.working_df is None:
        raise DatasetStateError("No working dataset is loaded.")
    return clone_dataframe(state.working_df) if copy_frame else state.working_df


def current_dataset_revision() -> int:
    return int(_state().revision)


def update_dataset_context(
    *,
    expected_revision: int | None = None,
    **updates: Any,
) -> None:
    """Update non-DataFrame dataset context through the authoritative state."""
    reconcile_working_state()
    state = _state()
    if expected_revision is not None and int(expected_revision) != int(state.revision):
        raise StaleDatasetRevisionError(
            "The dataset changed while this page was open. Refresh the page and try again."
        )
    update_state_context(state, updates)
    _sync_state(state)


def refresh_quality_report(
    *,
    policy: Mapping[str, Any] | None = None,
    set_baseline: bool = False,
    expected_revision: int | None = None,
) -> Dict[str, Any]:
    """Re-evaluate Quality Engine V2 without changing the working dataset."""
    reconcile_working_state()
    state = _state()
    if state.working_df is None:
        raise DatasetStateError("No working dataset is loaded.")
    if expected_revision is not None and int(expected_revision) != int(state.revision):
        raise StaleDatasetRevisionError(
            "The dataset changed while this page was open. Refresh the page and try again."
        )

    selected_policy = state.quality_policy if policy is None else policy
    resolved_policy = normalise_quality_policy(
        state.working_df, selected_policy, state.semantic_profiles
    )
    previous = copy.deepcopy(state.quality_report)
    report = _run_quality_scan(
        state.working_df,
        policy=resolved_policy,
        semantic_profiles=state.semantic_profiles,
        revision=int(state.revision),
        fingerprint=str(state.working_fingerprint or ""),
    )
    state.quality_previous_report = previous
    state.quality_policy = resolved_policy
    state.quality_report = report
    if set_baseline or not state.quality_baseline_report:
        state.quality_baseline_report = copy.deepcopy(report)
    _sync_state(state)
    append_audit_event(
        {
            "event": "quality_scan",
            "action": "Quality Engine V2 policy applied" if policy is not None else "Quality Engine V2 refreshed",
            "revision": int(state.revision),
            "quality_score": report.get("quality_score"),
            "baseline_reset": bool(set_baseline),
        }
    )
    return report


def _capture_atomic_rollback() -> Dict[str, Any]:
    state = _state()
    return {
        "working_df": state.working_df,
        "working_fingerprint": state.working_fingerprint,
        "revision": state.revision,
        "last_action": state.last_action,
        "mapper_approved": state.mapper_approved,
        "column_mappings": copy.deepcopy(state.column_mappings),
        "mapping_confidence": copy.deepcopy(state.mapping_confidence),
        "semantic_profiles": copy.deepcopy(state.semantic_profiles),
        "ml_readiness_report": copy.deepcopy(state.ml_readiness_report),
        "feature_derivation_recipe": copy.deepcopy(state.feature_derivation_recipe),
        "feature_pipeline_spec": copy.deepcopy(state.feature_pipeline_spec),
        "feature_pipeline_report": copy.deepcopy(state.feature_pipeline_report),
        "ml_experiment_report": copy.deepcopy(state.ml_experiment_report),
        "explainability_report": copy.deepcopy(state.explainability_report),
        "quality_report": copy.deepcopy(state.quality_report),
        "quality_policy": copy.deepcopy(state.quality_policy),
        "quality_baseline_report": copy.deepcopy(state.quality_baseline_report),
        "quality_previous_report": copy.deepcopy(state.quality_previous_report),
        "kpi_targets": copy.deepcopy(state.kpi_targets),
        "data_clean_report": copy.deepcopy(state.data_clean_report),
        "show_import_report": state.show_import_report,
        "ai_messages": copy.deepcopy(st.session_state.get("ai_messages", [])),
        "ml_experiment_artifact": st.session_state.get("ml_experiment_artifact"),
        "ml_clustering_result": st.session_state.get("ml_clustering_result"),
        "model_explainability_artifact": st.session_state.get("model_explainability_artifact"),
        "local_explanation_result": st.session_state.get("local_explanation_result"),
        "shap_explanation_result": st.session_state.get("shap_explanation_result"),
    }


def _restore_atomic_rollback(rollback: Mapping[str, Any]) -> None:
    state = _state()
    state.working_df = rollback["working_df"]
    state.working_fingerprint = rollback["working_fingerprint"]
    state.revision = int(rollback["revision"])
    state.last_action = str(rollback["last_action"])
    state.mapper_approved = bool(rollback["mapper_approved"])
    state.column_mappings = copy.deepcopy(rollback["column_mappings"])
    state.mapping_confidence = copy.deepcopy(rollback["mapping_confidence"])
    state.semantic_profiles = copy.deepcopy(rollback["semantic_profiles"])
    state.ml_readiness_report = copy.deepcopy(rollback["ml_readiness_report"])
    state.feature_derivation_recipe = copy.deepcopy(rollback["feature_derivation_recipe"])
    state.feature_pipeline_spec = copy.deepcopy(rollback["feature_pipeline_spec"])
    state.feature_pipeline_report = copy.deepcopy(rollback["feature_pipeline_report"])
    state.ml_experiment_report = copy.deepcopy(rollback["ml_experiment_report"])
    state.explainability_report = copy.deepcopy(rollback["explainability_report"])
    state.quality_report = copy.deepcopy(rollback["quality_report"])
    state.quality_policy = copy.deepcopy(rollback["quality_policy"])
    state.quality_baseline_report = copy.deepcopy(rollback["quality_baseline_report"])
    state.quality_previous_report = copy.deepcopy(rollback["quality_previous_report"])
    state.kpi_targets = copy.deepcopy(rollback["kpi_targets"])
    state.data_clean_report = copy.deepcopy(rollback["data_clean_report"])
    state.show_import_report = bool(rollback["show_import_report"])
    st.session_state.ai_messages = copy.deepcopy(rollback["ai_messages"])
    st.session_state.ml_experiment_artifact = rollback.get("ml_experiment_artifact")
    st.session_state.ml_clustering_result = rollback.get("ml_clustering_result")
    st.session_state.model_explainability_artifact = rollback.get("model_explainability_artifact")
    st.session_state.local_explanation_result = rollback.get("local_explanation_result")
    st.session_state.shap_explanation_result = rollback.get("shap_explanation_result")
    st.session_state.pending_dataset_change = None
    _sync_state(state)


def apply_dataset_change(
    action: str,
    transform: Callable[[pd.DataFrame], Optional[pd.DataFrame]],
    *,
    details: Optional[Dict[str, Any]] = None,
    expected_revision: int | None = None,
    context_updates: Optional[Mapping[str, Any]] = None,
) -> DatasetChangeResult:
    """
    Apply one dataset mutation atomically.

    The transform receives an isolated working copy. The active dataset is not
    changed unless transformation, validation, full history snapshot, dependent
    analysis refresh, and raw-integrity checks all succeed.
    """
    if not callable(transform):
        raise TypeError("transform must be callable.")

    reconcile_working_state()
    state = _state()
    if state.working_df is None:
        raise DatasetStateError("No working dataset is loaded.")
    if not raw_dataset_is_intact():
        raise DatasetStateError(
            "The protected raw dataset integrity check failed. The change was blocked."
        )
    if expected_revision is not None and int(expected_revision) != int(state.revision):
        raise StaleDatasetRevisionError(
            "The dataset changed while this page was open. Refresh the page and try again."
        )

    action_label = (action or "Dataset change").strip()[:240]
    before_df = state.working_df
    before_fp = state.working_fingerprint or dataframe_fingerprint(before_df)
    before_shape = tuple(before_df.shape)
    before_schema = _schema_signature(before_df)
    before_revision = int(state.revision)

    isolated = clone_dataframe(before_df)
    try:
        transformed = transform(isolated)
    except Exception as exc:
        raise DatasetMutationError(
            f"'{action_label}' failed before any data was changed: {exc}"
        ) from exc

    candidate = isolated if transformed is None else transformed
    _validate_working_dataframe(candidate)
    candidate = clone_dataframe(candidate)
    after_fp = dataframe_fingerprint(candidate)
    after_shape = tuple(candidate.shape)

    if after_fp == before_fp:
        _sync_state(state)
        sync_history_mirror()
        return DatasetChangeResult(
            changed=False,
            action=action_label,
            revision=before_revision,
            before_shape=before_shape,
            after_shape=after_shape,
            before_fingerprint=before_fp,
            after_fingerprint=after_fp,
        )

    manager = st.session_state.get("history_manager")
    if not isinstance(manager, SmartHistoryManager):
        raise DatasetStateError("The history manager is unavailable. The change was blocked.")

    rollback = _capture_atomic_rollback()
    snapshot_id = ""
    try:
        snapshot_id = manager.push(
            before_df,
            action=action_label,
            fingerprint=before_fp,
            context=_history_context(),
            clear_redo=False,
        )

        schema_changed = before_schema != _schema_signature(candidate)
        state.working_df = candidate
        state.working_fingerprint = after_fp
        state.revision = before_revision + 1
        state.last_action = action_label
        st.session_state.pending_dataset_change = None
        _sync_state(state)
        _refresh_dependent_state(candidate, schema_changed=schema_changed)

        if context_updates:
            update_state_context(state, context_updates)
            _sync_state(state)

        if not raw_dataset_is_intact():
            raise DatasetStateError(
                "The protected raw dataset changed during the operation. The working change was rolled back."
            )

        if not manager.commit_push(snapshot_id):
            raise HistoryError("The undo snapshot could not be committed safely.")
        sync_history_mirror()
        st.session_state.history_warning = ""

        append_audit_event(
            {
                "event": "atomic_change",
                "action": action_label,
                "details": copy.deepcopy(details or {}),
                "before_shape": before_shape,
                "after_shape": after_shape,
                "before_fingerprint": before_fp,
                "after_fingerprint": after_fp,
                "schema_changed": schema_changed,
                "revision": state.revision,
            }
        )
    except Exception as exc:
        if snapshot_id:
            manager.cancel_push(snapshot_id)
        _restore_atomic_rollback(rollback)
        sync_history_mirror()
        if isinstance(exc, DatasetStateError):
            raise
        if isinstance(exc, HistoryError):
            raise
        raise DatasetMutationError(
            f"'{action_label}' was rolled back safely: {exc}"
        ) from exc

    return DatasetChangeResult(
        changed=True,
        action=action_label,
        revision=state.revision,
        before_shape=before_shape,
        after_shape=after_shape,
        before_fingerprint=before_fp,
        after_fingerprint=after_fp,
        snapshot_id=snapshot_id,
    )


def replace_working_dataframe(
    new_df: pd.DataFrame,
    *,
    action: str,
    details: Optional[Dict[str, Any]] = None,
    expected_revision: int | None = None,
    context_updates: Optional[Mapping[str, Any]] = None,
) -> DatasetChangeResult:
    """Atomically replace the working dataset with a validated DataFrame."""
    if not isinstance(new_df, pd.DataFrame):
        raise DatasetMutationError("The replacement dataset is not a pandas DataFrame.")
    prepared = clone_dataframe(new_df)
    return apply_dataset_change(
        action,
        lambda _working: prepared,
        details=details,
        expected_revision=expected_revision,
        context_updates=context_updates,
    )


def reconcile_working_state() -> bool:
    """
    Synchronize legacy direct mutations and compatibility mirrors.

    DataBridge Stage 5 pages no longer use this path. It remains as a guarded
    compatibility layer for older extensions and previous automated tests.
    """
    state = _state()
    legacy_df = st.session_state.get("df")

    if state.working_df is None and legacy_df is None:
        _sync_state(state)
        return False
    if legacy_df is None:
        _sync_state(state)
        return False
    if not isinstance(legacy_df, pd.DataFrame):
        raise DatasetStateError("The working dataset compatibility mirror is invalid.")
    if state.raw_df is not None and not raw_dataset_is_intact():
        raise DatasetStateError("The protected raw dataset integrity check failed.")

    current_fingerprint = dataframe_fingerprint(legacy_df)
    expected_fingerprint = state.working_fingerprint
    pending = st.session_state.get("pending_dataset_change")
    manager = st.session_state.get("history_manager")

    if expected_fingerprint is None:
        state.working_df = legacy_df
        state.working_fingerprint = current_fingerprint
        _sync_state(state)
        sync_history_mirror()
        return False

    if current_fingerprint == expected_fingerprint:
        # Repair any replaced compatibility aliases without altering the state.
        if legacy_df is not state.working_df:
            _sync_state(state)
        if pending:
            if isinstance(manager, SmartHistoryManager):
                manager.cancel_push(str(pending.get("snapshot_id", "")))
            st.session_state.pending_dataset_change = None
            sync_history_mirror()
        return False

    before_shape = tuple(pending.get("before_shape", ())) if pending else None
    before_schema = pending.get("before_schema", {}) if pending else {}
    schema_changed = before_schema != _schema_signature(legacy_df) if before_schema else True
    action = pending.get("action", "Legacy untracked dataset change") if pending else "Legacy untracked dataset change"
    details = copy.deepcopy(pending.get("details", {})) if pending else {}
    before_fp = pending.get("before_fingerprint", expected_fingerprint) if pending else expected_fingerprint

    state.working_df = legacy_df
    state.working_fingerprint = current_fingerprint
    state.revision = int(state.revision) + 1
    state.last_action = action
    st.session_state.pending_dataset_change = None
    if pending and isinstance(manager, SmartHistoryManager):
        manager.commit_push(str(pending.get("snapshot_id", "")))
    _refresh_dependent_state(legacy_df, schema_changed=schema_changed)
    _sync_state(state)
    sync_history_mirror()

    append_audit_event(
        {
            "event": "legacy_change" if pending else "untracked_change",
            "action": action,
            "details": details,
            "before_shape": before_shape,
            "after_shape": tuple(legacy_df.shape),
            "before_fingerprint": before_fp,
            "after_fingerprint": current_fingerprint,
            "schema_changed": schema_changed,
            "revision": state.revision,
            "duration_seconds": round(
                time.time() - float(pending.get("started_at", time.time())), 4
            ) if pending else None,
        }
    )
    return True


def _apply_history_restore(restore: HistoryRestore, direction: str) -> None:
    restored_df = clone_dataframe(restore.dataframe)
    actual_fingerprint = dataframe_fingerprint(restored_df)
    if restore.fingerprint and actual_fingerprint != restore.fingerprint:
        raise HistoryError("History integrity verification failed. The restore was blocked.")

    state = _state()
    if state.working_df is None:
        raise DatasetStateError("No working dataset is loaded.")
    before_shape = tuple(state.working_df.shape)
    before_fingerprint = state.working_fingerprint
    _install_working_dataframe(
        restored_df,
        reset_analysis=True,
        action_label=f"{direction}: {restore.action}",
        restored_context=restore.context,
    )
    state = _state()
    sync_history_mirror()
    append_audit_event(
        {
            "event": direction.lower(),
            "action": restore.action,
            "before_shape": before_shape,
            "after_shape": tuple(restored_df.shape),
            "before_fingerprint": before_fingerprint,
            "after_fingerprint": state.working_fingerprint,
            "snapshot_id": restore.snapshot_id,
            "revision": state.revision,
        }
    )


def perform_undo() -> bool:
    """Atomically restore the previous complete working state."""
    reconcile_working_state()
    manager = st.session_state.get("history_manager")
    state = _state()
    if (
        not isinstance(manager, SmartHistoryManager)
        or not manager.can_undo
        or state.working_df is None
    ):
        return False

    restore = manager.undo(
        state.working_df,
        current_context=_history_context(),
        current_fingerprint=state.working_fingerprint or dataframe_fingerprint(state.working_df),
    )
    if restore is None:
        return False
    _apply_history_restore(restore, "Undo")
    st.session_state.history_warning = ""
    return True


def perform_redo() -> bool:
    """Atomically restore the next complete working state."""
    reconcile_working_state()
    manager = st.session_state.get("history_manager")
    state = _state()
    if (
        not isinstance(manager, SmartHistoryManager)
        or not manager.can_redo
        or state.working_df is None
    ):
        return False

    restore = manager.redo(
        state.working_df,
        current_context=_history_context(),
        current_fingerprint=state.working_fingerprint or dataframe_fingerprint(state.working_df),
    )
    if restore is None:
        return False
    _apply_history_restore(restore, "Redo")
    st.session_state.history_warning = ""
    return True


def restore_raw_dataset() -> None:
    """Rebuild the working dataset from the protected source safely."""
    state = _state()
    raw_df = state.raw_df
    if raw_df is None or not isinstance(raw_df, pd.DataFrame):
        raise RuntimeError("No protected original dataset is available.")
    if not raw_dataset_is_intact():
        raise DatasetStateError("The protected original failed its integrity check.")

    old_manager = st.session_state.get("history_manager")
    if isinstance(old_manager, SmartHistoryManager):
        old_manager.close()
    st.session_state.history_manager = SmartHistoryManager()
    st.session_state.df_history = []
    st.session_state.pending_dataset_change = None

    safe_working, structural_steps, proposals = prepare_safe_working_dataframe(raw_df)
    before_shape = state.shape
    state.feature_derivation_recipe = []
    _sync_state(state)

    _install_working_dataframe(
        safe_working,
        reset_analysis=True,
        action_label="Protected original restored safely",
    )
    state = _state()

    report = copy.deepcopy(state.data_clean_report or {})
    retained_steps = [
        step
        for step in report.get("cleaning_steps", [])
        if not (isinstance(step, dict) and step.get("signature"))
    ]
    existing_operations = {
        str(step.get("operation"))
        for step in retained_steps
        if isinstance(step, dict) and step.get("operation")
    }
    for step in structural_steps:
        operation = str(step.get("operation", ""))
        if operation not in existing_operations:
            retained_steps.append(step)
            existing_operations.add(operation)

    report["cleaning_steps"] = retained_steps
    report[PROPOSED_IMPORT_ACTIONS_KEY] = proposals
    report[COMPLETED_IMPORT_SIGNATURES_KEY] = []
    report["approved_import_actions"] = []
    report["safe_import_policy"] = "review_required"
    report["automatic_value_changes"] = 0
    report["automatic_rows_removed"] = 0
    report["automatic_columns_removed"] = 0
    report["initial_working_shape"] = tuple(safe_working.shape)
    report["import_review_fingerprint"] = state.working_fingerprint
    report["import_review_shape"] = tuple(safe_working.shape)
    report["import_review_status"] = "pending" if proposals else "clean"
    state.data_clean_report = report
    state.show_import_report = True
    _sync_state(state)

    st.session_state.restore_original_confirm = False
    st.session_state.dataset_audit_log = []
    append_audit_event(
        {
            "event": "restore_original",
            "action": "Protected original restored safely",
            "before_shape": before_shape,
            "after_shape": state.shape,
            "after_fingerprint": state.working_fingerprint,
            "revision": state.revision,
        }
    )


def get_raw_dataset_copy() -> pd.DataFrame:
    state = _state()
    if state.raw_df is None or not isinstance(state.raw_df, pd.DataFrame):
        raise RuntimeError("No protected original dataset is available.")
    return clone_dataframe(state.raw_df)


def raw_dataset_is_intact() -> bool:
    state = _state()
    if state.raw_df is None or state.raw_fingerprint is None:
        return False
    try:
        return dataframe_fingerprint(state.raw_df) == state.raw_fingerprint
    except Exception:
        return False


def dataset_state_health(*, deep: bool = False) -> Dict[str, Any]:
    """Return state/mirror consistency information for diagnostics and tests."""
    state = _state()
    working_loaded = state.working_df is not None
    mirror_ok = (
        st.session_state.get("df") is state.working_df
        and st.session_state.get("hdf") is state.working_df
        and st.session_state.get("raw_df") is state.raw_df
        and st.session_state.get("quality_report") is state.quality_report
        and st.session_state.get("quality_policy") is state.quality_policy
        and st.session_state.get("quality_baseline_report") is state.quality_baseline_report
        and st.session_state.get("quality_previous_report") is state.quality_previous_report
        and st.session_state.get("column_mappings") is state.column_mappings
        and st.session_state.get("mapping_confidence") is state.mapping_confidence
        and st.session_state.get("semantic_profiles") is state.semantic_profiles
        and st.session_state.get("ml_readiness_report") is state.ml_readiness_report
        and st.session_state.get("feature_derivation_recipe") is state.feature_derivation_recipe
        and st.session_state.get("feature_pipeline_spec") is state.feature_pipeline_spec
        and st.session_state.get("feature_pipeline_report") is state.feature_pipeline_report
        and st.session_state.get("ml_experiment_report") is state.ml_experiment_report
        and st.session_state.get("explainability_report") is state.explainability_report
        and st.session_state.get("kpi_targets") is state.kpi_targets
        and st.session_state.get("data_clean_report") is state.data_clean_report
    )
    fingerprint_ok = True
    raw_ok = True
    if deep and working_loaded:
        fingerprint_ok = dataframe_fingerprint(state.working_df) == state.working_fingerprint
    if deep and state.raw_df is not None:
        raw_ok = raw_dataset_is_intact()
    return {
        "loaded": working_loaded,
        "revision": int(state.revision),
        "shape": state.shape,
        "mirrors_consistent": mirror_ok,
        "working_fingerprint_valid": fingerprint_ok,
        "raw_intact": raw_ok,
        "ok": mirror_ok and fingerprint_ok and raw_ok,
    }


def assert_dataset_state_consistent(*, deep: bool = True) -> None:
    health = dataset_state_health(deep=deep)
    if not health["ok"]:
        raise DatasetStateError(f"Dataset state consistency check failed: {health}")
