# ════════════════════════════════════════════════════════
#  DataBridge AI — Session State Initialiser
#  Stage 11: unified state + quality policy/baseline context
# ════════════════════════════════════════════════════════
from __future__ import annotations

import copy
import time
from typing import Any, Dict, Optional

import streamlit as st

from core.audit import sanitise_audit_event, write_persistent_audit_event
from core.dataset_state import DatasetState, ensure_dataset_state, sync_state_to_session
from core.history import HistoryError, SmartHistoryManager


AUDIT_LOG_LIMIT = 250


def init_session_state() -> None:
    """Initialise every session-state key exactly once."""
    defaults = {
        "is_authenticated": False,
        "current_user_id": "",
        "current_user": "",
        "current_display_name": "",
        "current_role": "",
        "current_permissions": [],
        "current_user_revision": 0,
        "current_user_must_change_password": False,
        "current_user_env_managed": False,
        "dataset_state": None,
        "raw_df": None,
        "df": None,
        "hdf": None,
        "standard_df": None,
        "file_name": None,
        "raw_shape": None,
        "initial_working_shape": None,
        "raw_fingerprint": None,
        "working_fingerprint": None,
        "df_fingerprint": None,
        "dataset_revision": 0,
        "last_dataset_action": "",
        "restore_original_confirm": False,
        "df_history": [],
        "pending_dataset_change": None,
        "dataset_audit_log": [],
        "history_warning": "",
        "data_clean_report": {},
        "mapper_approved": False,
        "column_mappings": {},
        "mapping_confidence": {},
        "semantic_profiles": {},
        "ml_readiness_report": {},
        "feature_derivation_recipe": [],
        "feature_pipeline_spec": {},
        "feature_pipeline_report": {},
        "feature_pipeline_preview": None,
        "ml_experiment_report": {},
        "ml_experiment_artifact": None,
        "ml_clustering_result": None,
        "explainability_report": {},
        "model_explainability_artifact": None,
        "local_explanation_result": None,
        "shap_explanation_result": None,
        "model_package_bytes": None,
        "model_package_manifest": {},
        "model_package_file_name": "",
        "loaded_model_package": None,
        "loaded_model_package_bytes": None,
        "prediction_input_df": None,
        "prediction_input_report": {},
        "prediction_input_name": "",
        "prediction_result": None,
        "model_monitoring_input_df": None,
        "model_monitoring_input_name": "",
        "model_monitoring_report": {},
        "model_monitoring_feature_table": None,
        "model_governance_assessment": None,
        "retraining_candidate_result": None,
        "retraining_last_report": {},
        "quality_report": {},
        "quality_policy": {},
        "quality_baseline_report": {},
        "quality_previous_report": {},
        "kpi_targets": {},
        "ai_mode": "demo",
        "ai_engine": None,
        "ai_messages": [],
        "show_import_report": False,
        "ai_privacy_mode": "metadata",
        "ai_raw_transfer_confirm": False,
        "ai_remote_endpoint_confirm": False,
        "ai_secret_source": "not_configured",
        "gemini_allow_data": False,
        "gemini_mask_pii": True,
    }

    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = copy.deepcopy(default)

    state = ensure_dataset_state(st.session_state)
    sync_state_to_session(st.session_state, state)

    if "history_manager" not in st.session_state or not isinstance(
        st.session_state.history_manager, SmartHistoryManager
    ):
        st.session_state.history_manager = SmartHistoryManager()


def _history_context() -> Dict[str, Any]:
    """Capture state that must follow a DataFrame through Undo/Redo."""
    state = ensure_dataset_state(st.session_state)
    return {
        "mapper_approved": bool(state.mapper_approved),
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
        "show_import_report": bool(state.show_import_report),
    }


def sync_history_mirror() -> None:
    """Keep df_history as lightweight metadata, never DataFrames."""
    manager = st.session_state.get("history_manager")
    st.session_state.df_history = manager.metadata() if isinstance(manager, SmartHistoryManager) else []


def append_audit_event(event: Dict[str, Any]) -> None:
    """Append a bounded, sanitized audit event and persist a safe JSONL copy."""
    log = list(st.session_state.get("dataset_audit_log", []))
    source = dict(event)
    source.setdefault("timestamp", time.time())
    source.setdefault("actor", str(st.session_state.get("current_user") or "system"))
    source.setdefault("actor_role", str(st.session_state.get("current_role") or "system"))
    item = sanitise_audit_event(source)
    log.append(item)
    st.session_state.dataset_audit_log = log[-AUDIT_LOG_LIMIT:]
    write_persistent_audit_event(item)


def save_history(
    action: str = "Dataset change",
    *,
    details: Optional[Dict[str, Any]] = None,
) -> str:
    """Legacy pre-change transaction API for older third-party pages."""
    state = ensure_dataset_state(st.session_state)
    if state.working_df is None:
        raise HistoryError("No working dataset is loaded.")

    from core.dataset import dataframe_fingerprint, reconcile_working_state

    reconcile_working_state()
    state = ensure_dataset_state(st.session_state)
    df = state.working_df
    if df is None:
        raise HistoryError("No working dataset is loaded.")

    before_fingerprint = dataframe_fingerprint(df)
    before_schema = {
        "columns": tuple(map(str, df.columns)),
        "dtypes": tuple(map(str, df.dtypes)),
    }

    manager = st.session_state.history_manager
    try:
        snapshot_id = manager.push(
            df,
            action=action,
            fingerprint=before_fingerprint,
            context=_history_context(),
            clear_redo=False,
        )
    except HistoryError as exc:
        st.session_state.history_warning = str(exc)
        raise

    st.session_state.pending_dataset_change = {
        "snapshot_id": snapshot_id,
        "action": (action or "Dataset change").strip(),
        "details": copy.deepcopy(details or {}),
        "before_fingerprint": before_fingerprint,
        "before_shape": tuple(df.shape),
        "before_schema": before_schema,
        "started_at": time.time(),
    }
    st.session_state.history_warning = ""
    sync_history_mirror()
    return snapshot_id


def reset_file_state() -> None:
    """Clear all dataset-related state before activating a new source."""
    old_manager = st.session_state.get("history_manager")
    if isinstance(old_manager, SmartHistoryManager):
        old_manager.close()

    state = DatasetState()
    st.session_state.dataset_state = state
    sync_state_to_session(st.session_state, state)
    st.session_state.restore_original_confirm = False

    st.session_state.history_manager = SmartHistoryManager()
    st.session_state.df_history = []
    st.session_state.pending_dataset_change = None
    st.session_state.dataset_audit_log = []
    st.session_state.history_warning = ""
    st.session_state.ai_messages = []
    st.session_state.feature_pipeline_preview = None
    st.session_state.ml_experiment_artifact = None
    st.session_state.ml_clustering_result = None
    st.session_state.model_explainability_artifact = None
    st.session_state.local_explanation_result = None
    st.session_state.shap_explanation_result = None
    st.session_state.prediction_input_df = None
    st.session_state.prediction_input_report = {}
    st.session_state.prediction_input_name = ""
    st.session_state.prediction_result = None
    st.session_state.model_monitoring_input_df = None
    st.session_state.model_monitoring_input_name = ""
    st.session_state.model_monitoring_report = {}
    st.session_state.model_monitoring_feature_table = None
    st.session_state.model_governance_assessment = None
