# ════════════════════════════════════════════════════════
#  DataBridge AI — Professional Grouped Sidebar
# ════════════════════════════════════════════════════════
import streamlit as st

from config.settings import NAV_PAGE_KEYS
from core.auth       import logout
from core.credential_store import SecretStoreError, get_credential_store
from core.security   import safe_error_message, safe_html
from core.session import append_audit_event
from core.team_access import (
    PERM_DATA_IMPORT,
    PERM_DATA_TRANSFORM,
    PERM_SECRETS_MANAGE,
    ROLE_LABELS,
    can_access_page,
    first_accessible_page,
    has_permission,
)
from modules.import_engine  import smart_parse_file, SUPPORTED_FILE_TYPES
from core.dataset    import (
    activate_dataset,
    current_dataset_revision,
    dataset_state_health,
    perform_redo,
    perform_undo,
    restore_raw_dataset,
)
from ai import (
    DataBridgeAIEngine,
    GeminiCloudEngine,
    AnthropicCloudEngine,
    OllamaLocalEngine,
    DemoEngine,
    PRIVACY_MASKED,
    PRIVACY_METADATA,
    PRIVACY_RAW,
)


NAV_GROUPS = [
    (
        "DATA",
        [
            ("overview", "Overview"),
            ("data_sources", "Data Sources"),
            ("data_mapper", "Data Mapper"),
            ("quality_engine", "Quality Engine"),
        ],
    ),
    (
        "TRANSFORM",
        [
            ("filter_search", "Filter & Search"),
            ("cleaning", "Clean Nulls"),
            ("data_types", "Data Types"),
            ("replace_values", "Replace Values"),
            ("feature_engineering", "Feature Engineering"),
        ],
    ),
    (
        "ANALYZE",
        [
            ("visualization", "Visualize"),
            ("outlier_detection", "Outlier Detection"),
            ("kpi_tracker", "KPI Tracker"),
            ("ml_studio", "ML Studio V2"),
            ("explainability_studio", "Explainability Studio"),
            ("prediction_studio", "Prediction Studio"),
            ("model_monitoring", "Model Monitoring"),
            ("monitoring_scheduler", "Scheduled Monitoring"),
            ("deployment_api", "Deployment API"),
            ("remote_model_registry", "Remote Model Registry"),
            ("retraining_workflow", "Safe Auto Retraining"),
            ("model_governance", "Model Governance"),
            ("ai_assistant", "AI Assistant"),
        ],
    ),
    (
        "MANAGE",
        [
            ("export", "Export"),
            ("delete_dedupe", "Delete & Dedupe"),
            ("settings", "Settings"),
            ("team_admin", "Team Administration"),
        ],
    ),
]


def _render_nav_group(group_label: str, pages: list[tuple[str, str]], current_page: str) -> None:
    st.markdown(
        f"<div class='nav-group-label'>{safe_html(group_label)}</div>",
        unsafe_allow_html=True,
    )

    for page_key, label in pages:
        if not can_access_page(page_key, session_state=st.session_state):
            continue
        if page_key == current_page:
            st.markdown(
                f"<div class='nav-item active'>{safe_html(label)}</div>",
                unsafe_allow_html=True,
            )
        else:
            if st.button(label, key=f"nav_{page_key}", width="stretch"):
                st.session_state["current_page"] = page_key
                st.rerun()


def _render_status_block() -> None:
    if st.session_state.df is None:
        return

    df = st.session_state.df
    file_name = safe_html(st.session_state.get("file_name") or "Untitled dataset")
    quality_report = st.session_state.get("quality_report", {}) or {}
    quality = quality_report.get("quality_score")
    quality_baseline = st.session_state.get("quality_baseline_report", {}) or {}
    quality_delta = (
        round(float(quality or 0.0) - float(quality_baseline.get("quality_score", quality or 0.0) or 0.0), 1)
        if quality is not None and quality_baseline
        else None
    )
    mapper_status = "Approved" if st.session_state.get("mapper_approved") else "Pending"
    mapper_class = "ok" if st.session_state.get("mapper_approved") else "warn"
    readiness = st.session_state.get("ml_readiness_report", {}) or {}
    readiness_status = str(readiness.get("status", "Pending"))
    readiness_score = readiness.get("score")
    readiness_class = "ok" if readiness_status == "Ready" else "danger" if readiness_status == "Blocked" else "warn"
    pipeline_report = st.session_state.get("feature_pipeline_report", {}) or {}
    pipeline_status = str(pipeline_report.get("status", "Pending"))
    pipeline_class = "ok" if pipeline_status == "Configured" else "danger" if pipeline_status == "Invalid" else "warn"
    experiment_report = st.session_state.get("ml_experiment_report", {}) or {}
    experiment_status = str(experiment_report.get("status", "Pending"))
    experiment_class = "ok" if experiment_status == "Completed" else "danger" if experiment_status == "Stale" else "warn"
    explain_report = st.session_state.get("explainability_report", {}) or {}
    explain_status = str(explain_report.get("status", "Pending"))
    explain_class = "ok" if explain_status == "Completed" else "danger" if explain_status == "Stale" else "warn"
    loaded_package = st.session_state.get("loaded_model_package")
    package_status = "Verified" if loaded_package is not None else "Pending"
    package_class = "ok" if loaded_package is not None else "warn"
    monitoring_report = st.session_state.get("model_monitoring_report", {}) or {}
    monitoring_status = str(monitoring_report.get("overall_status", "Pending"))
    monitoring_class = (
        "ok" if monitoring_status == "Stable"
        else "warn" if monitoring_status in {"Pending", "Watch"}
        else "danger"
    )
    retraining = st.session_state.get("retraining_last_report", {}) or {}
    retraining_status = "Challenger Ready" if retraining.get("challenger_id") else "Pending"
    retraining_class = "ok" if retraining.get("challenger_id") else "warn"
    revision = current_dataset_revision()

    quality_html = ""
    if quality is not None:
        q_class = "ok" if quality >= 85 else "warn" if quality >= 60 else "danger"
        delta_text = f" {quality_delta:+.1f}" if quality_delta not in (None, 0.0) else ""
        quality_html = f"<span class='mini-chip {q_class}'>Quality V2 {quality}%{delta_text}</span>"

    raw_shape = st.session_state.get("raw_shape")
    if raw_shape:
        raw_shape_html = f"Original: {raw_shape[0]:,} rows × {raw_shape[1]} cols"
    else:
        raw_shape_html = "Original snapshot unavailable"

    st.markdown(
        f"""
        <div class='sidebar-status'>
            <div class='status-label'>OPEN DATASET</div>
            <div class='status-file' title='{file_name}'>{file_name}</div>
            <div class='status-shape'>Working: {df.shape[0]:,} rows × {df.shape[1]} cols · Revision {revision}</div>
            <div class='status-shape' style='margin-top:.2rem;color:#777;'>{raw_shape_html}</div>
            <div class='status-chips'>
                {quality_html}
                <span class='mini-chip {mapper_class}'>Mapping {mapper_status}</span>
                <span class='mini-chip {readiness_class}'>ML {readiness_status}{f" {readiness_score}%" if readiness_score is not None else ""}</span>
                <span class='mini-chip {pipeline_class}'>Pipeline {pipeline_status}</span>
                <span class='mini-chip {experiment_class}'>Experiment {experiment_status}</span>
                <span class='mini-chip {explain_class}'>Explain {explain_status}</span>
                <span class='mini-chip {package_class}'>Package {package_status}</span>
                <span class='mini-chip {monitoring_class}'>Monitor {monitoring_status}</span>
                <span class='mini-chip {retraining_class}'>Retrain {retraining_status}</span>
                <span class='mini-chip ok'>Original protected</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> str:
    """Render full sidebar. Returns selected internal page key."""
    with st.sidebar:
        st.markdown(
            """
            <div class='sidebar-brand'>
                <div class='sidebar-logo'>DataBridge AI</div>
                <div class='sidebar-subtitle'>Data Intelligence Platform</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        current_page = st.session_state.get("current_page", NAV_PAGE_KEYS[0])
        if current_page not in NAV_PAGE_KEYS or not can_access_page(current_page, session_state=st.session_state):
            current_page = first_accessible_page(NAV_PAGE_KEYS, session_state=st.session_state)
            st.session_state["current_page"] = current_page

        with st.container(key="sidebar_nav"):
            for group_label, pages in NAV_GROUPS:
                _render_nav_group(group_label, pages, current_page)

        st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)

        # ── Account ──────────────────────────────────────
        user_name = st.session_state.get("current_user", "") or "admin"
        user_role = str(st.session_state.get("current_role") or "viewer")
        role_label = ROLE_LABELS.get(user_role, user_role)
        st.markdown(
            f"<div class='account-pill'>{safe_html(user_name)} · {safe_html(role_label)}</div>",
            unsafe_allow_html=True,
        )
        if st.button("Logout", width="stretch", key="logout_btn"):
            logout()
            st.rerun()

        can_transform = has_permission(PERM_DATA_TRANSFORM, session_state=st.session_state)
        can_import = has_permission(PERM_DATA_IMPORT, session_state=st.session_state)
        can_manage_secrets = has_permission(PERM_SECRETS_MANAGE, session_state=st.session_state)

        # ── File Info ────────────────────────────────────
        _render_status_block()

        if st.session_state.df is not None:
            manager = st.session_state.history_manager
            c1, c2 = st.columns(2)
            with c1:
                undo_label = "Undo" if not manager.next_undo_action else f"Undo ({manager.undo_count})"
                if st.button(
                    undo_label,
                    disabled=not manager.can_undo or not can_transform,
                    width="stretch",
                    help=(f"Undo: {manager.next_undo_action}" if manager.can_undo else "No previous change"),
                    key="history_undo_btn",
                ):
                    try:
                        if perform_undo():
                            st.rerun()
                    except Exception as exc:
                        st.session_state.history_warning = str(exc)
                        st.error(f"Undo failed safely: {exc}")
            with c2:
                redo_label = "Redo" if not manager.next_redo_action else f"Redo ({manager.redo_count})"
                if st.button(
                    redo_label,
                    disabled=not manager.can_redo or not can_transform,
                    width="stretch",
                    help=(f"Redo: {manager.next_redo_action}" if manager.can_redo else "No change to redo"),
                    key="history_redo_btn",
                ):
                    try:
                        if perform_redo():
                            st.rerun()
                    except Exception as exc:
                        st.session_state.history_warning = str(exc)
                        st.error(f"Redo failed safely: {exc}")

            history_warning = st.session_state.get("history_warning", "")
            if history_warning:
                st.warning(history_warning)

            with st.expander("History & Audit", expanded=False):
                health = dataset_state_health(deep=False)
                state_label = "Consistent" if health["ok"] else "Needs attention"
                st.caption(
                    f"Unified state: {state_label} · Revision: {health['revision']}"
                )
                storage = manager.storage_summary()
                st.caption(
                    f"Undo: {storage['undo_count']} · Redo: {storage['redo_count']} · "
                    f"Memory: {storage['memory_bytes'] / (1024 * 1024):.1f} MB · "
                    f"Disk: {storage['disk_bytes'] / (1024 * 1024):.1f} MB"
                )
                if manager.can_undo:
                    st.markdown(f"**Next undo:** {safe_html(manager.next_undo_action)}")
                if manager.can_redo:
                    st.markdown(f"**Next redo:** {safe_html(manager.next_redo_action)}")

                audit_log = st.session_state.get("dataset_audit_log", [])
                if audit_log:
                    st.markdown("**Recent dataset actions**")
                    for event in reversed(audit_log[-5:]):
                        action = safe_html(str(event.get("action", event.get("event", "Change"))))
                        after_shape = event.get("after_shape")
                        shape_text = (
                            f" — {after_shape[0]:,} × {after_shape[1]}"
                            if isinstance(after_shape, (tuple, list)) and len(after_shape) == 2
                            else ""
                        )
                        st.caption(f"{action}{shape_text}")
                else:
                    st.caption("No dataset actions recorded yet.")

            with st.expander("🛡️ Dataset Safety", expanded=False):
                raw_shape = st.session_state.get("raw_shape")
                if raw_shape:
                    st.caption(
                        f"Protected original: {raw_shape[0]:,} rows × {raw_shape[1]} columns. "
                        "Cleaning and transformation pages only receive the working copy."
                    )
                else:
                    st.warning("No protected original snapshot is available for this dataset.")

                confirmed = st.checkbox(
                    "I understand that restoring will discard all current working changes.",
                    key="restore_original_confirm",
                )
                if st.button(
                    "Restore Protected Original",
                    type="secondary",
                    width="stretch",
                    disabled=not confirmed or st.session_state.get("raw_df") is None or not can_transform,
                    key="restore_original_btn",
                ):
                    try:
                        restore_raw_dataset()
                        st.success("The working dataset was restored from the protected original.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Restore failed: {exc}")

        # ── Load New File ────────────────────────────────
        st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
        new_file = st.file_uploader(
            "Load new data source",
            type=SUPPORTED_FILE_TYPES,
            label_visibility="visible",
            help="CSV, Excel, JSON, JSONL, Parquet, SQLite DB",
            disabled=not can_import,
        )
        if not can_import:
            st.caption("Your team role cannot replace the active dataset.")
        if can_import and new_file and new_file.name != st.session_state.file_name:
            try:
                parsed_df, clean_report = smart_parse_file(new_file)

                activate_dataset(parsed_df, clean_report, display_name=new_file.name)
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

        # ── AI Settings ──────────────────────────────────
        st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
        st.markdown("<div class='sidebar-section-title'>AI ENGINE</div>", unsafe_allow_html=True)
        engine_choice = st.selectbox(
            "Engine",
            ["Demo", "Claude", "Ollama", "Gemini"],
            label_visibility="collapsed",
            key="ai_engine_choice",
        )

        privacy_labels = {
            "Metadata only — no sample rows": PRIVACY_METADATA,
            "Masked sample — recommended": PRIVACY_MASKED,
            "Raw sample — explicit consent": PRIVACY_RAW,
        }
        privacy_reverse = {value: key for key, value in privacy_labels.items()}
        current_privacy = st.session_state.get("ai_privacy_mode", PRIVACY_METADATA)
        if current_privacy not in privacy_reverse:
            current_privacy = PRIVACY_METADATA

        store = get_credential_store()
        active_strategy = None
        engine_ready = False
        remote_endpoint = False
        raw_confirmed = False
        secret_source = "not_configured"

        def _resolve_provider_key(secret_name: str, label: str, placeholder: str) -> str:
            nonlocal secret_source
            widget_key = f"{secret_name}_temporary_widget"
            if st.session_state.pop(f"clear_{widget_key}", False):
                st.session_state.pop(widget_key, None)
            stored_value = None
            try:
                stored_value, secret_source = store.get_secret(secret_name)
            except SecretStoreError:
                secret_source = "unavailable"
            temp_key = st.text_input(
                label,
                type="password",
                placeholder=placeholder,
                key=widget_key,
                help="Temporary keys remain only in this Streamlit session unless saved securely.",
            )
            c_save, c_delete = st.columns(2)
            with c_save:
                if st.button(
                    "Save securely",
                    key=f"save_{secret_name}",
                    width="stretch",
                    disabled=not bool(temp_key) or not can_manage_secrets,
                ):
                    try:
                        store.set_secret(secret_name, temp_key)
                        append_audit_event({
                            "event": "credential_saved",
                            "action": f"{secret_name} saved to OS credential manager",
                            "provider": secret_name.split("_", 1)[0],
                        })
                        st.session_state[f"clear_{widget_key}"] = True
                        st.success("Saved in the operating-system credential manager.")
                        st.rerun()
                    except SecretStoreError as exc:
                        st.error(safe_error_message(exc))
            with c_delete:
                if st.button(
                    "Delete saved",
                    key=f"delete_{secret_name}",
                    width="stretch",
                    disabled=not bool(stored_value) or not can_manage_secrets,
                ):
                    try:
                        store.delete_secret(secret_name)
                        append_audit_event({
                            "event": "credential_deleted",
                            "action": f"{secret_name} removed from OS credential manager",
                            "provider": secret_name.split("_", 1)[0],
                        })
                        st.success("Saved credential deleted.")
                        st.rerun()
                    except SecretStoreError as exc:
                        st.error(safe_error_message(exc))
            backend = store.backend_name()
            if not can_manage_secrets:
                st.caption("Saved credential changes require Admin permission; this session may still use configured or temporary credentials.")
            if stored_value:
                st.caption(f"Credential source: {secret_source} · backend: {backend}")
            elif backend == "Unavailable":
                st.warning("OS credential storage is unavailable. Install requirements and use a temporary key only for this session.")
            else:
                st.caption(f"No saved credential · backend: {backend}")
            return str(temp_key or stored_value or "")

        if engine_choice == "Claude":
            api_key = _resolve_provider_key("anthropic_api_key", "Anthropic API Key", "sk-ant-...")
            privacy_label = st.selectbox(
                "Cloud privacy mode",
                list(privacy_labels),
                index=list(privacy_labels).index(privacy_reverse[current_privacy]),
                key="claude_privacy_mode_widget",
            )
            privacy_mode = privacy_labels[privacy_label]
            if privacy_mode == PRIVACY_RAW:
                raw_confirmed = st.checkbox(
                    "I explicitly approve sending raw sample rows to Anthropic for this session.",
                    value=False,
                    key="claude_raw_confirm",
                )
            if api_key and (privacy_mode != PRIVACY_RAW or raw_confirmed):
                active_strategy = AnthropicCloudEngine(api_key)
                st.session_state.ai_engine = DataBridgeAIEngine(
                    active_strategy,
                    privacy_mode=privacy_mode,
                    raw_data_confirmed=raw_confirmed,
                    semantic_profiles=st.session_state.get("semantic_profiles", {}),
                )
                st.session_state.ai_mode = "anthropic"
                engine_ready = True

        elif engine_choice == "Gemini":
            api_key = _resolve_provider_key("gemini_api_key", "Gemini API Key", "AIza...")
            privacy_label = st.selectbox(
                "Cloud privacy mode",
                list(privacy_labels),
                index=list(privacy_labels).index(privacy_reverse[current_privacy]),
                key="gemini_privacy_mode_widget",
            )
            privacy_mode = privacy_labels[privacy_label]
            if privacy_mode == PRIVACY_RAW:
                raw_confirmed = st.checkbox(
                    "I explicitly approve sending raw sample rows to Google Gemini for this session.",
                    value=False,
                    key="gemini_raw_confirm",
                )
            if api_key and (privacy_mode != PRIVACY_RAW or raw_confirmed):
                active_strategy = GeminiCloudEngine(api_key, mask_pii=(privacy_mode == PRIVACY_MASKED))
                st.session_state.ai_engine = DataBridgeAIEngine(
                    active_strategy,
                    privacy_mode=privacy_mode,
                    raw_data_confirmed=raw_confirmed,
                    semantic_profiles=st.session_state.get("semantic_profiles", {}),
                )
                st.session_state.ai_mode = "gemini"
                engine_ready = True

        elif engine_choice == "Ollama":
            host = st.text_input("Host", value="http://localhost:11434", key="ollama_host")
            model = st.text_input("Model", value="llama3", key="ollama_model")
            privacy_label = st.selectbox(
                "Data mode",
                list(privacy_labels),
                index=list(privacy_labels).index(privacy_reverse.get(current_privacy, privacy_reverse[PRIVACY_MASKED])),
                key="ollama_privacy_mode_widget",
            )
            privacy_mode = privacy_labels[privacy_label]
            try:
                active_strategy = OllamaLocalEngine(host, model)
                remote_endpoint = active_strategy.get_engine_type() == "remote"
                remote_confirmed = True
                if remote_endpoint:
                    st.warning("This Ollama host is not loopback. Data will leave this computer.")
                    remote_confirmed = st.checkbox(
                        "I trust this remote Ollama host and approve network transfer for this session.",
                        value=False,
                        key="ollama_remote_confirm",
                    )
                if privacy_mode == PRIVACY_RAW and remote_endpoint:
                    raw_confirmed = st.checkbox(
                        "I explicitly approve sending raw sample rows to the remote Ollama host.",
                        value=False,
                        key="ollama_raw_confirm",
                    )
                else:
                    raw_confirmed = privacy_mode == PRIVACY_RAW
                if remote_confirmed and (privacy_mode != PRIVACY_RAW or raw_confirmed):
                    st.session_state.ai_engine = DataBridgeAIEngine(
                        active_strategy,
                        privacy_mode=privacy_mode,
                        raw_data_confirmed=raw_confirmed,
                        allow_remote_endpoint=remote_confirmed,
                        semantic_profiles=st.session_state.get("semantic_profiles", {}),
                    )
                    st.session_state.ai_mode = "ollama"
                    engine_ready = True
            except Exception as exc:
                st.error(f"Ollama configuration blocked safely: {safe_error_message(exc)}")
                active_strategy = None

        else:
            privacy_mode = PRIVACY_METADATA
            active_strategy = DemoEngine()
            st.session_state.ai_mode = "demo"
            st.session_state.ai_engine = DataBridgeAIEngine(active_strategy, privacy_mode=privacy_mode)
            engine_ready = True

        st.session_state.ai_privacy_mode = privacy_mode
        st.session_state.ai_raw_transfer_confirm = bool(raw_confirmed)
        st.session_state.ai_remote_endpoint_confirm = bool(remote_endpoint and engine_ready)
        st.session_state.ai_secret_source = secret_source

        if not engine_ready and engine_choice != "Demo":
            st.session_state.ai_mode = "demo"
            st.session_state.ai_engine = DataBridgeAIEngine(DemoEngine(), privacy_mode=PRIVACY_METADATA)

        if active_strategy is not None and engine_ready:
            if st.button("Test Connection", width="stretch", key="test_ai_connection"):
                with st.spinner("Testing connection..."):
                    ok, message = active_strategy.test_connection()
                append_audit_event({
                    "event": "ai_connection_test",
                    "action": "AI engine connection tested",
                    "engine": engine_choice,
                    "success": bool(ok),
                    "privacy_mode": privacy_mode,
                    "remote_endpoint": bool(remote_endpoint),
                })
                if ok:
                    st.success(message)
                else:
                    st.error(safe_error_message(message))

        st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='sidebar-footer'>DataBridge AI © 2026</div>",
            unsafe_allow_html=True,
        )

    return st.session_state.get("current_page", NAV_PAGE_KEYS[0])
