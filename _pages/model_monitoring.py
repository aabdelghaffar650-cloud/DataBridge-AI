# ════════════════════════════════════════════════════════
#  DataBridge AI — Page: Model Monitoring & Drift
# ════════════════════════════════════════════════════════
from __future__ import annotations

import json
from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st
from ui.streamlit_compat import safe_dataframe

from core.security import safe_html
from core.session import append_audit_event
from modules.import_engine import smart_parse_file
from modules.model_monitoring import ModelMonitoringError, run_monitoring_analysis
from modules.model_governance import default_family_name
from modules.model_package import (
    LoadedModelPackage,
    ModelPackageError,
    get_or_create_signing_key,
    load_signed_model_package,
    signing_key_id,
    validate_prediction_frame,
)
from modules.model_registry import (
    ModelRegistryError,
    delete_registered_package,
    list_registered_packages,
    load_registered_package_bytes,
    register_signed_package,
    save_monitoring_report,
)
from ui.cards import section_header


MONITOR_FILE_TYPES = ["csv", "xlsx", "xls", "json", "jsonl", "ndjson", "parquet"]


def _load_package(raw: bytes, source: str) -> LoadedModelPackage:
    package = load_signed_model_package(raw)
    st.session_state.loaded_model_package = package
    st.session_state.loaded_model_package_bytes = bytes(raw)
    st.session_state.model_monitoring_report = {}
    append_audit_event(
        {
            "event": "monitoring_package_loaded",
            "action": "Verified model package loaded for monitoring",
            "package_id": package.package_id,
            "source": source,
        }
    )
    return package


def _render_package_section() -> Optional[LoadedModelPackage]:
    st.markdown("### 1. Monitoring model")
    generated = st.session_state.get("model_package_bytes")
    loaded = st.session_state.get("loaded_model_package")

    left, right = st.columns(2)
    with left:
        st.markdown("#### Current / generated package")
        if isinstance(loaded, LoadedModelPackage):
            has_ref = bool(loaded.manifest.get("monitoring", {}).get("training_reference"))
            ref_text = "Stage 14 reference ready" if has_ref else "rebuild package after Stage 14"
            st.success(
                f"Verified: {loaded.package_id} · {loaded.task} · {loaded.target} · {ref_text}"
            )
        elif isinstance(generated, (bytes, bytearray)):
            if st.button("Verify ML Studio package", width="stretch", key="monitor_verify_generated"):
                try:
                    _load_package(bytes(generated), "ML Studio session")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Package verification failed safely: {exc}")
        else:
            st.info("Create a signed model package in ML Studio V2 or upload one here.")

        upload = st.file_uploader(
            "Upload signed .dbmlpkg",
            type=["dbmlpkg"],
            key="monitor_package_upload",
        )
        if upload is not None and st.button(
            "Verify & Load Package", width="stretch", key="monitor_load_upload"
        ):
            try:
                _load_package(upload.getvalue(), upload.name)
                st.success("Package verified before deserialization and loaded.")
                st.rerun()
            except Exception as exc:
                st.error(f"Package load blocked safely: {exc}")

    with right:
        st.markdown("#### Local signed model registry")
        registry = list_registered_packages()
        if registry:
            labels = {
                f"{row.get('target','—')} · {row.get('selected_model','—')} · {row.get('package_id','')}": row
                for row in registry
            }
            choice = st.selectbox(
                "Registered model",
                list(labels),
                key="monitor_registry_choice",
            )
            selected = labels[choice]
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Load Registered", width="stretch", key="monitor_registry_load"):
                    try:
                        raw = load_registered_package_bytes(str(selected["package_id"]))
                        _load_package(raw, "Local model registry")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Registry load blocked: {exc}")
            with c2:
                delete_confirm = st.checkbox("Confirm remove", key="monitor_registry_delete_confirm")
                if st.button(
                    "Remove",
                    disabled=not delete_confirm,
                    width="stretch",
                    key="monitor_registry_delete",
                ):
                    try:
                        delete_registered_package(str(selected["package_id"]))
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Registry removal failed: {exc}")
        else:
            st.caption("Registry is empty. Packages are stored only after explicit registration.")

        raw = st.session_state.get("loaded_model_package_bytes")
        current = st.session_state.get("loaded_model_package")
        if isinstance(current, LoadedModelPackage) and isinstance(raw, (bytes, bytearray)):
            label = st.text_input("Registry label (optional)", key="monitor_registry_label")
            family_default = default_family_name(current.task, current.target)
            model_family = st.text_input(
                "Model family",
                value=family_default,
                help="Packages in the same family participate in Champion/Challenger governance.",
                key=f"monitor_registry_family_{current.package_id}",
            )
            if st.button("Register Verified Package", width="stretch", key="monitor_registry_add"):
                try:
                    meta = register_signed_package(bytes(raw), label=label, model_family=model_family)
                    append_audit_event(
                        {
                            "event": "model_registered",
                            "action": "Verified signed model registered locally",
                            "package_id": meta.get("package_id"),
                        }
                    )
                    st.success("Verified package registered locally as a governance Candidate.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Registration failed safely: {exc}")

    package = st.session_state.get("loaded_model_package")
    if not isinstance(package, LoadedModelPackage):
        return None

    monitoring = package.manifest.get("monitoring", {}) or {}
    has_reference = bool(monitoring.get("training_reference"))
    tone = "#6bff8e" if has_reference else "#ffb86b"
    st.markdown(
        f"<div class='info-box' style='border-left:4px solid {tone};'>"
        f"<b>{safe_html(package.package_id)}</b> · {safe_html(package.task)} · target "
        f"<code>{safe_html(package.target)}</code> · "
        f"{'training-only drift reference available' if has_reference else 'no Stage 14 drift reference — rebuild package in ML Studio V2'}"
        "</div>",
        unsafe_allow_html=True,
    )
    return package


def _store_monitor_input(frame: pd.DataFrame, name: str) -> None:
    st.session_state.model_monitoring_input_df = frame.copy(deep=True)
    st.session_state.model_monitoring_input_name = str(name)
    st.session_state.model_monitoring_report = {}


def _render_input_section(active_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    st.markdown("### 2. Production / scoring data")
    st.caption("Monitoring data is held as an independent copy and never replaces the active working dataset.")
    left, right = st.columns(2)
    with left:
        if st.button("Copy Active Dataset", width="stretch", key="monitor_copy_active"):
            _store_monitor_input(active_df, st.session_state.get("file_name") or "Active dataset")
            st.rerun()
    with right:
        uploaded = st.file_uploader(
            "Upload monitoring data",
            type=MONITOR_FILE_TYPES,
            key="monitor_data_upload",
        )
        if uploaded is not None and st.button(
            "Read Monitoring File Safely", width="stretch", key="monitor_read_upload"
        ):
            try:
                frame, report = smart_parse_file(uploaded)
                if len(report.get("json_array_candidates") or []) > 1:
                    raise ModelMonitoringError(
                        "Monitoring JSON must resolve to one unambiguous array."
                    )
                _store_monitor_input(frame, uploaded.name)
                st.rerun()
            except Exception as exc:
                st.error(f"Monitoring import failed safely: {exc}")

    frame = st.session_state.get("model_monitoring_input_df")
    if not isinstance(frame, pd.DataFrame):
        return None
    name = st.session_state.get("model_monitoring_input_name") or "Monitoring data"
    st.markdown(
        f"<div class='info-box'><b>{safe_html(name)}</b> · {len(frame):,} rows × {frame.shape[1]:,} columns · independent monitoring copy</div>",
        unsafe_allow_html=True,
    )
    safe_dataframe(frame.head(15), width="stretch", height=260)
    return frame


def _render_schema(package: LoadedModelPackage, frame: pd.DataFrame) -> bool:
    report = validate_prediction_frame(frame, package)
    status = report.get("status", "Blocked")
    if report.get("valid"):
        st.success("Prediction schema is compatible with the signed model contract.")
    else:
        st.error("Schema drift blocks safe scoring: " + "; ".join(report.get("blockers", [])))
    if report.get("warnings"):
        for item in report["warnings"]:
            st.warning(str(item))
    with st.expander("Schema checks", expanded=False):
        safe_dataframe(pd.DataFrame(report.get("column_checks", [])), width="stretch")
    return bool(report.get("valid"))


def _status_color(status: str) -> str:
    return {
        "Stable": "#6bff8e",
        "Watch": "#ffcf6b",
        "Drifted": "#ff9f43",
        "Critical": "#ff6b6b",
    }.get(status, "#ff6b6b")


def _render_report(report: Dict[str, Any], feature_table: pd.DataFrame) -> None:
    status = str(report.get("overall_status", "Critical"))
    color = _status_color(status)
    pred = report.get("prediction_drift", {}) or {}
    perf = report.get("observed_performance", {}) or {}
    st.markdown(
        f"""
        <div class="metrics-row" style="grid-template-columns:repeat(4,1fr);">
          <div class="metric-card"><div class="label">Overall</div>
            <div class="value" style="font-size:1.05rem;color:{color}">{safe_html(status)}</div>
            <div class="sub">worst monitored signal</div></div>
          <div class="metric-card"><div class="label">Drift Score</div>
            <div class="value">{float(report.get('drift_score',0)):.1f}</div><div class="sub">0 stable · 100 critical</div></div>
          <div class="metric-card"><div class="label">Prediction Drift</div>
            <div class="value" style="font-size:1.05rem;color:{_status_color(str(pred.get('status','Stable')))}">{safe_html(str(pred.get('status','N/A')))}</div>
            <div class="sub">output distribution</div></div>
          <div class="metric-card"><div class="label">Labelled Performance</div>
            <div class="value" style="font-size:1.05rem">{'Available' if perf.get('available') else 'Not supplied'}</div>
            <div class="sub">optional actual outcomes</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Feature drift")
    if feature_table.empty:
        st.info("No monitored feature rows.")
    else:
        display = feature_table.copy()
        safe_dataframe(display, width="stretch", hide_index=True, height=360)

    if pred.get("available"):
        st.markdown("#### Prediction drift")
        st.json(pred, expanded=False)

    if perf.get("available"):
        st.markdown("#### Observed model performance")
        metrics = perf.get("metrics", {}) or {}
        baseline = perf.get("signed_holdout_metrics", {}) or {}
        rows = []
        for name, value in metrics.items():
            rows.append(
                {
                    "Metric": name,
                    "Current labelled data": value,
                    "Signed holdout": baseline.get(name),
                }
            )
        safe_dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.caption(
            "Observed performance is descriptive. A drop can come from concept drift, data-quality changes, label changes, or a genuinely weaker model."
        )

    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(
            "Download Monitoring JSON",
            data=payload,
            file_name=f"{report.get('package_id','model')}_monitoring.json",
            mime="application/json",
            width="stretch",
            key="monitor_download_json",
        )
    with c2:
        st.download_button(
            "Download Feature Drift CSV",
            data=feature_table.to_csv(index=False).encode("utf-8-sig"),
            file_name=f"{report.get('package_id','model')}_feature_drift.csv",
            mime="text/csv",
            width="stretch",
            key="monitor_download_csv",
        )
    with c3:
        if st.button("Save Report Locally", width="stretch", key="monitor_save_report"):
            try:
                path = save_monitoring_report(report)
                append_audit_event(
                    {
                        "event": "monitoring_report_saved",
                        "action": "Model monitoring report saved locally",
                        "package_id": report.get("package_id"),
                        "overall_status": status,
                    }
                )
                st.success(f"Saved securely in user data: {path.name}")
            except Exception as exc:
                st.error(f"Could not save report: {exc}")


def render(df: pd.DataFrame) -> None:
    st.markdown(
        section_header("📡", "Model Monitoring & Drift", "Stage 14 · production model health"),
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="info-box">Compare new scoring data with the <b>training-only</b> reference stored in the signed model package. '
        'The package persists aggregate distributions only; categorical feature labels are stored as local HMAC tokens, not raw values.</div>',
        unsafe_allow_html=True,
    )

    package = _render_package_section()
    if package is None:
        return
    if not package.manifest.get("monitoring", {}).get("training_reference"):
        st.warning("Rebuild this model package in ML Studio V2 after Stage 14 to enable drift monitoring.")
        return

    frame = _render_input_section(df)
    if frame is None:
        return

    st.markdown("### 3. Contract & monitoring run")
    valid = _render_schema(package, frame)

    actual_options = ["— No actual outcome column —"] + list(map(str, frame.columns))
    default_idx = actual_options.index(package.target) if package.target in actual_options else 0
    actual_choice = st.selectbox(
        "Actual outcome column (optional, enables observed performance)",
        actual_options,
        index=default_idx,
        key="monitor_actual_target",
    )
    actual_col = None if actual_choice.startswith("—") else actual_choice

    if st.button(
        "Run Drift & Performance Monitor",
        type="primary",
        width="stretch",
        disabled=not valid,
        key="monitor_run",
    ):
        try:
            key = get_or_create_signing_key()
            if signing_key_id(key) != package.signer_key_id:
                raise ModelMonitoringError("The active local signing key does not match this verified package.")
            with st.spinner("Comparing schema, training distributions, prediction distribution, and optional labelled performance..."):
                result = run_monitoring_analysis(
                    package,
                    frame,
                    signing_key=key,
                    actual_target_column=actual_col,
                    include_prediction_output=False,
                )
            st.session_state.model_monitoring_report = result.report
            st.session_state.model_monitoring_feature_table = result.feature_table
            append_audit_event(
                {
                    "event": "model_monitoring",
                    "action": "Model drift monitoring completed",
                    "package_id": package.package_id,
                    "rows": len(frame),
                    "overall_status": result.report.get("overall_status"),
                    "drift_score": result.report.get("drift_score"),
                }
            )
            st.rerun()
        except Exception as exc:
            st.error(f"Monitoring was blocked safely: {exc}")

    report = st.session_state.get("model_monitoring_report", {}) or {}
    table = st.session_state.get("model_monitoring_feature_table")
    if report and report.get("package_id") == package.package_id and isinstance(table, pd.DataFrame):
        st.markdown("---")
        _render_report(report, table)
        if str(report.get("overall_status", "")) in {"Watch", "Drifted", "Critical"}:
            if st.button("♻️ Open Safe Auto Retraining", width="stretch", key="monitor_open_retraining"):
                st.session_state["current_page"] = "retraining_workflow"
                st.rerun()
