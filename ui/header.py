# ════════════════════════════════════════════════════════
#  DataBridge AI — Professional Top Header
# ════════════════════════════════════════════════════════
import streamlit as st
from config.constants import APP_VERSION
from core.security import safe_html
from core.team_access import ROLE_LABELS


def _chip(label: str, value: str = "", tone: str = "neutral") -> str:
    text = f"{safe_html(label)} {safe_html(value)}".strip()
    return f"<span class='top-chip {tone}'>{text}</span>"


def render_header() -> None:
    df = st.session_state.get("df")
    file_name = st.session_state.get("file_name")
    ai_mode = st.session_state.get("ai_mode", "demo")
    current_user = str(st.session_state.get("current_user") or "")
    current_role = str(st.session_state.get("current_role") or "viewer")
    role_label = ROLE_LABELS.get(current_role, current_role)

    if df is not None:
        dataset_title = safe_html(file_name or "Untitled dataset")
        dataset_meta = f"{df.shape[0]:,} rows × {df.shape[1]} cols"
        quality_report = st.session_state.get("quality_report", {}) or {}
        quality = quality_report.get("quality_score")
        quality_baseline = st.session_state.get("quality_baseline_report", {}) or {}
        quality_delta = (
            round(float(quality or 0.0) - float(quality_baseline.get("quality_score", quality or 0.0) or 0.0), 1)
            if quality is not None and quality_baseline
            else None
        )
        mapper_ok = bool(st.session_state.get("mapper_approved"))
        readiness = st.session_state.get("ml_readiness_report", {}) or {}
        readiness_status = str(readiness.get("status", "Pending"))
        readiness_score = readiness.get("score")
        readiness_tone = "ok" if readiness_status == "Ready" else "danger" if readiness_status == "Blocked" else "warn"
        pipeline_report = st.session_state.get("feature_pipeline_report", {}) or {}
        pipeline_status = str(pipeline_report.get("status", "Pending"))
        pipeline_tone = "ok" if pipeline_status == "Configured" else "danger" if pipeline_status == "Invalid" else "warn"
        experiment_report = st.session_state.get("ml_experiment_report", {}) or {}
        experiment_status = str(experiment_report.get("status", "Pending"))
        experiment_tone = "ok" if experiment_status == "Completed" else "danger" if experiment_status == "Stale" else "warn"
        package_loaded = st.session_state.get("loaded_model_package") is not None
        quality_tone = "ok" if quality is not None and quality >= 85 else "warn" if quality is not None and quality >= 60 else "danger"
        chips = [
            _chip("User", f"{current_user} · {role_label}", "neutral"),
            _chip("Rows", f"{df.shape[0]:,}", "neutral"),
            _chip("Cols", f"{df.shape[1]:,}", "neutral"),
        ]
        if quality is not None:
            quality_value = f"{quality}%" + (f" ({quality_delta:+.1f})" if quality_delta not in (None, 0.0) else "")
            chips.append(_chip("Quality V2", quality_value, quality_tone))
        chips.append(_chip("Mapping", "Approved" if mapper_ok else "Pending", "ok" if mapper_ok else "warn"))
        if readiness_score is not None:
            chips.append(_chip("ML", f"{readiness_status} {readiness_score}%", readiness_tone))
        chips.append(_chip("Pipeline", pipeline_status, pipeline_tone))
        chips.append(_chip("Experiment", experiment_status, experiment_tone))
        chips.append(_chip("Package", "Verified" if package_loaded else "Pending", "ok" if package_loaded else "warn"))
        chips.append(_chip("AI", ai_mode.title(), "neutral"))
        chips_html = "".join(chips)
    else:
        dataset_title = "No dataset loaded"
        dataset_meta = "Upload CSV or Excel to start profiling and analysis"
        chips_html = "".join([
            _chip("User", f"{current_user} · {role_label}", "neutral"),
            _chip("Version", f"v{APP_VERSION}", "neutral"),
        ])

    st.markdown(f"""
<div class="top-header">
  <div class="top-brand">
    <div class="logo">DataBridge <span>AI</span></div>
    <div class="subtitle">{safe_html("Universal Data Intelligence & Analysis Platform")}</div>
  </div>
  <div class="dataset-head">
    <div class="dataset-title" title="{dataset_title}">{dataset_title}</div>
    <div class="dataset-meta">{safe_html(dataset_meta)}</div>
  </div>
  <div class="top-chips">{chips_html}</div>
</div>
""", unsafe_allow_html=True)
