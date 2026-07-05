# ════════════════════════════════════════════════════════
#  DataBridge AI — Professional Top Header
# ════════════════════════════════════════════════════════
import streamlit as st
from config.constants import APP_VERSION
from core.security import safe_html


def _chip(label: str, value: str = "", tone: str = "neutral") -> str:
    text = f"{safe_html(label)} {safe_html(value)}".strip()
    return f"<span class='top-chip {tone}'>{text}</span>"


def render_header() -> None:
    df = st.session_state.get("df")
    file_name = st.session_state.get("file_name")
    ai_mode = st.session_state.get("ai_mode", "demo")

    if df is not None:
        dataset_title = safe_html(file_name or "Untitled dataset")
        dataset_meta = f"{df.shape[0]:,} rows × {df.shape[1]} cols"
        quality = st.session_state.get("quality_report", {}).get("quality_score")
        mapper_ok = bool(st.session_state.get("mapper_approved"))
        quality_tone = "ok" if quality is not None and quality >= 85 else "warn" if quality is not None and quality >= 60 else "danger"
        chips = [
            _chip("Rows", f"{df.shape[0]:,}", "neutral"),
            _chip("Cols", f"{df.shape[1]:,}", "neutral"),
        ]
        if quality is not None:
            chips.append(_chip("Quality", f"{quality}%", quality_tone))
        chips.append(_chip("Mapping", "Approved" if mapper_ok else "Pending", "ok" if mapper_ok else "warn"))
        chips.append(_chip("AI", ai_mode.title(), "neutral"))
        chips_html = "".join(chips)
    else:
        dataset_title = "No dataset loaded"
        dataset_meta = "Upload CSV or Excel to start profiling and analysis"
        chips_html = "".join([
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
