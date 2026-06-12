# ════════════════════════════════════════════════════════
#  DataBridge AI — Page: Data Mapper
# ════════════════════════════════════════════════════════
import pandas as pd
import streamlit as st

from config.constants import SEMANTIC_GROUPS
from modules.data_mapper import confidence_label, build_confidence_summary
from ui.cards import section_header
from core.security import safe_html


def render(df: pd.DataFrame) -> None:
    st.markdown(
        section_header("🗺️", "Auto Data Mapper & Mapping Review Center", "Human-in-the-loop"),
        unsafe_allow_html=True,
    )

    mappings = st.session_state.mapping_confidence

    if not mappings:
        st.info("Upload a file first to enable the Auto Mapper.")
        return

    summary = build_confidence_summary(mappings)

    # ── Confidence Dashboard ──
    st.markdown("#### 📊 Mapping Confidence Dashboard")
    st.markdown(f"""
    <div class="metrics-row" style="grid-template-columns: repeat(5,1fr);">
      <div class="metric-card"><div class="label">Auto Accepted</div>
        <div class="value" style="color:#6bff8e">{summary['auto_accepted_pct']}%</div>
        <div class="sub">{summary['auto_accepted']} columns</div></div>
      <div class="metric-card"><div class="label">Verify</div>
        <div class="value" style="color:#ffb86b">{summary['verify_pct']}%</div>
        <div class="sub">{summary['verify']} columns</div></div>
      <div class="metric-card"><div class="label">Suspicious</div>
        <div class="value" style="color:#ff6b6b">{summary['suspicious_pct']}%</div>
        <div class="sub">{summary['suspicious']} columns</div></div>
      <div class="metric-card"><div class="label">Unknown</div>
        <div class="value" style="color:#555">{summary['unknown_pct']}%</div>
        <div class="sub">{summary['unknown']} columns</div></div>
      <div class="metric-card"><div class="label">Overall Confidence</div>
        <div class="value" style="color:#7c6aff">{summary['overall_pct']}%</div>
        <div class="sub">mapping quality</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 🔍 Review & Edit Column Mappings")
    st.markdown(
        '<div class="info-box">📋 Review the auto-detected mappings. '
        'Override any mapping using the dropdowns, then click '
        '<b>Approve Mappings & Run Analysis</b> to unlock all pages.</div>',
        unsafe_allow_html=True,
    )

    group_options = list(SEMANTIC_GROUPS.keys())
    edited_mappings = {}
    lbl_colors = {
        "auto_accepted": "#6bff8e",
        "verify":        "#ffb86b",
        "suspicious":    "#ff6b6b",
        "unknown":       "#555",
    }
    lbl_emojis = {
        "auto_accepted": "✅",
        "verify":        "⚠️",
        "suspicious":    "🔴",
        "unknown":       "❓",
    }

    for col, (group, score) in mappings.items():
        lbl   = confidence_label(score)
        color = lbl_colors.get(lbl, "#555")
        emoji = lbl_emojis.get(lbl, "❓")

        c1, c2, c3 = st.columns([2, 2, 1])
        with c1:
            st.markdown(
                f'<div style="padding:.5rem 0;font-family:JetBrains Mono,monospace;'
                f'font-size:.85rem;color:#e0e0f0;">{safe_html(col)}</div>',
                unsafe_allow_html=True,
            )
        with c2:
            idx      = group_options.index(group) if group in group_options else len(group_options) - 1
            selected = st.selectbox(
                "", group_options, index=idx,
                key=f"mapper_{col}", label_visibility="collapsed",
            )
            edited_mappings[col] = selected
        with c3:
            st.markdown(
                f'<div style="padding:.5rem 0;font-size:.78rem;color:{color};">'
                f'{emoji} {round(score * 100)}%</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    if st.session_state.mapper_approved:
        st.success("✅ Mappings already approved — you can re-approve after edits.")

    if st.button("✅ Approve Mappings & Run Analysis", type="primary", use_container_width=True):
        st.session_state.column_mappings  = edited_mappings
        st.session_state.mapper_approved  = True
        st.success("✅ Mappings approved! All pages are now unlocked.")
        st.balloons()
        st.rerun()

    if not st.session_state.mapper_approved:
        st.warning("⚠️ Full analysis is blocked until you approve mappings above.")
