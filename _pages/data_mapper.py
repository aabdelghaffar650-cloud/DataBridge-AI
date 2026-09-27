# ════════════════════════════════════════════════════════
#  DataBridge AI — Semantic Mapper & ML Readiness
#  Stage 6: content-aware typing with human approval
# ════════════════════════════════════════════════════════
from __future__ import annotations

from typing import Any, Dict

import pandas as pd
import streamlit as st
from ui.streamlit_compat import safe_dataframe

from config.constants import ML_SEMANTIC_TYPES
from core.dataset import current_dataset_revision, update_dataset_context
from core.security import safe_html
from modules.data_mapper import (
    analyze_dataframe,
    apply_manual_mappings,
    build_confidence_summary,
    confidence_label,
)
from ui.cards import section_header


def _status_color(status: str) -> str:
    return {
        "Ready": "#6bff8e",
        "Needs Review": "#ffb86b",
        "Blocked": "#ff6b6b",
    }.get(status, "#7c6aff")


def _profile_table(profiles: Dict[str, Dict[str, Any]], mappings: Dict[str, str]) -> pd.DataFrame:
    rows = []
    for column, profile in profiles.items():
        flags = []
        if profile.get("high_cardinality"):
            flags.append("High cardinality")
        if profile.get("leakage_risk"):
            flags.append("Leakage/ID risk")
        if profile.get("is_constant"):
            flags.append("Constant")
        if profile.get("all_missing"):
            flags.append("All missing")
        if profile.get("manual_override"):
            flags.append("Manual override")

        signals = profile.get("signals", {}) or {}
        reasons = profile.get("reasons", []) or []
        rows.append(
            {
                "Column": column,
                "Approved Type": mappings.get(column, profile.get("effective_semantic_type", "Unknown")),
                "Auto Type": profile.get("semantic_type", "Unknown"),
                "Confidence": f"{float(profile.get('confidence', 0.0)) * 100:.0f}%",
                "Business Role": profile.get("business_role", "Unknown"),
                "Pandas Type": profile.get("dtype", ""),
                "Missing %": float(profile.get("missing_pct", 0.0)),
                "Unique": int(profile.get("unique_count", 0)),
                "Unique %": round(float(profile.get("unique_ratio", 0.0)) * 100, 1),
                "Numeric Match": f"{float(signals.get('numeric_ratio', 0.0)) * 100:.0f}%",
                "Date Match": f"{float(signals.get('date_ratio', 0.0)) * 100:.0f}%",
                "Target": (
                    f"{profile.get('target_task')} {float(profile.get('target_candidate_score', 0.0)) * 100:.0f}%"
                    if profile.get("target_task")
                    else "—"
                ),
                "Flags": ", ".join(flags) if flags else "—",
                "Evidence": "; ".join(map(str, reasons[:2])) if reasons else "—",
            }
        )
    return pd.DataFrame(rows)


def _render_readiness(readiness: Dict[str, Any]) -> None:
    status = str(readiness.get("status", "Needs Review"))
    score = float(readiness.get("score", 0.0))
    counts = readiness.get("counts", {}) or {}
    color = _status_color(status)

    st.markdown("#### 🧪 ML Readiness Assessment")
    st.markdown(
        f"""
        <div class="quality-score-banner" style="border-left:4px solid {color};">
          <div>
            <div style="font-size:.72rem;color:#777;text-transform:uppercase;letter-spacing:.08em;">ML Readiness</div>
            <div style="font-size:2.7rem;font-weight:700;color:{color};font-family:'JetBrains Mono',monospace;">{score:.1f}%</div>
          </div>
          <div style="flex:1;">
            <div style="font-size:1rem;font-weight:700;color:{color};margin-bottom:.35rem;">{safe_html(status)}</div>
            <div style="font-size:.82rem;color:#999;line-height:1.65;">
              Usable features: <b style="color:#e0e0f0">{int(counts.get('usable_features', 0))}</b> &nbsp;·&nbsp;
              Target candidates: <b style="color:#e0e0f0">{int(counts.get('target_candidates', 0))}</b> &nbsp;·&nbsp;
              Review mappings: <b style="color:#e0e0f0">{int(counts.get('low_confidence', 0))}</b> &nbsp;·&nbsp;
              Leakage/ID risks: <b style="color:#e0e0f0">{int(counts.get('leakage_risks', 0))}</b>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    blockers = readiness.get("blockers", []) or []
    warnings_list = readiness.get("warnings", []) or []
    actions = readiness.get("recommended_actions", []) or []

    for message in blockers:
        st.error(f"⛔ {message}")
    if warnings_list:
        with st.expander(f"⚠️ Review warnings ({len(warnings_list)})", expanded=bool(blockers)):
            for message in warnings_list:
                st.markdown(f"- {message}")
    if actions:
        with st.expander(f"✅ Recommended preparation actions ({len(actions)})", expanded=False):
            for index, action in enumerate(actions, start=1):
                st.markdown(f"{index}. {action}")

    candidates = readiness.get("target_candidates", []) or []
    if candidates:
        candidate_rows = []
        for item in candidates[:12]:
            candidate_rows.append(
                {
                    "Column": item.get("column"),
                    "Suggested Task": str(item.get("task", "")).title(),
                    "Suitability": f"{float(item.get('score', 0.0)) * 100:.0f}%",
                    "Unique": int(item.get("unique_count", 0)),
                    "Missing %": round(float(item.get("missing_pct", 0.0)), 1),
                    "Reason": "; ".join(map(str, (item.get("reasons", []) or [])[:2])),
                }
            )
        st.markdown("**Suggested target candidates — final selection remains manual**")
        safe_dataframe(pd.DataFrame(candidate_rows), width="stretch", hide_index=True)

    plan = readiness.get("preprocessing_plan", {}) or {}
    with st.expander("🧭 Detected preprocessing plan", expanded=False):
        plan_rows = []
        labels = {
            "numeric": "Numeric",
            "categorical": "Categorical",
            "ordinal": "Ordinal",
            "boolean": "Boolean",
            "datetime": "Datetime",
            "text": "Free text",
            "excluded_by_default": "Excluded by default",
        }
        for key, label in labels.items():
            columns = plan.get(key, []) or []
            plan_rows.append({"Group": label, "Count": len(columns), "Columns": ", ".join(map(str, columns)) or "—"})
        safe_dataframe(pd.DataFrame(plan_rows), width="stretch", hide_index=True)


def render(df: pd.DataFrame) -> None:
    st.markdown(
        section_header(
            "🗺️",
            "Semantic Mapper & ML Readiness",
            "Content-aware · Human-in-the-loop",
        ),
        unsafe_allow_html=True,
    )

    revision = current_dataset_revision()
    mappings = dict(st.session_state.get("mapping_confidence", {}) or {})
    profiles = dict(st.session_state.get("semantic_profiles", {}) or {})
    readiness = dict(st.session_state.get("ml_readiness_report", {}) or {})
    approved_mappings = dict(st.session_state.get("column_mappings", {}) or {})

    if not mappings or set(mappings) != set(map(str, df.columns)):
        st.error("Semantic analysis is unavailable or stale. Re-open the dataset to rebuild it safely.")
        return

    summary = build_confidence_summary(mappings)
    st.markdown(
        '<div class="info-box">The mapper combines column names, pandas types, value patterns, '
        'cardinality, missingness, uniqueness, and sampled content. Name-only guesses are never '
        'auto-approved. Review all outcome, identifier, and low-confidence columns before ML.</div>',
        unsafe_allow_html=True,
    )

    st.markdown("#### 📊 Semantic Confidence Dashboard")
    st.markdown(
        f"""
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
            <div class="value" style="color:#777">{summary['unknown_pct']}%</div>
            <div class="sub">{summary['unknown']} columns</div></div>
          <div class="metric-card"><div class="label">Overall Confidence</div>
            <div class="value" style="color:#7c6aff">{summary['overall_pct']}%</div>
            <div class="sub">content-aware score</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _render_readiness(readiness)

    st.markdown("---")
    st.markdown("#### 🔬 Column Semantic Profiles")
    if profiles:
        safe_dataframe(
            _profile_table(profiles, approved_mappings),
            width="stretch",
            height=390,
            hide_index=True,
        )

    st.markdown("---")
    st.markdown("#### ✅ Human Mapping Review")
    st.markdown(
        '<div class="info-box">Choose the actual ML semantic type for every column. '
        'Approval stores the reviewed schema and rebuilds the readiness report without changing data.</div>',
        unsafe_allow_html=True,
    )

    edited_mappings: Dict[str, str] = {}
    label_colors = {
        "auto_accepted": "#6bff8e",
        "verify": "#ffb86b",
        "suspicious": "#ff6b6b",
        "unknown": "#777",
    }
    label_emojis = {
        "auto_accepted": "✅",
        "verify": "⚠️",
        "suspicious": "🔴",
        "unknown": "❓",
    }

    for index, column in enumerate(map(str, df.columns)):
        auto_group, raw_score = mappings[column]
        score = float(raw_score)
        label = confidence_label(score)
        color = label_colors.get(label, "#777")
        emoji = label_emojis.get(label, "❓")
        current = approved_mappings.get(column, auto_group)
        selected_index = ML_SEMANTIC_TYPES.index(current) if current in ML_SEMANTIC_TYPES else ML_SEMANTIC_TYPES.index("Unknown")
        profile = profiles.get(column, {}) or {}
        flags = []
        if profile.get("leakage_risk"):
            flags.append("risk")
        if profile.get("high_cardinality"):
            flags.append("high-card")
        flag_text = f" · {', '.join(flags)}" if flags else ""

        c1, c2, c3 = st.columns([2.2, 2.2, 1.1])
        with c1:
            st.markdown(
                f'<div style="padding:.5rem 0;font-family:JetBrains Mono,monospace;'
                f'font-size:.84rem;color:#e0e0f0;">{safe_html(column)}</div>',
                unsafe_allow_html=True,
            )
        with c2:
            selected = st.selectbox(
                f"Semantic type for {column}",
                ML_SEMANTIC_TYPES,
                index=selected_index,
                key=f"semantic_mapper_{index}",
                label_visibility="collapsed",
            )
            edited_mappings[column] = selected
        with c3:
            st.markdown(
                f'<div style="padding:.5rem 0;font-size:.76rem;color:{color};">'
                f'{emoji} {score * 100:.0f}%{safe_html(flag_text)}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    button_col, rescan_col = st.columns([2, 1])
    with button_col:
        approve_clicked = st.button(
            "✅ Approve Semantic Mappings & ML Readiness",
            type="primary",
            width="stretch",
        )
    with rescan_col:
        rescan_clicked = st.button(
            "🔄 Re-run Analysis",
            width="stretch",
            help="Rebuild automatic content profiles and clear previous approval.",
        )

    if approve_clicked:
        try:
            reviewed_profiles, reviewed_readiness = apply_manual_mappings(
                df, profiles, edited_mappings
            )
            update_dataset_context(
                expected_revision=revision,
                column_mappings=edited_mappings,
                semantic_profiles=reviewed_profiles,
                ml_readiness_report=reviewed_readiness,
                mapper_approved=True,
            )
            st.success("Semantic mappings approved and ML readiness rebuilt safely.")
            st.rerun()
        except Exception as exc:
            st.error(f"Mapping approval blocked safely: {exc}")

    if rescan_clicked:
        try:
            analysis = analyze_dataframe(df)
            automatic_mappings = {
                column: semantic_type
                for column, (semantic_type, _) in analysis["mappings"].items()
            }
            update_dataset_context(
                expected_revision=revision,
                mapping_confidence=analysis["mappings"],
                column_mappings=automatic_mappings,
                semantic_profiles=analysis["profiles"],
                ml_readiness_report=analysis["readiness"],
                mapper_approved=False,
            )
            st.success("Semantic analysis rebuilt. Review and approve the mappings.")
            st.rerun()
        except Exception as exc:
            st.error(f"Semantic analysis failed safely: {exc}")

    if st.session_state.get("mapper_approved"):
        st.success("✅ Semantic mappings are approved for the current dataset revision.")
    else:
        st.warning("⚠️ Downstream analysis remains locked until semantic mappings are approved.")
