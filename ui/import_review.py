# ════════════════════════════════════════════════════════
#  DataBridge AI — Safe Import Review UI
#  Stage 3: explicit approval for value/type/delete actions
# ════════════════════════════════════════════════════════
from __future__ import annotations

import copy
from typing import Any, Dict, List

import pandas as pd
import streamlit as st
from ui.streamlit_compat import safe_dataframe

from core.dataset import (
    current_dataset_revision,
    dataframe_fingerprint,
    replace_working_dataframe,
    update_dataset_context,
)
from core.security import safe_html
from modules.import_engine import (
    COMPLETED_IMPORT_SIGNATURES_KEY,
    PROPOSED_IMPORT_ACTIONS_KEY,
    apply_import_actions,
    build_import_action_proposals,
)


RISK_STYLE = {
    "low": ("LOW", "#6bff8e"),
    "medium": ("MEDIUM", "#ffb86b"),
    "high": ("HIGH", "#ff6b6b"),
}


def _clear_action_widget_state(
    key_prefix: str,
    proposals: List[Dict[str, Any]],
) -> None:
    """
    Widget keys are versioned instead of mutated after instantiation.

    Streamlit blocks changing a widget's session-state key after that widget has
    rendered in the current run. A new review generation safely resets controls
    on the next rerun without touching live widget keys.
    """
    return None


def _refresh_report_proposals(
    report: Dict[str, Any],
    df: pd.DataFrame,
) -> Dict[str, Any]:
    updated = copy.deepcopy(report)
    completed = list(updated.get(COMPLETED_IMPORT_SIGNATURES_KEY, []))
    proposals = build_import_action_proposals(
        df,
        excluded_signatures=completed,
    )
    updated[PROPOSED_IMPORT_ACTIONS_KEY] = proposals
    updated["import_review_status"] = "pending" if proposals else "complete"
    updated["import_review_fingerprint"] = dataframe_fingerprint(df)
    updated["import_review_shape"] = tuple(df.shape)
    updated["import_review_generation"] = int(
        updated.get("import_review_generation", 0)
    ) + 1
    return updated


def _render_preview(action: Dict[str, Any]) -> None:
    preview = action.get("preview") or []
    if not preview:
        return
    with st.expander("Preview", expanded=False):
        rows = [
            {
                "Before": str(item.get("before", "")),
                "After": str(item.get("after", "")),
            }
            for item in preview[:3]
        ]
        safe_dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def render_import_safety_review(
    df: pd.DataFrame,
    *,
    key_prefix: str = "safe_import",
    heading: bool = True,
) -> None:
    """Render opt-in import recommendations for the active working dataset."""
    report = st.session_state.get("data_clean_report", {}) or {}
    revision = current_dataset_revision()
    if report.get("safe_import_policy") != "review_required":
        return

    if heading:
        st.markdown("#### 🛡️ Safe Import Review")

    proposals = list(report.get(PROPOSED_IMPORT_ACTIONS_KEY, []) or [])
    expected_fingerprint = report.get("import_review_fingerprint")
    current_fingerprint = dataframe_fingerprint(df)
    generation = int(report.get("import_review_generation", 0))
    widget_prefix = f"{key_prefix}_{generation}"

    st.markdown(
        "<div class='info-box'><b>No value conversion or deletion was applied automatically.</b> "
        "Every item below is optional. Date and boolean recognition creates a new derived "
        "column and keeps the source column unchanged.</div>",
        unsafe_allow_html=True,
    )

    if expected_fingerprint and expected_fingerprint != current_fingerprint:
        st.warning(
            "The working dataset changed after this review was generated. "
            "The old recommendations are blocked until you re-scan the current data."
        )
        if st.button(
            "Re-scan current working data",
            width="stretch",
            key=f"{widget_prefix}_rescan_stale",
        ):
            _clear_action_widget_state(key_prefix, proposals)
            try:
                update_dataset_context(
                    expected_revision=revision,
                    data_clean_report=_refresh_report_proposals(report, df),
                )
                st.rerun()
            except Exception as exc:
                st.error(f"Re-scan blocked safely: {exc}")
        return

    if not proposals:
        st.success(
            "✅ No pending import actions. The import engine did not alter values, types, rows, or source columns automatically."
        )
        if st.button(
            "Re-scan current working data",
            width="stretch",
            key=f"{widget_prefix}_rescan_clean",
        ):
            try:
                update_dataset_context(
                    expected_revision=revision,
                    data_clean_report=_refresh_report_proposals(report, df),
                )
                st.rerun()
            except Exception as exc:
                st.error(f"Re-scan blocked safely: {exc}")
        return

    risk_counts = {
        risk: sum(1 for action in proposals if action.get("risk") == risk)
        for risk in ("high", "medium", "low")
    }
    st.caption(
        f"Pending recommendations: {len(proposals)} · "
        f"High risk: {risk_counts['high']} · Medium: {risk_counts['medium']} · Low: {risk_counts['low']}"
    )

    selected_ids: List[str] = []
    for action in proposals:
        action_id = str(action.get("id", ""))
        risk = str(action.get("risk", "low"))
        risk_label, risk_color = RISK_STYLE.get(risk, RISK_STYLE["low"])
        title = safe_html(str(action.get("title", "Import action")))
        detail = safe_html(str(action.get("detail", "")))
        count = int(action.get("count", 0) or 0)

        left, right = st.columns([0.9, 0.1])
        with left:
            selected = st.checkbox(
                str(action.get("title", "Import action")),
                value=False,
                key=f"{widget_prefix}_select_{action_id}",
            )
            st.markdown(
                f"<div style='margin:-.45rem 0 .35rem 1.8rem;font-size:.77rem;color:#8f8fa7;line-height:1.55;'>"
                f"{detail}</div>",
                unsafe_allow_html=True,
            )
            _render_preview(action)
        with right:
            st.markdown(
                f"<div style='text-align:right;padding-top:.3rem;'>"
                f"<span style='font-size:.64rem;font-weight:700;color:{risk_color};'>{risk_label}</span><br>"
                f"<span style='font-size:.72rem;color:#777;'>{count:,}</span></div>",
                unsafe_allow_html=True,
            )
        if selected:
            selected_ids.append(action_id)
        st.markdown("<div style='height:.15rem'></div>", unsafe_allow_html=True)

    destructive_selected = any(
        action.get("id") in selected_ids and bool(action.get("destructive"))
        for action in proposals
    )
    confirm_label = (
        "I reviewed the selected actions and understand that selected row/column deletion will change the working dataset."
        if destructive_selected
        else "I reviewed the selected actions and approve applying them to the working dataset."
    )
    confirmed = st.checkbox(confirm_label, key=f"{widget_prefix}_confirm")

    c_apply, c_rescan = st.columns([1.2, 0.8])
    with c_apply:
        if st.button(
            "Apply selected approved actions",
            type="primary",
            width="stretch",
            disabled=not selected_ids or not confirmed,
            key=f"{widget_prefix}_apply",
        ):
            try:
                # Build the complete result first. If review validation fails, no
                # history transaction or dataset mutation is opened.
                updated_df, applied_steps, completed = apply_import_actions(
                    df,
                    proposals,
                    selected_ids,
                )

                updated_report = copy.deepcopy(report)
                updated_report.setdefault("cleaning_steps", []).extend(applied_steps)
                signatures = list(
                    updated_report.get(COMPLETED_IMPORT_SIGNATURES_KEY, [])
                )
                signatures.extend(completed)
                updated_report[COMPLETED_IMPORT_SIGNATURES_KEY] = list(
                    dict.fromkeys(signatures)
                )
                updated_report.setdefault("approved_import_actions", []).extend(
                    [
                        {
                            "id": action.get("id"),
                            "signature": action.get("signature"),
                            "operation": action.get("operation"),
                            "title": action.get("title"),
                            "count": action.get("count", 0),
                        }
                        for action in proposals
                        if action.get("id") in selected_ids
                    ]
                )
                updated_report = _refresh_report_proposals(
                    updated_report,
                    updated_df,
                )

                _clear_action_widget_state(key_prefix, proposals)
                replace_working_dataframe(
                    updated_df,
                    action="Apply approved import actions",
                    details={
                        "selected_action_ids": list(selected_ids),
                        "selected_count": len(selected_ids),
                        "contains_deletion": destructive_selected,
                    },
                    expected_revision=revision,
                    context_updates={
                        "data_clean_report": updated_report,
                        "show_import_report": True,
                    },
                )
                st.success("Approved import actions were applied to the working copy only.")
                st.rerun()
            except Exception as exc:
                st.error(f"Safe import action blocked: {exc}")

    with c_rescan:
        if st.button(
            "Re-scan",
            width="stretch",
            key=f"{widget_prefix}_rescan_pending",
        ):
            _clear_action_widget_state(key_prefix, proposals)
            try:
                update_dataset_context(
                    expected_revision=revision,
                    data_clean_report=_refresh_report_proposals(report, df),
                )
                st.rerun()
            except Exception as exc:
                st.error(f"Re-scan blocked safely: {exc}")
