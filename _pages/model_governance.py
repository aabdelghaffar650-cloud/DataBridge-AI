# ════════════════════════════════════════════════════════
# DataBridge AI — Model Governance
# Stage 15: Champion / Challenger + human approval workflow
# ════════════════════════════════════════════════════════
from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st
from ui.streamlit_compat import safe_dataframe

from core.security import safe_html
from core.session import append_audit_event
from core.team_access import (
    PERM_MODEL_APPROVE,
    PERM_MODEL_PREDICT,
    PERM_MODEL_SUBMIT,
    ROLE_LABELS,
    has_permission,
)
from modules.model_governance import (
    ModelGovernanceError,
    PromotionAssessment,
    STATUS_ARCHIVED,
    STATUS_CANDIDATE,
    STATUS_CHALLENGER,
    STATUS_CHAMPION,
    STATUS_REJECTED,
    assess_promotion,
    ensure_registered_candidate,
    governance_history,
    list_family_models,
    list_governance_families,
    promote_challenger,
    reject_challenger,
    resubmit_archived_as_challenger,
    submit_as_challenger,
)
from modules.model_package import load_signed_model_package
from modules.model_registry import (
    list_registered_packages,
    load_registered_package_bytes,
)
from ui.cards import section_header


def _actor() -> str:
    return str(st.session_state.get("current_user") or "local-user")[:100]


def _bootstrap_registry_governance() -> None:
    failures = []
    for row in list_registered_packages():
        package_id = str(row.get("package_id", ""))
        if not package_id:
            continue
        try:
            ensure_registered_candidate(package_id)
        except Exception as exc:
            failures.append(f"{package_id}: {exc}")
    if failures:
        st.warning(
            "Some registry packages could not enter the governance lifecycle safely. "
            "They remain unavailable for promotion until verification succeeds."
        )
        with st.expander("Governance bootstrap details", expanded=False):
            for item in failures:
                st.caption(item)


def _status_tone(status: str) -> str:
    return {
        STATUS_CHAMPION: "#6bff8e",
        STATUS_CHALLENGER: "#ffcf6b",
        STATUS_CANDIDATE: "#7c6aff",
        STATUS_ARCHIVED: "#6bb8ff",
        STATUS_REJECTED: "#ff6b6b",
    }.get(status, "#999")


def _metric(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def _render_family_header(family: Dict[str, Any], models: list[Dict[str, Any]]) -> None:
    champion = next((row for row in models if row.get("status") == STATUS_CHAMPION), None)
    challengers = sum(1 for row in models if row.get("status") == STATUS_CHALLENGER)
    candidates = sum(1 for row in models if row.get("status") == STATUS_CANDIDATE)
    archived = sum(1 for row in models if row.get("status") == STATUS_ARCHIVED)
    champion_model = safe_html(str(champion.get("model", "—"))) if champion else "None"
    st.markdown(
        f"""
        <div class="metrics-row" style="grid-template-columns:repeat(5,1fr);">
          <div class="metric-card"><div class="label">Active Champion</div>
            <div class="value" style="font-size:.95rem;color:#6bff8e">{champion_model}</div>
            <div class="sub">{safe_html(str(family.get('champion_id') or 'not promoted yet'))}</div></div>
          <div class="metric-card"><div class="label">Challengers</div>
            <div class="value">{challengers}</div><div class="sub">awaiting approval</div></div>
          <div class="metric-card"><div class="label">Candidates</div>
            <div class="value">{candidates}</div><div class="sub">not submitted</div></div>
          <div class="metric-card"><div class="label">Archived</div>
            <div class="value">{archived}</div><div class="sub">former Champions</div></div>
          <div class="metric-card"><div class="label">Governance Revision</div>
            <div class="value">{int(family.get('revision', 0))}</div><div class="sub">stale approvals are rejected</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _models_table(models: list[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in models:
        rows.append(
            {
                "Status": item.get("status", ""),
                "Package": item.get("package_id", ""),
                "Label": item.get("label", ""),
                "Model": item.get("model", ""),
                "Primary Holdout": item.get("metric", ""),
                "Value": item.get("holdout_value"),
                "Train Rows": item.get("training_rows"),
                "Holdout Rows": item.get("holdout_rows"),
                "Created": item.get("created_at", ""),
                "Verified": not bool(item.get("verification_error")),
            }
        )
    return pd.DataFrame(rows)


def _render_assessment(assessment: PromotionAssessment) -> None:
    ready = assessment.ready
    tone = "#6bff8e" if ready else "#ff6b6b"
    st.markdown(
        f"<div class='info-box' style='border-left:4px solid {tone};'>"
        f"<b>{'Promotion gate READY' if ready else 'Promotion gate BLOCKED'}</b> · "
        f"family revision {assessment.state_revision} · assessment becomes invalid after any governance change."
        "</div>",
        unsafe_allow_html=True,
    )
    for item in assessment.blockers:
        st.error(item)
    for item in assessment.warnings:
        st.warning(item)

    comparison = assessment.comparison or {}
    ch = comparison.get("challenger") or {}
    cp = comparison.get("champion") or {}
    if cp:
        delta = comparison.get("metric_delta")
        delta_text = _metric(delta) if delta is not None else "—"
        strength = str(comparison.get("metric_comparison_strength", "indicative")).title()
        st.markdown("#### Champion vs Challenger evidence")
        safe_dataframe(
            pd.DataFrame(
                [
                    {
                        "Role": "Champion",
                        "Package": cp.get("package_id"),
                        "Model": cp.get("model"),
                        "Metric": cp.get("metric"),
                        "Holdout Value": cp.get("holdout_value"),
                        "Holdout Rows": cp.get("holdout_rows"),
                        "Dataset Fingerprint": cp.get("dataset_fingerprint"),
                        "Split": cp.get("split_strategy"),
                    },
                    {
                        "Role": "Challenger",
                        "Package": ch.get("package_id"),
                        "Model": ch.get("model"),
                        "Metric": ch.get("metric"),
                        "Holdout Value": ch.get("holdout_value"),
                        "Holdout Rows": ch.get("holdout_rows"),
                        "Dataset Fingerprint": ch.get("dataset_fingerprint"),
                        "Split": ch.get("split_strategy"),
                    },
                ]
            ),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            f"Comparison strength: {strength} · favourable primary-metric delta: {delta_text}. "
            "Different dataset fingerprints are never presented as an apples-to-apples benchmark."
        )
    else:
        st.info("No Champion exists yet. This Challenger is being assessed as the first production Champion.")


def _load_champion_for_prediction(package_id: str) -> None:
    raw = load_registered_package_bytes(package_id)
    package = load_signed_model_package(raw)
    st.session_state.loaded_model_package = package
    st.session_state.loaded_model_package_bytes = raw
    st.session_state.prediction_result = None
    append_audit_event(
        {
            "event": "champion_loaded_for_prediction",
            "action": "Governed Champion loaded into Prediction Studio",
            "package_id": package_id,
            "task": package.task,
            "target": package.target,
        }
    )


def render(df: pd.DataFrame) -> None:
    st.markdown(
        section_header(
            "🏆",
            "Model Governance",
            "Champion · Challenger · Human Approval",
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="info-box">A signed package enters as <b>Candidate</b>. '
        'It must be explicitly submitted as <b>Challenger</b>, assessed against the current Champion, '
        'and manually approved before promotion. Promotion never happens automatically.</div>',
        unsafe_allow_html=True,
    )

    _bootstrap_registry_governance()
    families = list_governance_families()
    if not families:
        st.info(
            "No governed model packages are registered yet. Build a signed package in ML Studio V2, "
            "then register it from Model Monitoring."
        )
        return

    family_labels = {
        f"{row['family_name']} · {row['task']} → {row['target']}": row for row in families
    }
    choice = st.selectbox("Model family", list(family_labels), key="governance_family_select")
    family = family_labels[choice]
    family_id = str(family["family_id"])
    models = list_family_models(family_id)
    _render_family_header(family, models)

    st.markdown("### Registry lifecycle")
    table = _models_table(models)
    if not table.empty:
        safe_dataframe(table, width="stretch", hide_index=True, height=300)

    available = [row for row in models if not row.get("verification_error")]
    if not available:
        st.error("No package in this family currently passes signed-package verification.")
        return
    labels = {
        f"{row.get('status')} · {row.get('model')} · {row.get('package_id')}": row
        for row in available
    }
    selected_label = st.selectbox("Review model", list(labels), key=f"governance_model_{family_id}")
    selected = labels[selected_label]
    package_id = str(selected["package_id"])
    status = str(selected.get("status", STATUS_CANDIDATE))
    tone = _status_tone(status)
    st.markdown(
        f"<div class='info-box' style='border-left:4px solid {tone};'>"
        f"Status: <b style='color:{tone}'>{safe_html(status)}</b> · "
        f"Package <code>{safe_html(package_id)}</code> · Model {safe_html(str(selected.get('model','—')))} · "
        f"{safe_html(str(selected.get('metric','')))} {_metric(selected.get('holdout_value'))}"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("### Decision workflow")
    actor = _actor()
    can_submit = has_permission(PERM_MODEL_SUBMIT, session_state=st.session_state)
    can_approve = has_permission(PERM_MODEL_APPROVE, session_state=st.session_state)
    can_predict = has_permission(PERM_MODEL_PREDICT, session_state=st.session_state)
    role = str(st.session_state.get("current_role") or "viewer")
    st.caption(f"Signed in as {actor} · {ROLE_LABELS.get(role, role)}. Submission and production approval are separate permissions.")

    if status == STATUS_CANDIDATE:
        note = st.text_input(
            "Submission note",
            placeholder="Why should this package enter Champion/Challenger review?",
            key=f"gov_submit_note_{package_id}",
        )
        if not can_submit:
            st.info("Your role can review governance evidence but cannot submit Candidates as Challengers.")
        if st.button("Submit as Challenger", type="primary", width="stretch", disabled=not can_submit, key=f"gov_submit_{package_id}"):
            try:
                submit_as_challenger(package_id, actor=actor, note=note)
                append_audit_event(
                    {
                        "event": "model_challenger_submitted",
                        "action": "Candidate submitted as Challenger",
                        "package_id": package_id,
                        "model_family": family["family_name"],
                    }
                )
                st.session_state.model_governance_assessment = None
                st.success("Candidate entered Challenger review. No production promotion occurred.")
                st.rerun()
            except Exception as exc:
                st.error(f"Submission blocked safely: {exc}")

    elif status == STATUS_CHALLENGER:
        left, right = st.columns(2)
        with left:
            if st.button("Run Promotion Assessment", type="primary", width="stretch", key=f"gov_assess_{package_id}"):
                try:
                    assessment = assess_promotion(package_id)
                    st.session_state.model_governance_assessment = assessment
                    append_audit_event(
                        {
                            "event": "model_promotion_assessed",
                            "action": "Champion/Challenger promotion gate assessed",
                            "package_id": package_id,
                            "ready": assessment.ready,
                            "family_id": assessment.family_id,
                        }
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(f"Assessment failed safely: {exc}")
        with right:
            reject_reason = st.text_input(
                "Rejection reason",
                placeholder="Required only when rejecting",
                key=f"gov_reject_reason_{package_id}",
            )
            reject_confirm = st.checkbox("Confirm rejection", key=f"gov_reject_confirm_{package_id}")
            if st.button(
                "Reject Challenger",
                width="stretch",
                disabled=not reject_confirm or not can_approve,
                key=f"gov_reject_{package_id}",
            ):
                try:
                    reject_challenger(package_id, reason=reject_reason, actor=actor)
                    append_audit_event(
                        {
                            "event": "model_challenger_rejected",
                            "action": "Challenger rejected by human review",
                            "package_id": package_id,
                            "family_id": family_id,
                        }
                    )
                    st.session_state.model_governance_assessment = None
                    st.rerun()
                except Exception as exc:
                    st.error(f"Rejection blocked safely: {exc}")

        assessment = st.session_state.get("model_governance_assessment")
        if isinstance(assessment, PromotionAssessment) and assessment.challenger_id == package_id:
            _render_assessment(assessment)
            if assessment.ready:
                if not can_approve:
                    st.warning("Promotion requires Manager / Approver or Admin permission. You may review the evidence, but cannot approve production promotion.")
                approval_note = st.text_area(
                    "Approval note",
                    placeholder="Record why this Challenger is approved for production.",
                    key=f"gov_approval_note_{package_id}",
                )
                expected = f"PROMOTE {package_id}"
                confirmation = st.text_input(
                    f"Type exactly: {expected}",
                    key=f"gov_promote_confirm_text_{package_id}",
                )
                if st.button(
                    "🏆 Promote Challenger to Champion",
                    type="primary",
                    width="stretch",
                    disabled=confirmation.strip() != expected or not can_approve,
                    key=f"gov_promote_{package_id}",
                ):
                    try:
                        result = promote_challenger(
                            package_id,
                            approval_token=assessment.approval_token,
                            approval_note=approval_note,
                            actor=actor,
                        )
                        append_audit_event(
                            {
                                "event": "model_challenger_promoted",
                                "action": "Human-approved Challenger promoted to Champion",
                                "package_id": package_id,
                                "previous_champion_id": result.get("previous_champion_id", ""),
                                "family_id": result.get("family_id", ""),
                            }
                        )
                        st.session_state.model_governance_assessment = None
                        st.success("Promotion completed. Previous Champion was archived atomically.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Promotion blocked safely: {exc}")

    elif status == STATUS_CHAMPION:
        st.success("This package is the active production Champion for the selected model family.")
        if st.button(
            "Load Champion into Prediction Studio",
            type="primary",
            width="stretch",
            disabled=not can_predict,
            key=f"gov_load_champion_{package_id}",
        ):
            try:
                _load_champion_for_prediction(package_id)
                st.session_state.current_page = "prediction_studio"
                st.rerun()
            except Exception as exc:
                st.error(f"Champion load failed safely: {exc}")

    elif status == STATUS_ARCHIVED:
        st.info("This is a former Champion. Rollback is never direct; it must re-enter as a Challenger and pass a fresh assessment.")
        reason = st.text_input(
            "Rollback / resubmission reason",
            placeholder="Why should this former Champion be reconsidered?",
            key=f"gov_rollback_reason_{package_id}",
        )
        confirm = st.checkbox("Confirm re-submission for rollback review", key=f"gov_rollback_confirm_{package_id}")
        if st.button(
            "Re-submit Archived Model as Challenger",
            disabled=not confirm or not can_submit,
            width="stretch",
            key=f"gov_rollback_{package_id}",
        ):
            try:
                resubmit_archived_as_challenger(package_id, reason=reason, actor=actor)
                append_audit_event(
                    {
                        "event": "model_archived_resubmitted",
                        "action": "Archived Champion re-submitted through Challenger gate",
                        "package_id": package_id,
                        "family_id": family_id,
                    }
                )
                st.session_state.model_governance_assessment = None
                st.rerun()
            except Exception as exc:
                st.error(f"Rollback review submission blocked safely: {exc}")

    elif status == STATUS_REJECTED:
        st.warning("This immutable package was rejected. Train and register a new package instead of reusing the rejected artifact.")

    st.markdown("---")
    st.markdown("### Authenticated decision history")
    try:
        history = governance_history(family_id)
        if history:
            history_df = pd.DataFrame(history)
            visible = [col for col in ["timestamp", "event", "package_id", "actor", "note", "event_hash", "previous_event_hash"] if col in history_df.columns]
            safe_dataframe(history_df[visible].iloc[::-1], width="stretch", hide_index=True, height=320)
            st.caption("The governance file is HMAC-authenticated and each event contains a hash link to the previous event.")
        else:
            st.caption("No governance decisions recorded yet.")
    except ModelGovernanceError as exc:
        st.error(f"Governance history failed authentication: {exc}")
