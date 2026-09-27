# ════════════════════════════════════════════════════════
# DataBridge AI — Safe Auto Retraining Workflow
# Stage 16 UI: Champion-triggered contract replay that always stops at Challenger.
# ════════════════════════════════════════════════════════
from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd
import streamlit as st
from ui.streamlit_compat import safe_dataframe

from core.dataset import current_dataset_revision, dataframe_fingerprint
from core.security import safe_html
from core.session import append_audit_event
from modules.model_governance import current_champion, list_governance_families
from modules.model_package import LoadedModelPackage, load_signed_model_package
from modules.model_registry import load_registered_package_bytes
from modules.retraining_workflow import (
    RetrainingWorkflowError,
    assess_retraining_need,
    create_and_register_retraining_challenger,
    run_safe_retraining_experiment,
)
from ui.cards import section_header


def _verified_champion(family_id: str) -> tuple[Optional[LoadedModelPackage], Dict[str, Any]]:
    summary = current_champion(family_id)
    if not summary:
        return None, {}
    package_id = str(summary.get("package_id", ""))
    raw = load_registered_package_bytes(package_id)
    package = load_signed_model_package(raw)
    return package, summary


def _source_semantics(package: LoadedModelPackage, active: bool) -> Dict[str, Dict[str, Any]]:
    if active:
        profiles = st.session_state.get("semantic_profiles", {}) or {}
        if profiles:
            return dict(profiles)
    contract = getattr(package, "retraining_contract", None) or {}
    spec = dict(contract.get("feature_pipeline_spec", {}) or {})
    semantics = dict(spec.get("semantic_types", {}) or {})
    return {
        str(column): {"effective_semantic_type": str(semantic), "semantic_type": str(semantic)}
        for column, semantic in semantics.items()
    }


def _render_contract(assessment) -> None:
    summary = assessment.contract_summary or {}
    if not summary:
        return
    st.markdown("### Locked retraining contract")
    st.caption(
        "Stage 16 reuses the Champion's saved feature logic, split strategy, model candidates, random seed, CV policy, and tuning policy. "
        "Only the dataset revision/fingerprint is rebound after validation."
    )
    rows = [
        ("Task", summary.get("task")),
        ("Target", summary.get("target")),
        ("Input features", summary.get("feature_count")),
        ("Split", summary.get("split_strategy")),
        ("Split column", summary.get("split_column") or "—"),
        ("Holdout", f"{float(summary.get('holdout_size', 0))*100:.0f}%"),
        ("CV folds", summary.get("cv_folds")),
        ("Random seed", summary.get("random_state")),
        ("Class weighting", summary.get("class_weight_mode")),
        ("Tune winner", "Yes" if summary.get("tune_best") else "No"),
    ]
    safe_dataframe(pd.DataFrame(rows, columns=["Contract", "Value"]), width="stretch", hide_index=True)
    models = summary.get("candidate_models", []) or []
    if models:
        st.caption("Candidate models: " + ", ".join(map(str, models)))


def _render_assessment(assessment) -> None:
    status = assessment.monitoring_status
    if assessment.eligible and assessment.recommended:
        st.warning(f"Retraining recommended · monitoring status: {status}")
    elif assessment.eligible:
        st.info(f"Retraining gate is technically eligible, but no automatic trigger is active · monitoring status: {status}")
    else:
        st.error("Retraining is blocked until the contract/data issues below are resolved.")

    for reason in assessment.reasons:
        st.markdown(f"- ✅ {safe_html(reason)}", unsafe_allow_html=True)
    for warning in assessment.warnings:
        st.warning(warning)
    for blocker in assessment.blockers:
        st.error(blocker)

    perf = assessment.performance_signal or {}
    if perf.get("available"):
        metric = safe_html(str(perf.get("metric", "Metric")))
        st.markdown(
            f"<div class='info-box'><b>{metric}</b> · signed holdout: {perf.get('signed_holdout')} · "
            f"current: {perf.get('current')} · trigger crossed: <b>{'Yes' if perf.get('triggered') else 'No'}</b></div>",
            unsafe_allow_html=True,
        )


def render(df: pd.DataFrame) -> None:
    st.markdown(
        section_header(
            "♻️",
            "Safe Auto Retraining",
            "Champion → contract-locked retrain → Challenger → human approval",
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="info-box"><b>Safety rule:</b> Stage 16 can train and register a new Challenger, but it has no code path that promotes it. '
        'The current Champion remains active until Model Governance performs a fresh human approval.</div>',
        unsafe_allow_html=True,
    )

    families = [row for row in list_governance_families() if row.get("champion_id")]
    if not families:
        st.warning("No governed model family has an active Champion. Promote a verified Challenger in Model Governance first.")
        return

    labels = [f"{row['family_name']} · {row['task']} · {row['target']}" for row in families]
    choice = st.selectbox("Champion model family", labels, key="retrain_family")
    family = families[labels.index(choice)]
    try:
        champion, champion_summary = _verified_champion(str(family["family_id"]))
    except Exception as exc:
        st.error(f"Champion verification failed safely: {exc}")
        return
    if champion is None:
        st.error("Active Champion could not be resolved.")
        return

    has_contract = bool(getattr(champion, "retraining_contract", None))
    st.markdown(
        f"<div class='info-box'><b>Champion:</b> {safe_html(champion.package_id)} · "
        f"{safe_html(str(champion_summary.get('model','')))} · target <code>{safe_html(champion.target)}</code> · "
        f"Stage 16 contract: <b>{'Ready' if has_contract else 'Missing'}</b></div>",
        unsafe_allow_html=True,
    )
    if not has_contract:
        st.warning(
            "This Champion was packaged before Stage 16. Rebuild a signed package from a fresh ML Studio V2 experiment after installing Stage 16, "
            "then register/promote it once so future retraining can replay an authenticated contract."
        )
        return

    st.markdown("### Retraining data")
    monitoring_df = st.session_state.get("model_monitoring_input_df")
    source_options = ["Active working dataset"]
    if isinstance(monitoring_df, pd.DataFrame):
        source_options.append("Latest independent monitoring dataset")
    source_choice = st.radio("Data source", source_options, horizontal=True, key="retrain_source")
    use_active = source_choice == "Active working dataset"
    frame = df if use_active else monitoring_df
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        st.error("Selected retraining data is unavailable.")
        return

    if use_active:
        revision = current_dataset_revision()
        fingerprint = str(st.session_state.get("working_fingerprint", "") or dataframe_fingerprint(frame))
        source_name = st.session_state.get("file_name") or "Active dataset"
    else:
        revision = 1
        fingerprint = dataframe_fingerprint(frame)
        source_name = st.session_state.get("model_monitoring_input_name") or "Monitoring dataset"

    st.caption(f"{source_name} · {len(frame):,} rows × {frame.shape[1]:,} columns · source data will not be mutated")
    report = st.session_state.get("model_monitoring_report", {}) or {}
    if report and str(report.get("package_id", "")) != champion.package_id:
        st.warning("The latest monitoring report belongs to another package and will not be used as a trigger.")
        report = {}

    try:
        assessment = assess_retraining_need(
            champion,
            frame,
            monitoring_report=report,
            require_champion=True,
        )
    except Exception as exc:
        st.error(f"Retraining assessment failed safely: {exc}")
        return

    _render_assessment(assessment)
    _render_contract(assessment)

    manual_override = False
    if assessment.eligible and not assessment.recommended:
        manual_override = st.checkbox(
            "Manual override: I reviewed the evidence and explicitly want to retrain even though the automatic trigger is not active.",
            value=False,
            key=f"retrain_override_{champion.package_id}",
        )

    confirm = st.checkbox(
        "I understand that retraining creates only a Challenger and cannot replace the Champion automatically.",
        value=False,
        key=f"retrain_confirm_{champion.package_id}",
    )
    can_run = assessment.eligible and confirm and (assessment.recommended or manual_override)

    if st.button(
        "🚀 Run Safe Retraining & Register Challenger",
        type="primary",
        width="stretch",
        disabled=not can_run,
        key=f"retrain_run_{champion.package_id}_{fingerprint[:10]}",
    ):
        source_copy = frame.copy(deep=True)
        try:
            with st.spinner("Replaying the signed training contract, running training-only CV, evaluating one untouched holdout, then signing a Challenger..."):
                result, rebound_spec, fresh_assessment = run_safe_retraining_experiment(
                    champion,
                    source_copy,
                    dataset_revision=revision,
                    dataset_fingerprint=fingerprint,
                    monitoring_report=report,
                    manual_override=manual_override,
                )
                candidate = create_and_register_retraining_challenger(
                    champion,
                    source_copy,
                    result,
                    rebound_spec,
                    fresh_assessment,
                    semantic_profiles=_source_semantics(champion, use_active),
                    actor=st.session_state.get("current_user") or "local-user",
                    note=f"Safe retraining from Champion {champion.package_id}; trigger={fresh_assessment.monitoring_status}",
                )
            st.session_state.retraining_candidate_result = candidate
            st.session_state.retraining_last_report = {
                "source_champion_id": champion.package_id,
                "challenger_id": candidate.registry_metadata.get("package_id"),
                "experiment_id": result.experiment_id,
                "selected_model": result.selected_model,
                "holdout_metrics": result.holdout_metrics,
                "monitoring_status": fresh_assessment.monitoring_status,
                "recommended": fresh_assessment.recommended,
            }
            append_audit_event(
                {
                    "event": "safe_retraining_challenger_created",
                    "action": "Stage 16 safe retraining created a governed Challenger",
                    "source_champion_id": champion.package_id,
                    "challenger_id": candidate.registry_metadata.get("package_id"),
                    "experiment_id": result.experiment_id,
                    "selected_model": result.selected_model,
                    "monitoring_status": fresh_assessment.monitoring_status,
                }
            )
            st.success("Retraining completed. The new signed package is registered as Challenger; the Champion was not changed.")
            st.rerun()
        except RetrainingWorkflowError as exc:
            st.error(f"Retraining was blocked safely: {exc}")
        except Exception as exc:
            st.error(f"Retraining failed safely: {exc}")

    last = st.session_state.get("retraining_last_report", {}) or {}
    if last and last.get("source_champion_id") == champion.package_id:
        st.markdown("---")
        st.markdown("### Latest retraining candidate")
        st.success(
            f"Challenger {last.get('challenger_id')} · experiment {last.get('experiment_id')} · model {last.get('selected_model')}"
        )
        metrics = last.get("holdout_metrics", {}) or {}
        if metrics:
            safe_dataframe(
                pd.DataFrame([{"Metric": key, "Value": value} for key, value in metrics.items()]),
                width="stretch",
                hide_index=True,
            )
        st.info("Open Model Governance to assess and approve/reject the Challenger. Stage 16 cannot promote it.")
