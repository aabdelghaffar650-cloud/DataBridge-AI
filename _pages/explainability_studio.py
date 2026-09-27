# ════════════════════════════════════════════════════════
#  DataBridge AI — Page: Explainability Studio
#  Stage 10: global, local, optional SHAP, and error analysis
# ════════════════════════════════════════════════════════
from __future__ import annotations

import json
from typing import Any, Dict, Mapping

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from ui.streamlit_compat import safe_dataframe

from core.dataset import current_dataset_revision, update_dataset_context
from core.security import safe_html
from core.session import append_audit_event
from modules.feature_pipeline import feature_pipeline_spec_fingerprint
from modules.ml_engine import CLASSIFICATION, SupervisedExperimentResult
from modules.model_explainability import (
    GlobalExplainabilityResult,
    LocalExplanationResult,
    ShapExplanationResult,
    build_error_analysis,
    compute_segment_performance,
    reconcile_explainability_report,
    run_global_explainability,
    run_local_sensitivity,
    run_optional_shap,
)
from ui.cards import section_header


def _metric_text(value: Any, *, percent: bool = False) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(number):
        return "—"
    return f"{number * 100:.1f}%" if percent else f"{number:.4f}"


def _experiment_guard(
    df: pd.DataFrame,
) -> tuple[SupervisedExperimentResult | None, Dict[str, Any], Dict[str, Any], int, str]:
    artifact = st.session_state.get("ml_experiment_artifact")
    spec = st.session_state.get("feature_pipeline_spec", {}) or {}
    experiment_report = st.session_state.get("ml_experiment_report", {}) or {}
    revision = current_dataset_revision()
    fingerprint = str(st.session_state.get("working_fingerprint") or "")

    if not isinstance(artifact, SupervisedExperimentResult):
        return None, spec, experiment_report, revision, fingerprint
    valid = (
        artifact.dataset_revision == revision
        and artifact.dataset_fingerprint == fingerprint
        and artifact.pipeline_spec_fingerprint == feature_pipeline_spec_fingerprint(spec)
        and not bool(experiment_report.get("stale"))
    )
    return (artifact if valid else None), spec, experiment_report, revision, fingerprint


def _sync_explainability_status(
    result: SupervisedExperimentResult | None,
    spec: Mapping[str, Any],
    *,
    revision: int,
    fingerprint: str,
) -> Dict[str, Any]:
    artifact = st.session_state.get("model_explainability_artifact")
    artifact_valid = (
        isinstance(result, SupervisedExperimentResult)
        and isinstance(artifact, GlobalExplainabilityResult)
        and artifact.experiment_id == result.experiment_id
    )
    report = reconcile_explainability_report(
        st.session_state.get("explainability_report", {}) or {},
        current_revision=revision,
        current_fingerprint=fingerprint,
        current_experiment_id=result.experiment_id if result is not None else "",
        current_pipeline_spec_fingerprint=(
            feature_pipeline_spec_fingerprint(spec) if spec else ""
        ),
        artifact_available=artifact_valid,
    )
    if report != (st.session_state.get("explainability_report", {}) or {}):
        try:
            update_dataset_context(
                expected_revision=revision,
                explainability_report=report,
            )
        except Exception:
            st.session_state.explainability_report = report
    if report.get("stale"):
        st.session_state.model_explainability_artifact = None
        st.session_state.local_explanation_result = None
        st.session_state.shap_explanation_result = None
    return report


def _render_context_banner(result: SupervisedExperimentResult) -> None:
    task = safe_html(result.task.title())
    target = safe_html(result.target)
    model = safe_html(result.selected_model)
    st.markdown(
        f"""
        <div class="metrics-row" style="grid-template-columns:repeat(4,1fr);">
          <div class="metric-card"><div class="label">Model</div>
            <div class="value" style="font-size:1.05rem">{model}</div>
            <div class="sub">fitted winner</div></div>
          <div class="metric-card"><div class="label">Task</div>
            <div class="value" style="font-size:1.05rem">{task}</div>
            <div class="sub">supervised experiment</div></div>
          <div class="metric-card"><div class="label">Target</div>
            <div class="value" style="font-size:1.05rem">{target}</div>
            <div class="sub">prediction objective</div></div>
          <div class="metric-card"><div class="label">Holdout</div>
            <div class="value">{result.holdout_rows:,}</div>
            <div class="sub">untouched evaluation rows</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.warning(
        "Explainability describes how this fitted model behaves. Importance, sensitivity, and SHAP values do not establish causality."
    )


def _render_global_result(artifact: GlobalExplainabilityResult) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Holdout rows used", f"{artifact.rows_used:,}")
    c2.metric("Source features", f"{len(artifact.source_importance):,}")
    c3.metric("Permutation repeats", artifact.permutation_repeats)
    c4.metric("Native method", artifact.native_method)

    for warning in artifact.warnings:
        st.warning(warning)

    source = artifact.source_importance.copy()
    top_n = st.slider(
        "Source features to display",
        5,
        max(5, min(50, len(source))),
        min(20, max(5, len(source))),
        key="explain_global_top_n",
    ) if len(source) > 5 else len(source)
    chart_data = source.head(int(top_n)).sort_values("Absolute Importance", ascending=True)
    fig = px.bar(
        chart_data,
        x="Importance Mean",
        y="Feature",
        orientation="h",
        error_x="Importance Std",
        title=f"Global source-feature importance — {artifact.permutation_metric}",
        template="plotly_dark",
        hover_data=["Direction", "Normalized Importance"],
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,13,26,1)",
        height=max(360, 28 * max(5, len(chart_data))),
    )
    st.plotly_chart(fig, width="stretch")
    safe_dataframe(source.round(6), width="stretch", hide_index=True, height=330)

    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            "Download source importance CSV",
            data=source.to_csv(index=False).encode("utf-8-sig"),
            file_name="databridge_source_feature_importance.csv",
            mime="text/csv",
            width="stretch",
            key="explain_source_csv",
        )
    with d2:
        payload = {
            "report": artifact.report(),
            "source_importance": source.to_dict(orient="records"),
            "native_importance": artifact.native_importance.head(500).to_dict(orient="records"),
        }
        st.download_button(
            "Download explanation JSON",
            data=json.dumps(payload, indent=2, ensure_ascii=False, default=str).encode("utf-8"),
            file_name="databridge_model_explanation.json",
            mime="application/json",
            width="stretch",
            key="explain_json",
        )

    if not artifact.native_importance.empty:
        with st.expander("Transformed-feature native importance", expanded=False):
            st.caption(
                "This view reflects model coefficients or tree importance after encoding/scaling. Use source-level permutation importance as the primary model-agnostic view."
            )
            native = artifact.native_importance.head(100).copy()
            safe_dataframe(native.round(6), width="stretch", hide_index=True, height=380)
            grouped = (
                artifact.native_importance.groupby("Source Feature", as_index=False)["Importance"]
                .sum()
                .sort_values("Importance", ascending=False)
            )
            st.markdown("**Native importance aggregated back to source columns**")
            safe_dataframe(grouped.round(6), width="stretch", hide_index=True)


def _global_tab(
    df: pd.DataFrame,
    result: SupervisedExperimentResult,
    spec: Mapping[str, Any],
    *,
    revision: int,
    fingerprint: str,
) -> None:
    st.markdown("#### Model-agnostic global importance")
    st.caption(
        "Each original source column is shuffled on untouched holdout rows while the fitted preprocessing and model remain frozen."
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        max_rows = st.number_input(
            "Maximum holdout rows",
            min_value=100,
            max_value=5000,
            value=min(2000, max(100, result.holdout_rows)),
            step=100,
            key="explain_max_rows",
        )
    with c2:
        repeats = st.slider(
            "Permutation repeats",
            2,
            15,
            5,
            key="explain_repeats",
        )
    with c3:
        random_state = st.number_input(
            "Random seed",
            min_value=0,
            max_value=2_147_483_647,
            value=int(result.random_state),
            key="explain_seed",
        )

    if st.button(
        "Generate Global Explanation",
        type="primary",
        width="stretch",
        key="explain_global_run",
    ):
        try:
            with st.spinner("Running holdout-only permutation importance without refitting the model..."):
                artifact = run_global_explainability(
                    df,
                    result,
                    spec,
                    current_revision=revision,
                    current_fingerprint=fingerprint,
                    max_rows=int(max_rows),
                    n_repeats=int(repeats),
                    random_state=int(random_state),
                )
            st.session_state.model_explainability_artifact = artifact
            st.session_state.local_explanation_result = None
            st.session_state.shap_explanation_result = None
            update_dataset_context(
                expected_revision=revision,
                explainability_report=artifact.report(),
            )
            append_audit_event(
                {
                    "event": "model_explainability",
                    "action": "Global model explanation generated",
                    "revision": revision,
                    "experiment_id": result.experiment_id,
                    "explanation_id": artifact.explanation_id,
                    "rows_used": artifact.rows_used,
                    "feature_count": len(artifact.source_importance),
                }
            )
            st.success("Global explanation completed on holdout rows only.")
            st.rerun()
        except Exception as exc:
            st.error(f"Explainability was blocked safely: {exc}")

    artifact = st.session_state.get("model_explainability_artifact")
    if isinstance(artifact, GlobalExplainabilityResult) and artifact.experiment_id == result.experiment_id:
        st.markdown("---")
        _render_global_result(artifact)


def _local_tab(
    df: pd.DataFrame,
    result: SupervisedExperimentResult,
    spec: Mapping[str, Any],
    *,
    revision: int,
    fingerprint: str,
) -> None:
    st.markdown("#### Explain one untouched holdout prediction")
    predictions = result.predictions.reset_index(names="Source Index")
    if predictions.empty:
        st.info("No holdout predictions are available.")
        return

    positions = list(range(len(predictions)))

    def _row_label(position: int) -> str:
        row = predictions.iloc[position]
        return (
            f"Row {row['Source Index']} · Actual: {row['Actual']} · Predicted: {row['Predicted']}"
        )

    selected_position = st.selectbox(
        "Holdout row",
        positions,
        format_func=_row_label,
        key="explain_local_position",
    )
    selected = predictions.iloc[int(selected_position)]
    source_index = selected["Source Index"]

    p1, p2, p3 = st.columns(3)
    p1.metric("Actual", str(selected["Actual"]))
    p2.metric("Predicted", str(selected["Predicted"]))
    if "Confidence" in selected.index:
        p3.metric("Confidence", _metric_text(selected["Confidence"], percent=True))
    elif "Absolute Error" in selected.index:
        p3.metric("Absolute Error", _metric_text(selected["Absolute Error"]))
    else:
        p3.metric("Correct", str(selected.get("Correct", "—")))

    c1, c2 = st.columns(2)
    with c1:
        background_rows = st.number_input(
            "Reference background rows",
            min_value=20,
            max_value=1000,
            value=min(500, max(20, result.holdout_rows)),
            step=20,
            key="explain_local_background",
        )
    with c2:
        local_seed = st.number_input(
            "Local reference seed",
            min_value=0,
            max_value=2_147_483_647,
            value=int(result.random_state),
            key="explain_local_seed",
        )

    if st.button(
        "Calculate Local Sensitivity",
        type="primary",
        width="stretch",
        key="explain_local_run",
    ):
        try:
            local = run_local_sensitivity(
                df,
                result,
                spec,
                source_index=source_index,
                current_revision=revision,
                current_fingerprint=fingerprint,
                background_rows=int(background_rows),
                random_state=int(local_seed),
            )
            st.session_state.local_explanation_result = local
            st.session_state.shap_explanation_result = None
            st.rerun()
        except Exception as exc:
            st.error(f"Local explanation was blocked safely: {exc}")

    local = st.session_state.get("local_explanation_result")
    if isinstance(local, LocalExplanationResult) and local.experiment_id == result.experiment_id and local.source_index == source_index:
        st.markdown("---")
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted output", local.predicted_label)
        m2.metric(local.score_name, _metric_text(local.score_value))
        m3.metric("Actual", str(local.actual_value))
        for warning in local.warnings:
            st.warning(warning)

        top_n = min(25, len(local.sensitivity))
        chart = local.sensitivity.head(top_n).sort_values("Sensitivity", ascending=True)
        fig = px.bar(
            chart,
            x="Sensitivity",
            y="Feature",
            orientation="h",
            color="Effect",
            title="One-at-a-time local sensitivity",
            template="plotly_dark",
            hover_data=["Observed Value", "Reference Value", "Score After Replacement"],
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(13,13,26,1)",
            height=max(360, 27 * max(5, len(chart))),
        )
        st.plotly_chart(fig, width="stretch")
        safe_dataframe(local.sensitivity.round(6), width="stretch", hide_index=True, height=360)
        st.download_button(
            "Download local sensitivity CSV",
            data=local.sensitivity.to_csv(index=False).encode("utf-8-sig"),
            file_name="databridge_local_sensitivity.csv",
            mime="text/csv",
            width="stretch",
            key="explain_local_csv",
        )

        with st.expander("Optional SHAP explanation", expanded=False):
            st.caption(
                "SHAP is optional and not required by DataBridge AI. It runs only if the shap package is installed and the transformed matrix is within safety limits."
            )
            if st.button("Try Optional SHAP", width="stretch", key="explain_shap_run"):
                try:
                    with st.spinner("Attempting optional SHAP on the frozen transformed model..."):
                        shap_result = run_optional_shap(
                            df,
                            result,
                            spec,
                            source_index=source_index,
                            current_revision=revision,
                            current_fingerprint=fingerprint,
                            background_rows=50,
                            random_state=int(local_seed),
                        )
                    st.session_state.shap_explanation_result = shap_result
                    st.rerun()
                except Exception as exc:
                    st.error(f"Optional SHAP was blocked safely: {exc}")

            shap_result = st.session_state.get("shap_explanation_result")
            if isinstance(shap_result, ShapExplanationResult):
                if shap_result.warning:
                    st.warning(shap_result.warning)
                if shap_result.available and not shap_result.feature_values.empty:
                    st.caption(
                        f"Method: {shap_result.method} · Base: {_metric_text(shap_result.base_value)} · Reconstructed output: {_metric_text(shap_result.output_value)}"
                    )
                    safe_dataframe(
                        shap_result.feature_values.head(100).round(6),
                        width="stretch",
                        hide_index=True,
                        height=380,
                    )


def _error_tab(df: pd.DataFrame, result: SupervisedExperimentResult) -> None:
    st.markdown("#### Holdout error analysis")
    errors = build_error_analysis(result, limit=500)
    if errors.empty:
        st.success("No holdout classification errors were found.")
    else:
        safe_dataframe(errors, width="stretch", hide_index=True, height=360)
        st.download_button(
            "Download holdout errors CSV",
            data=errors.to_csv(index=False).encode("utf-8-sig"),
            file_name="databridge_holdout_errors.csv",
            mime="text/csv",
            width="stretch",
            key="explain_errors_csv",
        )

    st.markdown("---")
    st.markdown("#### Segment performance")
    candidates: list[str] = []
    holdout_count = max(1, result.holdout_rows)
    for column in map(str, df.columns):
        if column == result.target:
            continue
        unique = int(df[column].nunique(dropna=True))
        if 2 <= unique <= min(50, max(2, holdout_count // 2)):
            candidates.append(column)
    if not candidates:
        st.info("No low-cardinality segment column is available.")
        return
    segment = st.selectbox("Segment column", candidates, key="explain_segment_column")
    min_rows = st.slider("Minimum holdout rows per segment", 2, 20, 3, key="explain_segment_min")
    try:
        performance = compute_segment_performance(
            df,
            result,
            segment_column=segment,
            min_group_rows=min_rows,
        )
        if performance.empty:
            st.info("No segment has enough holdout rows under the selected threshold.")
        else:
            display = performance.copy()
            numeric = display.select_dtypes(include="number").columns
            display[numeric] = display[numeric].round(4)
            safe_dataframe(display, width="stretch", hide_index=True)
            y_col = "Error Rate" if result.task == CLASSIFICATION else "RMSE"
            fig = px.bar(
                performance,
                x="Segment",
                y=y_col,
                title=f"Holdout {y_col} by {segment}",
                template="plotly_dark",
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(13,13,26,1)",
                height=360,
            )
            st.plotly_chart(fig, width="stretch")
    except Exception as exc:
        st.error(f"Segment analysis was blocked safely: {exc}")


def render(df: pd.DataFrame) -> None:
    st.markdown(
        section_header("🔬", "Explainability Studio", "Holdout-only model behavior analysis"),
        unsafe_allow_html=True,
    )
    result, spec, experiment_report, revision, fingerprint = _experiment_guard(df)
    _sync_explainability_status(
        result,
        spec,
        revision=revision,
        fingerprint=fingerprint,
    )

    if result is None:
        st.error(
            "No current supervised ML Studio V2 artifact is available. Train a fresh classification or regression experiment after saving the Feature Engineering Pipeline."
        )
        if experiment_report:
            with st.expander("Experiment status", expanded=False):
                st.json(experiment_report, expanded=False)
        return

    _render_context_banner(result)
    tab_global, tab_local, tab_errors = st.tabs(
        ["🌍 Global Importance", "🔎 Local Prediction", "🧪 Error & Segment Analysis"]
    )
    with tab_global:
        _global_tab(
            df,
            result,
            spec,
            revision=revision,
            fingerprint=fingerprint,
        )
    with tab_local:
        _local_tab(
            df,
            result,
            spec,
            revision=revision,
            fingerprint=fingerprint,
        )
    with tab_errors:
        _error_tab(df, result)
