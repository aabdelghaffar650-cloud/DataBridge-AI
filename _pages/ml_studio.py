# ════════════════════════════════════════════════════════
#  DataBridge AI — Page: ML Studio V2
#  Stage 10: leakage-safe ML plus package export and explanation invalidation
# ════════════════════════════════════════════════════════
from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from ui.streamlit_compat import safe_dataframe

from core.dataset import (
    apply_dataset_change,
    current_dataset_revision,
    update_dataset_context,
)
from core.session import append_audit_event
from core.security import safe_html
from modules.feature_pipeline import feature_pipeline_spec_fingerprint
from modules.ml_engine import (
    CLASSIFICATION,
    REGRESSION,
    SPLIT_GROUP,
    SPLIT_RANDOM,
    SPLIT_STRATIFIED,
    SPLIT_TIME,
    ClusteringExperimentResult,
    SupervisedExperimentResult,
    available_model_names,
    reconcile_ml_experiment_report,
    run_clustering_experiment,
    run_supervised_experiment,
    xgboost_available,
)
from modules.model_package import (
    ModelPackageError,
    create_signed_model_package,
    load_signed_model_package,
)
from ui.cards import section_header


_SPLIT_LABELS = {
    "Stratified Random Holdout": SPLIT_STRATIFIED,
    "Random Holdout": SPLIT_RANDOM,
    "Time-Ordered Holdout": SPLIT_TIME,
    "Group Holdout": SPLIT_GROUP,
}


def _safe_metric(value: Any, *, percent: bool = False, digits: int = 3) -> str:
    if value is None:
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(number):
        return "—"
    if percent:
        return f"{number * 100:.1f}%"
    return f"{number:.{digits}f}"


def _pipeline_banner(report: Mapping[str, Any]) -> None:
    status = str(report.get("status", "Pending"))
    color = "#6bff8e" if status == "Configured" else "#ffb86b" if status == "Stale" else "#ff6b6b"
    target = safe_html(str(report.get("spec", {}).get("target", "") or "—"))
    task = safe_html(str(report.get("spec", {}).get("task", "") or "—"))
    features = int(report.get("input_feature_count", 0) or 0)
    projected = int(report.get("estimated_features", {}).get("total", 0) or 0)
    st.markdown(
        f"""
        <div class="metrics-row" style="grid-template-columns:repeat(4,1fr);">
          <div class="metric-card"><div class="label">Pipeline Status</div>
            <div class="value" style="font-size:1.25rem;color:{color}">{status}</div>
            <div class="sub">must be Configured</div></div>
          <div class="metric-card"><div class="label">Target</div>
            <div class="value" style="font-size:1.1rem">{target}</div>
            <div class="sub">prediction objective</div></div>
          <div class="metric-card"><div class="label">Task</div>
            <div class="value" style="font-size:1.1rem">{task.title()}</div>
            <div class="sub">configured problem type</div></div>
          <div class="metric-card"><div class="label">Features</div>
            <div class="value">{features:,}</div>
            <div class="sub">~{projected:,} transformed</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _time_candidates(df: pd.DataFrame, profiles: Mapping[str, Mapping[str, Any]], target: str) -> list[str]:
    candidates: list[str] = []
    for column in map(str, df.columns):
        if column == target:
            continue
        profile = profiles.get(column, {}) or {}
        semantic = str(
            profile.get("effective_semantic_type")
            or profile.get("semantic_type")
            or ""
        )
        if semantic == "Datetime" or pd.api.types.is_datetime64_any_dtype(df[column]):
            candidates.append(column)
    return candidates


def _group_candidates(df: pd.DataFrame, target: str) -> list[str]:
    result: list[str] = []
    for column in map(str, df.columns):
        if column == target:
            continue
        unique = int(df[column].nunique(dropna=True))
        if 2 <= unique < max(3, len(df)):
            result.append(column)
    return result


def _sync_experiment_status(
    *,
    revision: int,
    fingerprint: str,
    spec: Mapping[str, Any],
) -> Dict[str, Any]:
    artifact = st.session_state.get("ml_experiment_artifact")
    artifact_valid = isinstance(artifact, SupervisedExperimentResult)
    report = reconcile_ml_experiment_report(
        st.session_state.get("ml_experiment_report", {}) or {},
        current_revision=revision,
        current_fingerprint=fingerprint,
        current_pipeline_spec_fingerprint=(
            feature_pipeline_spec_fingerprint(spec) if spec else ""
        ),
        artifact_available=artifact_valid,
    )
    if report != (st.session_state.get("ml_experiment_report", {}) or {}):
        try:
            update_dataset_context(
                expected_revision=revision,
                ml_experiment_report=report,
            )
        except Exception:
            pass
    if report.get("stale"):
        st.session_state.ml_experiment_artifact = None
        st.session_state.model_explainability_artifact = None
        st.session_state.local_explanation_result = None
        st.session_state.shap_explanation_result = None
    return report


def _render_experiment_summary(result: SupervisedExperimentResult) -> None:
    improvement = result.improvement_vs_baseline
    improvement_text = "—"
    if improvement is not None:
        improvement_text = (
            f"{improvement * 100:+.1f} pts"
            if result.task == CLASSIFICATION
            else f"{improvement:+.3f} RMSE"
        )
    st.markdown(
        f"""
        <div class="metrics-row" style="grid-template-columns:repeat(5,1fr);">
          <div class="metric-card"><div class="label">Selected Model</div>
            <div class="value" style="font-size:1rem">{result.selected_model}</div>
            <div class="sub">chosen by CV only</div></div>
          <div class="metric-card"><div class="label">Training Rows</div>
            <div class="value">{result.train_rows:,}</div>
            <div class="sub">preprocessing fitted here</div></div>
          <div class="metric-card"><div class="label">Holdout Rows</div>
            <div class="value">{result.holdout_rows:,}</div>
            <div class="sub">untouched until final test</div></div>
          <div class="metric-card"><div class="label">CV Folds</div>
            <div class="value">{result.actual_cv_folds}</div>
            <div class="sub">split-aware validation</div></div>
          <div class="metric-card"><div class="label">Vs Baseline</div>
            <div class="value" style="font-size:1.05rem">{improvement_text}</div>
            <div class="sub">positive is better</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if result.sampled:
        st.warning(
            f"Safety sampling was active: {result.modelling_rows:,} rows were used from "
            f"{result.source_rows:,} source rows."
        )
    for warning in result.warnings:
        st.warning(warning)
    if result.task == CLASSIFICATION:
        train_distribution = result.split_summary.get("training_class_distribution", {}) or {}
        holdout_distribution = result.split_summary.get("holdout_class_distribution", {}) or {}
        if train_distribution:
            with st.expander("Class distribution by split", expanded=False):
                class_rows = []
                for label in result.classes:
                    class_rows.append(
                        {
                            "Class": label,
                            "Training": int(train_distribution.get(label, 0)),
                            "Holdout": int(holdout_distribution.get(label, 0)),
                        }
                    )
                safe_dataframe(pd.DataFrame(class_rows), width="stretch", hide_index=True)
                ratio = result.split_summary.get("training_imbalance_ratio")
                if ratio is not None:
                    st.caption(f"Training imbalance ratio: {float(ratio):.2f}:1")
    if result.failures:
        with st.expander("Model failures and tuning fallbacks", expanded=False):
            for name, error in result.failures.items():
                st.error(f"{name}: {error}")


def _leaderboard_view(result: SupervisedExperimentResult) -> pd.DataFrame:
    board = result.leaderboard.copy()
    hidden = ["Selection Score", "Error"]
    board = board.drop(columns=[column for column in hidden if column in board.columns])
    numeric_columns = board.select_dtypes(include="number").columns
    board[numeric_columns] = board[numeric_columns].round(4)
    return board


def _render_classification_results(result: SupervisedExperimentResult) -> None:
    metrics = result.holdout_metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Holdout F1 Weighted", _safe_metric(metrics.get("F1 Weighted"), percent=True))
    c2.metric("Holdout F1 Macro", _safe_metric(metrics.get("F1 Macro"), percent=True))
    c3.metric("Balanced Accuracy", _safe_metric(metrics.get("Balanced Accuracy"), percent=True))
    c4.metric("ROC AUC", _safe_metric(metrics.get("ROC AUC")))
    c5.metric("PR AUC", _safe_metric(metrics.get("PR AUC")))

    left, right = st.columns([1.05, 0.95])
    with left:
        if result.confusion is not None:
            fig = px.imshow(
                result.confusion,
                text_auto=True,
                x=result.classes,
                y=result.classes,
                labels={"x": "Predicted", "y": "Actual", "color": "Count"},
                title="Untouched Holdout Confusion Matrix",
                template="plotly_dark",
                color_continuous_scale="Blues",
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(13,13,26,1)",
                height=430,
            )
            st.plotly_chart(fig, width="stretch")
    with right:
        st.markdown("**Holdout classification report**")
        if result.classification_report_df is not None:
            safe_dataframe(
                result.classification_report_df.round(4),
                width="stretch",
                height=390,
            )


def _render_regression_results(result: SupervisedExperimentResult) -> None:
    metrics = result.holdout_metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Holdout RMSE", _safe_metric(metrics.get("RMSE")))
    c2.metric("Holdout MAE", _safe_metric(metrics.get("MAE")))
    c3.metric("Holdout R²", _safe_metric(metrics.get("R²")))
    c4.metric("Median Abs Error", _safe_metric(metrics.get("Median Absolute Error")))
    c5.metric("MAPE", _safe_metric(metrics.get("MAPE"), percent=True))

    prediction_df = result.predictions.reset_index(names="Source Index")
    left, right = st.columns(2)
    with left:
        fig = px.scatter(
            prediction_df,
            x="Actual",
            y="Predicted",
            title="Untouched Holdout: Actual vs Predicted",
            template="plotly_dark",
            opacity=0.7,
        )
        minimum = float(min(prediction_df["Actual"].min(), prediction_df["Predicted"].min()))
        maximum = float(max(prediction_df["Actual"].max(), prediction_df["Predicted"].max()))
        fig.add_shape(
            type="line",
            x0=minimum,
            y0=minimum,
            x1=maximum,
            y1=maximum,
            line={"dash": "dash"},
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(13,13,26,1)",
            height=390,
        )
        st.plotly_chart(fig, width="stretch")
    with right:
        fig = px.histogram(
            prediction_df,
            x="Residual",
            nbins=30,
            title="Holdout Residual Distribution",
            template="plotly_dark",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(13,13,26,1)",
            height=390,
        )
        st.plotly_chart(fig, width="stretch")


def _render_saved_metadata(report: Mapping[str, Any]) -> None:
    if not report or report.get("status") == "Pending":
        return
    with st.expander("Last experiment metadata", expanded=False):
        st.json(dict(report), expanded=False)
        st.download_button(
            "Download experiment metadata JSON",
            data=json.dumps(report, indent=2, ensure_ascii=False, default=str).encode("utf-8"),
            file_name="databridge_ml_experiment_metadata.json",
            mime="application/json",
            width="stretch",
            key="ml_metadata_download",
        )


def _render_supervised_lab(
    df: pd.DataFrame,
    *,
    revision: int,
    fingerprint: str,
    spec: Mapping[str, Any],
    pipeline_report: Mapping[str, Any],
    profiles: Mapping[str, Mapping[str, Any]],
    experiment_report: Mapping[str, Any],
) -> None:
    _pipeline_banner(pipeline_report)
    for blocker in pipeline_report.get("blockers", []) or []:
        st.error(blocker)
    for warning in pipeline_report.get("warnings", []) or []:
        st.warning(warning)

    pipeline_ready = (
        bool(spec)
        and bool(pipeline_report.get("valid"))
        and not bool(pipeline_report.get("stale"))
        and str(pipeline_report.get("status")) == "Configured"
    )
    if not pipeline_ready:
        st.info(
            "Configure and save a valid Feature Engineering Pipeline first. ML Studio V2 will not bypass or refit preprocessing outside that contract."
        )
        _render_saved_metadata(experiment_report)
        return

    task = str(spec.get("task"))
    target = str(spec.get("target"))
    st.markdown("#### 1. Honest evaluation design")
    st.caption(
        "The holdout is created before model comparison. Models are ranked only by cross-validation on training rows. "
        "The selected model sees the holdout once for final evaluation."
    )

    split_options = list(_SPLIT_LABELS.keys())
    if task == REGRESSION:
        split_options.remove("Stratified Random Holdout")
    default_label = "Stratified Random Holdout" if task == CLASSIFICATION else "Random Holdout"
    split_label = st.selectbox(
        "Holdout strategy",
        split_options,
        index=split_options.index(default_label),
        key=f"ml_split_{revision}_{target}",
    )
    split_strategy = _SPLIT_LABELS[split_label]
    split_column = ""
    if split_strategy == SPLIT_TIME:
        candidates = _time_candidates(df, profiles, target)
        if not candidates:
            st.error("No semantic Datetime column is available for a time-ordered split.")
        else:
            split_column = st.selectbox(
                "Time order column",
                candidates,
                key=f"ml_time_col_{revision}_{target}",
                help="Training uses earlier rows and holdout uses the latest period.",
            )
    elif split_strategy == SPLIT_GROUP:
        candidates = _group_candidates(df, target)
        if not candidates:
            st.error("No suitable group column is available.")
        else:
            split_column = st.selectbox(
                "Group column",
                candidates,
                key=f"ml_group_col_{revision}_{target}",
                help="Entire groups are kept in either training or holdout, never both.",
            )

    s1, s2, s3, s4 = st.columns(4)
    with s1:
        holdout_percent = st.slider(
            "Holdout %",
            10,
            40,
            20,
            key=f"ml_holdout_{revision}_{target}",
        )
    with s2:
        cv_folds = st.slider(
            "CV folds",
            2,
            10,
            5,
            key=f"ml_cv_{revision}_{target}",
        )
    with s3:
        random_state = st.number_input(
            "Random seed",
            min_value=0,
            max_value=2_147_483_647,
            value=42,
            step=1,
            key=f"ml_seed_{revision}_{target}",
        )
    with s4:
        max_rows = st.number_input(
            "Safety row cap",
            min_value=1_000,
            max_value=250_000,
            value=min(100_000, max(1_000, len(df))),
            step=1_000,
            key=f"ml_rowcap_{revision}_{target}",
            help="A deterministic sample is used only when target-valid rows exceed this explicit cap.",
        )

    st.markdown("#### 2. Models and imbalance handling")
    class_weight_mode = "none"
    if task == CLASSIFICATION:
        class_weight_mode = st.radio(
            "Class weighting",
            ["none", "balanced"],
            horizontal=True,
            key=f"ml_class_weight_{revision}_{target}",
            help="Balanced weights are fitted inside each training fold. No resampling touches the holdout.",
        )

    all_models = available_model_names(
        task,
        random_state=int(random_state),
        class_weight="balanced" if class_weight_mode == "balanced" else None,
        include_xgboost=True,
    )
    non_baseline = [name for name in all_models if not name.startswith("Baseline")]
    if task == CLASSIFICATION:
        default_models = [name for name in ["Logistic Regression", "Random Forest", "Linear SVM", "XGBoost"] if name in non_baseline]
    else:
        default_models = [name for name in ["Ridge Regression", "Random Forest", "Extra Trees", "XGBoost"] if name in non_baseline]
    selected_models = st.multiselect(
        "Candidate models",
        non_baseline,
        default=default_models,
        key=f"ml_models_{revision}_{target}",
        help="A task-appropriate dummy baseline is always included automatically.",
    )
    if xgboost_available():
        st.caption("XGBoost CPU is available and participates in the same training-only CV / untouched-holdout workflow as the other candidate models.")
    else:
        st.warning("XGBoost is unavailable in this runtime. Rebuild the production runtime before relying on XGBoost experiments.")

    tune_best = st.checkbox(
        "Tune the CV winner before final holdout evaluation",
        value=False,
        key=f"ml_tune_{revision}_{target}",
        help="Randomized search runs only on training folds. The holdout remains untouched.",
    )
    tuning_iterations = 8
    if tune_best:
        tuning_iterations = st.slider(
            "Tuning iterations",
            2,
            30,
            8,
            key=f"ml_tune_iter_{revision}_{target}",
        )

    split_control_valid = not (
        split_strategy in {SPLIT_TIME, SPLIT_GROUP} and not split_column
    )
    train_disabled = not selected_models or not split_control_valid
    if st.button(
        "🚀 Run Leakage-Safe Experiment",
        type="primary",
        width="stretch",
        disabled=train_disabled,
        key=f"ml_run_{revision}_{target}",
    ):
        try:
            with st.spinner(
                "Splitting first, comparing models by training-only CV, then evaluating the winner once on holdout..."
            ):
                result = run_supervised_experiment(
                    df,
                    spec,
                    dataset_revision=revision,
                    dataset_fingerprint=fingerprint,
                    split_strategy=split_strategy,
                    split_column=split_column,
                    holdout_size=holdout_percent / 100.0,
                    cv_folds=cv_folds,
                    random_state=int(random_state),
                    model_names=selected_models,
                    class_weight_mode=class_weight_mode,
                    tune_best=tune_best,
                    tuning_iterations=tuning_iterations,
                    max_rows=int(max_rows),
                    include_xgboost=True,
                )
            st.session_state.ml_experiment_artifact = result
            st.session_state.model_explainability_artifact = None
            st.session_state.local_explanation_result = None
            st.session_state.shap_explanation_result = None
            update_dataset_context(
                expected_revision=revision,
                ml_experiment_report=result.report(),
                explainability_report={},
            )
            append_audit_event(
                {
                    "event": "ml_experiment",
                    "action": "ML Studio V2 experiment completed",
                    "revision": revision,
                    "experiment_id": result.experiment_id,
                    "task": result.task,
                    "target": result.target,
                    "selected_model": result.selected_model,
                    "split_strategy": result.split_strategy,
                    "train_rows": result.train_rows,
                    "holdout_rows": result.holdout_rows,
                }
            )
            st.success("Experiment completed without using the holdout for model selection.")
            st.rerun()
        except Exception as exc:
            st.error(f"Experiment was blocked safely: {exc}")

    result = st.session_state.get("ml_experiment_artifact")
    if isinstance(result, SupervisedExperimentResult):
        current_pipeline_fp = feature_pipeline_spec_fingerprint(spec)
        valid_artifact = (
            result.dataset_fingerprint == fingerprint
            and result.pipeline_spec_fingerprint == current_pipeline_fp
        )
        if valid_artifact:
            st.markdown("---")
            st.markdown("### Experiment Results")
            _render_experiment_summary(result)
            st.markdown("#### Cross-validation leaderboard — training rows only")
            safe_dataframe(
                _leaderboard_view(result),
                width="stretch",
                hide_index=True,
                height=300,
            )
            st.caption(
                f"Winner selection metric: {result.primary_metric} ({result.primary_direction} is better). "
                "No holdout metric participated in this ranking."
            )
            st.markdown("#### Final untouched holdout evaluation")
            if result.task == CLASSIFICATION:
                _render_classification_results(result)
            else:
                _render_regression_results(result)

            with st.expander("Holdout predictions", expanded=False):
                prediction_view = result.predictions.reset_index(names="Source Index")
                safe_dataframe(prediction_view, width="stretch", height=320)
                st.download_button(
                    "Download holdout predictions CSV",
                    data=prediction_view.to_csv(index=False).encode("utf-8-sig"),
                    file_name="databridge_holdout_predictions.csv",
                    mime="text/csv",
                    width="stretch",
                    key=f"ml_pred_download_{result.experiment_id}",
                )
            with st.expander("Generated feature contract", expanded=False):
                st.caption(
                    "These are the fitted output features from the training-only preprocessing pipeline."
                )
                safe_dataframe(
                    pd.DataFrame(
                        {
                            "Index": range(len(result.feature_names)),
                            "Feature": result.feature_names,
                        }
                    ),
                    width="stretch",
                    height=300,
                )
            st.markdown("#### Signed Model Package")
            st.caption(
                "The package contains the fitted preprocessing Pipeline, model, target decoder, input schema, training-only aggregate monitoring reference, evaluation metadata, SHA-256 integrity data, and a per-installation HMAC signature. Raw datasets, raw categorical monitoring labels, and row-level holdout predictions are not included."
            )
            package_manifest = st.session_state.get("model_package_manifest", {}) or {}
            package_matches = (
                str(package_manifest.get("experiment", {}).get("experiment_id", ""))
                == result.experiment_id
            )
            if st.button(
                "🔐 Build & Verify Signed Model Package",
                type="primary",
                width="stretch",
                key=f"ml_build_package_{result.experiment_id}",
            ):
                try:
                    with st.spinner("Serializing, signing, and verifying the model package..."):
                        build = create_signed_model_package(
                            result,
                            df,
                            spec,
                            semantic_profiles=profiles,
                            feature_derivation_recipe=st.session_state.get(
                                "feature_derivation_recipe", []
                            ),
                        )
                        verified = load_signed_model_package(build.package_bytes)
                    st.session_state.model_package_bytes = build.package_bytes
                    st.session_state.model_package_manifest = build.manifest
                    st.session_state.model_package_file_name = build.file_name
                    st.session_state.loaded_model_package = verified
                    st.session_state.loaded_model_package_bytes = build.package_bytes
                    st.session_state.prediction_result = None
                    st.session_state.model_monitoring_report = {}
                    st.session_state.model_monitoring_feature_table = None
                    append_audit_event(
                        {
                            "event": "model_package_created",
                            "action": "Signed model package created and verified",
                            "revision": revision,
                            "experiment_id": result.experiment_id,
                            "package_id": verified.package_id,
                            "task": verified.task,
                            "target": verified.target,
                        }
                    )
                    st.success("Signed package created and verified before download.")
                    st.rerun()
                except ModelPackageError as exc:
                    st.error(f"Package creation was blocked safely: {exc}")
                except Exception as exc:
                    st.error(f"Package creation failed safely: {exc}")

            package_bytes = st.session_state.get("model_package_bytes")
            package_manifest = st.session_state.get("model_package_manifest", {}) or {}
            package_matches = (
                isinstance(package_bytes, (bytes, bytearray))
                and str(package_manifest.get("experiment", {}).get("experiment_id", ""))
                == result.experiment_id
            )
            if package_matches:
                pkg_id = safe_html(str(package_manifest.get("package_id", "")))
                signer = safe_html(str(package_manifest.get("signer_key_id", "")))
                st.success(f"✅ Package ready · ID {pkg_id} · signer {signer}")
                st.download_button(
                    "📦 Download Signed Model Package (.dbmlpkg)",
                    data=bytes(package_bytes),
                    file_name=(
                        st.session_state.get("model_package_file_name")
                        or f"databridge_{result.experiment_id}.dbmlpkg"
                    ),
                    mime="application/zip",
                    width="stretch",
                    key=f"ml_package_download_{result.experiment_id}",
                )
                with st.expander("Signed package manifest", expanded=False):
                    st.json(package_manifest, expanded=False)
                    st.info(
                        "Use Prediction Studio to verify this package and score new files. Raw .pkl/joblib imports remain disabled."
                    )

    _render_saved_metadata(experiment_report)


def _render_clustering_lab(df: pd.DataFrame, revision: int) -> None:
    st.markdown(
        '<div class="info-box">Clustering V2 uses median imputation and Standard Scaling inside a fitted pipeline, '
        'then reports cluster quality. It never overwrites source columns.</div>',
        unsafe_allow_html=True,
    )
    numeric = list(map(str, df.select_dtypes(include="number").columns))
    if len(numeric) < 2:
        st.warning("Clustering requires at least two numeric columns.")
        return

    features = st.multiselect(
        "Numeric clustering features",
        numeric,
        default=numeric[: min(4, len(numeric))],
        key=f"clv2_features_{revision}",
    )
    algorithm_label = st.radio(
        "Algorithm",
        ["K-Means", "DBSCAN"],
        horizontal=True,
        key=f"clv2_algorithm_{revision}",
    )
    algorithm = "kmeans" if algorithm_label == "K-Means" else "dbscan"
    n_clusters = 3
    eps = 0.5
    min_samples = 5
    if algorithm == "kmeans":
        n_clusters = st.slider(
            "Number of clusters",
            2,
            20,
            3,
            key=f"clv2_k_{revision}",
        )
    else:
        c1, c2 = st.columns(2)
        with c1:
            eps = st.slider(
                "DBSCAN eps",
                0.05,
                5.0,
                0.5,
                0.05,
                key=f"clv2_eps_{revision}",
            )
        with c2:
            min_samples = st.slider(
                "DBSCAN min_samples",
                2,
                50,
                5,
                key=f"clv2_min_{revision}",
            )
    c1, c2 = st.columns(2)
    with c1:
        seed = st.number_input(
            "Random seed",
            min_value=0,
            max_value=2_147_483_647,
            value=42,
            step=1,
            key=f"clv2_seed_{revision}",
        )
    with c2:
        row_cap = st.number_input(
            "Clustering row cap",
            min_value=1_000,
            max_value=100_000,
            value=min(50_000, max(1_000, len(df))),
            step=1_000,
            key=f"clv2_cap_{revision}",
        )

    if st.button(
        "Run Clustering V2",
        type="primary",
        width="stretch",
        disabled=len(features) < 2,
        key=f"clv2_run_{revision}",
    ):
        try:
            with st.spinner("Fitting scaled clustering pipeline..."):
                result = run_clustering_experiment(
                    df,
                    features,
                    algorithm=algorithm,
                    n_clusters=n_clusters,
                    eps=eps,
                    min_samples=min_samples,
                    random_state=int(seed),
                    max_rows=int(row_cap),
                )
            st.session_state.ml_clustering_result = {
                "revision": revision,
                "result": result,
            }
            st.rerun()
        except Exception as exc:
            st.error(f"Clustering was blocked safely: {exc}")

    stored = st.session_state.get("ml_clustering_result") or {}
    result = stored.get("result")
    if int(stored.get("revision", -1)) != revision or not isinstance(
        result, ClusteringExperimentResult
    ):
        return

    for warning in result.warnings:
        st.warning(warning)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows Used", f"{result.rows_used:,}")
    m2.metric("Clusters", _safe_metric(result.metrics.get("Clusters"), digits=0))
    m3.metric("Silhouette", _safe_metric(result.metrics.get("Silhouette")))
    m4.metric("Davies–Bouldin", _safe_metric(result.metrics.get("Davies-Bouldin")))

    left, right = st.columns([1.15, 0.85])
    with left:
        fig = px.scatter(
            result.projection.reset_index(names="Source Index"),
            x="Component 1",
            y="Component 2",
            color="Cluster",
            title=f"{result.algorithm} — 2D diagnostic projection",
            template="plotly_dark",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(13,13,26,1)",
            height=420,
        )
        st.plotly_chart(fig, width="stretch")
    with right:
        fig = px.bar(
            result.cluster_sizes,
            x="Cluster",
            y="Count",
            title="Cluster Sizes",
            template="plotly_dark",
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(13,13,26,1)",
            height=420,
        )
        st.plotly_chart(fig, width="stretch")

    cluster_column = st.text_input(
        "New cluster column name",
        value="Cluster",
        key=f"clv2_colname_{revision}",
    ).strip()
    if st.button(
        "Add Cluster Labels to Working Dataset",
        width="stretch",
        disabled=not cluster_column,
        key=f"clv2_add_{revision}",
    ):
        indices = result.source_indices.copy()
        labels = result.labels.astype(str).copy()

        def add_labels(working: pd.DataFrame) -> pd.DataFrame:
            working[cluster_column] = pd.Series(pd.NA, index=working.index, dtype="string")
            working.loc[indices, cluster_column] = labels
            return working

        try:
            change = apply_dataset_change(
                "Add ML Studio V2 cluster labels",
                add_labels,
                details={
                    "algorithm": result.algorithm,
                    "features": result.feature_columns,
                    "rows_clustered": result.rows_used,
                    "column": cluster_column,
                },
                expected_revision=revision,
            )
            st.success(
                f"Added '{cluster_column}' to the working dataset."
                if change.changed
                else "The cluster labels were already up to date."
            )
            st.rerun()
        except Exception as exc:
            st.error(f"Cluster update was blocked safely: {exc}")


def render(df: pd.DataFrame) -> None:
    st.markdown(
        section_header(
            "🧠",
            "ML Studio V2",
            "Training-only CV · untouched holdout · reproducible experiments",
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="info-box">ML Studio V2 consumes the saved Feature Engineering Pipeline. '
        'Imputation, encoding, scaling, datetime extraction, and TF-IDF are fitted inside each training fold. '
        'The final holdout is never used to choose a model or tune hyperparameters.</div>',
        unsafe_allow_html=True,
    )

    revision = current_dataset_revision()
    fingerprint = str(st.session_state.get("working_fingerprint", "") or "")
    spec = st.session_state.get("feature_pipeline_spec", {}) or {}
    pipeline_report = st.session_state.get("feature_pipeline_report", {}) or {}
    profiles = st.session_state.get("semantic_profiles", {}) or {}
    experiment_report = _sync_experiment_status(
        revision=revision,
        fingerprint=fingerprint,
        spec=spec,
    )

    supervised_tab, clustering_tab = st.tabs(
        ["🎯 Supervised Experiment Lab", "🔵 Clustering V2"]
    )
    with supervised_tab:
        _render_supervised_lab(
            df,
            revision=revision,
            fingerprint=fingerprint,
            spec=spec,
            pipeline_report=pipeline_report,
            profiles=profiles,
            experiment_report=experiment_report,
        )
    with clustering_tab:
        _render_clustering_lab(df, revision)
