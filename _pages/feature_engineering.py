# ════════════════════════════════════════════════════════
#  DataBridge AI — Page: Leakage-Safe Feature Engineering
#  Stage 8: declarative sklearn preprocessing pipeline + experiment invalidation
# ════════════════════════════════════════════════════════
from __future__ import annotations

import copy
import json
from typing import Any, Dict, Mapping

import pandas as pd
import streamlit as st
from ui.streamlit_compat import safe_dataframe

from core.dataset import current_dataset_revision, update_dataset_context
from core.session import append_audit_event
from modules.feature_pipeline import (
    create_default_feature_pipeline_spec,
    feature_pipeline_spec_fingerprint,
    normalise_feature_pipeline_spec,
    safe_training_preview,
    validate_feature_pipeline_spec,
)
from modules.ml_engine import reconcile_ml_experiment_report
from ui.cards import section_header
from ui.feature_derivation_builder import render_replayable_feature_builder


_TRANSFORM_OPTIONS = [
    "Exclude",
    "Numeric",
    "One-Hot",
    "Frequency",
    "Ordinal",
    "Boolean",
    "Datetime",
    "Text TF-IDF",
]
_GROUP_TO_TRANSFORM = {
    "numeric": "Numeric",
    "categorical": "One-Hot",
    "frequency": "Frequency",
    "ordinal": "Ordinal",
    "boolean": "Boolean",
    "datetime": "Datetime",
    "text": "Text TF-IDF",
}
_TRANSFORM_TO_GROUP = {value: key for key, value in _GROUP_TO_TRANSFORM.items()}


def _effective_type(profile: Mapping[str, Any]) -> str:
    return str(
        profile.get("effective_semantic_type")
        or profile.get("semantic_type")
        or "Unknown"
    )


def _target_default(df: pd.DataFrame, saved: Mapping[str, Any], readiness: Mapping[str, Any]) -> str:
    saved_target = str(saved.get("target", ""))
    if saved_target in df.columns:
        return saved_target
    for item in readiness.get("target_candidates", []) or []:
        candidate = str(item.get("column", ""))
        if candidate in df.columns:
            return candidate
    return str(df.columns[-1])


def _task_default(df: pd.DataFrame, target: str, saved: Mapping[str, Any]) -> str:
    if str(saved.get("target", "")) == target and saved.get("task") in {
        "classification",
        "regression",
    }:
        return str(saved["task"])
    clean = df[target].dropna()
    if not pd.api.types.is_numeric_dtype(clean.dtype):
        return "classification"
    unique = int(clean.nunique(dropna=True))
    return "classification" if unique <= max(20, int(len(clean) * 0.05)) else "regression"


def _saved_assignment(saved: Mapping[str, Any]) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for group_name, transform in _GROUP_TO_TRANSFORM.items():
        for column in saved.get("groups", {}).get(group_name, []) or []:
            result[str(column)] = transform
    return result


def _default_assignment(default_spec: Mapping[str, Any]) -> Dict[str, str]:
    return _saved_assignment(default_spec)


def _feature_assignment_table(
    df: pd.DataFrame,
    profiles: Mapping[str, Mapping[str, Any]],
    target: str,
    assignment: Mapping[str, str],
) -> pd.DataFrame:
    rows = []
    for column in map(str, df.columns):
        if column == target:
            continue
        profile = dict(profiles.get(column, {}))
        risk_flags = []
        if profile.get("leakage_risk"):
            risk_flags.append("Leakage risk")
        if profile.get("high_cardinality"):
            risk_flags.append("High cardinality")
        if profile.get("is_constant"):
            risk_flags.append("Constant")
        if profile.get("all_missing") or int(profile.get("non_null_count", 1) or 0) == 0:
            risk_flags.append("All missing")
        rows.append(
            {
                "Feature": column,
                "Semantic Type": _effective_type(profile),
                "Transformation": assignment.get(column, "Exclude"),
                "Missing": int(profile.get("missing_count", int(df[column].isna().sum())) or 0),
                "Unique": int(profile.get("unique_count", int(df[column].nunique(dropna=True))) or 0),
                "Risk": ", ".join(risk_flags) if risk_flags else "—",
            }
        )
    return pd.DataFrame(rows)


def _groups_from_editor(editor_df: pd.DataFrame) -> Dict[str, list[str]]:
    groups = {name: [] for name in _GROUP_TO_TRANSFORM}
    for row in editor_df.to_dict(orient="records"):
        transform = str(row.get("Transformation", "Exclude"))
        group_name = _TRANSFORM_TO_GROUP.get(transform)
        if group_name:
            groups[group_name].append(str(row["Feature"]))
    return groups


def _base_spec(
    df: pd.DataFrame,
    profiles: Mapping[str, Mapping[str, Any]],
    target: str,
    task: str,
    revision: int,
    fingerprint: str,
    saved: Mapping[str, Any],
) -> Dict[str, Any]:
    if str(saved.get("target", "")) == target:
        base = normalise_feature_pipeline_spec(saved)
        base["task"] = task
        base["configured_revision"] = revision
        base["configured_fingerprint"] = fingerprint
        return normalise_feature_pipeline_spec(base)
    return create_default_feature_pipeline_spec(
        df,
        profiles,
        target,
        task=task,
        configured_revision=revision,
        configured_fingerprint=fingerprint,
    )


def _metric_row(report: Mapping[str, Any]) -> None:
    estimated = report.get("estimated_features", {}) or {}
    status = str(report.get("status", "Pending"))
    status_color = "#6bff8e" if status == "Configured" else "#ffb86b" if status == "Stale" else "#ff6b6b"
    st.markdown(
        f"""
        <div class="metrics-row" style="grid-template-columns:repeat(4,1fr);">
          <div class="metric-card"><div class="label">Pipeline Status</div>
            <div class="value" style="font-size:1.25rem;color:{status_color}">{status}</div>
            <div class="sub">configuration validation</div></div>
          <div class="metric-card"><div class="label">Input Features</div>
            <div class="value">{int(report.get('input_feature_count', 0)):,}</div>
            <div class="sub">selected source columns</div></div>
          <div class="metric-card"><div class="label">Projected Features</div>
            <div class="value">{int(estimated.get('total', 0)):,}</div>
            <div class="sub">estimated transformed width</div></div>
          <div class="metric-card"><div class="label">Usable Target Rows</div>
            <div class="value">{int(report.get('usable_target_rows', 0)):,}</div>
            <div class="sub">rows available for splitting</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render(df: pd.DataFrame) -> None:
    st.markdown(
        section_header(
            "🔧",
            "Feature Engineering Pipeline",
            "Fit on training data only · reusable in ML Studio",
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="info-box"><b>Two feature layers:</b> replayable source derivations can add new '
        'columns to the Working dataset while protected Raw stays immutable; train-only preprocessing '
        'then fits imputers, encoders, scalers, date references, and TF-IDF vocabulary only inside the '
        'training split to prevent leakage.</div>',
        unsafe_allow_html=True,
    )

    render_replayable_feature_builder(df, key_prefix="feature_engineering_source")
    st.markdown("---")

    revision = current_dataset_revision()
    fingerprint = str(st.session_state.get("working_fingerprint", "") or "")
    profiles = st.session_state.get("semantic_profiles", {}) or {}
    readiness = st.session_state.get("ml_readiness_report", {}) or {}
    saved_spec = st.session_state.get("feature_pipeline_spec", {}) or {}
    saved_report = st.session_state.get("feature_pipeline_report", {}) or {}

    if not profiles:
        st.warning("Run the Data Mapper semantic analysis before configuring the ML pipeline.")
        return

    setup_left, setup_right = st.columns([1.25, 0.75])
    with setup_left:
        target_default = _target_default(df, saved_spec, readiness)
        target_index = list(map(str, df.columns)).index(target_default)
        target = st.selectbox(
            "Target column",
            list(map(str, df.columns)),
            index=target_index,
            key=f"fe_target_r{revision}",
            help="The target is never included in the feature matrix.",
        )
    with setup_right:
        default_task = _task_default(df, target, saved_spec)
        task = st.radio(
            "Problem type",
            ["classification", "regression"],
            index=0 if default_task == "classification" else 1,
            horizontal=True,
            key=f"fe_task_r{revision}_{target}",
        )

    default_spec = create_default_feature_pipeline_spec(
        df,
        profiles,
        target,
        task=task,
        configured_revision=revision,
        configured_fingerprint=fingerprint,
    )
    base = _base_spec(
        df,
        profiles,
        target,
        task,
        revision,
        fingerprint,
        saved_spec,
    )
    assignment = (
        _saved_assignment(base)
        if str(saved_spec.get("target", "")) == target
        else _default_assignment(default_spec)
    )

    st.markdown("#### 2. Assign every source column to a safe transformation")
    st.caption(
        "Identifiers, contact fields, constants, and suspected leakage columns are excluded by default. "
        "You can override the proposal explicitly."
    )
    assignment_df = _feature_assignment_table(df, profiles, target, assignment)
    edited = st.data_editor(
        assignment_df,
        width="stretch",
        hide_index=True,
        disabled=["Feature", "Semantic Type", "Missing", "Unique", "Risk"],
        column_config={
            "Transformation": st.column_config.SelectboxColumn(
                "Transformation",
                options=_TRANSFORM_OPTIONS,
                required=True,
                help="Choose how this column will be processed inside the ML pipeline.",
            )
        },
        key=f"fe_assignment_editor_r{revision}_{target}",
    )
    groups = _groups_from_editor(edited)

    st.markdown("#### 3. Configure train-only transformers")
    tab_num, tab_cat, tab_date, tab_text, tab_ordinal = st.tabs(
        ["Numeric", "Categorical", "Datetime", "Text", "Ordinal"]
    )

    with tab_num:
        n1, n2, n3 = st.columns(3)
        with n1:
            numeric_imputer = st.selectbox(
                "Numeric imputation",
                ["median", "mean", "most_frequent", "constant"],
                index=["median", "mean", "most_frequent", "constant"].index(
                    str(base.get("numeric", {}).get("imputer", "median"))
                ),
                key=f"fe_num_imputer_{revision}_{target}",
            )
        with n2:
            scaler_options = ["standard", "robust", "minmax", "maxabs", "none"]
            scaler_default = str(base.get("numeric", {}).get("scaler", "standard"))
            numeric_scaler = st.selectbox(
                "Numeric scaler",
                scaler_options,
                index=scaler_options.index(scaler_default) if scaler_default in scaler_options else 0,
                key=f"fe_num_scaler_{revision}_{target}",
            )
        with n3:
            add_num_indicator = st.checkbox(
                "Add missing indicators",
                value=bool(base.get("numeric", {}).get("add_missing_indicator", True)),
                key=f"fe_num_indicator_{revision}_{target}",
            )
            percent_as_fraction = st.checkbox(
                "Convert explicit 45% to 0.45",
                value=bool(base.get("numeric", {}).get("percent_as_fraction", True)),
                key=f"fe_percent_fraction_{revision}_{target}",
            )
        st.caption("Numeric, currency, and percentage strings are parsed inside the pipeline; source columns remain unchanged.")

    with tab_cat:
        c1, c2, c3 = st.columns(3)
        with c1:
            cat_imputer_options = ["most_frequent", "constant"]
            cat_default = str(base.get("categorical", {}).get("imputer", "most_frequent"))
            cat_imputer = st.selectbox(
                "Categorical imputation",
                cat_imputer_options,
                index=cat_imputer_options.index(cat_default) if cat_default in cat_imputer_options else 0,
                key=f"fe_cat_imputer_{revision}_{target}",
            )
        with c2:
            min_frequency = st.number_input(
                "Minimum category frequency",
                min_value=1,
                max_value=max(2, min(1000, len(df))),
                value=int(base.get("categorical", {}).get("min_frequency", 2)),
                step=1,
                key=f"fe_cat_minfreq_{revision}_{target}",
                help="Rare categories are grouped by OneHotEncoder instead of creating unstable columns.",
            )
        with c3:
            max_categories = st.number_input(
                "Maximum categories per feature",
                min_value=2,
                max_value=1000,
                value=int(base.get("categorical", {}).get("max_categories", 100)),
                step=1,
                key=f"fe_cat_max_{revision}_{target}",
            )
        st.caption("One-Hot uses unknown-category handling. Frequency encoding learns category frequencies from training rows only.")

    with tab_date:
        available_parts = [
            "year",
            "quarter",
            "month",
            "day",
            "dayofweek",
            "dayofyear",
            "weekofyear",
            "is_weekend",
            "elapsed_days",
        ]
        configured_parts = [
            part
            for part in base.get("datetime", {}).get("parts", [])
            if part in available_parts
        ]
        date_parts = st.multiselect(
            "Date features",
            available_parts,
            default=configured_parts or ["year", "month", "dayofweek", "elapsed_days"],
            key=f"fe_date_parts_{revision}_{target}",
        )
        d1, d2, d3 = st.columns(3)
        with d1:
            date_cyclical = st.checkbox(
                "Add cyclical sin/cos",
                value=bool(base.get("datetime", {}).get("cyclical", True)),
                key=f"fe_date_cyc_{revision}_{target}",
            )
        with d2:
            date_dayfirst = st.checkbox(
                "Parse day first",
                value=bool(base.get("datetime", {}).get("dayfirst", True)),
                key=f"fe_date_dayfirst_{revision}_{target}",
            )
        with d3:
            date_scaler_options = ["standard", "robust", "minmax", "maxabs", "none"]
            date_scaler_default = str(base.get("datetime", {}).get("scaler", "standard"))
            date_scaler = st.selectbox(
                "Date feature scaler",
                date_scaler_options,
                index=date_scaler_options.index(date_scaler_default) if date_scaler_default in date_scaler_options else 0,
                key=f"fe_date_scaler_{revision}_{target}",
            )
        st.caption("Elapsed days use the earliest valid training date as the reference; the holdout cannot influence it.")

    with tab_text:
        text_active = bool(groups["text"])
        if not text_active:
            st.info("Assign at least one column to Text TF-IDF to enable these settings.")
        t1, t2, t3 = st.columns(3)
        with t1:
            text_max_features = st.number_input(
                "Maximum TF-IDF features per column",
                min_value=10,
                max_value=10000,
                value=int(base.get("text", {}).get("max_features", 1000)),
                step=50,
                disabled=not text_active,
                key=f"fe_text_max_{revision}_{target}",
            )
        with t2:
            ngram_min = st.selectbox(
                "Minimum n-gram",
                [1, 2, 3],
                index=max(0, min(2, int(base.get("text", {}).get("ngram_min", 1)) - 1)),
                disabled=not text_active,
                key=f"fe_text_ngmin_{revision}_{target}",
            )
        with t3:
            possible_max = [value for value in [1, 2, 3] if value >= int(ngram_min)]
            configured_max = int(base.get("text", {}).get("ngram_max", 2))
            text_ngram_max = st.selectbox(
                "Maximum n-gram",
                possible_max,
                index=possible_max.index(configured_max) if configured_max in possible_max else len(possible_max) - 1,
                disabled=not text_active,
                key=f"fe_text_ngmax_{revision}_{target}",
            )
        text_lowercase = st.checkbox(
            "Lowercase text inside the pipeline",
            value=bool(base.get("text", {}).get("lowercase", True)),
            disabled=not text_active,
            key=f"fe_text_lower_{revision}_{target}",
        )
        st.caption("Arabic characters are retained. Vocabulary is fitted from training rows only and unseen words are ignored safely.")

    ordinal_orders: Dict[str, list[str]] = {}
    with tab_ordinal:
        if not groups["ordinal"]:
            st.info("No feature is assigned to Ordinal.")
        for column in groups["ordinal"]:
            existing = base.get("ordinal", {}).get("orders", {}).get(column, [])
            default_text = ", ".join(map(str, existing))
            raw_order = st.text_input(
                f"{column} order — lowest to highest",
                value=default_text,
                placeholder="Low, Medium, High",
                key=f"fe_ord_{revision}_{target}_{column}",
            )
            values = [value.strip() for value in raw_order.split(",") if value.strip()]
            if values:
                ordinal_orders[column] = list(dict.fromkeys(values))
        st.caption("Without an explicit order, an ordinal column falls back to safe One-Hot encoding instead of inventing an order.")

    draft = copy.deepcopy(base)
    draft["target"] = target
    draft["task"] = task
    draft["groups"] = groups
    draft["semantic_types"] = {
        column: _effective_type(profiles.get(column, {}))
        for columns in groups.values()
        for column in columns
    }
    selected_columns = {column for columns in groups.values() for column in columns}
    draft["excluded_columns"] = sorted(
        column for column in map(str, df.columns) if column != target and column not in selected_columns
    )
    draft["numeric"] = {
        "imputer": numeric_imputer,
        "scaler": numeric_scaler,
        "add_missing_indicator": add_num_indicator,
        "percent_as_fraction": percent_as_fraction,
    }
    draft["categorical"] = {
        "imputer": cat_imputer,
        "encoding": "one_hot",
        "min_frequency": int(min_frequency),
        "max_categories": int(max_categories),
        "add_missing_indicator": False,
    }
    draft["frequency"] = {"enabled": True}
    draft["ordinal"] = {
        "imputer": "most_frequent",
        "orders": ordinal_orders,
    }
    draft["boolean"] = {"imputer": "most_frequent"}
    draft["datetime"] = {
        "parts": date_parts,
        "cyclical": date_cyclical,
        "imputer": "median",
        "scaler": date_scaler,
        "dayfirst": date_dayfirst,
    }
    draft["text"] = {
        "enabled": bool(groups["text"]),
        "columns": groups["text"],
        "max_features": int(text_max_features),
        "ngram_min": int(ngram_min),
        "ngram_max": int(text_ngram_max),
        "min_df": 1,
        "max_df": 1.0,
        "lowercase": text_lowercase,
    }
    draft["output"] = {"sparse": bool(groups["categorical"] or groups["text"])}
    draft["configured_revision"] = revision
    draft["configured_fingerprint"] = fingerprint
    draft = normalise_feature_pipeline_spec(draft)

    report = validate_feature_pipeline_spec(
        df,
        draft,
        require_target=True,
        current_revision=revision,
        current_fingerprint=fingerprint,
    )

    st.markdown("#### 4. Validate and save")
    _metric_row(report)
    for blocker in report.get("blockers", []):
        st.error(blocker)
    for warning in report.get("warnings", []):
        st.warning(warning)

    action_left, action_mid, action_right = st.columns([1.1, 1.1, 0.8])
    with action_left:
        if st.button(
            "💾 Save Pipeline Configuration",
            type="primary",
            width="stretch",
            disabled=not report.get("valid", False),
            key=f"fe_save_{revision}_{target}",
        ):
            try:
                experiment_report = reconcile_ml_experiment_report(
                    st.session_state.get("ml_experiment_report", {}) or {},
                    current_revision=revision,
                    current_fingerprint=fingerprint,
                    current_pipeline_spec_fingerprint=feature_pipeline_spec_fingerprint(draft),
                    artifact_available=False,
                )
                update_dataset_context(
                    expected_revision=revision,
                    feature_pipeline_spec=draft,
                    feature_pipeline_report=report,
                    ml_experiment_report=experiment_report,
                    explainability_report={},
                )
                st.session_state.feature_pipeline_preview = None
                st.session_state.ml_experiment_artifact = None
                st.session_state.model_explainability_artifact = None
                st.session_state.local_explanation_result = None
                st.session_state.shap_explanation_result = None
                append_audit_event(
                    {
                        "event": "pipeline_configuration",
                        "action": "Feature pipeline configured",
                        "revision": revision,
                        "feature_count": len(draft.get("feature_columns", [])),
                        "projected_features": int(report.get("estimated_features", {}).get("total", 0)),
                        "target": target,
                        "task": task,
                    }
                )
                st.success("Feature pipeline configuration saved safely.")
                st.rerun()
            except Exception as exc:
                st.error(f"Pipeline save was blocked safely: {exc}")

    with action_mid:
        if st.button(
            "🧪 Fit Training-Only Preview",
            width="stretch",
            disabled=not report.get("valid", False),
            key=f"fe_preview_{revision}_{target}",
        ):
            try:
                with st.spinner("Fitting transformers on the training split only..."):
                    preview_result = safe_training_preview(df, draft)
                st.session_state.feature_pipeline_preview = {
                    "revision": revision,
                    "spec_fingerprint": preview_result["spec_fingerprint"],
                    "source_rows": preview_result["source_rows"],
                    "sampled_rows": preview_result["sampled_rows"],
                    "preview_sampled": preview_result["preview_sampled"],
                    "train_rows": preview_result["train_rows"],
                    "holdout_rows": preview_result["holdout_rows"],
                    "output_features": preview_result["output_features"],
                    "sparse_output": preview_result["sparse_output"],
                    "feature_names": preview_result["feature_names"],
                    "preview": preview_result["preview"],
                }
                st.rerun()
            except Exception as exc:
                st.error(f"Training-only preview failed safely: {exc}")

    with action_right:
        st.download_button(
            "⬇ Export Config JSON",
            data=json.dumps(draft, indent=2, ensure_ascii=False, default=str).encode("utf-8"),
            file_name="databridge_feature_pipeline.json",
            mime="application/json",
            width="stretch",
            key=f"fe_export_config_{revision}_{target}",
        )

    preview_state = st.session_state.get("feature_pipeline_preview") or {}
    current_spec_fp = feature_pipeline_spec_fingerprint(draft)
    if (
        preview_state
        and int(preview_state.get("revision", -1)) == revision
        and preview_state.get("spec_fingerprint") == current_spec_fp
    ):
        st.markdown("---")
        st.markdown("#### Training-only transformed preview")
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Training Rows", f"{int(preview_state['train_rows']):,}")
        p2.metric("Holdout Rows", f"{int(preview_state['holdout_rows']):,}")
        p3.metric("Output Features", f"{int(preview_state['output_features']):,}")
        p4.metric("Matrix", "Sparse" if preview_state.get("sparse_output") else "Dense")
        if preview_state.get("preview_sampled"):
            st.caption(
                f"Safety preview used a deterministic sample of {int(preview_state['sampled_rows']):,} "
                f"from {int(preview_state['source_rows']):,} target-valid rows."
            )
        st.caption("The table below is transformed holdout data. Holdout values were not used to fit any transformer.")
        safe_dataframe(preview_state["preview"], width="stretch", height=300)
        with st.expander("Generated feature names", expanded=False):
            names = pd.DataFrame(
                {
                    "Index": range(len(preview_state["feature_names"])),
                    "Feature": preview_state["feature_names"],
                }
            )
            safe_dataframe(names, width="stretch", height=350)

    if saved_spec:
        st.markdown("---")
        st.caption(
            f"Saved pipeline: {saved_report.get('status', 'Configured')} · "
            f"target {saved_spec.get('target', '—')} · "
            f"{len(saved_spec.get('feature_columns', [])):,} input features"
        )
