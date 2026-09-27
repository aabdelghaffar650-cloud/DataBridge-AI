# ════════════════════════════════════════════════════════
#  DataBridge AI — Page: Prediction Studio
#  Stage 9: verified model packages, schema gates, and batch scoring
# ════════════════════════════════════════════════════════
from __future__ import annotations

import json
from typing import Any, Mapping

import pandas as pd
import streamlit as st
from ui.streamlit_compat import safe_dataframe

from core.session import append_audit_event
from core.security import safe_html
from modules.import_engine import smart_parse_file
from modules.model_governance import current_champion, list_governance_families
from modules.model_registry import load_registered_package_bytes
from modules.model_package import (
    DEFAULT_MAX_PREDICTION_ROWS,
    LoadedModelPackage,
    ModelPackageError,
    PredictionRunResult,
    load_signed_model_package,
    run_batch_prediction,
    validate_prediction_frame,
)
from ui.cards import section_header


PREDICTION_FILE_TYPES = ["csv", "xlsx", "xls", "json", "jsonl", "ndjson", "parquet"]


def _metric_value(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _package_cards(package: LoadedModelPackage) -> None:
    manifest = package.manifest
    model = manifest.get("model", {}) or {}
    experiment = manifest.get("experiment", {}) or {}
    evaluation = manifest.get("evaluation", {}) or {}
    columns = package.required_columns
    st.markdown(
        f"""
        <div class="metrics-row" style="grid-template-columns:repeat(5,1fr);">
          <div class="metric-card"><div class="label">Trust</div>
            <div class="value" style="font-size:1.05rem;color:#6bff8e">Verified</div>
            <div class="sub">hash + local signature</div></div>
          <div class="metric-card"><div class="label">Task</div>
            <div class="value" style="font-size:1.05rem">{safe_html(package.task.title())}</div>
            <div class="sub">signed contract</div></div>
          <div class="metric-card"><div class="label">Target</div>
            <div class="value" style="font-size:1.05rem">{safe_html(package.target)}</div>
            <div class="sub">prediction output</div></div>
          <div class="metric-card"><div class="label">Model</div>
            <div class="value" style="font-size:.95rem">{safe_html(str(model.get('selected_model', '—')))}</div>
            <div class="sub">CV-selected</div></div>
          <div class="metric-card"><div class="label">Input Columns</div>
            <div class="value">{len(columns):,}</div>
            <div class="sub">required schema</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        f"Package {package.package_id} · signer {package.signer_key_id} · "
        f"experiment {experiment.get('experiment_id', '—')} · "
        f"primary metric {evaluation.get('primary_metric', '—')}"
    )


def _load_package_bytes(raw: bytes, source_label: str) -> None:
    with st.spinner("Verifying archive, model hash, local signer, and HMAC before deserialization..."):
        package = load_signed_model_package(raw)
    st.session_state.loaded_model_package = package
    st.session_state.loaded_model_package_bytes = bytes(raw)
    st.session_state.prediction_result = None
    append_audit_event(
        {
            "event": "model_package_loaded",
            "action": "Signed model package verified and loaded",
            "package_id": package.package_id,
            "task": package.task,
            "target": package.target,
            "source": source_label,
        }
    )


def _render_package_loader() -> LoadedModelPackage | None:
    st.markdown("### 1. Load a trusted model package")
    st.markdown(
        '<div class="info-box">Only <b>.dbmlpkg</b> packages signed by this DataBridge AI installation are accepted. '
        'Raw <code>.pkl</code>/<code>.joblib</code> files and packages from an unknown signer are rejected before deserialization.</div>',
        unsafe_allow_html=True,
    )

    generated = st.session_state.get("model_package_bytes")
    generated_manifest = st.session_state.get("model_package_manifest", {}) or {}
    left, right = st.columns(2)
    with left:
        st.markdown("#### Package created in ML Studio")
        if isinstance(generated, (bytes, bytearray)) and generated_manifest:
            st.caption(
                f"Ready: {generated_manifest.get('package_id', '—')} · "
                f"{generated_manifest.get('model', {}).get('selected_model', '—')}"
            )
            if st.button(
                "Use ML Studio package",
                type="primary",
                width="stretch",
                key="prediction_use_generated_package",
            ):
                try:
                    _load_package_bytes(bytes(generated), "ML Studio session")
                    st.success("Package verified and loaded.")
                    st.rerun()
                except ModelPackageError as exc:
                    st.error(f"Package load blocked safely: {exc}")
        else:
            st.info("Build a signed package from a completed ML Studio V2 experiment first.")

    with right:
        st.markdown("#### Upload a signed package")
        uploaded_package = st.file_uploader(
            "Signed DataBridge AI package",
            type=["dbmlpkg"],
            key="prediction_package_upload",
            help="Packages are installation-bound by a private local signing key.",
        )
        if uploaded_package is not None and st.button(
            "Verify & Load Uploaded Package",
            width="stretch",
            key="prediction_load_uploaded_package",
        ):
            try:
                _load_package_bytes(uploaded_package.getvalue(), uploaded_package.name)
                st.success("Uploaded package verified and loaded.")
                st.rerun()
            except ModelPackageError as exc:
                st.error(f"Package load blocked safely: {exc}")
            except Exception as exc:
                st.error(f"Package load failed safely: {exc}")

    st.markdown("#### Governed production Champion")
    try:
        families = [row for row in list_governance_families() if row.get("champion_id")]
    except Exception as exc:
        families = []
        st.caption(f"Champion registry unavailable: {exc}")
    if families:
        family_labels = {
            f"{row.get('family_name')} · {row.get('task')} → {row.get('target')}": row
            for row in families
        }
        gov_choice = st.selectbox(
            "Production model family",
            list(family_labels),
            key="prediction_governed_family",
        )
        gov_family = family_labels[gov_choice]
        try:
            champion = current_champion(str(gov_family["family_id"]))
        except Exception as exc:
            champion = None
            st.error(f"Champion verification failed safely: {exc}")
        if champion:
            st.caption(
                f"Champion: {champion.get('model','—')} · package {champion.get('package_id','—')} · "
                f"{champion.get('metric','')} {_metric_value(champion.get('holdout_value'))}"
            )
            if st.button(
                "🏆 Load Verified Champion",
                type="primary",
                width="stretch",
                key=f"prediction_load_champion_{champion.get('package_id')}",
            ):
                try:
                    raw = load_registered_package_bytes(str(champion["package_id"]))
                    _load_package_bytes(raw, "Governed Champion registry")
                    st.success("Verified Champion loaded from the governed registry.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Champion load blocked safely: {exc}")
    else:
        st.caption("No Champion has been promoted yet. Use Model Governance to approve a Challenger.")

    package = st.session_state.get("loaded_model_package")
    if not isinstance(package, LoadedModelPackage):
        return None

    st.markdown("---")
    _package_cards(package)
    p1, p2 = st.columns([3, 1])
    with p1:
        with st.expander("Verified package manifest", expanded=False):
            st.json(package.manifest, expanded=False)
            st.download_button(
                "Download manifest JSON",
                data=json.dumps(package.manifest, indent=2, ensure_ascii=False).encode("utf-8"),
                file_name=f"{package.package_id}_manifest.json",
                mime="application/json",
                width="stretch",
                key=f"prediction_manifest_{package.package_id}",
            )
    with p2:
        if st.button(
            "Unload Package",
            width="stretch",
            key=f"prediction_unload_{package.package_id}",
        ):
            st.session_state.loaded_model_package = None
            st.session_state.loaded_model_package_bytes = None
            st.session_state.prediction_result = None
            st.rerun()
    return package


def _store_prediction_input(frame: pd.DataFrame, report: Mapping[str, Any], name: str) -> None:
    st.session_state.prediction_input_df = frame.copy(deep=True)
    st.session_state.prediction_input_report = dict(report)
    st.session_state.prediction_input_name = str(name)
    st.session_state.prediction_result = None


def _render_input_loader(active_df: pd.DataFrame) -> pd.DataFrame | None:
    st.markdown("### 2. Load prediction data")
    st.caption(
        "Prediction data is held separately and never replaces or mutates the active working dataset. Import parsers do not coerce cell values or delete rows automatically."
    )
    left, right = st.columns(2)
    with left:
        st.markdown("#### Use current working dataset")
        st.caption(f"{active_df.shape[0]:,} rows × {active_df.shape[1]:,} columns")
        if st.button(
            "Copy Active Dataset for Prediction",
            width="stretch",
            key="prediction_use_active_df",
        ):
            _store_prediction_input(
                active_df,
                {"source_type": "Active working dataset", "automatic_value_changes": 0},
                st.session_state.get("file_name") or "Active dataset",
            )
            st.success("An independent prediction copy was created.")
            st.rerun()

    with right:
        st.markdown("#### Upload new prediction data")
        uploaded = st.file_uploader(
            "CSV, Excel, JSON, JSON Lines, or Parquet",
            type=PREDICTION_FILE_TYPES,
            key="prediction_data_upload",
        )
        if uploaded is not None and st.button(
            "Read Prediction File Safely",
            width="stretch",
            key="prediction_read_file",
        ):
            try:
                with st.spinner("Reading prediction file without automatic value mutation..."):
                    frame, report = smart_parse_file(uploaded)
                candidates = report.get("json_array_candidates") or []
                if len(candidates) > 1:
                    raise ModelPackageError(
                        "This JSON contains multiple top-level arrays. Import a single-array JSON/JSONL file for prediction so the selected records are unambiguous."
                    )
                _store_prediction_input(frame, report, uploaded.name)
                st.success("Prediction file loaded independently.")
                st.rerun()
            except Exception as exc:
                st.error(f"Prediction file import failed safely: {exc}")

    frame = st.session_state.get("prediction_input_df")
    if not isinstance(frame, pd.DataFrame):
        return None
    name = safe_html(st.session_state.get("prediction_input_name") or "Prediction input")
    st.markdown(
        f'<div class="info-box"><b>{safe_html(name)}</b> · {frame.shape[0]:,} rows × {frame.shape[1]:,} columns · independent copy</div>',
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns([4, 1])
    with c1:
        safe_dataframe(frame.head(20), width="stretch", height=300)
    with c2:
        if st.button("Clear Input", width="stretch", key="prediction_clear_input"):
            st.session_state.prediction_input_df = None
            st.session_state.prediction_input_report = {}
            st.session_state.prediction_input_name = ""
            st.session_state.prediction_result = None
            st.rerun()
    return frame


def _render_schema_report(report: Mapping[str, Any]) -> None:
    status = str(report.get("status", "Blocked"))
    tone = "#6bff8e" if status == "Ready" else "#ff6b6b"
    st.markdown(
        f'<div class="quality-score-banner" style="border-left:4px solid {tone};">'
        f'<div><div style="font-size:.72rem;color:#777;text-transform:uppercase;">Prediction Schema</div>'
        f'<div style="font-size:2rem;font-weight:700;color:{tone};">{safe_html(status)}</div></div>'
        f'<div style="font-size:.82rem;color:#aaa;">Rows: {int(report.get("rows", 0)):,} · '
        f'Required: {len(report.get("required_columns", [])):,} · '
        f'Missing: {len(report.get("missing_columns", [])):,} · '
        f'Extra: {len(report.get("extra_columns", [])):,}</div></div>',
        unsafe_allow_html=True,
    )
    for blocker in report.get("blockers", []) or []:
        st.error(blocker)
    for warning in report.get("warnings", []) or []:
        st.warning(warning)
    checks = report.get("column_checks", []) or []
    if checks:
        safe_dataframe(pd.DataFrame(checks), width="stretch", hide_index=True, height=330)


def _render_predictions(result: PredictionRunResult) -> None:
    st.markdown("---")
    st.markdown("### Prediction Results")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows Scored", f"{result.rows:,}")
    c2.metric("Task", result.task.title())
    c3.metric("Output Column", result.prediction_column)
    c4.metric("Probability Columns", len(result.probability_columns))
    for warning in result.warnings:
        st.warning(warning)
    safe_dataframe(result.output.head(100), width="stretch", height=430)
    st.download_button(
        "📥 Download Predictions CSV",
        data=result.output.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"databridge_predictions_{result.package_id}.csv",
        mime="text/csv",
        width="stretch",
        key=f"prediction_download_{result.package_id}_{result.created_at}",
    )


def render(df: pd.DataFrame) -> None:
    st.markdown(
        section_header(
            "🔮",
            "Prediction Studio",
            "Signed packages · schema validation · batch scoring",
        ),
        unsafe_allow_html=True,
    )

    package = _render_package_loader()
    if package is None:
        st.stop()

    st.markdown("---")
    prediction_df = _render_input_loader(df)
    if prediction_df is None:
        return

    st.markdown("---")
    st.markdown("### 3. Validate and score")
    cap_max = max(1_000, min(DEFAULT_MAX_PREDICTION_ROWS, max(1_000, len(prediction_df))))
    max_rows = st.number_input(
        "Prediction safety row cap",
        min_value=1_000,
        max_value=DEFAULT_MAX_PREDICTION_ROWS,
        value=cap_max,
        step=1_000,
        key=f"prediction_row_cap_{package.package_id}",
    )
    schema_report = validate_prediction_frame(
        prediction_df,
        package,
        max_rows=int(max_rows),
    )
    _render_schema_report(schema_report)

    include_probabilities = False
    if package.task == "classification":
        include_probabilities = st.checkbox(
            "Add probability/confidence columns when supported",
            value=True,
            key=f"prediction_probabilities_{package.package_id}",
        )

    if st.button(
        "🚀 Run Verified Batch Prediction",
        type="primary",
        width="stretch",
        disabled=not bool(schema_report.get("valid")),
        key=f"prediction_run_{package.package_id}",
    ):
        try:
            with st.spinner("Applying the exact fitted preprocessing pipeline and model..."):
                result = run_batch_prediction(
                    package,
                    prediction_df,
                    max_rows=int(max_rows),
                    include_probabilities=include_probabilities,
                )
            st.session_state.prediction_result = result
            append_audit_event(
                {
                    "event": "batch_prediction",
                    "action": "Verified batch prediction completed",
                    "package_id": package.package_id,
                    "task": package.task,
                    "target": package.target,
                    "rows": result.rows,
                    "input_name": st.session_state.get("prediction_input_name", ""),
                }
            )
            st.success("Prediction completed without modifying the active dataset or prediction input.")
            st.rerun()
        except ModelPackageError as exc:
            st.error(f"Prediction was blocked safely: {exc}")
        except Exception as exc:
            st.error(f"Prediction failed safely: {exc}")

    result = st.session_state.get("prediction_result")
    if isinstance(result, PredictionRunResult) and result.package_id == package.package_id:
        _render_predictions(result)
