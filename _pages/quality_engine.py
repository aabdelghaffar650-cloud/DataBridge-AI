# ════════════════════════════════════════════════════════
#  DataBridge AI — Quality Engine V2 & Repair Center
#  Stage 11: weighted score, critical columns, true comparisons
# ════════════════════════════════════════════════════════
from __future__ import annotations

from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd
import streamlit as st
from ui.streamlit_compat import safe_dataframe

from core.dataset import (
    apply_dataset_change,
    current_dataset_revision,
    refresh_quality_report,
)
from core.utils import vectorized_fill_nulls
from modules.quality_engine import (
    QUALITY_WEIGHT_PRESETS,
    compare_quality_reports,
    normalise_quality_policy,
    run_quality_engine,
)
from ui.cards import section_header


def _actual_column(df: pd.DataFrame, display_name: str) -> Any:
    for column in df.columns:
        if str(column) == str(display_name):
            return column
    raise KeyError(f"Column '{display_name}' no longer exists.")


def _delta_text(value: float, *, inverse: bool = False) -> str:
    effective = -value if inverse else value
    if effective > 0:
        return f"+{value:.1f}"
    if effective < 0:
        return f"{value:.1f}"
    return "0.0"


def _quality_tone(score: float) -> str:
    if score >= 90:
        return "#6bff8e"
    if score >= 75:
        return "#b9ff6b"
    if score >= 60:
        return "#ffb86b"
    return "#ff6b6b"


def _issues_frame(report: Dict[str, Any]) -> pd.DataFrame:
    rows = []
    for issue in report.get("issues", []) or []:
        rows.append(
            {
                "Severity": str(issue.get("severity", "")).title(),
                "Dimension": str(issue.get("dimension", "")).title(),
                "Issue": str(issue.get("issue_type", "")).replace("_", " ").title(),
                "Column": issue.get("column") or "— Dataset —",
                "Count": int(issue.get("count", 0) or 0),
                "Rate %": float(issue.get("percentage", 0.0) or 0.0),
                "Critical": "Yes" if issue.get("critical_column") else "No",
                "Recommendation": str(issue.get("recommended_action", "")),
            }
        )
    return pd.DataFrame(rows)


def _comparison_table(reference: Dict[str, Any], current: Dict[str, Any]) -> pd.DataFrame:
    comparison = compare_quality_reports(reference, current)
    if not comparison.get("available"):
        return pd.DataFrame()
    reference_dimensions = reference.get("dimension_scores", {}) or {}
    current_dimensions = current.get("dimension_scores", {}) or {}
    rows = [
        {
            "Metric": "Overall Quality",
            "Reference": float(reference.get("quality_score", 0.0) or 0.0),
            "Current": float(current.get("quality_score", 0.0) or 0.0),
            "Change": float(comparison.get("score_delta", 0.0) or 0.0),
        }
    ]
    for dimension in ("completeness", "validity", "uniqueness"):
        rows.append(
            {
                "Metric": dimension.title(),
                "Reference": float(reference_dimensions.get(dimension, 0.0) or 0.0),
                "Current": float(current_dimensions.get(dimension, 0.0) or 0.0),
                "Change": float(comparison.get("dimension_deltas", {}).get(dimension, 0.0) or 0.0),
            }
        )
    return pd.DataFrame(rows)


def _render_score_banner(report: Dict[str, Any], baseline: Dict[str, Any]) -> None:
    score = float(report.get("quality_score", 0.0) or 0.0)
    status = str(report.get("status", "Pending"))
    color = _quality_tone(score)
    baseline_comparison = compare_quality_reports(baseline, report)
    baseline_delta = float(baseline_comparison.get("score_delta", 0.0) or 0.0)
    baseline_text = (
        f"{baseline_delta:+.1f} points from import baseline"
        if baseline_comparison.get("available")
        else "Import baseline unavailable"
    )
    critical_issues = int((report.get("severity_counts", {}) or {}).get("critical", 0) or 0)

    st.markdown(
        f"""
        <div class="quality-score-banner" style="border-left:4px solid {color};">
          <div>
            <div style="font-size:.72rem;color:#777;text-transform:uppercase;letter-spacing:.08em;">Quality Engine V2</div>
            <div style="font-size:3rem;font-weight:700;color:{color};font-family:'JetBrains Mono',monospace;">{score:.1f}%</div>
            <div style="font-size:.82rem;color:{color};">{status}</div>
          </div>
          <div style="flex:1;">
            <div style="font-size:.9rem;color:#e0e0f0;margin-bottom:.35rem;"><b>{baseline_text}</b></div>
            <div style="font-size:.82rem;color:#888;line-height:1.7;">
              Unique defective cells: <b>{int(report.get('unique_defect_cells', 0) or 0):,}</b> &nbsp;·&nbsp;
              Duplicate rows: <b>{int(report.get('duplicate_count', 0) or 0):,}</b> &nbsp;·&nbsp;
              Affected rows: <b>{float(report.get('affected_row_pct', 0.0) or 0.0):.1f}%</b><br>
              Critical issue groups: <b style="color:#ff6b6b">{critical_issues}</b> &nbsp;·&nbsp;
              Double-counted validity events avoided: <b>{int(report.get('overlap_avoided', 0) or 0):,}</b>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_dimensions(report: Dict[str, Any]) -> None:
    dimensions = report.get("dimensions", {}) or {}
    cols = st.columns(3)
    labels = {
        "completeness": "Completeness",
        "validity": "Validity",
        "uniqueness": "Uniqueness",
    }
    for container, name in zip(cols, labels):
        payload = dimensions.get(name, {}) or {}
        score = float(payload.get("score", 0.0) or 0.0)
        weight = float(payload.get("weight", 0.0) or 0.0) * 100
        defect_pct = float(payload.get("defect_percentage", 0.0) or 0.0)
        container.metric(
            labels[name],
            f"{score:.1f}%",
            delta=f"Weight {weight:.0f}% · defects {defect_pct:.1f}%",
            delta_color="off",
        )


def _render_summary(
    df: pd.DataFrame,
    report: Dict[str, Any],
    baseline: Dict[str, Any],
    previous: Dict[str, Any],
) -> None:
    _render_dimensions(report)

    st.markdown("#### Prioritised issue register")
    issues_df = _issues_frame(report)
    if issues_df.empty:
        st.success("✅ No quality defects were detected under the active policy.")
    else:
        severity_order = pd.CategoricalDtype(
            ["Critical", "High", "Medium", "Low"], ordered=True
        )
        issues_df["Severity"] = issues_df["Severity"].astype(severity_order)
        issues_df = issues_df.sort_values(
            ["Severity", "Count"], ascending=[True, False]
        ).reset_index(drop=True)
        safe_dataframe(issues_df, width="stretch", height=360)

    st.markdown("#### True before / after comparison")
    comp_tab1, comp_tab2 = st.tabs(["Import baseline", "Previous scan / revision"])
    with comp_tab1:
        baseline_df = _comparison_table(baseline, report)
        if baseline_df.empty:
            comparison = compare_quality_reports(baseline, report)
            if comparison.get("reason") == "policy_mismatch":
                st.info("The import baseline used a different quality policy. Reset the baseline only after approving the current policy.")
            else:
                st.info("No import baseline is available for this dataset.")
        else:
            safe_dataframe(baseline_df, width="stretch", hide_index=True)
            comparison = compare_quality_reports(baseline, report)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Score change", f"{comparison['score_delta']:+.1f}")
            c2.metric("Missing change", f"{comparison['null_delta']:+,}", delta_color="inverse")
            c3.metric("Validity change", f"{comparison['validity_delta']:+,}", delta_color="inverse")
            c4.metric("Duplicates change", f"{comparison['duplicate_delta']:+,}", delta_color="inverse")
    with comp_tab2:
        previous_df = _comparison_table(previous, report)
        if previous_df.empty:
            comparison = compare_quality_reports(previous, report)
            if comparison.get("reason") == "policy_mismatch":
                st.info("The previous scan used a different quality policy, so a score delta would not be comparable.")
            else:
                st.info("No previous quality scan is available yet.")
        else:
            safe_dataframe(previous_df, width="stretch", hide_index=True)

    st.caption(str(report.get("scoring_note", "")))


def _render_duplicates(df: pd.DataFrame, report: Dict[str, Any], revision: int) -> None:
    duplicate_count = int(report.get("duplicate_count", 0) or 0)
    if duplicate_count == 0:
        st.success("✅ No exact duplicate rows found.")
        return

    st.warning(
        f"Found {duplicate_count:,} rows that duplicate an earlier full record. "
        "Repeated business events may still be valid, so review before deletion."
    )
    sample = report.get("duplicate_sample", pd.DataFrame())
    if isinstance(sample, pd.DataFrame) and not sample.empty:
        safe_dataframe(sample, width="stretch", height=300)

    confirm = st.checkbox(
        "I reviewed the duplicate definition and want to remove exact full-row duplicates.",
        key="qv2_confirm_dedupe",
    )
    if st.button(
        "Remove exact duplicates",
        type="primary",
        disabled=not confirm,
        key="qv2_dedupe",
    ):
        before = len(df)
        try:
            result = apply_dataset_change(
                "Remove exact duplicate rows from Quality Engine V2",
                lambda working: working.drop_duplicates().reset_index(drop=True),
                details={"duplicates_detected": duplicate_count},
                expected_revision=revision,
            )
            st.success(f"Removed {before - result.after_shape[0]:,} duplicate rows.")
            st.rerun()
        except Exception as exc:
            st.error(f"Repair blocked safely: {exc}")


def _render_missing(df: pd.DataFrame, report: Dict[str, Any], revision: int) -> None:
    null_by_col = report.get("null_by_col", {}) or {}
    if not null_by_col:
        st.success("✅ No missing values detected.")
        return

    critical = set(report.get("critical_columns", []) or [])
    rows = []
    for column, count in null_by_col.items():
        rows.append(
            {
                "Column": column,
                "Missing": int(count),
                "Missing %": round(int(count) / max(len(df), 1) * 100, 2),
                "Critical": "Yes" if column in critical else "No",
            }
        )
    safe_dataframe(
        pd.DataFrame(rows).sort_values(["Critical", "Missing"], ascending=[True, False]),
        width="stretch",
        hide_index=True,
    )

    st.markdown("#### Controlled batch repair")
    c1, c2 = st.columns(2)
    with c1:
        strategy = st.selectbox(
            "Fill strategy",
            ["median", "mean", "mode", "forward", "backward", "custom"],
            key="qv2_fill_strategy",
        )
        custom_fill = None
        if strategy == "custom":
            custom_fill = st.text_input("Custom fill value", key="qv2_custom_fill")
    with c2:
        selected_display = st.multiselect(
            "Columns",
            list(null_by_col.keys()),
            default=[],
            key="qv2_fill_columns",
        )

    selected_actual = [_actual_column(df, column) for column in selected_display]
    incompatible = []
    if strategy in {"mean", "median"}:
        incompatible = [
            str(column)
            for column in selected_actual
            if not pd.api.types.is_numeric_dtype(df[column].dtype)
        ]
    if incompatible:
        st.error(
            f"{strategy.title()} can only be applied to numeric columns. Remove: "
            + ", ".join(incompatible)
        )

    count_to_fill = int(
        sum(int(df[column].isna().sum()) for column in selected_actual)
    ) if selected_actual else 0
    st.caption(f"Selected repair would fill {count_to_fill:,} cells. The protected raw dataset remains unchanged.")

    if st.button(
        "Apply missing-value repair",
        type="primary",
        disabled=not selected_actual or bool(incompatible),
        key="qv2_apply_fill",
    ):
        def transform(working: pd.DataFrame) -> pd.DataFrame:
            if strategy == "custom":
                raw_value = custom_fill
                try:
                    fill_value: Any = float(raw_value) if raw_value not in (None, "") else raw_value
                except (TypeError, ValueError):
                    fill_value = raw_value
                for column in selected_actual:
                    working[column] = working[column].fillna(fill_value)
                return working
            return vectorized_fill_nulls(working, strategy, selected_actual)

        try:
            result = apply_dataset_change(
                f"Quality V2 missing-value repair ({strategy})",
                transform,
                details={
                    "strategy": strategy,
                    "columns": list(map(str, selected_actual)),
                    "cells_targeted": count_to_fill,
                },
                expected_revision=revision,
            )
            st.success("Repair applied." if result.changed else "No values changed.")
            st.rerun()
        except Exception as exc:
            st.error(f"Repair blocked safely: {exc}")


def _render_validity(df: pd.DataFrame, report: Dict[str, Any], revision: int) -> None:
    type_errors = report.get("type_errors", {}) or {}
    boolean_errors = report.get("boolean_errors", {}) or {}
    non_finite = report.get("non_finite_by_col", {}) or {}

    validity_issues = [
        issue
        for issue in report.get("issues", []) or []
        if issue.get("dimension") == "validity"
        and issue.get("issue_type") not in {"invalid_date", "future_date"}
    ]
    if not validity_issues:
        st.success("✅ No non-date validity defects detected.")
        return

    table = pd.DataFrame(
        [
            {
                "Issue": str(issue.get("issue_type", "")).replace("_", " ").title(),
                "Column": issue.get("column"),
                "Count": int(issue.get("count", 0) or 0),
                "Rate %": float(issue.get("percentage", 0.0) or 0.0),
                "Severity": str(issue.get("severity", "")).title(),
                "Critical": "Yes" if issue.get("critical_column") else "No",
            }
            for issue in validity_issues
        ]
    )
    safe_dataframe(table, width="stretch", hide_index=True)

    if type_errors:
        st.markdown("#### Reviewed numeric coercion")
        selected = st.multiselect(
            "Mixed numeric/text columns",
            list(type_errors.keys()),
            default=[],
            key="qv2_numeric_coerce_cols",
        )
        new_nulls = sum(int(type_errors[column]) for column in selected)
        confirm = st.checkbox(
            f"I understand that coercion may create {new_nulls:,} new missing values.",
            key="qv2_confirm_coerce",
        )
        if st.button(
            "Coerce selected columns to numeric",
            disabled=not selected or not confirm,
            key="qv2_coerce_numeric",
        ):
            actual = [_actual_column(df, column) for column in selected]

            def coerce(working: pd.DataFrame) -> pd.DataFrame:
                for column in actual:
                    working[column] = pd.to_numeric(working[column], errors="coerce")
                return working

            try:
                apply_dataset_change(
                    "Coerce reviewed mixed columns to numeric",
                    coerce,
                    details={"columns": selected, "expected_new_nulls": new_nulls},
                    expected_revision=revision,
                )
                st.success("Numeric coercion applied.")
                st.rerun()
            except Exception as exc:
                st.error(f"Repair blocked safely: {exc}")

    if non_finite:
        st.markdown("#### Non-finite numeric values")
        selected_inf = st.multiselect(
            "Columns containing ±infinity",
            list(non_finite.keys()),
            default=[],
            key="qv2_inf_cols",
        )
        if st.button(
            "Replace selected infinities with missing values",
            disabled=not selected_inf,
            key="qv2_replace_inf",
        ):
            actual = [_actual_column(df, column) for column in selected_inf]

            def replace_inf(working: pd.DataFrame) -> pd.DataFrame:
                for column in actual:
                    numeric = pd.to_numeric(working[column], errors="coerce")
                    working[column] = numeric.replace([np.inf, -np.inf], np.nan)
                return working

            try:
                apply_dataset_change(
                    "Replace reviewed infinite values with missing values",
                    replace_inf,
                    details={"columns": selected_inf},
                    expected_revision=revision,
                )
                st.success("Infinities were replaced with missing values for later imputation.")
                st.rerun()
            except Exception as exc:
                st.error(f"Repair blocked safely: {exc}")

    if boolean_errors:
        st.info(
            "Boolean defects require an explicit value mapping in Replace Values; "
            "Quality Engine does not guess the meaning of unknown tokens."
        )


def _render_dates(report: Dict[str, Any]) -> None:
    date_errors = report.get("date_errors", {}) or {}
    if not date_errors:
        st.success("✅ No date validity defects detected under the active policy.")
        return

    rows = []
    for column, payload in date_errors.items():
        rows.append(
            {
                "Column": column,
                "Invalid dates": int(payload.get("invalid_dates", 0) or 0),
                "Future dates": int(payload.get("future_dates", 0) or 0),
                "Future allowed": "Yes"
                if column in set((report.get("policy", {}) or {}).get("future_dates_allowed", []) or [])
                else "No",
            }
        )
    safe_dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.warning(
        "Date errors are reported but never repaired automatically. Preview and correct formats in Data Types, "
        "or allow legitimate forecast dates in Quality Policy."
    )


def _preset_name_for_weights(weights: Dict[str, Any]) -> str:
    for name, preset in QUALITY_WEIGHT_PRESETS.items():
        if all(abs(float(weights.get(key, 0.0)) - float(value)) < 1e-9 for key, value in preset.items()):
            return name
    return "Balanced"


def _render_policy(
    df: pd.DataFrame,
    report: Dict[str, Any],
    baseline: Dict[str, Any],
    revision: int,
) -> None:
    policy = normalise_quality_policy(
        df,
        report.get("policy", {}) or st.session_state.get("quality_policy", {}),
        st.session_state.get("semantic_profiles", {}) or {},
    )
    st.markdown("#### Critical-column weighting")
    auto_critical = st.checkbox(
        "Automatically include identifiers, strong target candidates, and key dates",
        value=bool(policy.get("auto_critical_columns", True)),
        key="qv2_auto_critical",
    )
    explicit_critical = st.multiselect(
        "Additional critical columns",
        [str(column) for column in df.columns],
        default=list(policy.get("critical_columns", []) or []),
        key="qv2_critical_cols",
    )
    multiplier = st.slider(
        "Critical-column multiplier",
        min_value=1.0,
        max_value=5.0,
        value=float(policy.get("critical_multiplier", 2.0) or 2.0),
        step=0.25,
        key="qv2_critical_multiplier",
    )
    inferred = list(policy.get("inferred_critical_columns", []) or [])
    if inferred:
        st.caption("Automatically inferred: " + ", ".join(inferred))

    st.markdown("#### Scoring priorities")
    current_preset = _preset_name_for_weights(policy.get("dimension_weights", {}) or {})
    preset_name = st.selectbox(
        "Weight preset",
        list(QUALITY_WEIGHT_PRESETS.keys()),
        index=list(QUALITY_WEIGHT_PRESETS.keys()).index(current_preset),
        key="qv2_weight_preset",
    )
    preset = QUALITY_WEIGHT_PRESETS[preset_name]
    st.caption(
        f"Completeness {preset['completeness']:.0%} · Validity {preset['validity']:.0%} · "
        f"Uniqueness {preset['uniqueness']:.0%}"
    )

    date_columns = sorted((report.get("date_errors", {}) or {}).keys())
    semantic_profiles = st.session_state.get("semantic_profiles", {}) or {}
    for column, profile in semantic_profiles.items():
        semantic_type = str(
            profile.get("effective_semantic_type")
            or profile.get("semantic_type")
            or ""
        )
        if semantic_type == "Datetime" and column not in date_columns:
            date_columns.append(column)
    future_allowed = st.multiselect(
        "Columns where future dates are legitimate",
        date_columns,
        default=[
            column for column in policy.get("future_dates_allowed", []) or []
            if column in date_columns
        ],
        key="qv2_future_allowed",
    )

    new_policy = {
        "critical_columns": explicit_critical,
        "auto_critical_columns": auto_critical,
        "critical_multiplier": multiplier,
        "future_dates_allowed": future_allowed,
        "dimension_weights": preset,
    }
    if st.button("Apply quality policy and rescan", type="primary", key="qv2_apply_policy"):
        try:
            refresh_quality_report(
                policy=new_policy,
                expected_revision=revision,
            )
            st.success("Quality policy applied without changing the dataset.")
            st.rerun()
        except Exception as exc:
            st.error(f"Quality policy update failed safely: {exc}")

    st.markdown("---")
    st.markdown("#### Baseline control")
    if baseline:
        st.caption(
            f"Current baseline score: {float(baseline.get('quality_score', 0.0) or 0.0):.1f}% "
            f"at revision {int(baseline.get('dataset_revision', 0) or 0)}."
        )
    reset_confirm = st.checkbox(
        "Use the current scan as the new comparison baseline.",
        key="qv2_baseline_confirm",
    )
    if st.button(
        "Reset quality baseline",
        disabled=not reset_confirm,
        key="qv2_reset_baseline",
    ):
        try:
            refresh_quality_report(
                policy=new_policy,
                set_baseline=True,
                expected_revision=revision,
            )
            st.success("The current scan is now the quality baseline.")
            st.rerun()
        except Exception as exc:
            st.error(f"Baseline reset failed safely: {exc}")


def render(df: pd.DataFrame) -> None:
    st.markdown(
        section_header(
            "🛡️",
            "Data Quality Engine V2",
            "Weighted · Deduplicated · Policy-aware",
        ),
        unsafe_allow_html=True,
    )
    revision = current_dataset_revision()
    report = st.session_state.get("quality_report", {}) or {}
    if not report or str(report.get("engine_version", "")) != "11.0":
        try:
            report = refresh_quality_report(expected_revision=revision)
        except Exception:
            report = run_quality_engine(
                df,
                policy=st.session_state.get("quality_policy", {}) or {},
                semantic_profiles=st.session_state.get("semantic_profiles", {}) or {},
                dataset_revision=revision,
                dataset_fingerprint=str(st.session_state.get("working_fingerprint", "") or ""),
            )

    baseline = st.session_state.get("quality_baseline_report", {}) or {}
    previous = st.session_state.get("quality_previous_report", {}) or {}
    _render_score_banner(report, baseline)

    tabs = st.tabs(
        [
            "  📊 Summary  ",
            "  🟡 Missing  ",
            "  🔴 Duplicates  ",
            "  🟠 Validity  ",
            "  📅 Dates  ",
            "  ⚙️ Policy  ",
        ]
    )
    with tabs[0]:
        _render_summary(df, report, baseline, previous)
    with tabs[1]:
        _render_missing(df, report, revision)
    with tabs[2]:
        _render_duplicates(df, report, revision)
    with tabs[3]:
        _render_validity(df, report, revision)
    with tabs[4]:
        _render_dates(report)
    with tabs[5]:
        _render_policy(df, report, baseline, revision)
