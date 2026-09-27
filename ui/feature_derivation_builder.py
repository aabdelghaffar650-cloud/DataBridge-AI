"""Reusable source-level replayable feature derivation controls.

These controls mutate only the Working dataset through the atomic dataset API.
Protected Raw remains immutable. Every created feature is also appended to the
signed feature-derivation recipe so model packages can recreate it at inference.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from core.dataset import apply_dataset_change, current_dataset_revision
from core.feature_derivation import (
    FeatureDerivationError,
    append_derivation_step,
    build_formula_derivation_step,
    build_regex_extract_derivation_step,
    build_value_map_derivation_step,
    replay_feature_derivations,
)
from core.formula import safe_eval_formula
from ui.streamlit_compat import safe_dataframe


def _parse_mapping_lines(text: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for line_number, raw_line in enumerate(str(text or "").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise FeatureDerivationError(
                f"Mapping line {line_number} must use Source = Replacement."
            )
        source, replacement = [part.strip() for part in line.split("=", 1)]
        if not source:
            raise FeatureDerivationError(
                f"Mapping line {line_number} has an empty source value."
            )
        if source in mapping:
            raise FeatureDerivationError(
                f"Mapping source '{source}' is defined more than once."
            )
        mapping[source] = replacement
    if not mapping:
        raise FeatureDerivationError(
            "Enter at least one Source = Replacement mapping line."
        )
    return mapping


def render_replayable_feature_builder(
    df: pd.DataFrame,
    *,
    key_prefix: str = "feature_engineering_derivation",
) -> None:
    """Render Computed / Regex Extract / Value Mapping controls."""
    revision = current_dataset_revision()
    recipe_count = len(st.session_state.get("feature_derivation_recipe", []) or [])

    st.markdown("#### 1. Create replayable source features (optional)")
    st.caption(
        "These operations update the Working dataset atomically while protected Raw stays unchanged. "
        "Their recipes are signed into model packages and replayed from raw prediction inputs."
    )
    if recipe_count:
        st.info(
            f"Current signed derivation recipe: {recipe_count} step(s). "
            "New steps are appended in dependency order."
        )

    tab_formula, tab_regex, tab_mapping = st.tabs(
        ["Computed Column", "Regex Extract", "Value Mapping"]
    )

    with tab_formula:
        st.markdown(
            '<div class="info-box">Create a replayable numeric/boolean feature with a safe formula. '
            'Example: <code>SibSp + Parch + 1</code> or <code>FamilySize == 1</code>. '
            'For columns with spaces use backticks.</div>',
            unsafe_allow_html=True,
        )
        output_name = st.text_input(
            "New column name:",
            placeholder="FamilySize",
            key=f"{key_prefix}_formula_output",
        )
        formula = st.text_area(
            "Formula:",
            placeholder="SibSp + Parch + 1",
            height=80,
            key=f"{key_prefix}_formula_text",
        )
        st.caption("Available columns: " + ", ".join([f"`{c}`" for c in df.columns]))

        if st.button(
            "➕ Add Computed Column",
            key=f"{key_prefix}_formula_add",
        ):
            cleaned_name = output_name.strip()
            if not cleaned_name or not formula.strip():
                st.error("Enter a new column name and a formula.")
            else:
                try:
                    step = build_formula_derivation_step(
                        df,
                        cleaned_name,
                        formula,
                        require_new_column=True,
                    )
                    next_recipe = append_derivation_step(
                        st.session_state.get("feature_derivation_recipe", []),
                        step,
                    )

                    def add_computed_column(working: pd.DataFrame) -> pd.DataFrame:
                        working[cleaned_name] = safe_eval_formula(working, formula)
                        return working

                    result = apply_dataset_change(
                        f"Create computed column {cleaned_name}",
                        add_computed_column,
                        details={
                            "column": cleaned_name,
                            "formula": formula.strip(),
                            "replayable": True,
                            "inputs": list(step.get("inputs", [])),
                        },
                        expected_revision=revision,
                        context_updates={"feature_derivation_recipe": next_recipe},
                    )
                    st.success(
                        f"Column '{cleaned_name}' added with a replayable production recipe."
                        if result.changed
                        else "The computed result did not change the dataset."
                    )
                    st.rerun()
                except FeatureDerivationError as exc:
                    st.error(f"Formula blocked safely: {exc}")
                except Exception as exc:
                    st.error(f"Formula blocked safely: {exc}")

    with tab_regex:
        st.markdown(
            '<div class="info-box">Extract one capture group from a text column into a new '
            'replayable feature. The source column is not overwritten.</div>',
            unsafe_allow_html=True,
        )
        rx1, rx2 = st.columns(2)
        with rx1:
            source = st.selectbox(
                "Source column:",
                list(df.columns),
                key=f"{key_prefix}_regex_source",
            )
        with rx2:
            output = st.text_input(
                "New column name:",
                value="Title",
                key=f"{key_prefix}_regex_output",
            )
        pattern = st.text_input(
            "Regex pattern (must contain a capture group):",
            value=r",\s*([^.]*)\.",
            key=f"{key_prefix}_regex_pattern",
        )
        rx3, rx4, rx5 = st.columns(3)
        with rx3:
            group = st.number_input(
                "Capture group:",
                min_value=1,
                max_value=20,
                value=1,
                step=1,
                key=f"{key_prefix}_regex_group",
            )
        with rx4:
            ignore_case = st.checkbox(
                "Ignore case",
                value=False,
                key=f"{key_prefix}_regex_ignore_case",
            )
        with rx5:
            fill_no_match = st.checkbox(
                "Fill no-match values",
                value=True,
                key=f"{key_prefix}_regex_fill",
            )
        no_match = None
        if fill_no_match:
            no_match = st.text_input(
                "No-match value:",
                value="Unknown",
                key=f"{key_prefix}_regex_no_match",
            )

        if st.button(
            "🔎 Preview Regex Extraction",
            key=f"{key_prefix}_regex_preview",
        ):
            try:
                step = build_regex_extract_derivation_step(
                    df,
                    output,
                    source,
                    pattern,
                    capture_group=int(group),
                    ignore_case=bool(ignore_case),
                    no_match_value=no_match,
                )
                preview = replay_feature_derivations(df, [step])[[source, output]].head(25)
                safe_dataframe(preview, width="stretch", hide_index=True)
            except Exception as exc:
                st.error(f"Regex extraction blocked safely: {exc}")

        if st.button(
            "➕ Add Regex Extract Feature",
            key=f"{key_prefix}_regex_add",
        ):
            try:
                step = build_regex_extract_derivation_step(
                    df,
                    output,
                    source,
                    pattern,
                    capture_group=int(group),
                    ignore_case=bool(ignore_case),
                    no_match_value=no_match,
                )
                next_recipe = append_derivation_step(
                    st.session_state.get("feature_derivation_recipe", []), step
                )

                def add_regex_feature(working: pd.DataFrame) -> pd.DataFrame:
                    return replay_feature_derivations(working, [step])

                result = apply_dataset_change(
                    f"Create regex-derived column {output.strip()}",
                    add_regex_feature,
                    details={
                        "column": output.strip(),
                        "source_column": str(source),
                        "operation": "regex_extract",
                        "capture_group": int(group),
                        "replayable": True,
                    },
                    expected_revision=revision,
                    context_updates={"feature_derivation_recipe": next_recipe},
                )
                st.success(
                    f"Column '{output.strip()}' added with a replayable regex recipe."
                    if result.changed
                    else "The extracted feature did not change the dataset."
                )
                st.rerun()
            except Exception as exc:
                st.error(f"Regex extraction blocked safely: {exc}")

    with tab_mapping:
        st.markdown(
            '<div class="info-box">Normalize exact values into a new replayable feature. '
            'Unlisted values can be preserved, replaced by a constant, or set missing.</div>',
            unsafe_allow_html=True,
        )
        mp1, mp2 = st.columns(2)
        with mp1:
            source = st.selectbox(
                "Source column:",
                list(df.columns),
                key=f"{key_prefix}_map_source",
            )
        with mp2:
            output = st.text_input(
                "New column name:",
                value="Title_Normalized",
                key=f"{key_prefix}_map_output",
            )
        mapping_lines = st.text_area(
            "Exact mappings — one Source = Replacement per line:",
            value=(
                "Mlle = Miss\n"
                "Ms = Miss\n"
                "Mme = Mrs\n"
                "Dr = Rare\n"
                "Rev = Rare\n"
                "Col = Rare\n"
                "Major = Rare\n"
                "Don = Rare\n"
                "Lady = Rare\n"
                "Sir = Rare\n"
                "Capt = Rare\n"
                "Countess = Rare\n"
                "Jonkheer = Rare"
            ),
            height=220,
            key=f"{key_prefix}_map_lines",
        )
        unmatched_label = st.selectbox(
            "Values not listed above:",
            ["Preserve original value", "Replace with a constant", "Set to missing"],
            key=f"{key_prefix}_map_unmatched",
        )
        unmatched_strategy = {
            "Preserve original value": "preserve",
            "Replace with a constant": "constant",
            "Set to missing": "null",
        }[unmatched_label]
        unmatched_value = None
        if unmatched_strategy == "constant":
            unmatched_value = st.text_input(
                "Unmatched replacement:",
                value="Rare",
                key=f"{key_prefix}_map_default",
            )

        if st.button(
            "🔎 Preview Value Mapping",
            key=f"{key_prefix}_map_preview",
        ):
            try:
                mapping = _parse_mapping_lines(mapping_lines)
                step = build_value_map_derivation_step(
                    df,
                    output,
                    source,
                    mapping,
                    unmatched_strategy=unmatched_strategy,
                    unmatched_value=unmatched_value,
                )
                preview = replay_feature_derivations(df, [step])[[source, output]].head(25)
                safe_dataframe(preview, width="stretch", hide_index=True)
            except Exception as exc:
                st.error(f"Value Mapping blocked safely: {exc}")

        if st.button(
            "➕ Add Value-Mapped Feature",
            key=f"{key_prefix}_map_add",
        ):
            try:
                mapping = _parse_mapping_lines(mapping_lines)
                step = build_value_map_derivation_step(
                    df,
                    output,
                    source,
                    mapping,
                    unmatched_strategy=unmatched_strategy,
                    unmatched_value=unmatched_value,
                )
                next_recipe = append_derivation_step(
                    st.session_state.get("feature_derivation_recipe", []), step
                )

                def add_mapped_feature(working: pd.DataFrame) -> pd.DataFrame:
                    return replay_feature_derivations(working, [step])

                result = apply_dataset_change(
                    f"Create value-mapped column {output.strip()}",
                    add_mapped_feature,
                    details={
                        "column": output.strip(),
                        "source_column": str(source),
                        "operation": "value_map",
                        "mapping_entries": len(mapping),
                        "unmatched_strategy": unmatched_strategy,
                        "replayable": True,
                    },
                    expected_revision=revision,
                    context_updates={"feature_derivation_recipe": next_recipe},
                )
                st.success(
                    f"Column '{output.strip()}' added with a replayable value-mapping recipe."
                    if result.changed
                    else "The mapped feature did not change the dataset."
                )
                st.rerun()
            except Exception as exc:
                st.error(f"Value Mapping blocked safely: {exc}")
