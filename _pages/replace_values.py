# ════════════════════════════════════════════════════════
#  DataBridge AI — Page: Replace Values
#  Stage 5: atomic dataset mutations
# ════════════════════════════════════════════════════════
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
from ui.cards import section_header


def _cast_value(value: str):
    try:
        return float(value) if "." in value else int(value)
    except (TypeError, ValueError):
        return value


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
            raise FeatureDerivationError(f"Mapping line {line_number} has an empty source value.")
        if source in mapping:
            raise FeatureDerivationError(f"Mapping source '{source}' is defined more than once.")
        mapping[source] = replacement
    if not mapping:
        raise FeatureDerivationError("Enter at least one Source = Replacement mapping line.")
    return mapping


def render(df: pd.DataFrame) -> None:
    st.markdown(section_header("✏️", "Replace & Transform Values"), unsafe_allow_html=True)
    revision = current_dataset_revision()

    tab1, tab2, tab3, tab4 = st.tabs([
        "  Find & Replace  ",
        "  Quick: Nulls → Value  ",
        "  Computed Column  ",
        "  Text / Regex Feature  ",
    ])

    with tab1:
        r1, r2, r3 = st.columns(3)
        with r1:
            rep_col_choice = st.selectbox(
                "Column:", ["— All Columns —"] + list(df.columns), key="rep_col"
            )
        with r2:
            find_vals = st.text_input(
                "Find (comma-separated):", placeholder="None, -, unknown, -1"
            )
        with r3:
            rep_val = st.text_input("Replace with:", placeholder="0")

        use_regex = st.checkbox("Use regex pattern")

        if st.button("🔁 Replace"):
            if not find_vals or rep_val == "":
                st.error("Enter both the values to find and the replacement value.")
            else:
                values = [value.strip() for value in find_vals.split(",")]
                replacement = _cast_value(rep_val)

                def replace_values(working: pd.DataFrame) -> pd.DataFrame:
                    if rep_col_choice == "— All Columns —":
                        return working.replace(values, replacement, regex=use_regex)

                    series = working[rep_col_choice].astype(str)
                    if use_regex:
                        for value in values:
                            series = series.str.replace(value, str(replacement), regex=True)
                    else:
                        for value in values:
                            series = series.replace(value, str(replacement))
                    try:
                        working[rep_col_choice] = pd.to_numeric(series, errors="raise")
                    except (TypeError, ValueError):
                        working[rep_col_choice] = series
                    return working

                try:
                    result = apply_dataset_change(
                        "Find and replace values",
                        replace_values,
                        details={
                            "column": str(rep_col_choice),
                            "find_count": len(values),
                            "regex": bool(use_regex),
                        },
                        expected_revision=revision,
                    )
                    st.success("Replacement applied!" if result.changed else "No matching values were found.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Replacement blocked safely: {exc}")

    with tab2:
        st.markdown(
            '<div class="info-box">Quickly replace null / NaN values in a column with any value.</div>',
            unsafe_allow_html=True,
        )
        qc1, qc2, qc3 = st.columns(3)
        with qc1:
            q_col = st.selectbox("Column:", df.columns, key="qc")
        with qc2:
            q_val = st.text_input("Replace nulls with:", value="0", key="qv")
        with qc3:
            q_scope = st.radio("Apply to:", ["Selected column", "All columns"], key="qs")

        if st.button("⚡ Apply Quick Replace"):
            replacement = _cast_value(q_val)

            def quick_replace(working: pd.DataFrame) -> pd.DataFrame:
                if q_scope == "Selected column":
                    working[q_col] = working[q_col].fillna(replacement)
                    return working
                return working.fillna(replacement)

            try:
                result = apply_dataset_change(
                    "Quick replace missing values",
                    quick_replace,
                    details={"scope": q_scope, "column": str(q_col)},
                    expected_revision=revision,
                )
                st.success("Null replacement applied." if result.changed else "No null values required replacement.")
                st.rerun()
            except Exception as exc:
                st.error(f"Replacement blocked safely: {exc}")

    with tab3:
        st.markdown(
            '<div class="info-box">Create a computed column using safe numeric formulas only.<br>'
            'Example: <code>Price * 0.14</code>. For columns with spaces use backticks: '
            '<code>`Total Amount` * 0.14</code>.</div>',
            unsafe_allow_html=True,
        )
        new_col_name = st.text_input("New column name:", placeholder="Tax_Amount")
        formula = st.text_area("Formula:", placeholder="Price * 0.14", height=80)
        st.caption("Available columns: " + ", ".join([f"`{c}`" for c in df.columns]))
        recipe_count = len(st.session_state.get("feature_derivation_recipe", []) or [])
        if recipe_count:
            st.caption(
                f"Replayable production recipe: {recipe_count} derived feature step(s). "
                "Signed model packages will recreate required derived columns automatically."
            )

        if st.button("➕ Add Computed Column"):
            cleaned_name = new_col_name.strip()
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
                        if result.changed else "The computed result did not change the dataset."
                    )
                    st.rerun()
                except FeatureDerivationError as exc:
                    st.error(f"Formula blocked safely: {exc}")
                except Exception as exc:
                    st.error(f"Formula blocked safely: {exc}")

    with tab4:
        st.markdown(
            '<div class="info-box">Create replayable text features without changing the source column. '
            'Regex Extract writes a capture group into a new column; Value Mapping normalizes exact values. '
            'Both steps are signed into the model package and replayed automatically on raw prediction data.</div>',
            unsafe_allow_html=True,
        )
        derivation_mode = st.radio(
            "Text feature operation:",
            ["Regex Extract", "Value Mapping"],
            horizontal=True,
            key="text_derivation_mode",
        )

        if derivation_mode == "Regex Extract":
            rx1, rx2 = st.columns(2)
            with rx1:
                rx_source = st.selectbox("Source column:", list(df.columns), key="regex_extract_source")
            with rx2:
                rx_output = st.text_input("New column name:", value="Title", key="regex_extract_output")
            rx_pattern = st.text_input(
                "Regex pattern (must contain a capture group):",
                value=r",\s*([^.]*)\.",
                key="regex_extract_pattern",
            )
            rx3, rx4, rx5 = st.columns(3)
            with rx3:
                rx_group = st.number_input(
                    "Capture group:", min_value=1, max_value=20, value=1, step=1, key="regex_extract_group"
                )
            with rx4:
                rx_ignore_case = st.checkbox("Ignore case", value=False, key="regex_extract_ignore_case")
            with rx5:
                rx_fill_no_match = st.checkbox("Fill no-match values", value=True, key="regex_extract_fill")
            rx_no_match = None
            if rx_fill_no_match:
                rx_no_match = st.text_input("No-match value:", value="Unknown", key="regex_extract_no_match")

            if st.button("🔎 Preview Regex Extraction", key="preview_regex_extract"):
                try:
                    step = build_regex_extract_derivation_step(
                        df,
                        rx_output,
                        rx_source,
                        rx_pattern,
                        capture_group=int(rx_group),
                        ignore_case=bool(rx_ignore_case),
                        no_match_value=rx_no_match,
                    )
                    preview = replay_feature_derivations(df, [step])[[rx_source, rx_output]].head(25)
                    st.dataframe(preview, width="stretch", hide_index=True)
                except Exception as exc:
                    st.error(f"Regex extraction blocked safely: {exc}")

            if st.button("➕ Add Regex Extract Feature", key="add_regex_extract"):
                try:
                    step = build_regex_extract_derivation_step(
                        df,
                        rx_output,
                        rx_source,
                        rx_pattern,
                        capture_group=int(rx_group),
                        ignore_case=bool(rx_ignore_case),
                        no_match_value=rx_no_match,
                    )
                    next_recipe = append_derivation_step(
                        st.session_state.get("feature_derivation_recipe", []), step
                    )

                    def add_regex_feature(working: pd.DataFrame) -> pd.DataFrame:
                        return replay_feature_derivations(working, [step])

                    result = apply_dataset_change(
                        f"Create regex-derived column {rx_output.strip()}",
                        add_regex_feature,
                        details={
                            "column": rx_output.strip(),
                            "source_column": str(rx_source),
                            "operation": "regex_extract",
                            "capture_group": int(rx_group),
                            "replayable": True,
                        },
                        expected_revision=revision,
                        context_updates={"feature_derivation_recipe": next_recipe},
                    )
                    st.success(
                        f"Column '{rx_output.strip()}' added with a replayable regex recipe."
                        if result.changed else "The extracted feature did not change the dataset."
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(f"Regex extraction blocked safely: {exc}")

        else:
            mp1, mp2 = st.columns(2)
            with mp1:
                map_source = st.selectbox("Source column:", list(df.columns), key="value_map_source")
            with mp2:
                map_output = st.text_input(
                    "New column name:", value="Title_Normalized", key="value_map_output"
                )
            map_lines = st.text_area(
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
                key="value_map_lines",
            )
            unmatched_label = st.selectbox(
                "Values not listed above:",
                ["Preserve original value", "Replace with a constant", "Set to missing"],
                key="value_map_unmatched",
            )
            unmatched_strategy = {
                "Preserve original value": "preserve",
                "Replace with a constant": "constant",
                "Set to missing": "null",
            }[unmatched_label]
            unmatched_value = None
            if unmatched_strategy == "constant":
                unmatched_value = st.text_input("Unmatched replacement:", value="Rare", key="value_map_default")

            if st.button("🔎 Preview Value Mapping", key="preview_value_map"):
                try:
                    mapping = _parse_mapping_lines(map_lines)
                    step = build_value_map_derivation_step(
                        df,
                        map_output,
                        map_source,
                        mapping,
                        unmatched_strategy=unmatched_strategy,
                        unmatched_value=unmatched_value,
                    )
                    preview = replay_feature_derivations(df, [step])[[map_source, map_output]].head(25)
                    st.dataframe(preview, width="stretch", hide_index=True)
                except Exception as exc:
                    st.error(f"Value Mapping blocked safely: {exc}")

            if st.button("➕ Add Value-Mapped Feature", key="add_value_map"):
                try:
                    mapping = _parse_mapping_lines(map_lines)
                    step = build_value_map_derivation_step(
                        df,
                        map_output,
                        map_source,
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
                        f"Create value-mapped column {map_output.strip()}",
                        add_mapped_feature,
                        details={
                            "column": map_output.strip(),
                            "source_column": str(map_source),
                            "operation": "value_map",
                            "mapping_entries": len(mapping),
                            "unmatched_strategy": unmatched_strategy,
                            "replayable": True,
                        },
                        expected_revision=revision,
                        context_updates={"feature_derivation_recipe": next_recipe},
                    )
                    st.success(
                        f"Column '{map_output.strip()}' added with a replayable value-mapping recipe."
                        if result.changed else "The mapped feature did not change the dataset."
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(f"Value Mapping blocked safely: {exc}")