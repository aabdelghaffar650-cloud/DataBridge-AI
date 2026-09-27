# ════════════════════════════════════════════════════════
#  DataBridge AI — Page: Clean & Fix Nulls
#  Stage 5: atomic dataset mutations
# ════════════════════════════════════════════════════════
import pandas as pd
import streamlit as st
from ui.streamlit_compat import safe_dataframe

from core.dataset import apply_dataset_change, current_dataset_revision
from ui.cards import section_header


def _cast_input(value: str):
    try:
        return float(value) if "." in value else int(value)
    except (TypeError, ValueError):
        return value


def render(df: pd.DataFrame) -> None:
    st.markdown(section_header("🧹", "Clean & Fix Missing Values"), unsafe_allow_html=True)
    revision = current_dataset_revision()

    null_info = df.isnull().sum()
    denominator = max(len(df), 1)
    null_df = pd.DataFrame({
        "Column": null_info.index,
        "Missing": null_info.values,
        "Pct": (null_info.values / denominator * 100).round(1),
    })
    null_df = null_df[null_df["Missing"] > 0].sort_values("Missing", ascending=False)

    if null_df.empty:
        st.success("✓ No missing values detected!")
    else:
        safe_dataframe(null_df, width="stretch", height=250)

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Quick Fix — Fake Nulls")
            st.markdown(
                '<div class="info-box">Converts text like None, null, N/A, -, ? to real NaN.</div>',
                unsafe_allow_html=True,
            )
            if st.button("✨ Convert Fake Nulls to NaN"):
                fake = [
                    r"^\s*$", "None", "none", "null", "Null", "NaN", "nan",
                    "NA", "N/A", "n/a", "-", "--", "?", "missing", "MISSING",
                ]
                try:
                    result = apply_dataset_change(
                        "Convert fake null text to missing values",
                        lambda working: working.replace(fake, None, regex=True),
                        details={"patterns": len(fake)},
                        expected_revision=revision,
                    )
                    st.success("Done!" if result.changed else "No matching fake null values were found.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Change blocked safely: {exc}")

        with col_b:
            st.subheader("Global Actions")
            null_action = st.radio(
                "Action:",
                [
                    "Drop rows with nulls",
                    "Fill all nulls with value",
                    "Fill numeric with mean",
                    "Fill numeric with median",
                ],
            )
            if null_action == "Drop rows with nulls":
                if st.button("Apply Drop"):
                    before = len(df)
                    try:
                        result = apply_dataset_change(
                            "Drop rows containing missing values",
                            lambda working: working.dropna().reset_index(drop=True),
                            details={"scope": "all columns"},
                            expected_revision=revision,
                        )
                        st.success(f"Dropped {before - result.after_shape[0]:,} rows.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Change blocked safely: {exc}")
            elif null_action == "Fill all nulls with value":
                fv = st.text_input("Fill value:")
                if st.button("Apply Fill") and fv:
                    fill_value = _cast_input(fv)
                    try:
                        result = apply_dataset_change(
                            "Fill all missing values",
                            lambda working: working.fillna(fill_value),
                            details={"scope": "all columns", "value_type": type(fill_value).__name__},
                            expected_revision=revision,
                        )
                        st.success("Done!" if result.changed else "No missing values required filling.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Change blocked safely: {exc}")
            elif null_action == "Fill numeric with mean":
                if st.button("Apply Mean Fill"):
                    def fill_means(working: pd.DataFrame) -> pd.DataFrame:
                        for column in working.select_dtypes(include="number").columns:
                            working[column] = working[column].fillna(working[column].mean())
                        return working

                    try:
                        result = apply_dataset_change(
                            "Fill numeric missing values with mean",
                            fill_means,
                            details={"strategy": "mean"},
                            expected_revision=revision,
                        )
                        st.success("Done!" if result.changed else "No numeric missing values required filling.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Change blocked safely: {exc}")
            elif null_action == "Fill numeric with median":
                if st.button("Apply Median Fill"):
                    def fill_medians(working: pd.DataFrame) -> pd.DataFrame:
                        for column in working.select_dtypes(include="number").columns:
                            working[column] = working[column].fillna(working[column].median())
                        return working

                    try:
                        result = apply_dataset_change(
                            "Fill numeric missing values with median",
                            fill_medians,
                            details={"strategy": "median"},
                            expected_revision=revision,
                        )
                        st.success("Done!" if result.changed else "No numeric missing values required filling.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Change blocked safely: {exc}")

    st.markdown("---")
    st.subheader("Per-Column Fix")
    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        per_col = st.selectbox("Column:", df.columns, key="per_col")
    with pc2:
        per_action = st.selectbox(
            "Fill with:",
            ["Custom value", "Mean", "Median", "Mode", "Forward fill", "Backward fill"],
        )
    with pc3:
        per_val = ""
        if per_action == "Custom value":
            per_val = st.text_input("Value:", key="per_val")

    if st.button("Apply to Column"):
        if per_action == "Custom value" and per_val == "":
            st.error("Enter a custom fill value.")
            return

        def fill_column(working: pd.DataFrame) -> pd.DataFrame:
            series = working[per_col]
            if per_action == "Custom value":
                working[per_col] = series.fillna(_cast_input(per_val))
            elif per_action == "Mean":
                working[per_col] = series.fillna(series.mean())
            elif per_action == "Median":
                working[per_col] = series.fillna(series.median())
            elif per_action == "Mode":
                mode = series.mode()
                if not mode.empty:
                    working[per_col] = series.fillna(mode.iloc[0])
            elif per_action == "Forward fill":
                working[per_col] = series.ffill()
            elif per_action == "Backward fill":
                working[per_col] = series.bfill()
            return working

        try:
            result = apply_dataset_change(
                f"Fill missing values in {per_col}",
                fill_column,
                details={"column": str(per_col), "strategy": per_action},
                expected_revision=revision,
            )
            st.success(
                f"Applied '{per_action}' to '{per_col}'!"
                if result.changed
                else f"No missing values in '{per_col}' required changes."
            )
            st.rerun()
        except Exception as exc:
            st.error(f"Change blocked safely: {exc}")
