# ════════════════════════════════════════════════════════
#  DataBridge AI — Page: Data Types
#  Stage 5: atomic dataset mutations
# ════════════════════════════════════════════════════════
import pandas as pd
import streamlit as st
from ui.streamlit_compat import safe_dataframe

from core.dataset import apply_dataset_change, current_dataset_revision
from ui.cards import section_header


def render(df: pd.DataFrame) -> None:
    st.markdown(section_header("🔄", "Data Type Management"), unsafe_allow_html=True)
    revision = current_dataset_revision()

    dtype_df = pd.DataFrame({
        "Column": df.dtypes.index,
        "Type": df.dtypes.astype(str).values,
        "Non-Null": df.count().values,
        "Null": df.isnull().sum().values,
        "Unique": [df[c].nunique() for c in df.columns],
        "Sample": [str(df[c].dropna().iloc[0])[:30] if not df[c].dropna().empty else "—" for c in df.columns],
    })
    safe_dataframe(dtype_df, width="stretch", height=260)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        target_col = st.selectbox("Column:", df.columns, key="dt_col")
        st.markdown(
            f'<div class="info-box">Current type: <code>{df[target_col].dtype}</code> · '
            f'{df[target_col].nunique()} unique values</div>',
            unsafe_allow_html=True,
        )

    with c2:
        type_groups = {
            "Numeric": ["integer (int64)", "small int (int32)", "tiny int (int8)", "float64", "float32"],
            "Text": ["text (object)", "category"],
            "Date & Time": ["datetime (auto)", "date only", "time only"],
            "Boolean": ["boolean"],
            "Text Tools": [
                "✨ Extract Numbers", "✨ Extract First Date", "✨ UPPERCASE",
                "✨ lowercase", "✨ Title Case", "✨ Strip Whitespace",
                "✨ Clean Arabic Text", "✨ Remove Special Chars",
            ],
        }
        flat_types = [item for items in type_groups.values() for item in items]
        new_type = st.selectbox("Convert to:", flat_types)

    dt_fmt = None
    if "datetime" in new_type or "date only" in new_type:
        dt_fmt = st.text_input("Date format (optional):", placeholder="%d/%m/%Y")

    if st.button("🔄 Convert Type", type="primary"):
        def convert_type(working: pd.DataFrame) -> pd.DataFrame:
            series = working[target_col]
            if new_type == "integer (int64)":
                working[target_col] = pd.to_numeric(series, errors="coerce").fillna(0).astype("int64")
            elif new_type == "small int (int32)":
                working[target_col] = pd.to_numeric(series, errors="coerce").fillna(0).astype("int32")
            elif new_type == "tiny int (int8)":
                working[target_col] = pd.to_numeric(series, errors="coerce").fillna(0).astype("int8")
            elif new_type == "float64":
                working[target_col] = pd.to_numeric(series, errors="coerce")
            elif new_type == "float32":
                working[target_col] = pd.to_numeric(series, errors="coerce").astype("float32")
            elif new_type == "text (object)":
                working[target_col] = series.astype(str)
            elif new_type == "category":
                working[target_col] = series.astype("category")
            elif new_type == "datetime (auto)":
                working[target_col] = pd.to_datetime(series, format=dt_fmt or None, errors="coerce")
            elif new_type == "date only":
                working[target_col] = pd.to_datetime(series, format=dt_fmt or None, errors="coerce").dt.date
            elif new_type == "time only":
                working[target_col] = pd.to_datetime(series, errors="coerce").dt.time
            elif new_type == "boolean":
                trues = ["true", "1", "yes", "y", "نعم", "صح"]
                working[target_col] = series.astype(str).str.strip().str.lower().isin(trues)
            elif new_type == "✨ Extract Numbers":
                working[target_col] = pd.to_numeric(
                    series.astype(str).str.extract(r"([\d\.]+)")[0], errors="coerce"
                )
            elif new_type == "✨ Extract First Date":
                working[target_col] = series.astype(str).str.extract(
                    r"(\d{1,4}[-/\.]\d{1,2}[-/\.]\d{1,4})"
                )[0]
            elif new_type == "✨ UPPERCASE":
                working[target_col] = series.astype(str).str.upper()
            elif new_type == "✨ lowercase":
                working[target_col] = series.astype(str).str.lower()
            elif new_type == "✨ Title Case":
                working[target_col] = series.astype(str).str.title()
            elif new_type == "✨ Strip Whitespace":
                working[target_col] = series.astype(str).str.strip()
            elif new_type == "✨ Clean Arabic Text":
                working[target_col] = series.astype(str).str.replace(
                    r"[^\u0600-\u06FF\s]", "", regex=True
                ).str.strip()
            elif new_type == "✨ Remove Special Chars":
                working[target_col] = series.astype(str).str.replace(
                    r"[^a-zA-Z0-9\u0600-\u06FF\s]", "", regex=True
                ).str.strip()
            return working

        try:
            result = apply_dataset_change(
                f"Convert {target_col} to {new_type}",
                convert_type,
                details={"column": str(target_col), "target_type": new_type},
                expected_revision=revision,
            )
            st.success(
                f"✓ '{target_col}' converted to '{new_type}'"
                if result.changed
                else "The selected conversion produced no change."
            )
            st.rerun()
        except Exception as exc:
            st.error(f"Conversion blocked safely: {exc}")

    st.markdown("---")
    st.subheader("Bulk Rename Columns")
    old_name = st.selectbox("Column to rename:", df.columns, key="ren_col")
    new_name = st.text_input("New name:", value=old_name, key="ren_val")
    if st.button("Rename"):
        cleaned_name = new_name.strip()
        if not cleaned_name:
            st.error("Column name cannot be empty.")
        else:
            try:
                result = apply_dataset_change(
                    f"Rename column {old_name} to {cleaned_name}",
                    lambda working: working.rename(columns={old_name: cleaned_name}),
                    details={"old_name": str(old_name), "new_name": cleaned_name},
                    expected_revision=revision,
                )
                st.success(
                    f"Renamed '{old_name}' → '{cleaned_name}'"
                    if result.changed
                    else "The column name was unchanged."
                )
                st.rerun()
            except Exception as exc:
                st.error(f"Rename blocked safely: {exc}")
