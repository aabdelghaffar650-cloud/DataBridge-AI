# ════════════════════════════════════════════════════════
#  DataBridge AI — Page: Data Types
# ════════════════════════════════════════════════════════
import pandas as pd
import streamlit as st
from core.session import save_history
from ui.cards import section_header

def render(df) -> None:
    st.markdown(section_header("🔄", "Data Type Management"), unsafe_allow_html=True)


    dtype_df = pd.DataFrame({
        'Column': df.dtypes.index,
        'Type': df.dtypes.astype(str).values,
        'Non-Null': df.count().values,
        'Null': df.isnull().sum().values,
        'Unique': [df[c].nunique() for c in df.columns],
        'Sample': [str(df[c].dropna().iloc[0])[:30] if not df[c].dropna().empty else "—" for c in df.columns]
    })
    st.dataframe(dtype_df, use_container_width=True, height=260)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        target_col = st.selectbox("Column:", df.columns, key="dt_col")
        st.markdown(f'<div class="info-box">Current type: <code>{df[target_col].dtype}</code> · {df[target_col].nunique()} unique values</div>', unsafe_allow_html=True)

    with c2:
        type_groups = {
            "Numeric": ["integer (int64)", "small int (int32)", "tiny int (int8)", "float64", "float32"],
            "Text": ["text (object)", "category"],
            "Date & Time": ["datetime (auto)", "date only", "time only"],
            "Boolean": ["boolean"],
            "Text Tools": ["✨ Extract Numbers", "✨ Extract First Date", "✨ UPPERCASE", "✨ lowercase", "✨ Title Case", "✨ Strip Whitespace", "✨ Clean Arabic Text", "✨ Remove Special Chars"],
        }
        flat_types = []
        for g, types in type_groups.items():
            flat_types.extend(types)
        new_type = st.selectbox("Convert to:", flat_types)

    dt_fmt = None
    if "datetime" in new_type or "date only" in new_type:
        dt_fmt = st.text_input("Date format (optional):", placeholder="%d/%m/%Y")

    if st.button("🔄 Convert Type", type="primary"):
        save_history()
        try:
            s = st.session_state.df[target_col]
            if new_type == "integer (int64)":
                st.session_state.df[target_col] = pd.to_numeric(s, errors='coerce').fillna(0).astype('int64')
            elif new_type == "small int (int32)":
                st.session_state.df[target_col] = pd.to_numeric(s, errors='coerce').fillna(0).astype('int32')
            elif new_type == "tiny int (int8)":
                st.session_state.df[target_col] = pd.to_numeric(s, errors='coerce').fillna(0).astype('int8')
            elif new_type == "float64":
                st.session_state.df[target_col] = pd.to_numeric(s, errors='coerce')
            elif new_type == "float32":
                st.session_state.df[target_col] = pd.to_numeric(s, errors='coerce').astype('float32')
            elif new_type == "text (object)":
                st.session_state.df[target_col] = s.astype(str)
            elif new_type == "category":
                st.session_state.df[target_col] = s.astype('category')
            elif new_type == "datetime (auto)":
                fmt = dt_fmt if dt_fmt else None
                st.session_state.df[target_col] = pd.to_datetime(s, format=fmt, errors='coerce')
            elif new_type == "date only":
                fmt = dt_fmt if dt_fmt else None
                st.session_state.df[target_col] = pd.to_datetime(s, format=fmt, errors='coerce').dt.date
            elif new_type == "time only":
                st.session_state.df[target_col] = pd.to_datetime(s, errors='coerce').dt.time
            elif new_type == "boolean":
                trues = ['true','1','yes','y','نعم','صح','True','Yes']
                st.session_state.df[target_col] = s.astype(str).str.strip().isin(trues)
            elif new_type == "✨ Extract Numbers":
                st.session_state.df[target_col] = pd.to_numeric(s.astype(str).str.extract(r'([\d\.]+)')[0], errors='coerce')
            elif new_type == "✨ Extract First Date":
                st.session_state.df[target_col] = s.astype(str).str.extract(r'(\d{1,4}[-/\.]\d{1,2}[-/\.]\d{1,4})')[0]
            elif new_type == "✨ UPPERCASE":
                st.session_state.df[target_col] = s.astype(str).str.upper()
            elif new_type == "✨ lowercase":
                st.session_state.df[target_col] = s.astype(str).str.lower()
            elif new_type == "✨ Title Case":
                st.session_state.df[target_col] = s.astype(str).str.title()
            elif new_type == "✨ Strip Whitespace":
                st.session_state.df[target_col] = s.astype(str).str.strip()
            elif new_type == "✨ Clean Arabic Text":
                st.session_state.df[target_col] = s.astype(str).str.replace(r'[^\u0600-\u06FF\s]', '', regex=True).str.strip()
            elif new_type == "✨ Remove Special Chars":
                st.session_state.df[target_col] = s.astype(str).str.replace(r'[^a-zA-Z0-9\u0600-\u06FF\s]', '', regex=True).str.strip()

            st.success(f"✓ '{target_col}' converted to '{new_type}'")
            st.rerun()
        except Exception as e:
            st.error(f"Error: {e}")

    st.markdown("---")
    st.subheader("Bulk Rename Columns")
    old_name = st.selectbox("Column to rename:", df.columns, key="ren_col")
    new_name = st.text_input("New name:", value=old_name, key="ren_val")
    if st.button("Rename"):
        save_history()
        st.session_state.df = st.session_state.df.rename(columns={old_name: new_name})
        st.success(f"Renamed '{old_name}' → '{new_name}'")
        st.rerun()


# ════════════════════════════════════════════════════════
#  PAGE: REPLACE VALUES
# ════════════════════════════════════════════════════════
