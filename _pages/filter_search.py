# ════════════════════════════════════════════════════════
#  DataBridge AI — Page: Filter & Search
# ════════════════════════════════════════════════════════
import pandas as pd
import streamlit as st

from core.utils import secure_multi_condition_filter
from ui.cards   import section_header


def render(df: pd.DataFrame) -> None:
    st.markdown(section_header("🔍", "Filter & Search"), unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["  Quick Search  ", "  Column Filter  ", "  Advanced (Multi-condition)  "])

    with tab1:
        search_col = st.selectbox("Search in column:", ["— All Columns —"] + list(df.columns))
        search_q   = st.text_input("Search text:", placeholder="Type to search...")
        if search_q:
            if search_col == "— All Columns —":
                mask = df.astype(str).apply(lambda col: col.str.contains(search_q, case=False, na=False, regex=False)).any(axis=1)
            else:
                mask = df[search_col].astype(str).str.contains(search_q, case=False, na=False, regex=False)
            result = df[mask]
            st.markdown(f'<div class="info-box">Found <b style="color:#7c6aff">{len(result):,}</b> matching rows out of {len(df):,}</div>', unsafe_allow_html=True)
            st.dataframe(result, use_container_width=True)
            st.download_button("📥 Export results", result.to_csv(index=False).encode("utf-8-sig"), "search_results.csv", "text/csv")

    with tab2:
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            filter_col = st.selectbox("Column:", df.columns, key="fc")
        with fc2:
            if pd.api.types.is_numeric_dtype(df[filter_col]):
                op = st.selectbox("Condition:", ["=", "≠", ">", "<", "≥", "≤", "between"])
            else:
                op = st.selectbox("Condition:", ["contains", "equals", "starts with", "ends with", "not contains"])
        with fc3:
            if op == "between":
                v1  = st.number_input("Min:", value=float(df[filter_col].min()))
                v2  = st.number_input("Max:", value=float(df[filter_col].max()))
                val = (v1, v2)
            elif pd.api.types.is_numeric_dtype(df[filter_col]):
                val = st.number_input("Value:", value=0.0)
            else:
                val = st.text_input("Value:", placeholder="...")

        if st.button("Apply Filter", key="apply_filter"):
            try:
                s = df[filter_col]
                if op == "=":              mask = s == val
                elif op == "≠":            mask = s != val
                elif op == ">":            mask = s > val
                elif op == "<":            mask = s < val
                elif op == "≥":            mask = s >= val
                elif op == "≤":            mask = s <= val
                elif op == "between":      mask = s.between(val[0], val[1])
                elif op == "contains":     mask = s.astype(str).str.contains(str(val), case=False, na=False, regex=False)
                elif op == "equals":       mask = s.astype(str) == str(val)
                elif op == "starts with":  mask = s.astype(str).str.startswith(str(val))
                elif op == "ends with":    mask = s.astype(str).str.endswith(str(val))
                else:                      mask = ~s.astype(str).str.contains(str(val), case=False, na=False, regex=False)
                result = df[mask]
                st.markdown(f'<div class="info-box">Filtered: <b style="color:#7c6aff">{len(result):,}</b> rows match</div>', unsafe_allow_html=True)
                st.dataframe(result, use_container_width=True)
                st.download_button("📥 Export", result.to_csv(index=False).encode("utf-8-sig"), "filtered.csv", "text/csv")
            except Exception as e:
                st.error(f"Error: {e}")

    with tab3:
        st.markdown('<div class="info-box">🔒 Secure multi-condition filtering</div>', unsafe_allow_html=True)
        conditions     = []
        num_conditions = st.number_input("Number of conditions:", min_value=1, max_value=5, value=1)
        for i in range(int(num_conditions)):
            st.markdown(f"**Condition {i+1}**")
            c1, c2, c3 = st.columns(3)
            with c1: col_name  = st.selectbox("Column", df.columns, key=f"cond_col_{i}")
            with c2: op_type   = st.selectbox("Operator", ["==", "!=", ">", "<", ">=", "<=", "contains"], key=f"cond_op_{i}")
            with c3: val_input = st.text_input("Value", key=f"cond_val_{i}")
            if val_input:
                try:
                    parsed = float(val_input) if "." in val_input else int(val_input)
                except ValueError:
                    parsed = val_input
                conditions.append((col_name, op_type, parsed))

        if st.button("Run Query"):
            if conditions:
                result = secure_multi_condition_filter(df, conditions)
                st.markdown(f'<div class="info-box">Result: <b style="color:#7c6aff">{len(result):,}</b> rows</div>', unsafe_allow_html=True)
                st.dataframe(result, use_container_width=True)
                st.download_button("📥 Export", result.to_csv(index=False).encode("utf-8-sig"), "query_results.csv", "text/csv")
