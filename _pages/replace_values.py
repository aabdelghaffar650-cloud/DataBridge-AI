# ════════════════════════════════════════════════════════
#  DataBridge AI — Page: Replace Values
# ════════════════════════════════════════════════════════
import numpy as np
import pandas as pd
import streamlit as st
from core.session import save_history
from core.formula import safe_eval_formula
from ui.cards import section_header

def render(df) -> None:
    st.markdown(section_header("✏️", "Replace & Transform Values"), unsafe_allow_html=True)


    tab1, tab2, tab3 = st.tabs(["  Find & Replace  ", "  Quick: Nulls → Value  ", "  Computed Column  "])

    with tab1:
        r1, r2, r3 = st.columns(3)
        with r1:
            rep_col_choice = st.selectbox("Column:", ["— All Columns —"] + list(df.columns), key="rep_col")
        with r2:
            find_vals = st.text_input("Find (comma-separated):", placeholder="None, -, unknown, -1")
        with r3:
            rep_val = st.text_input("Replace with:", placeholder="0")

        use_regex = st.checkbox("Use regex pattern")

        rb1, rb2 = st.columns(2)
        with rb1:
            if st.button("🔁 Replace"):
                if find_vals and rep_val != "":
                    save_history()
                    vals = [v.strip() for v in find_vals.split(',')]
                    try:
                        rv = float(rep_val) if '.' in rep_val else int(rep_val)
                    except:
                        rv = rep_val

                    if rep_col_choice == "— All Columns —":
                        if use_regex:
                            st.session_state.df = st.session_state.df.replace(vals, rv, regex=True)
                        else:
                            st.session_state.df = st.session_state.df.replace(vals, rv)
                    else:
                        col_s = st.session_state.df[rep_col_choice].astype(str)
                        if use_regex:
                            for v in vals:
                                col_s = col_s.str.replace(v, str(rv), regex=True)
                        else:
                            for v in vals:
                                col_s = col_s.replace(v, str(rv))
                        try:
                            st.session_state.df[rep_col_choice] = pd.to_numeric(col_s, errors='raise')
                        except:
                            st.session_state.df[rep_col_choice] = col_s
                    st.success("Replacement applied!")
                    st.rerun()

    with tab2:
        st.markdown('<div class="info-box">Quickly replace null / NaN values in a column with any value — great for converting None → 0.</div>', unsafe_allow_html=True)
        qc1, qc2, qc3 = st.columns(3)
        with qc1:
            q_col = st.selectbox("Column:", df.columns, key="qc")
        with qc2:
            q_val = st.text_input("Replace nulls with:", value="0", key="qv")
        with qc3:
            q_scope = st.radio("Apply to:", ["Selected column", "All columns"], key="qs")

        if st.button("⚡ Apply Quick Replace"):
            save_history()
            try:
                rv = float(q_val) if '.' in q_val else int(q_val)
            except:
                rv = q_val
            if q_scope == "Selected column":
                st.session_state.df[q_col] = st.session_state.df[q_col].fillna(rv)
                st.success(f"Nulls in '{q_col}' → {rv}")
            else:
                st.session_state.df = st.session_state.df.fillna(rv)
                st.success(f"All nulls → {rv}")
            st.rerun()

    with tab3:
        st.markdown('<div class="info-box">Create a computed column using safe numeric formulas only.<br>Example: <code>Price * 0.14</code>. For columns with spaces use backticks: <code>`Total Amount` * 0.14</code>.</div>', unsafe_allow_html=True)
        new_col_name = st.text_input("New column name:", placeholder="Tax_Amount")
        formula = st.text_area("Formula:", placeholder="Price * 0.14", height=80)
        st.caption("Available columns: " + ", ".join([f"`{c}`" for c in df.columns]))

        if st.button("➕ Add Computed Column"):
            if new_col_name and formula:
                save_history()
                try:
                    result = safe_eval_formula(df, formula)
                    st.session_state.df[new_col_name] = result
                    st.success(f"Column '{new_col_name}' added!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Formula error: {e}")


# ════════════════════════════════════════════════════════
#  PAGE: VISUALIZE
# ════════════════════════════════════════════════════════
