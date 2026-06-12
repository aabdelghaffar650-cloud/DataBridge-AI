# ════════════════════════════════════════════════════════
#  DataBridge AI — Page: Clean & Fix Nulls
# ════════════════════════════════════════════════════════
import pandas as pd
import streamlit as st

from core.session import save_history
from ui.cards     import section_header


def render(df: pd.DataFrame) -> None:
    st.markdown(section_header("🧹", "Clean & Fix Missing Values"), unsafe_allow_html=True)

    null_info = df.isnull().sum()
    null_df   = pd.DataFrame({"Column": null_info.index, "Missing": null_info.values,
                               "Pct": (null_info.values / len(df) * 100).round(1)})
    null_df   = null_df[null_df["Missing"] > 0].sort_values("Missing", ascending=False)

    if null_df.empty:
        st.success("✓ No missing values detected!")
    else:
        st.dataframe(null_df, use_container_width=True, height=250)

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Quick Fix — Fake Nulls")
            st.markdown('<div class="info-box">Converts text like None, null, N/A, -, ? to real NaN.</div>', unsafe_allow_html=True)
            if st.button("✨ Convert Fake Nulls to NaN"):
                save_history()
                fake = [r"^\s*$", "None", "none", "null", "Null", "NaN", "nan", "NA", "N/A", "n/a", "-", "--", "?", "missing", "MISSING"]
                st.session_state.df = st.session_state.df.replace(fake, None, regex=True)
                st.success("Done!")
                st.rerun()

        with col_b:
            st.subheader("Global Actions")
            null_action = st.radio("Action:", ["Drop rows with nulls", "Fill all nulls with value", "Fill numeric with mean", "Fill numeric with median"])
            if null_action == "Drop rows with nulls":
                if st.button("Apply Drop"):
                    save_history()
                    before = len(st.session_state.df)
                    st.session_state.df = st.session_state.df.dropna()
                    st.success(f"Dropped {before - len(st.session_state.df):,} rows.")
                    st.rerun()
            elif null_action == "Fill all nulls with value":
                fv = st.text_input("Fill value:")
                if st.button("Apply Fill") and fv:
                    save_history()
                    try:    fv_cast = float(fv) if "." in fv else int(fv)
                    except: fv_cast = fv
                    st.session_state.df = st.session_state.df.fillna(fv_cast)
                    st.success("Done!"); st.rerun()
            elif null_action == "Fill numeric with mean":
                if st.button("Apply Mean Fill"):
                    save_history()
                    for c in st.session_state.df.select_dtypes(include="number").columns:
                        st.session_state.df[c] = st.session_state.df[c].fillna(st.session_state.df[c].mean())
                    st.success("Done!"); st.rerun()
            elif null_action == "Fill numeric with median":
                if st.button("Apply Median Fill"):
                    save_history()
                    for c in st.session_state.df.select_dtypes(include="number").columns:
                        st.session_state.df[c] = st.session_state.df[c].fillna(st.session_state.df[c].median())
                    st.success("Done!"); st.rerun()

    st.markdown("---")
    st.subheader("Per-Column Fix")
    pc1, pc2, pc3 = st.columns(3)
    with pc1: per_col    = st.selectbox("Column:", df.columns, key="per_col")
    with pc2: per_action = st.selectbox("Fill with:", ["Custom value", "Mean", "Median", "Mode", "Forward fill", "Backward fill"])
    with pc3:
        per_val = ""
        if per_action == "Custom value":
            per_val = st.text_input("Value:", key="per_val")

    if st.button("Apply to Column"):
        save_history()
        s = st.session_state.df[per_col]
        if per_action == "Custom value" and per_val:
            try:    v = float(per_val) if "." in per_val else int(per_val)
            except: v = per_val
            st.session_state.df[per_col] = s.fillna(v)
        elif per_action == "Mean":    st.session_state.df[per_col] = s.fillna(s.mean())
        elif per_action == "Median":  st.session_state.df[per_col] = s.fillna(s.median())
        elif per_action == "Mode":    st.session_state.df[per_col] = s.fillna(s.mode()[0] if not s.mode().empty else s)
        elif per_action == "Forward fill":  st.session_state.df[per_col] = s.ffill()
        elif per_action == "Backward fill": st.session_state.df[per_col] = s.bfill()
        st.success(f"Applied '{per_action}' to '{per_col}'!")
        st.rerun()
