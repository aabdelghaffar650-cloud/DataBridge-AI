# ════════════════════════════════════════════════════════
#  DataBridge AI — Page: Delete Dedupe
# ════════════════════════════════════════════════════════
import pandas as pd
import streamlit as st
from core.session import save_history
from ui.cards import section_header

def render(df) -> None:
    st.markdown(section_header("🗑️", "Delete & Deduplicate"), unsafe_allow_html=True)


    dc1, dc2 = st.columns(2)
    with dc1:
        st.subheader("Delete Column")
        col_del = st.selectbox("Column to delete:", df.columns, key="del_col")
        if st.button("❌ Delete Column"):
            save_history()
            st.session_state.df = df.drop(columns=[col_del])
            st.success(f"Deleted '{col_del}'")
            st.rerun()

        st.subheader("Delete Empty Columns")
        empty = [c for c in df.columns if df[c].isnull().all()]
        if empty:
            st.warning(f"Empty columns: {empty}")
            if st.button("🗑️ Delete All Empty Columns"):
                save_history()
                st.session_state.df = df.drop(columns=empty)
                st.rerun()
        else:
            st.info("No fully-empty columns.")

    with dc2:
        st.subheader("Delete Row by Index")
        row_idx = st.number_input("Row index:", min_value=0, max_value=len(df)-1, step=1)
        st.dataframe(df.iloc[[row_idx]], use_container_width=True)
        if st.button("❌ Delete Row"):
            save_history()
            st.session_state.df = df.drop(index=row_idx).reset_index(drop=True)
            st.success(f"Deleted row {row_idx}")
            st.rerun()

    st.markdown("---")
    st.subheader("Duplicate Rows")
    dups = df.duplicated().sum()
    if dups > 0:
        st.markdown(f'<div class="info-box">Found <b style="color:#ff6b6b">{dups:,}</b> duplicate rows ({round(dups/len(df)*100,1)}% of dataset)</div>', unsafe_allow_html=True)
        st.dataframe(df[df.duplicated()].head(20), use_container_width=True)
        if st.button("🗑️ Remove All Duplicates"):
            save_history()
            before = len(st.session_state.df)
            st.session_state.df = df.drop_duplicates().reset_index(drop=True)
            st.success(f"Removed {before - len(st.session_state.df):,} duplicate rows.")
            st.rerun()
    else:
        st.success("✓ No duplicate rows found!")

    st.markdown("---")
    st.subheader("Sort Dataset")
    s1, s2 = st.columns(2)
    with s1: sort_col = st.selectbox("Sort by:", df.columns)
    with s2: sort_dir = st.radio("Order:", ["Ascending", "Descending"], horizontal=True)
    if st.button("Sort"):
        save_history()
        st.session_state.df = df.sort_values(sort_col, ascending=(sort_dir == "Ascending")).reset_index(drop=True)
        st.success("Sorted!")
        st.rerun()


# ════════════════════════════════════════════════════════
#  PAGE: EXPORT
# ════════════════════════════════════════════════════════
