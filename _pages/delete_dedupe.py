# ════════════════════════════════════════════════════════
#  DataBridge AI — Page: Delete & Deduplicate
#  Stage 5: atomic dataset mutations
# ════════════════════════════════════════════════════════
import pandas as pd
import streamlit as st
from ui.streamlit_compat import safe_dataframe

from core.dataset import apply_dataset_change, current_dataset_revision
from ui.cards import section_header


def render(df: pd.DataFrame) -> None:
    st.markdown(section_header("🗑️", "Delete & Deduplicate"), unsafe_allow_html=True)
    revision = current_dataset_revision()

    dc1, dc2 = st.columns(2)
    with dc1:
        st.subheader("Delete Column")
        col_del = st.selectbox("Column to delete:", df.columns, key="del_col")
        if st.button("❌ Delete Column"):
            try:
                result = apply_dataset_change(
                    f"Delete column {col_del}",
                    lambda working: working.drop(columns=[col_del]),
                    details={"column": str(col_del)},
                    expected_revision=revision,
                )
                st.success(f"Deleted '{col_del}'" if result.changed else "No change was made.")
                st.rerun()
            except Exception as exc:
                st.error(f"Delete blocked safely: {exc}")

        st.subheader("Delete Empty Columns")
        empty = [c for c in df.columns if df[c].isnull().all()]
        if empty:
            st.warning(f"Empty columns: {empty}")
            if st.button("🗑️ Delete All Empty Columns"):
                try:
                    result = apply_dataset_change(
                        "Delete fully empty columns",
                        lambda working: working.drop(columns=empty),
                        details={"columns": [str(c) for c in empty]},
                        expected_revision=revision,
                    )
                    st.success(f"Deleted {len(empty)} empty column(s)." if result.changed else "No change was made.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Delete blocked safely: {exc}")
        else:
            st.info("No fully-empty columns.")

    with dc2:
        st.subheader("Delete Row by Position")
        if df.empty:
            st.info("The dataset has no rows to delete.")
        else:
            row_pos = st.number_input(
                "Row position:", min_value=0, max_value=len(df) - 1, step=1
            )
            safe_dataframe(df.iloc[[int(row_pos)]], width="stretch")
            if st.button("❌ Delete Row"):
                position = int(row_pos)

                def drop_row(working: pd.DataFrame) -> pd.DataFrame:
                    return working.drop(working.index[position]).reset_index(drop=True)

                try:
                    result = apply_dataset_change(
                        f"Delete row at position {position}",
                        drop_row,
                        details={"row_position": position},
                        expected_revision=revision,
                    )
                    st.success(f"Deleted row at position {position}" if result.changed else "No change was made.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Delete blocked safely: {exc}")

    st.markdown("---")
    st.subheader("Duplicate Rows")
    dups = int(df.duplicated().sum())
    if dups > 0:
        st.markdown(
            f'<div class="info-box">Found <b style="color:#ff6b6b">{dups:,}</b> '
            f'duplicate rows ({round(dups / max(len(df), 1) * 100, 1)}% of dataset)</div>',
            unsafe_allow_html=True,
        )
        safe_dataframe(df[df.duplicated()].head(20), width="stretch")
        if st.button("🗑️ Remove All Duplicates"):
            before = len(df)
            try:
                result = apply_dataset_change(
                    "Remove duplicate rows",
                    lambda working: working.drop_duplicates().reset_index(drop=True),
                    details={"duplicates_detected": dups},
                    expected_revision=revision,
                )
                st.success(f"Removed {before - result.after_shape[0]:,} duplicate rows.")
                st.rerun()
            except Exception as exc:
                st.error(f"Deduplication blocked safely: {exc}")
    else:
        st.success("✓ No duplicate rows found!")

    st.markdown("---")
    st.subheader("Sort Dataset")
    s1, s2 = st.columns(2)
    with s1:
        sort_col = st.selectbox("Sort by:", df.columns)
    with s2:
        sort_dir = st.radio("Order:", ["Ascending", "Descending"], horizontal=True)
    if st.button("Sort"):
        try:
            result = apply_dataset_change(
                f"Sort by {sort_col} ({sort_dir.lower()})",
                lambda working: working.sort_values(
                    sort_col,
                    ascending=(sort_dir == "Ascending"),
                ).reset_index(drop=True),
                details={"column": str(sort_col), "order": sort_dir},
                expected_revision=revision,
            )
            st.success("Sorted!" if result.changed else "The dataset was already in this order.")
            st.rerun()
        except Exception as exc:
            st.error(f"Sort blocked safely: {exc}")
