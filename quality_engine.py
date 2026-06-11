# ════════════════════════════════════════════════════════
#  DataBridge AI — Page: Quality Engine
# ════════════════════════════════════════════════════════
import pandas as pd
import streamlit as st

from modules.quality_engine import run_quality_engine
from core.utils import vectorized_fill_nulls
from core.session import save_history
from ui.cards import section_header


def render(df: pd.DataFrame) -> None:
    st.markdown(
        section_header("🛡️", "Data Quality Engine & Repair Center", "Generic Health Scanner"),
        unsafe_allow_html=True,
    )

    qr = st.session_state.quality_report
    if not qr:
        qr = run_quality_engine(df)
        st.session_state.quality_report = qr

    qs       = qr.get("quality_score", 100)
    qs_color = "#6bff8e" if qs >= 85 else "#ffb86b" if qs >= 60 else "#ff6b6b"

    _type_err_count = sum(qr.get("type_errors", {}).values())
    _total_cells    = qr.get("total_cells", 0)
    _total_errors   = qr.get("total_errors", 0)
    _dup_count      = qr.get("duplicate_count", 0)
    _null_count     = qr.get("total_nulls", 0)

    st.markdown(f"""
    <div class="quality-score-banner" style="border-left:4px solid {qs_color};">
      <div>
        <div style="font-size:.72rem;color:#555;text-transform:uppercase;letter-spacing:.08em;">Dataset Quality Score</div>
        <div style="font-size:3rem;font-weight:700;color:{qs_color};font-family:'JetBrains Mono',monospace;">{qs}%</div>
      </div>
      <div style="flex:1;">
        <div style="font-size:.82rem;color:#888;margin-bottom:.3rem;">
          Total cells: <b style="color:#e0e0f0">{_total_cells:,}</b> &nbsp;·&nbsp;
          Errors found: <b style="color:#ff6b6b">{_total_errors:,}</b>
        </div>
        <div style="font-size:.82rem;color:#888;">
          Duplicates: <b>{_dup_count}</b> &nbsp;·&nbsp;
          Nulls: <b>{_null_count:,}</b> &nbsp;·&nbsp;
          Type errors: <b>{_type_err_count}</b>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    q_tab1, q_tab2, q_tab3, q_tab4 = st.tabs([
        "  🔴 Duplicates  ",
        "  🟡 Missing Values  ",
        "  🟠 Type Errors  ",
        "  📅 Date Errors  ",
    ])

    # ── TAB 1 ─────────────────────────────────────────────
    with q_tab1:
        if _dup_count == 0:
            st.success("✅ No duplicate records found!")
        else:
            st.markdown(
                f'<div class="info-box">🔴 <b>{_dup_count:,}</b> duplicate rows '
                f'({round(_dup_count / len(df) * 100, 1)}% of dataset)</div>',
                unsafe_allow_html=True,
            )
            dup_rows = qr.get("duplicate_rows", pd.DataFrame())
            if not dup_rows.empty:
                st.dataframe(dup_rows.head(30), use_container_width=True, height=300)

            if st.button("✅ Remove All Duplicate Rows", key="q_dedup"):
                save_history()
                before = len(st.session_state.df)
                st.session_state.df = df.drop_duplicates().reset_index(drop=True)
                st.session_state.quality_report = run_quality_engine(st.session_state.df)
                st.success(f"Removed {before - len(st.session_state.df):,} duplicates.")
                st.rerun()

    # ── TAB 2 ─────────────────────────────────────────────
    with q_tab2:
        null_by_col = qr.get("null_by_col", {})
        if not null_by_col:
            st.success("✅ No missing values found!")
        else:
            null_display = pd.DataFrame([
                {
                    "Column":        c,
                    "Missing Count": v,
                    "Missing %":     round(v / len(df) * 100, 1),
                    "Severity":      "🔴 High" if v / len(df) > 0.3 else "🟡 Medium" if v / len(df) > 0.1 else "🟢 Low",
                }
                for c, v in null_by_col.items()
            ]).sort_values("Missing Count", ascending=False)
            st.dataframe(null_display, use_container_width=True)

            st.markdown("**🛠 Batch Repair:**")
            rc1, rc2 = st.columns(2)
            with rc1:
                strategy = st.selectbox(
                    "Fill strategy:",
                    ["mean", "median", "mode", "forward", "backward", "custom"],
                    key="q_fill_strat",
                )
                custom_fill = ""
                if strategy == "custom":
                    custom_fill = st.text_input("Custom fill value:", "0", key="q_custom_fill")
            with rc2:
                repair_cols = st.multiselect(
                    "Apply to columns:",
                    list(null_by_col.keys()),
                    default=list(null_by_col.keys()),
                    key="q_fill_cols",
                )

            if st.button("🔧 Apply Batch Fill", key="q_apply_fill"):
                save_history()
                if strategy == "custom":
                    try:
                        cv = float(custom_fill)
                    except Exception:
                        cv = custom_fill
                    for col in repair_cols:
                        st.session_state.df[col] = st.session_state.df[col].fillna(cv)
                else:
                    st.session_state.df = vectorized_fill_nulls(
                        st.session_state.df, strategy, repair_cols
                    )
                st.session_state.quality_report = run_quality_engine(st.session_state.df)
                st.success("✅ Batch fill applied!")
                st.rerun()

    # ── TAB 3 ─────────────────────────────────────────────
    with q_tab3:
        type_errors = qr.get("type_errors", {})
        if not type_errors:
            st.success("✅ No type errors detected!")
        else:
            st.markdown(
                f'<div class="info-box">🟠 Found type inconsistencies in '
                f'<b>{len(type_errors)}</b> columns.</div>',
                unsafe_allow_html=True,
            )
            te_df = pd.DataFrame([
                {"Column": c, "Non-numeric values": v}
                for c, v in type_errors.items()
            ])
            st.dataframe(te_df, use_container_width=True)

            if st.button("🔧 Coerce All to Numeric (errors → NaN)", key="q_coerce"):
                save_history()
                for col in type_errors:
                    st.session_state.df[col] = pd.to_numeric(
                        st.session_state.df[col], errors="coerce"
                    )
                st.session_state.quality_report = run_quality_engine(st.session_state.df)
                st.success("✅ Type coercion applied!")
                st.rerun()

    # ── TAB 4 ─────────────────────────────────────────────
    with q_tab4:
        date_errors = qr.get("date_errors", {})
        if not date_errors:
            st.success("✅ No date anomalies detected!")
        else:
            for col, issues in date_errors.items():
                st.markdown(f"**Column `{col}`:**")
                if "future_dates" in issues:
                    st.warning(f"⚠️ {issues['future_dates']} future dates found.")
