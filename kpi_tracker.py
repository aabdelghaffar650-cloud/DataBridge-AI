# ════════════════════════════════════════════════════════
#  DataBridge AI — Page: KPI Tracker
# ════════════════════════════════════════════════════════
import pandas as pd
import plotly.express as px
import streamlit as st

from modules.kpi_tracker import compute_kpi, compute_achievement
from ui.cards import section_header, info_box


def render(df: pd.DataFrame) -> None:
    st.markdown(
        section_header("🎯", "Dynamic KPI & Target Tracker", "Generic Metrics Engine"),
        unsafe_allow_html=True,
    )

    num_cols = df.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        st.warning("⚠️ No numeric columns found. KPI tracking requires numeric data.")
        return

    st.markdown(
        info_box("📊 Select numeric columns as KPIs, set annual/quarterly targets, and track achievement automatically."),
        unsafe_allow_html=True,
    )

    selected_kpis = st.multiselect(
        "Select KPI columns:",
        num_cols,
        default=num_cols[:min(3, len(num_cols))],
        key="kpi_select",
    )

    kpi_targets = st.session_state.kpi_targets.copy()

    if not selected_kpis:
        return

    st.markdown("#### 🎯 Set Targets")
    for kpi_col in selected_kpis:
        with st.expander(f"📌 {kpi_col}", expanded=True):
            t1, t2, t3 = st.columns(3)
            with t1:
                annual_target = st.number_input(
                    "Annual Target:",
                    value=float(kpi_targets.get(kpi_col, {}).get("annual", df[kpi_col].sum() * 1.2)),
                    key=f"kpi_annual_{kpi_col}",
                    format="%.2f",
                )
            with t2:
                quarterly_target = st.number_input(
                    "Quarterly Target:",
                    value=float(kpi_targets.get(kpi_col, {}).get("quarterly", annual_target / 4)),
                    key=f"kpi_quarterly_{kpi_col}",
                    format="%.2f",
                )
            with t3:
                agg = st.selectbox(
                    "Aggregation:",
                    ["Sum", "Mean", "Max", "Min", "Count"],
                    key=f"kpi_agg_{kpi_col}",
                )

            current    = compute_kpi(df, kpi_col, agg)
            ann_pct    = compute_achievement(current, annual_target)
            qtr_pct    = compute_achievement(current, quarterly_target)
            pct_color  = "#6bff8e" if ann_pct >= 100 else "#ffb86b" if ann_pct >= 70 else "#ff6b6b"
            status_lbl = "✅ Met" if ann_pct >= 100 else "⚠️ On Track" if ann_pct >= 70 else "🔴 Below target"

            st.markdown(f"""
            <div class="metrics-row" style="grid-template-columns:repeat(4,1fr);margin-top:.5rem;">
              <div class="metric-card"><div class="label">Current ({agg})</div>
                <div class="value">{current:,.2f}</div><div class="sub">actual value</div></div>
              <div class="metric-card"><div class="label">Annual Target</div>
                <div class="value">{annual_target:,.2f}</div><div class="sub">set target</div></div>
              <div class="metric-card"><div class="label">Annual Achievement</div>
                <div class="value" style="color:{pct_color}">{ann_pct}%</div>
                <div class="sub">{status_lbl}</div></div>
              <div class="metric-card"><div class="label">Quarterly Achievement</div>
                <div class="value" style="color:{pct_color}">{qtr_pct}%</div>
                <div class="sub">vs quarterly target</div></div>
            </div>
            """, unsafe_allow_html=True)

            # Trend line if date column exists
            date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
            if date_cols:
                date_c   = date_cols[0]
                trend_df = df[[date_c, kpi_col]].dropna().sort_values(date_c)
                if len(trend_df) > 1:
                    fig = px.line(
                        trend_df, x=date_c, y=kpi_col,
                        template="plotly_dark", title=f"{kpi_col} Trend",
                        color_discrete_sequence=["#7c6aff"],
                    )
                    fig.add_hline(
                        y=annual_target, line_dash="dash",
                        line_color="#6bff8e", annotation_text="Annual Target",
                    )
                    fig.update_layout(paper_bgcolor="#0d0d1a", plot_bgcolor="#0d0d1a", height=280)
                    st.plotly_chart(fig, use_container_width=True)

            kpi_targets[kpi_col] = {
                "annual":    annual_target,
                "quarterly": quarterly_target,
                "agg":       agg,
            }

    if st.button("💾 Save KPI Targets", key="kpi_save"):
        st.session_state.kpi_targets = kpi_targets
        st.success("✅ KPI Targets saved!")
