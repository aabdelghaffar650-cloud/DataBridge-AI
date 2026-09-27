# ════════════════════════════════════════════════════════
#  DataBridge AI — Page: Export Engine
# ════════════════════════════════════════════════════════
import io
from datetime import datetime
import pandas as pd
import streamlit as st
from ui.streamlit_compat import safe_dataframe
from core.security import safe_html
from ui.cards import section_header

def render(df) -> None:
    st.markdown(section_header("💾", "Export Cleaned Dataset"), unsafe_allow_html=True)


    st.markdown(f"""
    <div class="metrics-row" style="grid-template-columns: repeat(3, 1fr);">
      <div class="metric-card">
        <div class="label">Rows</div>
        <div class="value">{df.shape[0]:,}</div>
        <div class="sub">ready to export</div>
      </div>
      <div class="metric-card">
        <div class="label">Columns</div>
        <div class="value">{df.shape[1]}</div>
        <div class="sub">fields</div>
      </div>
      <div class="metric-card">
        <div class="label">Remaining Nulls</div>
        <div class="value">{int(df.isnull().sum().sum()):,}</div>
        <div class="sub">missing values</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    safe_dataframe(df.head(10), width="stretch")

    ex1, ex2, ex3 = st.columns(3)
    with ex1:
        csv_data = df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 Download CSV", csv_data, "smartmood_cleaned.csv", "text/csv", width="stretch")

    with ex2:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Cleaned')
        st.download_button("📥 Download Excel", buf.getvalue(), "smartmood_cleaned.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch")

    with ex3:
        json_data = df.to_json(orient='records', force_ascii=False).encode('utf-8')
        st.download_button("📥 Download JSON", json_data, "smartmood_cleaned.json", "application/json", width="stretch")

    st.markdown("---")
    st.subheader("Cleaning Summary")
    history_manager = st.session_state.history_manager
    st.markdown(f"""
    <div class="info-box">
    📋 File: <b>{safe_html(st.session_state.file_name or "Untitled")}</b><br>
    🔢 Shape: {df.shape[0]:,} rows × {df.shape[1]} columns<br>
    ↩️ Undo steps available: {history_manager.undo_count}<br>
    ↪️ Redo steps available: {history_manager.redo_count}<br>
    ⏱️ Exported at: {datetime.now().strftime('%Y-%m-%d %H:%M')}
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
#  PAGE: AI ASSISTANT
# ════════════════════════════════════════════════════════
