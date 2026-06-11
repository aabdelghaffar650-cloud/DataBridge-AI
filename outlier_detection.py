# ════════════════════════════════════════════════════════
#  DataBridge AI — Page: Outlier Detection
# ════════════════════════════════════════════════════════
import plotly.graph_objects as go
import pandas as pd
import streamlit as st
from core.session import save_history
from core.utils import vectorized_outlier_detection
from ui.cards import section_header

def render(df) -> None:
    st.markdown(section_header("🎯", "Outlier Detection"), unsafe_allow_html=True)


    num_cols = df.select_dtypes(include='number').columns.tolist()
    if not num_cols:
        st.warning("No numeric columns found.")
    else:
        oc1, oc2 = st.columns(2)
        with oc1:
            out_col = st.selectbox("Column:", num_cols)
        with oc2:
            method = st.selectbox("Method:", ["IQR (Interquartile Range)", "Z-Score"])

        s = df[out_col].dropna()

        if method == "IQR (Interquartile Range)":
            Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
            IQR    = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            outlier_mask = vectorized_outlier_detection(df, out_col, method='iqr')
        else:
            threshold = st.slider("Z-score threshold:", 1.5, 4.0, 3.0, 0.1)
            outlier_mask = vectorized_outlier_detection(df, out_col, method='zscore', threshold=threshold)
            mean_v, std_v = s.mean(), s.std()
            lower, upper  = mean_v - threshold * std_v, mean_v + threshold * std_v

        n_out = outlier_mask.sum()
        st.markdown(f'<div class="info-box">Found <b style="color:#ff6b6b">{n_out}</b> outliers ({round(n_out/len(df)*100,2)}% of data) in column <code>{out_col}</code></div>', unsafe_allow_html=True)

        fig = go.Figure()
        fig.add_trace(go.Box(y=df[out_col], name=out_col,
                             marker_color="#7c6aff", line_color="#7c6aff"))
        fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)',
                          plot_bgcolor='rgba(13,13,26,1)', height=350)
        st.plotly_chart(fig, use_container_width=True)

        if n_out > 0:
            st.dataframe(df[outlier_mask][[out_col]].head(50), use_container_width=True)

            ob1, ob2, ob3 = st.columns(3)
            with ob1:
                if st.button("🗑️ Remove Outlier Rows"):
                    save_history()
                    st.session_state.df = df[~outlier_mask].reset_index(drop=True)
                    st.success(f"Removed {n_out} outlier rows.")
                    st.rerun()
            with ob2:
                if st.button("📌 Cap to Bounds"):
                    save_history()
                    st.session_state.df[out_col] = df[out_col].clip(lower=lower, upper=upper)
                    st.success("Outliers capped!")
                    st.rerun()
            with ob3:
                if st.button("🔁 Replace with Median"):
                    save_history()
                    med = df[out_col].median()
                    st.session_state.df.loc[outlier_mask, out_col] = med
                    st.success(f"Outliers replaced with median ({med:.2f})")
                    st.rerun()


# ════════════════════════════════════════════════════════
#  PAGE: DELETE & DEDUPE
# ════════════════════════════════════════════════════════
