# ════════════════════════════════════════════════════════
#  DataBridge AI — Page: Outlier Detection
#  Stage 5: atomic dataset mutations
# ════════════════════════════════════════════════════════
import plotly.graph_objects as go
import pandas as pd
import streamlit as st
from ui.streamlit_compat import safe_dataframe

from core.dataset import apply_dataset_change, current_dataset_revision
from core.security import safe_error_message, safe_html
from core.utils import vectorized_outlier_detection
from ui.cards import section_header


def render(df: pd.DataFrame) -> None:
    st.markdown(section_header("🎯", "Outlier Detection"), unsafe_allow_html=True)
    revision = current_dataset_revision()

    num_cols = df.select_dtypes(include="number").columns.tolist()
    if not num_cols:
        st.warning("No numeric columns found.")
        return

    oc1, oc2 = st.columns(2)
    with oc1:
        out_col = st.selectbox("Column:", num_cols)
    with oc2:
        method = st.selectbox("Method:", ["IQR (Interquartile Range)", "Z-Score"])

    series = df[out_col].dropna()
    if series.empty:
        st.info("The selected column contains no numeric values to analyse.")
        return

    if method == "IQR (Interquartile Range)":
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_mask = vectorized_outlier_detection(df, out_col, method="iqr")
        method_key = "iqr"
    else:
        threshold = st.slider("Z-score threshold:", 1.5, 4.0, 3.0, 0.1)
        outlier_mask = vectorized_outlier_detection(
            df, out_col, method="zscore", threshold=threshold
        )
        mean_value, std_value = series.mean(), series.std()
        lower, upper = mean_value - threshold * std_value, mean_value + threshold * std_value
        method_key = "zscore"

    n_out = int(outlier_mask.sum())
    st.markdown(
        f'<div class="info-box">Found <b style="color:#ff6b6b">{n_out}</b> outliers '
        f'({round(n_out / max(len(df), 1) * 100, 2)}% of data) in column '
        f'<code>{safe_html(out_col)}</code></div>',
        unsafe_allow_html=True,
    )

    fig = go.Figure()
    fig.add_trace(
        go.Box(
            y=df[out_col],
            name=out_col,
            marker_color="#7c6aff",
            line_color="#7c6aff",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,13,26,1)",
        height=350,
    )
    st.plotly_chart(fig, width="stretch")

    if n_out <= 0:
        return

    safe_dataframe(df[outlier_mask][[out_col]].head(50), width="stretch")
    outlier_index = outlier_mask[outlier_mask].index

    ob1, ob2, ob3 = st.columns(3)
    with ob1:
        if st.button("🗑️ Remove Outlier Rows"):
            try:
                result = apply_dataset_change(
                    f"Remove outlier rows from {out_col}",
                    lambda working: working.drop(index=outlier_index).reset_index(drop=True),
                    details={"column": str(out_col), "method": method_key, "count": n_out},
                    expected_revision=revision,
                )
                st.success(f"Removed {result.before_shape[0] - result.after_shape[0]} outlier rows.")
                st.rerun()
            except Exception as exc:
                st.error(f"Outlier repair blocked safely: {safe_error_message(exc)}")
    with ob2:
        if st.button("📌 Cap to Bounds"):
            def cap_bounds(working: pd.DataFrame) -> pd.DataFrame:
                working[out_col] = working[out_col].clip(lower=lower, upper=upper)
                return working

            try:
                result = apply_dataset_change(
                    f"Cap outliers in {out_col}",
                    cap_bounds,
                    details={
                        "column": str(out_col),
                        "method": method_key,
                        "lower": float(lower),
                        "upper": float(upper),
                    },
                    expected_revision=revision,
                )
                st.success("Outliers capped!" if result.changed else "No values changed.")
                st.rerun()
            except Exception as exc:
                st.error(f"Outlier repair blocked safely: {safe_error_message(exc)}")
    with ob3:
        if st.button("🔁 Replace with Median"):
            median = df[out_col].median()

            def replace_median(working: pd.DataFrame) -> pd.DataFrame:
                working.loc[outlier_index, out_col] = median
                return working

            try:
                result = apply_dataset_change(
                    f"Replace outliers in {out_col} with median",
                    replace_median,
                    details={"column": str(out_col), "method": method_key, "median": float(median)},
                    expected_revision=revision,
                )
                st.success(
                    f"Outliers replaced with median ({median:.2f})"
                    if result.changed else "No values changed."
                )
                st.rerun()
            except Exception as exc:
                st.error(f"Outlier repair blocked safely: {safe_error_message(exc)}")
