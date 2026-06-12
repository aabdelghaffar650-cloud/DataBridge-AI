# ════════════════════════════════════════════════════════
#  DataBridge AI — Page: Visualization
# ════════════════════════════════════════════════════════
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import streamlit as st
from ui.cards import section_header

def render(df) -> None:
    st.markdown(section_header("📊", "Data Visualization"), unsafe_allow_html=True)


    num_columns = df.select_dtypes(include='number').columns.tolist()
    all_columns = df.columns.tolist()

    chart_type = st.selectbox("Chart type:", [
        "Histogram", "Bar Chart", "Line Chart", "Scatter Plot",
        "Box Plot", "Pie Chart", "Heatmap (Correlation)", "Distribution (all numeric)"
    ])

    plotly_theme = "plotly_dark"

    if chart_type == "Histogram":
        col = st.selectbox("Column:", num_columns or all_columns)
        bins = st.slider("Bins:", 5, 100, 30)
        fig = px.histogram(df, x=col, nbins=bins, template=plotly_theme,
                           color_discrete_sequence=["#7c6aff"])
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(13,13,26,1)')
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Bar Chart":
        c1, c2 = st.columns(2)
        with c1: x_col = st.selectbox("X-axis (category):", all_columns)
        with c2: y_col = st.selectbox("Y-axis (value):", num_columns if num_columns else all_columns)
        top_n = st.slider("Show top N:", 5, 50, 20)
        agg = df.groupby(x_col)[y_col].sum().nlargest(top_n).reset_index()
        fig = px.bar(agg, x=x_col, y=y_col, template=plotly_theme,
                     color_discrete_sequence=["#7c6aff"])
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(13,13,26,1)')
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Line Chart":
        c1, c2 = st.columns(2)
        with c1: x_col = st.selectbox("X-axis:", all_columns)
        with c2: y_col = st.selectbox("Y-axis:", num_columns if num_columns else all_columns)
        fig = px.line(df, x=x_col, y=y_col, template=plotly_theme,
                      color_discrete_sequence=["#7c6aff"])
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(13,13,26,1)')
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Scatter Plot":
        c1, c2, c3 = st.columns(3)
        with c1: x_col = st.selectbox("X-axis:", num_columns)
        with c2: y_col = st.selectbox("Y-axis:", num_columns)
        with c3: color_col = st.selectbox("Color by (optional):", ["None"] + all_columns)
        color = None if color_col == "None" else color_col
        fig = px.scatter(df, x=x_col, y=y_col, color=color, template=plotly_theme,
                         color_discrete_sequence=["#7c6aff"])
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(13,13,26,1)')
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Box Plot":
        c1, c2 = st.columns(2)
        with c1: y_col = st.selectbox("Value column:", num_columns)
        with c2: grp = st.selectbox("Group by (optional):", ["None"] + all_columns)
        x = None if grp == "None" else grp
        fig = px.box(df, x=x, y=y_col, template=plotly_theme,
                     color_discrete_sequence=["#7c6aff"])
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(13,13,26,1)')
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Pie Chart":
        c1, c2 = st.columns(2)
        with c1: name_col = st.selectbox("Category column:", all_columns)
        with c2: val_col_pie = st.selectbox("Value column:", num_columns if num_columns else all_columns)
        top_pie = st.slider("Top N slices:", 3, 20, 8)
        pie_data = df.groupby(name_col)[val_col_pie].sum().nlargest(top_pie).reset_index()
        fig = px.pie(pie_data, names=name_col, values=val_col_pie, template=plotly_theme,
                     color_discrete_sequence=px.colors.sequential.Purples_r)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Heatmap (Correlation)":
        if len(num_columns) < 2:
            st.warning("Need at least 2 numeric columns.")
        else:
            corr = df[num_columns].corr()
            fig = px.imshow(corr, template=plotly_theme, color_continuous_scale="RdBu_r",
                            text_auto=".2f", aspect="auto")
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)

    elif chart_type == "Distribution (all numeric)":
        if not num_columns:
            st.warning("No numeric columns found.")
        else:
            cols_to_plot = st.multiselect("Select columns:", num_columns, default=num_columns[:4])
            if cols_to_plot:
                fig = go.Figure()
                colors = ["#7c6aff","#ff6b9d","#6bffb8","#ffb86b","#6bb8ff"]
                for i, c in enumerate(cols_to_plot):
                    fig.add_trace(go.Violin(y=df[c].dropna(), name=c,
                                            fillcolor=colors[i % len(colors)],
                                            line_color=colors[i % len(colors)],
                                            opacity=0.7))
                fig.update_layout(template=plotly_theme, paper_bgcolor='rgba(0,0,0,0)',
                                  plot_bgcolor='rgba(13,13,26,1)')
                st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════
#  PAGE: OUTLIER DETECTION
# ════════════════════════════════════════════════════════
