# ════════════════════════════════════════════════════════
#  DataBridge AI — Page: Overview & Profile
# ════════════════════════════════════════════════════════
import pandas as pd
import streamlit as st

from ui.cards import section_header


def render(df: pd.DataFrame) -> None:
    st.markdown(
        section_header("🏠", "Overview & Profile", f"{df.shape[0]:,} rows"),
        unsafe_allow_html=True,
    )

    nulls_total = int(df.isnull().sum().sum())
    dups        = int(df.duplicated().sum())
    num_cols    = len(df.select_dtypes(include="number").columns)
    cat_cols    = len(df.select_dtypes(include="object").columns)
    null_pct    = round(nulls_total / max(df.shape[0] * df.shape[1], 1) * 100, 1)

    st.markdown(f"""
    <div class="metrics-row">
      <div class="metric-card"><div class="label">Rows</div><div class="value">{df.shape[0]:,}</div><div class="sub">records</div></div>
      <div class="metric-card"><div class="label">Columns</div><div class="value">{df.shape[1]}</div><div class="sub">fields</div></div>
      <div class="metric-card"><div class="label">Missing Values</div><div class="value">{nulls_total:,}</div><div class="sub">{null_pct}% of data</div></div>
      <div class="metric-card"><div class="label">Duplicates</div><div class="value">{dups:,}</div><div class="sub">duplicate rows</div></div>
      <div class="metric-card"><div class="label">Numeric / Text</div>
        <div class="value">{num_cols}<span style="color:#333;font-size:1rem"> / </span>{cat_cols}</div>
        <div class="sub">column types</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        section_header("👁", "Data Preview", "first 20 rows"),
        unsafe_allow_html=True,
    )
    st.dataframe(df.head(20), use_container_width=True, height=380)

    st.markdown(
        section_header("🔬", "Column Profiles", "all columns"),
        unsafe_allow_html=True,
    )
    profile_rows = []
    for col in df.columns:
        s        = df[col]
        n_null   = int(s.isnull().sum())
        n_unique = int(s.nunique())
        dtype    = str(s.dtype)
        sample   = str(s.dropna().iloc[0]) if not s.dropna().empty else "—"
        if pd.api.types.is_numeric_dtype(s):
            extra = f"min {s.min():.2g} · max {s.max():.2g} · mean {s.mean():.2g}"
        else:
            top   = s.value_counts().index[0] if n_unique > 0 else "—"
            extra = f"top: {str(top)[:25]}"
        profile_rows.append({
            "Column": col, "Type": dtype,
            "Non-Null": df.shape[0] - n_null, "Null": n_null,
            "Null %": f"{round(n_null / df.shape[0] * 100, 1)}%",
            "Unique": n_unique, "Sample": sample[:30], "Stats": extra,
        })
    st.dataframe(pd.DataFrame(profile_rows), use_container_width=True, height=350)
