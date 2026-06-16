# ════════════════════════════════════════════════════════
#  DataBridge AI — Page: Feature Engineering
# ════════════════════════════════════════════════════════
import plotly.express as px
import pandas as pd
import streamlit as st
from core.session import save_history
from ui.cards import section_header

def render(df) -> None:
    st.markdown(section_header("🔧", "Feature Engineering", "Prepare data for ML models"), unsafe_allow_html=True)
    from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler

    fe_tab1, fe_tab2, fe_tab3, fe_tab4, fe_tab5 = st.tabs([
        "  📅 Date → Columns  ",
        "  🔤 Encoding  ",
        "  📏 Scaling  ",
        "  ✂️ Binning  ",
        "  🔍 Text Extraction  ",
    ])

    # ── TAB 1: Date Extraction ──
    with fe_tab1:
        st.markdown('<div class="info-box">📅 Extract from a date column: day, month, year, day of week, quarter</div>', unsafe_allow_html=True)
        date_cols = [c for c in df.columns if 'date' in c.lower() or 'تاريخ' in c or pd.api.types.is_datetime64_any_dtype(df[c])]
        all_cols  = list(df.columns)
        date_col  = st.selectbox("Select the date column:", all_cols, index=all_cols.index(date_cols[0]) if date_cols else 0)

        parts = st.multiselect("Parts to extract:", ["year", "month", "day", "dayofweek", "quarter"], default=["year","month","day"])

        if st.button("✨ Extract", key="fe_date"):
            try:
                save_history()
                tmp = pd.to_datetime(df[date_col], errors='coerce')
                for p in parts:
                    df[f"{date_col}_{p}"] = getattr(tmp.dt, p)
                st.session_state.df = df
                st.success(f"✅ Added {len(parts)} new column(s)!")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")

    # ── TAB 2: Encoding ──
    with fe_tab2:
        st.markdown('<div class="info-box">🔤 Convert text columns to numbers so the model can understand them</div>', unsafe_allow_html=True)
        cat_cols_fe = df.select_dtypes(include='object').columns.tolist()
        if not cat_cols_fe:
            st.info("No text columns in this dataset.")
        else:
            enc_col    = st.selectbox("Select column:", cat_cols_fe, key="enc_col")
            enc_method = st.radio("Encoding type:", ["Label Encoding", "One-Hot Encoding"], horizontal=True)

            st.markdown(f"**Existing values:** {', '.join(df[enc_col].dropna().unique()[:10].astype(str))}")

            if st.button("✨ Apply Encoding", key="fe_enc"):
                save_history()
                if enc_method == "Label Encoding":
                    le = LabelEncoder()
                    df[f"{enc_col}_encoded"] = le.fit_transform(df[enc_col].astype(str))
                    st.session_state.df = df
                    st.success(f"✅ Added column `{enc_col}_encoded`")
                else:
                    dummies = pd.get_dummies(df[enc_col], prefix=enc_col)
                    df = pd.concat([df, dummies], axis=1)
                    st.session_state.df = df
                    st.success(f"✅ Added {len(dummies.columns)} column(s): {', '.join(dummies.columns.tolist())}")
                st.rerun()

    # ── TAB 3: Scaling ──
    with fe_tab3:
        st.markdown('<div class="info-box">📏 Scaling matters before models like SVM, KNN, and Logistic Regression</div>', unsafe_allow_html=True)
        num_cols_fe = df.select_dtypes(include='number').columns.tolist()
        if not num_cols_fe:
            st.info("No numeric columns.")
        else:
            scale_cols   = st.multiselect("Select columns:", num_cols_fe, default=num_cols_fe[:3])
            scale_method = st.radio("Scaling type:", ["Standard Scaler (Z-score)", "MinMax Scaler (0-1)"], horizontal=True)

            if scale_cols:
                # Preview
                preview_df = df[scale_cols].describe().round(3)
                st.markdown("**Before scaling:**")
                st.dataframe(preview_df, use_container_width=True, height=180)

            if st.button("✨ Apply Scaling", key="fe_scale") and scale_cols:
                save_history()
                scaler = StandardScaler() if "Standard" in scale_method else MinMaxScaler()
                df[scale_cols] = scaler.fit_transform(df[scale_cols])
                st.session_state.df = df
                st.success(f"✅ Applied {scale_method} to {len(scale_cols)} column(s)")
                st.rerun()

    # ── TAB 4: Binning ──
    with fe_tab4:
        st.markdown('<div class="info-box">✂️ Binning converts a numeric column into categories — e.g. age: young / middle-aged / senior</div>', unsafe_allow_html=True)
        num_cols_bin = df.select_dtypes(include='number').columns.tolist()
        if not num_cols_bin:
            st.info("No numeric columns.")
        else:
            bin_col   = st.selectbox("Select column:", num_cols_bin, key="bin_col")
            n_bins    = st.slider("Number of bins:", 2, 10, 3)
            bin_labels_input = st.text_input("Bin labels (optional):", placeholder="e.g. Low,Medium,High")

            col_min = float(df[bin_col].min())
            col_max = float(df[bin_col].max())
            st.markdown(f"**Column range:** {col_min:.2f} → {col_max:.2f}")

            if st.button("✨ Apply Binning", key="fe_bin"):
                save_history()
                labels = [l.strip() for l in bin_labels_input.split(',')] if bin_labels_input and len(bin_labels_input.split(',')) == n_bins else None
                df[f"{bin_col}_bin"] = pd.cut(df[bin_col], bins=n_bins, labels=labels)
                st.session_state.df = df
                dist = df[f"{bin_col}_bin"].value_counts().sort_index()
                st.success(f"✅ Created column `{bin_col}_bin`")
                st.dataframe(dist.reset_index().rename(columns={"index":"Bin", f"{bin_col}_bin":"Count"}), use_container_width=True)
                st.rerun()


    # ── TAB 5: Text Extraction ──
    with fe_tab5:
        st.markdown('<div class="info-box">🔍 Extract information from text columns — e.g. extracting a title from a name, like in Titanic</div>', unsafe_allow_html=True)

        cat_cols_tx = df.select_dtypes(include='object').columns.tolist()
        if not cat_cols_tx:
            st.warning("⚠️ No text columns in this dataset.")
        else:
            tx_col = st.selectbox("Select text column:", cat_cols_tx, key="tx_col")

            # Preview
            st.markdown(f"**Sample from column `{tx_col}`:**")
            st.write(df[tx_col].dropna().head(5).tolist())

            tx_method = st.radio("Extraction method:", [
                "🎩 Extract Title — Titanic style",
                "✂️ Split by separator",
                "🔎 Custom Regex",
                "🧹 Clean Text",
            ], key="tx_method")

            st.markdown("---")

            # ── Method 1: Title Extraction ──
            if tx_method == "🎩 Extract Title — Titanic style":
                st.markdown("""
                <div class="info-box">
                Extracts the title from a name — e.g.:<br>
                <code>Mr. John Smith</code> → <b>Mr</b><br>
                <code>Miss. Jane Doe</code> → <b>Miss</b><br>
                <code>Dr. James Brown</code> → <b>Dr</b>
                </div>
                """, unsafe_allow_html=True)

                new_col_name = st.text_input("New column name:", value=f"{tx_col}_title", key="tx_title_name")

                # Preview
                preview_titles = df[tx_col].str.extract(r' ([A-Za-z]+)\.', expand=False).dropna().value_counts().head(10)
                if not preview_titles.empty:
                    st.markdown("**Titles found in the data:**")
                    fig_t = px.bar(preview_titles.reset_index(), x='index', y=tx_col,
                                   template="plotly_dark", color=tx_col,
                                   color_continuous_scale="Purples",
                                   labels={"index": "Title", tx_col: "Count"})
                    fig_t.update_layout(paper_bgcolor='#0d0d1a', plot_bgcolor='#0d0d1a', showlegend=False)
                    st.plotly_chart(fig_t, use_container_width=True)

                group_rare = st.checkbox("Group rare titles into 'Rare'", value=True)
                rare_threshold = st.slider("Rarity threshold (below this):", 1, 20, 10) if group_rare else 0

                if st.button("✨ Extract Titles", key="tx_title_btn"):
                    save_history()
                    extracted = df[tx_col].str.extract(r' ([A-Za-z]+)\.', expand=False)
                    if group_rare:
                        counts = extracted.value_counts()
                        rare = counts[counts < rare_threshold].index
                        extracted = extracted.replace(rare, 'Rare')
                    df[new_col_name] = extracted
                    st.session_state.df = df
                    dist = df[new_col_name].value_counts()
                    st.success(f"✅ Created column `{new_col_name}` — {dist.nunique()} distinct title(s)")
                    st.dataframe(dist.reset_index().rename(columns={"index":"Title", new_col_name:"Count"}), use_container_width=True)
                    st.rerun()

            # ── Method 2: Split ──
            elif tx_method == "✂️ Split by separator":
                st.markdown('<div class="info-box">e.g.: <code>Smith, Mr. John</code> — split it by <code>,</code> and take the first or second part</div>', unsafe_allow_html=True)

                sep       = st.text_input("Separator:", value=",", key="tx_sep")
                part_idx  = st.number_input("Part index (0 = first part):", min_value=0, max_value=10, value=0, key="tx_part")
                new_col_s = st.text_input("New column name:", value=f"{tx_col}_part{part_idx}", key="tx_split_name")

                preview_split = df[tx_col].dropna().head(3).str.split(sep)
                st.markdown("**Preview:**")
                for orig, parts_list in zip(df[tx_col].dropna().head(3), preview_split):
                    parts_clean = [p.strip() for p in parts_list]
                    chosen = parts_clean[int(part_idx)] if int(part_idx) < len(parts_clean) else "—"
                    st.markdown(f'<span style="color:#555">{orig}</span> → <b style="color:#7c6aff">{chosen}</b>', unsafe_allow_html=True)

                if st.button("✨ Split", key="tx_split_btn"):
                    save_history()
                    df[new_col_s] = df[tx_col].str.split(sep).str[int(part_idx)].str.strip()
                    st.session_state.df = df
                    st.success(f"✅ Created column `{new_col_s}`")
                    st.rerun()

            # ── Method 3: Custom Regex ──
            elif tx_method == "🔎 Custom Regex":
                st.markdown('<div class="info-box">Write your own pattern — e.g.: ([A-Za-z]+) to extract the word before the dot</div>', unsafe_allow_html=True)

                regex_pattern = st.text_input("Regex Pattern:", value=r"([A-Za-z]+)\.", key="tx_regex")
                new_col_r     = st.text_input("New column name:", value=f"{tx_col}_regex", key="tx_regex_name")

                if regex_pattern:
                    try:
                        preview_regex = df[tx_col].str.extract(regex_pattern, expand=False).dropna().head(5).tolist()
                        st.markdown(f"**Preview:** {preview_regex}")
                    except Exception as e:
                        st.error(f"Regex error: {e}")

                if st.button("✨ Apply Regex", key="tx_regex_btn"):
                    try:
                        save_history()
                        df[new_col_r] = df[tx_col].str.extract(regex_pattern, expand=False)
                        st.session_state.df = df
                        st.success(f"✅ Created column `{new_col_r}`")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error: {e}")

            # ── Method 4: Text Cleaning ──
            elif tx_method == "🧹 Clean Text":
                st.markdown('<div class="info-box">Clean text before Encoding or NLP</div>', unsafe_allow_html=True)

                clean_ops = st.multiselect("Cleaning operations:", [
                    "Lowercase",
                    "Strip (remove whitespace)",
                    "Remove numbers",
                    "Remove special characters",
                    "Collapse repeated whitespace",
                ], default=["Lowercase", "Strip (remove whitespace)"])

                new_col_cl = st.text_input("New column name:", value=f"{tx_col}_clean", key="tx_clean_name")

                if st.button("✨ Clean", key="tx_clean_btn") and clean_ops:
                    save_history()
                    cleaned = df[tx_col].astype(str)
                    if "Lowercase" in clean_ops:
                        cleaned = cleaned.str.lower()
                    if "Strip (remove whitespace)" in clean_ops:
                        cleaned = cleaned.str.strip()
                    if "Remove numbers" in clean_ops:
                        cleaned = cleaned.str.replace(r'\d+', '', regex=True)
                    if "Remove special characters" in clean_ops:
                        cleaned = cleaned.str.replace(r'[^\w\s]', '', regex=True)
                    if "Collapse repeated whitespace" in clean_ops:
                        cleaned = cleaned.str.replace(r'\s+', ' ', regex=True).str.strip()
                    df[new_col_cl] = cleaned
                    st.session_state.df = df
                    st.success(f"✅ Created column `{new_col_cl}`")
                    st.rerun()


# ════════════════════════════════════════════════════════
#  PAGE: ML STUDIO
# ════════════════════════════════════════════════════════
