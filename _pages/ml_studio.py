# ════════════════════════════════════════════════════════
#  DataBridge AI — Page: Ml Studio
# ════════════════════════════════════════════════════════
import pickle

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from core.session import save_history
from ui.cards import section_header


def render(df) -> None:
<<<<<<< HEAD
    st.markdown(section_header("🧠", "ML Studio", "Model training"), unsafe_allow_html=True)
=======
    st.markdown(section_header("🧠", "ML Studio", "تدريب النماذج"), unsafe_allow_html=True)
>>>>>>> c8e725118d9c65808b1a67b5349827ad4e22458d

    try:
        from xgboost import XGBClassifier, XGBRegressor
        xgb_available = True
    except ImportError:
        xgb_available = False
<<<<<<< HEAD
        st.markdown('<div class="info-box">💡 XGBoost is not installed — run: <code>pip install xgboost</code></div>', unsafe_allow_html=True)
=======
        st.markdown('<div class="info-box">💡 XGBoost مش موجود — شغّل: <code>pip install xgboost</code></div>', unsafe_allow_html=True)
>>>>>>> c8e725118d9c65808b1a67b5349827ad4e22458d

    ml_tab1, ml_tab2, ml_tab3 = st.tabs(["  🎯 Classification  ", "  📈 Regression  ", "  🔵 Clustering  "])

    # ════════════════════
    #  CLASSIFICATION
    # ════════════════════
    with ml_tab1:
<<<<<<< HEAD
        st.markdown('<div class="info-box">🎯 Classification — use this when your target has categories (e.g.: pass/fail, alive/dead, positive/negative)</div>', unsafe_allow_html=True)
=======
        st.markdown('<div class="info-box">🎯 Classification — لما الـ target عندك فئات (مثلاً: نجح/رسب، حي/ميت، إيجابي/سلبي)</div>', unsafe_allow_html=True)
>>>>>>> c8e725118d9c65808b1a67b5349827ad4e22458d

        num_df_c = df.select_dtypes(include='number')
        all_cols_c = list(df.columns)

        if len(num_df_c.columns) < 2:
<<<<<<< HEAD
            st.warning("⚠️ You need at least two numeric columns. Try the Feature Engineering page first.")
        else:
            target_c = st.selectbox("🎯 Target Column (what you want to predict):", all_cols_c, key="clf_target")
=======
            st.warning("⚠️ محتاج على الأقل عمودين رقميين. جرّب صفحة Feature Engineering أول.")
        else:
            target_c = st.selectbox("🎯 Target Column (اللي عايز تتنبأ بيه):", all_cols_c, key="clf_target")
>>>>>>> c8e725118d9c65808b1a67b5349827ad4e22458d
            feature_options = [c for c in num_df_c.columns if c != target_c]
            features_c = st.multiselect("📊 Feature Columns:", feature_options, default=feature_options[:min(5, len(feature_options))], key="clf_features")

            clf_models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                "SVM": SVC(kernel='rbf', probability=True),
                "KNN": KNeighborsClassifier(n_neighbors=5),
            }
            if xgb_available:
                clf_models["XGBoost"] = XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss', verbosity=0)

<<<<<<< HEAD
            selected_clf = st.selectbox("🤖 Select model:", list(clf_models.keys()), key="clf_model")
            test_size_c  = st.slider("Test data size %:", 10, 40, 20, key="clf_split")

            if st.button("🚀 Train Model", key="clf_train") and features_c:
=======
            selected_clf = st.selectbox("🤖 اختار النموذج:", list(clf_models.keys()), key="clf_model")
            test_size_c  = st.slider("حجم بيانات الاختبار %:", 10, 40, 20, key="clf_split")

            if st.button("🚀 درّب النموذج", key="clf_train") and features_c:
>>>>>>> c8e725118d9c65808b1a67b5349827ad4e22458d
                try:
                    df_ml = df[features_c + [target_c]].dropna()
                    X = df_ml[features_c]

                    # Encode target if needed
                    y_raw = df_ml[target_c]
                    le_target = None
                    if y_raw.dtype == object:
                        le_target = LabelEncoder()
                        y = le_target.fit_transform(y_raw.astype(str))
                    else:
                        y = y_raw.values

                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size_c/100, random_state=42, stratify=y if len(set(y)) > 1 else None
                    )

                    model = clf_models[selected_clf]
<<<<<<< HEAD
                    with st.spinner(f"Training {selected_clf}..."):
=======
                    with st.spinner(f"جاري تدريب {selected_clf}..."):
>>>>>>> c8e725118d9c65808b1a67b5349827ad4e22458d
                        model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    acc = accuracy_score(y_test, y_pred)
                    st.session_state['last_clf_model'] = model
                    st.session_state['last_clf_features'] = features_c

                    # ── Results ──
                    m1, m2, m3 = st.columns(3)
                    m1.metric("✅ Accuracy", f"{acc*100:.1f}%")
                    m2.metric("📦 Train Size", f"{len(X_train):,}")
                    m3.metric("🧪 Test Size", f"{len(X_test):,}")

                    # Confusion Matrix
                    st.markdown("**Confusion Matrix:**")
                    cm = confusion_matrix(y_test, y_pred)
                    labels = le_target.classes_ if le_target else sorted(set(y))
                    fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                                       x=[str(l) for l in labels], y=[str(l) for l in labels],
                                       labels=dict(x="Predicted", y="Actual"), template="plotly_dark")
                    fig_cm.update_layout(paper_bgcolor='#0d0d1a', plot_bgcolor='#0d0d1a')
                    st.plotly_chart(fig_cm, use_container_width=True)

                    # Feature Importance
                    if hasattr(model, 'feature_importances_'):
                        fi = pd.DataFrame({'Feature': features_c, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=True)
                        fig_fi = px.bar(fi, x='Importance', y='Feature', orientation='h',
                                        color='Importance', color_continuous_scale='Purples',
                                        template="plotly_dark", title="Feature Importance")
                        fig_fi.update_layout(paper_bgcolor='#0d0d1a', plot_bgcolor='#0d0d1a')
                        st.plotly_chart(fig_fi, use_container_width=True)

                    # Classification Report
                    with st.expander("📋 Classification Report"):
                        report = classification_report(y_test, y_pred, target_names=[str(l) for l in labels], output_dict=True)
                        st.dataframe(pd.DataFrame(report).transpose().round(2), use_container_width=True)

                    # Download model
                    model_bytes = pickle.dumps(model)
<<<<<<< HEAD
                    st.download_button("💾 Download model (.pkl)", model_bytes, f"{selected_clf.replace(' ','_')}_model.pkl")

                except Exception as e:
                    st.error(f"❌ Error: {e}")
=======
                    st.download_button("💾 تحميل النموذج (.pkl)", model_bytes, f"{selected_clf.replace(' ','_')}_model.pkl")

                except Exception as e:
                    st.error(f"❌ خطأ: {e}")
>>>>>>> c8e725118d9c65808b1a67b5349827ad4e22458d

    # ════════════════════
    #  REGRESSION
    # ════════════════════
    with ml_tab2:
<<<<<<< HEAD
        st.markdown('<div class="info-box">📈 Regression — use this when your target is a continuous number (e.g.: price, score, age)</div>', unsafe_allow_html=True)
=======
        st.markdown('<div class="info-box">📈 Regression — لما الـ target عندك رقم مستمر (مثلاً: السعر، الدرجة، العمر)</div>', unsafe_allow_html=True)
>>>>>>> c8e725118d9c65808b1a67b5349827ad4e22458d

        num_df_r = df.select_dtypes(include='number')

        if len(num_df_r.columns) < 2:
<<<<<<< HEAD
            st.warning("⚠️ You need at least two numeric columns.")
=======
            st.warning("⚠️ محتاج على الأقل عمودين رقميين.")
>>>>>>> c8e725118d9c65808b1a67b5349827ad4e22458d
        else:
            target_r   = st.selectbox("🎯 Target Column:", list(num_df_r.columns), key="reg_target")
            features_r = st.multiselect("📊 Feature Columns:", [c for c in num_df_r.columns if c != target_r],
                                         default=[c for c in num_df_r.columns if c != target_r][:min(5, len(num_df_r.columns)-1)], key="reg_features")

            reg_models = {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            }
            if xgb_available:
                reg_models["XGBoost"] = XGBRegressor(n_estimators=100, random_state=42, verbosity=0)

<<<<<<< HEAD
            selected_reg  = st.selectbox("🤖 Select model:", list(reg_models.keys()), key="reg_model")
            test_size_r   = st.slider("Test data size %:", 10, 40, 20, key="reg_split")

            if st.button("🚀 Train Model", key="reg_train") and features_r:
=======
            selected_reg  = st.selectbox("🤖 اختار النموذج:", list(reg_models.keys()), key="reg_model")
            test_size_r   = st.slider("حجم بيانات الاختبار %:", 10, 40, 20, key="reg_split")

            if st.button("🚀 درّب النموذج", key="reg_train") and features_r:
>>>>>>> c8e725118d9c65808b1a67b5349827ad4e22458d
                try:
                    df_ml_r = df[features_r + [target_r]].dropna()
                    X_r = df_ml_r[features_r]
                    y_r = df_ml_r[target_r]

                    X_tr, X_te, y_tr, y_te = train_test_split(X_r, y_r, test_size=test_size_r/100, random_state=42)

                    model_r = reg_models[selected_reg]
<<<<<<< HEAD
                    with st.spinner(f"Training {selected_reg}..."):
=======
                    with st.spinner(f"جاري تدريب {selected_reg}..."):
>>>>>>> c8e725118d9c65808b1a67b5349827ad4e22458d
                        model_r.fit(X_tr, y_tr)
                    y_pred_r = model_r.predict(X_te)

                    r2  = r2_score(y_te, y_pred_r)
                    mae = mean_absolute_error(y_te, y_pred_r)
                    rmse= np.sqrt(mean_squared_error(y_te, y_pred_r))

                    m1, m2, m3 = st.columns(3)
                    m1.metric("📊 R² Score", f"{r2:.3f}")
                    m2.metric("📉 MAE", f"{mae:.3f}")
                    m3.metric("📉 RMSE", f"{rmse:.3f}")

                    # Actual vs Predicted
                    fig_avp = px.scatter(x=y_te, y=y_pred_r, template="plotly_dark",
                                          labels={"x": "Actual", "y": "Predicted"},
                                          title="Actual vs Predicted",
                                          color_discrete_sequence=["#7c6aff"])
                    fig_avp.add_shape(type='line', x0=y_te.min(), y0=y_te.min(), x1=y_te.max(), y1=y_te.max(),
                                      line=dict(color='#ff6b6b', dash='dash'))
                    fig_avp.update_layout(paper_bgcolor='#0d0d1a', plot_bgcolor='#0d0d1a')
                    st.plotly_chart(fig_avp, use_container_width=True)

                    # Feature Importance
                    if hasattr(model_r, 'feature_importances_'):
                        fi_r = pd.DataFrame({'Feature': features_r, 'Importance': model_r.feature_importances_}).sort_values('Importance', ascending=True)
                        fig_fi_r = px.bar(fi_r, x='Importance', y='Feature', orientation='h',
                                           color='Importance', color_continuous_scale='Purples',
                                           template="plotly_dark", title="Feature Importance")
                        fig_fi_r.update_layout(paper_bgcolor='#0d0d1a', plot_bgcolor='#0d0d1a')
                        st.plotly_chart(fig_fi_r, use_container_width=True)

                    model_bytes_r = pickle.dumps(model_r)
<<<<<<< HEAD
                    st.download_button("💾 Download model (.pkl)", model_bytes_r, f"{selected_reg.replace(' ','_')}_model.pkl")

                except Exception as e:
                    st.error(f"❌ Error: {e}")
=======
                    st.download_button("💾 تحميل النموذج (.pkl)", model_bytes_r, f"{selected_reg.replace(' ','_')}_model.pkl")

                except Exception as e:
                    st.error(f"❌ خطأ: {e}")
>>>>>>> c8e725118d9c65808b1a67b5349827ad4e22458d

    # ════════════════════
    #  CLUSTERING
    # ════════════════════
    with ml_tab3:
<<<<<<< HEAD
        st.markdown('<div class="info-box">🔵 Clustering — group data into similar clusters without a defined target</div>', unsafe_allow_html=True)
=======
        st.markdown('<div class="info-box">🔵 Clustering — تجميع البيانات لمجموعات متشابهة بدون target محدد</div>', unsafe_allow_html=True)
>>>>>>> c8e725118d9c65808b1a67b5349827ad4e22458d

        num_df_cl = df.select_dtypes(include='number')

        if len(num_df_cl.columns) < 2:
<<<<<<< HEAD
            st.warning("⚠️ You need at least two numeric columns.")
        else:
            features_cl     = st.multiselect("📊 Feature Columns:", list(num_df_cl.columns),
                                               default=list(num_df_cl.columns)[:min(4, len(num_df_cl.columns))], key="cl_features")
            cluster_method  = st.radio("Algorithm:", ["K-Means", "DBSCAN"], horizontal=True)

            if cluster_method == "K-Means":
                n_clusters = st.slider("Number of clusters:", 2, 10, 3)
=======
            st.warning("⚠️ محتاج على الأقل عمودين رقميين.")
        else:
            features_cl     = st.multiselect("📊 Feature Columns:", list(num_df_cl.columns),
                                               default=list(num_df_cl.columns)[:min(4, len(num_df_cl.columns))], key="cl_features")
            cluster_method  = st.radio("الخوارزمية:", ["K-Means", "DBSCAN"], horizontal=True)

            if cluster_method == "K-Means":
                n_clusters = st.slider("عدد الـ Clusters:", 2, 10, 3)
>>>>>>> c8e725118d9c65808b1a67b5349827ad4e22458d
            else:
                eps_val    = st.slider("DBSCAN eps:", 0.1, 5.0, 0.5, step=0.1)
                min_samp   = st.slider("min_samples:", 2, 20, 5)

<<<<<<< HEAD
            if st.button("🚀 Run Clustering", key="cl_run") and features_cl:
=======
            if st.button("🚀 شغّل Clustering", key="cl_run") and features_cl:
>>>>>>> c8e725118d9c65808b1a67b5349827ad4e22458d
                try:
                    df_cl = df[features_cl].dropna()

                    if cluster_method == "K-Means":
                        model_cl = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    else:
                        model_cl = DBSCAN(eps=eps_val, min_samples=min_samp)

<<<<<<< HEAD
                    with st.spinner("Clustering..."):
=======
                    with st.spinner("جاري التجميع..."):
>>>>>>> c8e725118d9c65808b1a67b5349827ad4e22458d
                        labels_cl = model_cl.fit_predict(df_cl)

                    df_cl = df_cl.copy()
                    df_cl['Cluster'] = labels_cl.astype(str)

                    n_found = len(set(labels_cl)) - (1 if -1 in labels_cl else 0)
                    noise   = (labels_cl == -1).sum() if -1 in labels_cl else 0

                    m1, m2 = st.columns(2)
                    m1.metric("🔵 Clusters Found", n_found)
                    if noise > 0:
                        m2.metric("⚫ Noise Points", noise)

<<<<<<< HEAD
                    # 2D scatter using the first two columns
=======
                    # 2D scatter باستخدام أول عمودين
>>>>>>> c8e725118d9c65808b1a67b5349827ad4e22458d
                    x_col, y_col = features_cl[0], features_cl[1] if len(features_cl) > 1 else features_cl[0]
                    fig_cl = px.scatter(df_cl, x=x_col, y=y_col, color='Cluster',
                                         template="plotly_dark", title=f"{cluster_method} Clustering",
                                         color_discrete_sequence=px.colors.qualitative.Bold)
                    fig_cl.update_layout(paper_bgcolor='#0d0d1a', plot_bgcolor='#0d0d1a')
                    st.plotly_chart(fig_cl, use_container_width=True)

                    # Cluster sizes
<<<<<<< HEAD
                    st.markdown("**Size of each cluster:**")
=======
                    st.markdown("**حجم كل Cluster:**")
>>>>>>> c8e725118d9c65808b1a67b5349827ad4e22458d
                    cluster_counts = pd.Series(labels_cl).value_counts().sort_index().reset_index()
                    cluster_counts.columns = ['Cluster', 'Count']
                    cluster_counts['Cluster'] = cluster_counts['Cluster'].astype(str)
                    fig_bar_cl = px.bar(cluster_counts, x='Cluster', y='Count',
                                         color='Cluster', template="plotly_dark",
                                         color_discrete_sequence=px.colors.qualitative.Bold)
                    fig_bar_cl.update_layout(paper_bgcolor='#0d0d1a', plot_bgcolor='#0d0d1a', showlegend=False)
                    st.plotly_chart(fig_bar_cl, use_container_width=True)

<<<<<<< HEAD
                    # Add the cluster column to the dataset
                    if st.button("➕ Add Cluster to Dataset", key="cl_add"):
                        save_history()
                        df.loc[df_cl.index, 'Cluster'] = labels_cl.astype(str)
                        st.session_state.df = df
                        st.success("✅ Added a Cluster column to the dataset!")
                        st.rerun()

                except Exception as e:
                    st.error(f"❌ Error: {e}")
=======
                    # إضافة الـ cluster للـ dataset
                    if st.button("➕ أضف Cluster للـ Dataset", key="cl_add"):
                        save_history()
                        df.loc[df_cl.index, 'Cluster'] = labels_cl.astype(str)
                        st.session_state.df = df
                        st.success("✅ تم إضافة عمود Cluster للـ dataset!")
                        st.rerun()

                except Exception as e:
                    st.error(f"❌ خطأ: {e}")
>>>>>>> c8e725118d9c65808b1a67b5349827ad4e22458d
