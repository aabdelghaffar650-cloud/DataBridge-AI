# ════════════════════════════════════════════════════════
#  DataBridge AI — Page: AI Assistant
# ════════════════════════════════════════════════════════
import html
import re
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st

from core.security import anonymise_df_for_ai
<<<<<<< HEAD
=======
from core.i18n import is_ar, t
>>>>>>> c8e725118d9c65808b1a67b5349827ad4e22458d
from ui.cards import section_header


# ── Dataset-aware helpers ────────────────────────────────────────────────────
def _date_columns(df: pd.DataFrame) -> List[str]:
    cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    for c in df.select_dtypes(include="object").columns:
        sample = df[c].dropna().head(40)
        if len(sample) and pd.to_datetime(sample, errors="coerce", dayfirst=True).notna().mean() >= 0.65:
            cols.append(c)
    return list(dict.fromkeys(cols))


def _categorical_columns(df: pd.DataFrame) -> List[str]:
    cats = []
    for c in df.columns:
        nunique = df[c].nunique(dropna=True)
        if df[c].dtype == "object" or nunique <= min(30, max(5, len(df) * 0.2)):
            cats.append(c)
    return cats


def _outlier_summary(df: pd.DataFrame) -> List[Dict[str, object]]:
    rows = []
    for col in df.select_dtypes(include="number").columns:
        s = df[col].dropna()
        if len(s) < 8:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        count = int(((df[col] < lower) | (df[col] > upper)).sum())
        if count:
            rows.append({"column": col, "count": count, "pct": round(count / len(df) * 100, 1)})
    return sorted(rows, key=lambda r: r["count"], reverse=True)


def _quality_snapshot(df: pd.DataFrame) -> Dict[str, object]:
    total_cells = max(df.shape[0] * df.shape[1], 1)
    nulls = df.isnull().sum()
    null_cols = nulls[nulls > 0].sort_values(ascending=False)
    duplicates = int(df.duplicated().sum())
    quality_report = st.session_state.get("quality_report", {}) or {}
    return {
        "rows": df.shape[0],
        "cols": df.shape[1],
        "null_total": int(nulls.sum()),
        "null_pct": round(int(nulls.sum()) / total_cells * 100, 1),
        "top_nulls": null_cols.head(7),
        "duplicates": duplicates,
        "duplicate_pct": round(duplicates / max(len(df), 1) * 100, 1),
        "quality_score": quality_report.get("quality_score", round((total_cells - int(nulls.sum()) - duplicates) / total_cells * 100, 1)),
        "type_errors": quality_report.get("type_errors", {}),
    }


def _risk_level(score: float) -> str:
    if score >= 85:
<<<<<<< HEAD
        return "Low"
    if score >= 60:
        return "Medium"
    return "High"
=======
        return "Low" if not is_ar() else "منخفض"
    if score >= 60:
        return "Medium" if not is_ar() else "متوسط"
    return "High" if not is_ar() else "مرتفع"
>>>>>>> c8e725118d9c65808b1a67b5349827ad4e22458d


def _build_suggestions(df: pd.DataFrame) -> List[Dict[str, str]]:
    """Create dynamic, dataset-aware suggestions instead of fixed generic questions."""
<<<<<<< HEAD
=======
    lang_ar = is_ar()
>>>>>>> c8e725118d9c65808b1a67b5349827ad4e22458d
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = _categorical_columns(df)
    date_cols = _date_columns(df)
    qs = _quality_snapshot(df)
    mappings = st.session_state.get("mapping_confidence", {}) or {}
    weak_mappings = [c for c, (_, score) in mappings.items() if score < 0.60]

    suggestions: List[Dict[str, str]] = []

<<<<<<< HEAD
    suggestions.append({
        "label": "🔎 Data quality diagnosis",
        "prompt": "Diagnose the current dataset quality and rank the top 5 risks by priority, impact, and recommended action.",
    })
    suggestions.append({
        "label": "🛠️ Safe cleaning plan",
        "prompt": "Suggest a safe cleaning plan without modifying the data. Identify columns requiring human approval before repair.",
    })
    if qs["null_total"]:
        suggestions.append({
            "label": "🟡 Critical missing values",
            "prompt": "Analyze missing values: which columns are actually risky, which can be ignored, and the best strategy for each.",
        })
    if qs["duplicates"]:
        suggestions.append({
            "label": "🔁 Duplicate risk review",
            "prompt": "Analyze duplicate rows and explain whether they look like true duplicates or valid repeated records.",
        })
    if num_cols:
        suggestions.append({
            "label": "🎯 KPI candidates",
            "prompt": "Recommend the best KPI candidates from numeric columns, aggregation methods, and where each KPI might mislead.",
        })
        suggestions.append({
            "label": "🚨 Outliers & anomalies",
            "prompt": "Find numeric outliers and unusual patterns, then rank them by likely impact on analysis.",
        })
    if cat_cols:
        suggestions.append({
            "label": "🧩 Category gaps",
            "prompt": "Compare key categories/regions/types and identify gaps, concentration, or imbalance in records.",
        })
    if date_cols:
        suggestions.append({
            "label": "📈 Time trend review",
            "prompt": "Analyze time trends and identify drops, spikes, or missing periods that need review.",
        })
    if weak_mappings:
        suggestions.append({
            "label": "🗺️ Mapping review",
            "prompt": "Review Auto Mapper decisions and identify low-confidence columns that need review before reporting.",
        })
    suggestions.append({
        "label": "📋 Management summary",
        "prompt": "Write a short management summary covering data status, risks, and the next 5 actions before dashboard/reporting.",
    })
=======
    if lang_ar:
        suggestions.append({
            "label": "🔎 تشخيص جودة البيانات",
            "prompt": "شخّص جودة البيانات الحالية، وحدد أخطر 5 مشاكل مرتبة حسب الأولوية مع سبب كل مشكلة وتأثيرها على التحليل.",
        })
        suggestions.append({
            "label": "🛠️ خطة تنظيف آمنة",
            "prompt": "اقترح خطة تنظيف آمنة لهذه البيانات بدون تنفيذ أي تعديل، واذكر الأعمدة التي تحتاج مراجعة بشرية قبل الإصلاح.",
        })
        if qs["null_total"]:
            suggestions.append({
                "label": "🟡 فراغات مؤثرة",
                "prompt": "حلل القيم الفارغة: أي أعمدة تمثل خطورة فعلية، وأي أعمدة يمكن تركها، وما أفضل استراتيجية لكل عمود؟",
            })
        if qs["duplicates"]:
            suggestions.append({
                "label": "🔁 تحليل التكرار",
                "prompt": "حلل الصفوف المكررة: هل تبدو تكرارات حقيقية أم سجلات متكررة منطقياً؟ وما القرار الآمن؟",
            })
        if num_cols:
            suggestions.append({
                "label": "🎯 مؤشرات KPI محتملة",
                "prompt": "اقترح أفضل مؤشرات KPI من الأعمدة الرقمية الموجودة، وحدد طريقة التجميع المناسبة لكل مؤشر ومتى يكون مضللاً.",
            })
            suggestions.append({
                "label": "🚨 قيم شاذة وأنماط غريبة",
                "prompt": "اكتشف القيم الشاذة والأنماط الرقمية غير الطبيعية، ورتبها حسب تأثيرها المحتمل على النتائج.",
            })
        if cat_cols:
            suggestions.append({
                "label": "🧩 فجوات بين الفئات",
                "prompt": "قارن الفئات والمناطق/الأنواع الموجودة في البيانات، وحدد أين توجد فجوات أو تركّز غير طبيعي في السجلات.",
            })
        if date_cols:
            suggestions.append({
                "label": "📈 اتجاه زمني",
                "prompt": "حلل الاتجاه الزمني في البيانات، وهل يوجد هبوط أو ارتفاع أو فترات مفقودة تحتاج مراجعة؟",
            })
        if weak_mappings:
            suggestions.append({
                "label": "🗺️ مراجعة الـ Mapping",
                "prompt": "راجع قرارات Auto Mapper، وحدد الأعمدة منخفضة الثقة التي يجب مراجعتها قبل أي تحليل أو تقرير.",
            })
        suggestions.append({
            "label": "📋 ملخص تقرير إداري",
            "prompt": "اكتب ملخصاً إدارياً قصيراً عن حالة البيانات، أهم المخاطر، وأفضل 5 خطوات تالية قبل بناء Dashboard أو تقرير مانح.",
        })
    else:
        suggestions.append({
            "label": "🔎 Data quality diagnosis",
            "prompt": "Diagnose the current dataset quality and rank the top 5 risks by priority, impact, and recommended action.",
        })
        suggestions.append({
            "label": "🛠️ Safe cleaning plan",
            "prompt": "Suggest a safe cleaning plan without modifying the data. Identify columns requiring human approval before repair.",
        })
        if qs["null_total"]:
            suggestions.append({
                "label": "🟡 Critical missing values",
                "prompt": "Analyze missing values: which columns are actually risky, which can be ignored, and the best strategy for each.",
            })
        if qs["duplicates"]:
            suggestions.append({
                "label": "🔁 Duplicate risk review",
                "prompt": "Analyze duplicate rows and explain whether they look like true duplicates or valid repeated records.",
            })
        if num_cols:
            suggestions.append({
                "label": "🎯 KPI candidates",
                "prompt": "Recommend the best KPI candidates from numeric columns, aggregation methods, and where each KPI might mislead.",
            })
            suggestions.append({
                "label": "🚨 Outliers & anomalies",
                "prompt": "Find numeric outliers and unusual patterns, then rank them by likely impact on analysis.",
            })
        if cat_cols:
            suggestions.append({
                "label": "🧩 Category gaps",
                "prompt": "Compare key categories/regions/types and identify gaps, concentration, or imbalance in records.",
            })
        if date_cols:
            suggestions.append({
                "label": "📈 Time trend review",
                "prompt": "Analyze time trends and identify drops, spikes, or missing periods that need review.",
            })
        if weak_mappings:
            suggestions.append({
                "label": "🗺️ Mapping review",
                "prompt": "Review Auto Mapper decisions and identify low-confidence columns that need review before reporting.",
            })
        suggestions.append({
            "label": "📋 Management summary",
            "prompt": "Write a short management summary covering data status, risks, and the next 5 actions before dashboard/reporting.",
        })
>>>>>>> c8e725118d9c65808b1a67b5349827ad4e22458d

    return suggestions[:10]


# ── Demo response engine ──────────────────────────────────────────────────────
def _demo_response(df: pd.DataFrame, question: str) -> str:
    q = question.lower().strip()
<<<<<<< HEAD
=======
    lang_ar = is_ar() or any("\u0600" <= ch <= "\u06FF" for ch in question)
>>>>>>> c8e725118d9c65808b1a67b5349827ad4e22458d

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = _categorical_columns(df)
    date_cols = _date_columns(df)
    qs = _quality_snapshot(df)
    outliers = _outlier_summary(df)
    mappings = st.session_state.get("mapping_confidence", {}) or {}
    weak_mappings = [(c, grp, score) for c, (grp, score) in mappings.items() if score < 0.60]

<<<<<<< HEAD
=======
    if lang_ar:
        if any(w in q for w in ["تشخيص", "quality", "جودة", "مخاطر", "risk"]):
            top_nulls = "\n".join([f"- **{c}**: {v:,} فراغ ({round(v/max(len(df),1)*100,1)}%)" for c, v in qs["top_nulls"].items()]) or "- لا توجد فراغات مؤثرة"
            type_errs = qs["type_errors"] or {}
            type_txt = "، ".join([f"{c}: {v}" for c, v in type_errs.items()]) or "لا توجد أخطاء نوع واضحة"
            return (
                f"## 🔎 تشخيص جودة البيانات\n\n"
                f"**درجة الجودة:** {qs['quality_score']}% — مستوى الخطر: **{_risk_level(float(qs['quality_score']))}**\n\n"
                f"**المشاكل الأساسية:**\n"
                f"- حجم البيانات: **{qs['rows']:,} صف × {qs['cols']} عمود**\n"
                f"- القيم الفارغة: **{qs['null_total']:,}** ({qs['null_pct']}%)\n"
                f"- الصفوف المكررة: **{qs['duplicates']:,}** ({qs['duplicate_pct']}%)\n"
                f"- أخطاء الأنواع: {type_txt}\n\n"
                f"**أكثر الأعمدة احتياجًا للمراجعة:**\n{top_nulls}\n\n"
                f"**الأولوية:** ابدأ بالأعمدة التي تؤثر على الهوية/التاريخ/KPI قبل أي Dashboard أو تقرير."
            )

        if any(w in q for w in ["تنظيف", "clean", "repair", "إصلاح"]):
            steps = []
            if qs["duplicates"]:
                steps.append(f"1. راجع **{qs['duplicates']:,}** صف مكرر قبل الحذف، خصوصًا لو البيانات تمثل زيارات أو معاملات متكررة.")
            if qs["null_total"]:
                top = list(qs["top_nulls"].index[:4])
                steps.append(f"2. عالج الفراغات في الأعمدة الأعلى تأثيرًا: **{', '.join(map(str, top))}**.")
            if weak_mappings:
                steps.append(f"3. راجع Mapping للأعمدة منخفضة الثقة وعددها **{len(weak_mappings)}** قبل التحليل.")
            if outliers:
                steps.append(f"4. افحص القيم الشاذة، خصوصًا في **{outliers[0]['column']}** ({outliers[0]['count']} حالة).")
            steps.append("5. لا تستبدل الفراغات بصفر إلا في أعمدة خدمية/كمية وبعد موافقة المستخدم.")
            return "## 🛠️ خطة تنظيف آمنة\n\n" + "\n".join(steps)

        if any(w in q for w in ["kpi", "مؤشر", "مؤشرات", "target"]):
            if not num_cols:
                return "⚠️ لا توجد أعمدة رقمية كافية لاستخراج KPI حقيقي."
            lines = ["## 🎯 مؤشرات KPI محتملة\n"]
            for col in num_cols[:8]:
                s = df[col].dropna()
                if not len(s):
                    continue
                lines.append(f"- **{col}**: يصلح كمؤشر إذا كان يمثل كمية/قيمة. التجميع المقترح: **Sum**؛ المتوسط مفيد للمقارنة بين الفئات. النطاق: {s.min():.2f} → {s.max():.2f}.")
            lines.append("\n**تنبيه:** أي KPI مبني على عمود فيه فراغات أو قيم شاذة يحتاج تنظيف أولًا.")
            return "\n".join(lines)

        if any(w in q for w in ["شاذ", "outlier", "anomal", "غريب"]):
            if not outliers:
                return "✅ لم تظهر قيم شاذة رقمية واضحة بطريقة IQR في الأعمدة الرقمية الحالية."
            lines = ["## 🚨 القيم الشاذة المحتملة\n"]
            for item in outliers[:8]:
                lines.append(f"- **{item['column']}**: {item['count']:,} قيمة شاذة ({item['pct']}% من الصفوف).")
            lines.append("\n**القرار الآمن:** لا تحذف القيم الشاذة مباشرة؛ افحص هل هي خطأ إدخال أم حالة حقيقية مهمة.")
            return "\n".join(lines)

        if any(w in q for w in ["فئات", "category", "gap", "فجوات", "مناطق", "compare"]):
            if not cat_cols:
                return "⚠️ لا توجد أعمدة فئوية واضحة للمقارنة."
            lines = ["## 🧩 فجوات وتركيز بين الفئات\n"]
            for col in cat_cols[:6]:
                vc = df[col].astype(str).replace("nan", np.nan).dropna().value_counts().head(3)
                if vc.empty:
                    continue
                top_share = round(vc.iloc[0] / max(df[col].notna().sum(), 1) * 100, 1)
                lines.append(f"- **{col}**: أعلى قيمة هي **{vc.index[0]}** وتمثل {top_share}% من السجلات. راجع التوازن لو النسبة مرتفعة.")
            return "\n".join(lines)

        if any(w in q for w in ["زمن", "time", "trend", "اتجاه", "تاريخ"]):
            if not date_cols:
                return "⚠️ لا يوجد عمود تاريخ واضح لتحليل الاتجاه الزمني."
            lines = ["## 📈 مراجعة الاتجاه الزمني\n"]
            for col in date_cols[:3]:
                s = pd.to_datetime(df[col], errors="coerce", dayfirst=True).dropna()
                if len(s):
                    lines.append(f"- **{col}**: المدى الزمني من **{s.min().date()}** إلى **{s.max().date()}**، وعدد التواريخ الصالحة {len(s):,}.")
            lines.append("\n**الخطوة التالية:** اعمل تجميع شهري/أسبوعي وقارن حجم السجلات والقيم الرقمية خلال الزمن.")
            return "\n".join(lines)

        if any(w in q for w in ["mapping", "mapper", "ربط", "خريطة"]):
            if not mappings:
                return "⚠️ لا توجد نتائج Mapping محفوظة لهذا الملف."
            if not weak_mappings:
                return "✅ أغلب قرارات الـ Mapping تبدو مقبولة. راجع يدويًا فقط الأعمدة الحساسة مثل ID / Date / KPI."
            lines = ["## 🗺️ أعمدة تحتاج مراجعة Mapping\n"]
            for col, grp, score in weak_mappings[:10]:
                lines.append(f"- **{col}** → {grp} بثقة {round(score*100)}%")
            lines.append("\n**القرار:** لا تعتمد على أي تقرير قبل اعتماد الأعمدة منخفضة الثقة.")
            return "\n".join(lines)

        if any(w in q for w in ["تقرير", "summary", "إداري", "مانح", "donor"]):
            return (
                f"## 📋 ملخص إداري\n\n"
                f"الملف يحتوي على **{qs['rows']:,}** صف و **{qs['cols']}** عمود. درجة الجودة الحالية **{qs['quality_score']}%**. "
                f"أهم المخاطر هي: **{qs['null_total']:,}** قيمة فارغة، **{qs['duplicates']:,}** صف مكرر، "
                f"و **{len(weak_mappings)}** عمود يحتاج مراجعة Mapping.\n\n"
                f"**أفضل 5 خطوات تالية:**\n"
                f"1. اعتماد Mapping للأعمدة المهمة.\n"
                f"2. مراجعة الفراغات في الأعمدة الأعلى تأثيرًا.\n"
                f"3. فحص التكرارات قبل حذفها.\n"
                f"4. فحص القيم الشاذة في الأعمدة الرقمية.\n"
                f"5. تحديد KPIs قبل بناء Dashboard أو تقرير."
            )

        return (
            "🤖 **مساعد ذكي حسب الملف الحالي**\n\n"
            "استخدم الأسئلة المقترحة بالأعلى؛ هي تتغير حسب وجود تواريخ، أرقام، فراغات، تكرارات، وأعمدة منخفضة الثقة.\n\n"
            "ممكن تسأل مثلًا: *ما أخطر مشاكل الداتا؟* أو *اقترح KPIs مناسبة* أو *اكتب تقرير إداري مختصر*."
        )

    # English responses
>>>>>>> c8e725118d9c65808b1a67b5349827ad4e22458d
    if any(w in q for w in ["diagnose", "quality", "risk"]):
        top_nulls = "\n".join([f"- **{c}**: {v:,} nulls ({round(v/max(len(df),1)*100,1)}%)" for c, v in qs["top_nulls"].items()]) or "- No major missing-value columns"
        return (
            f"## 🔎 Data quality diagnosis\n\n"
            f"**Quality score:** {qs['quality_score']}% — risk level: **{_risk_level(float(qs['quality_score']))}**\n\n"
            f"- Shape: **{qs['rows']:,} rows × {qs['cols']} columns**\n"
            f"- Missing cells: **{qs['null_total']:,}** ({qs['null_pct']}%)\n"
            f"- Duplicates: **{qs['duplicates']:,}** ({qs['duplicate_pct']}%)\n"
            f"- Type issues: {len(qs['type_errors'])}\n\n"
            f"**Highest-priority columns:**\n{top_nulls}\n\n"
            f"Start with identity/date/KPI columns before dashboarding or reporting."
        )

    if any(w in q for w in ["clean", "repair"]):
        steps = []
        if qs["duplicates"]:
            steps.append(f"1. Review **{qs['duplicates']:,}** duplicate rows before deletion.")
        if qs["null_total"]:
            top = list(qs["top_nulls"].index[:4])
            steps.append(f"2. Treat missing values first in: **{', '.join(map(str, top))}**.")
        if weak_mappings:
            steps.append(f"3. Review **{len(weak_mappings)}** low-confidence mapping decisions.")
        if outliers:
            steps.append(f"4. Investigate outliers, especially **{outliers[0]['column']}** ({outliers[0]['count']} cases).")
        steps.append("5. Fill blanks with zero only for quantity/service columns and after user approval.")
        return "## 🛠️ Safe cleaning plan\n\n" + "\n".join(steps)

    if any(w in q for w in ["kpi", "target"]):
        if not num_cols:
            return "⚠️ No numeric columns are available for real KPI candidates."
        lines = ["## 🎯 KPI candidates\n"]
        for col in num_cols[:8]:
            s = df[col].dropna()
            if len(s):
                lines.append(f"- **{col}**: candidate KPI if it represents volume/value. Suggested aggregation: **Sum**; use Mean for category comparison. Range: {s.min():.2f} → {s.max():.2f}.")
        return "\n".join(lines)

    if any(w in q for w in ["outlier", "anomal"]):
        if not outliers:
            return "✅ No obvious numeric outliers were detected using the IQR method."
        lines = ["## 🚨 Potential outliers\n"]
        for item in outliers[:8]:
            lines.append(f"- **{item['column']}**: {item['count']:,} outliers ({item['pct']}% of rows).")
        return "\n".join(lines) + "\n\nDo not delete automatically; validate whether these are errors or meaningful cases."

    if any(w in q for w in ["category", "gap", "compare", "imbalance"]):
        if not cat_cols:
            return "⚠️ No clear categorical columns are available for comparison."
        lines = ["## 🧩 Category gaps and concentration\n"]
        for col in cat_cols[:6]:
            vc = df[col].astype(str).replace("nan", np.nan).dropna().value_counts().head(3)
            if vc.empty:
                continue
            top_share = round(vc.iloc[0] / max(df[col].notna().sum(), 1) * 100, 1)
            lines.append(f"- **{col}**: top value **{vc.index[0]}** represents {top_share}% of records. Review balance if this is unexpectedly high.")
        return "\n".join(lines)

    if any(w in q for w in ["time", "trend", "date"]):
        if not date_cols:
            return "⚠️ No clear date column is available for time trend analysis."
        lines = ["## 📈 Time trend review\n"]
        for col in date_cols[:3]:
            s = pd.to_datetime(df[col], errors="coerce", dayfirst=True).dropna()
            if len(s):
                lines.append(f"- **{col}**: range from **{s.min().date()}** to **{s.max().date()}**, with {len(s):,} valid dates.")
        return "\n".join(lines)

    if any(w in q for w in ["mapping", "mapper"]):
        if not mappings:
            return "⚠️ No mapping results were found for this dataset."
        if not weak_mappings:
            return "✅ Most mapping decisions look acceptable. Still review ID, Date, and KPI columns manually."
        lines = ["## 🗺️ Mapping decisions requiring review\n"]
        for col, grp, score in weak_mappings[:10]:
            lines.append(f"- **{col}** → {grp} with {round(score*100)}% confidence")
        return "\n".join(lines)

    if any(w in q for w in ["summary", "management", "report", "donor"]):
        return (
            f"## 📋 Management summary\n\n"
            f"The dataset has **{qs['rows']:,}** rows and **{qs['cols']}** columns. Current quality score is **{qs['quality_score']}%**. "
            f"Key risks: **{qs['null_total']:,}** missing cells, **{qs['duplicates']:,}** duplicate rows, and **{len(weak_mappings)}** low-confidence mapping decisions.\n\n"
            f"**Next 5 actions:** approve mappings, review missing values, inspect duplicates, validate outliers, then define KPIs before dashboarding."
        )

    return (
        "🤖 **Dataset-aware assistant**\n\n"
        "Use the smart suggestions above. They change based on dates, numeric columns, missing values, duplicates, and mapping confidence."
    )


def _cloud_prompt(user_text: str, df: pd.DataFrame) -> str:
    """Strengthen cloud prompts with a fixed analytical instruction frame."""
    qs = _quality_snapshot(df)
    return (
        f"{user_text}\n\n"
        "Answer as a senior data analyst. Do not just repeat visible numbers. "
        "Prioritize risks, implications, and next actions. "
        f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns. "
        f"Missing cells: {qs['null_total']}. Duplicates: {qs['duplicates']}. "
        "Give a practical response with sections: Findings, Risks, Recommended Actions."
    )


def render(df: pd.DataFrame) -> None:
    st.markdown(
<<<<<<< HEAD
        section_header("🤖", "AI Data Assistant", "Dataset-aware intelligence"),
=======
        section_header("🤖", t("ai_title"), t("ai_subtitle")),
>>>>>>> c8e725118d9c65808b1a67b5349827ad4e22458d
        unsafe_allow_html=True,
    )

    mode      = st.session_state.get("ai_mode", "demo")
    ai_engine = st.session_state.get("ai_engine", None)

    banner_map = {
        "anthropic": '<div class="info-box">✅ <b style="color:#6bffb8">Claude AI connected</b> — Anthropic Cloud</div>',
        "ollama":    '<div class="info-box">🟢 <b style="color:#6bffb8">Ollama connected</b> — local engine</div>',
        "gemini":    '<div class="info-box">🔴 <b style="color:#6bffb8">Google Gemini connected</b> — 🛡️ PII masking active</div>',
<<<<<<< HEAD
        "demo":      '<div class="info-box">🟡 <b style="color:#ffb86b">Demo Mode</b> — no API key. Choose engine in sidebar.</div>',
=======
        "demo":      f'<div class="info-box">🟡 <b style="color:#ffb86b">{t("demo_mode")}</b> — no API key. Choose engine in sidebar.</div>',
>>>>>>> c8e725118d9c65808b1a67b5349827ad4e22458d
    }
    st.markdown(banner_map.get(mode, banner_map["demo"]), unsafe_allow_html=True)

    # ── Send handler ──
    def handle_send(user_text: str) -> None:
        st.session_state.ai_messages.append({"role": "user", "content": user_text})

        if mode in ("anthropic", "ollama", "gemini") and ai_engine is not None:
            try:
                history   = st.session_state.ai_messages[:-1]
                df_for_ai = df
                if mode == "gemini" and st.session_state.get("gemini_mask_pii", True):
                    df_for_ai = anonymise_df_for_ai(df)
                elif mode == "anthropic" and not ai_engine.allow_cloud_data:
                    df_for_ai = anonymise_df_for_ai(df)
<<<<<<< HEAD
                with st.spinner("Analysing..."):
=======
                with st.spinner("Analysing..." if not is_ar() else "جاري التحليل..."):
>>>>>>> c8e725118d9c65808b1a67b5349827ad4e22458d
                    reply = ai_engine.process_task(df_for_ai, _cloud_prompt(user_text, df_for_ai), history)
            except Exception as exc:
                reply = f"⚠️ Error: {exc}"
        else:
            reply = _demo_response(df, user_text)

        st.session_state.ai_messages.append({"role": "assistant", "content": reply})

    # ── Suggested prompts ──
    suggestions = _build_suggestions(df)
<<<<<<< HEAD
    st.markdown("**Smart suggestions based on this dataset**")
=======
    st.markdown(f"**{t('smart_suggestions')}**")
>>>>>>> c8e725118d9c65808b1a67b5349827ad4e22458d
    cols = st.columns(2)
    for i, sug in enumerate(suggestions):
        with cols[i % 2]:
            if st.button(sug["label"], key=f"smart_sug_{i}", use_container_width=True):
                handle_send(sug["prompt"])
                st.rerun()

    st.markdown("---")

    # ── Chat history ──
    for msg in st.session_state.ai_messages:
        if msg["role"] == "user":
            safe_content = html.escape(msg["content"])
            st.markdown(
                f'<div class="ai-message-user">{safe_content}</div>',
                unsafe_allow_html=True,
            )
        else:
            safe_raw = html.escape(msg["content"])
            content = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", safe_raw).replace("\n", "<br>")
            st.markdown(
                f'<div class="ai-message-bot">🤖 {content}</div>',
                unsafe_allow_html=True,
            )

    # ── Input ──
    ai_c1, ai_c2 = st.columns([5, 1])
    with ai_c1:
        user_input = st.text_input(
<<<<<<< HEAD
            "", placeholder="Ask anything about your data...",
            key="ai_input", label_visibility="collapsed",
        )
    with ai_c2:
        if st.button("Send →", use_container_width=True) and user_input.strip():
=======
            "", placeholder=t("ask_placeholder"),
            key="ai_input", label_visibility="collapsed",
        )
    with ai_c2:
        if st.button(t("send"), use_container_width=True) and user_input.strip():
>>>>>>> c8e725118d9c65808b1a67b5349827ad4e22458d
            handle_send(user_input.strip())
            st.rerun()

    if st.session_state.ai_messages:
        st.markdown("<br>", unsafe_allow_html=True)
<<<<<<< HEAD
        if st.button("🗑️ Clear Chat"):
=======
        if st.button(t("clear_chat")):
>>>>>>> c8e725118d9c65808b1a67b5349827ad4e22458d
            st.session_state.ai_messages = []
            st.rerun()
