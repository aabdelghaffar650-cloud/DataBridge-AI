# ════════════════════════════════════════════════════════
#  DataBridge AI — Page: AI Assistant
# ════════════════════════════════════════════════════════
import re

import pandas as pd
import streamlit as st

from core.security import anonymise_df_for_ai
from ui.cards import section_header


# ── Demo response engine ──────────────────────────────────────────────────────
def _demo_response(df: pd.DataFrame, question: str) -> str:
    q           = question.lower().strip()
    num_cols    = df.select_dtypes(include="number").columns.tolist()
    cat_cols    = df.select_dtypes(include="object").columns.tolist()
    nulls_total = int(df.isnull().sum().sum())
    dups_total  = int(df.duplicated().sum())
    null_series = df.isnull().sum()
    null_cols   = null_series[null_series > 0].sort_values(ascending=False)
    null_pct    = round(nulls_total / max(df.shape[0] * df.shape[1], 1) * 100, 1)

    if any(w in q for w in ["ملخص", "summarize", "summary", "overview", "نظرة"]):
        top_null = null_cols.head(3)
        null_info = "، ".join([f"**{c}** ({v})" for c, v in top_null.items()]) or "لا توجد فراغات"
        return (
            f"📊 **Dataset Summary**\n\n"
            f"- **File:** {st.session_state.file_name}\n"
            f"- **Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns\n"
            f"- **Numeric ({len(num_cols)}):** {', '.join(num_cols[:6])}{'...' if len(num_cols) > 6 else ''}\n"
            f"- **Text ({len(cat_cols)}):** {', '.join(cat_cols[:6])}{'...' if len(cat_cols) > 6 else ''}\n"
            f"- **Nulls:** {nulls_total:,} ({null_pct}%)\n"
            f"- **Duplicates:** {dups_total:,}\n\n"
            f"Most-null columns: {null_info}"
        )

    if any(w in q for w in ["null", "فراغ", "missing", "ناقص"]):
        if not null_cols.empty:
            lines = [f"**Null Analysis — {nulls_total:,} total nulls:**\n"]
            for col, cnt in null_cols.head(10).items():
                pct = round(cnt / df.shape[0] * 100, 1)
                bar = "█" * min(int(pct / 5), 20)
                sev = "🔴" if pct > 30 else "🟡" if pct > 10 else "🟢"
                lines.append(f"{sev} **{col}**: {cnt:,} ({pct}%) {bar}")
            return "\n".join(lines)
        return "✅ No nulls found!"

    if any(w in q for w in ["duplicate", "تكرار", "مكرر"]):
        if dups_total == 0:
            return "✅ No duplicate rows found."
        return (
            f"🟡 **{dups_total:,}** duplicate rows "
            f"({round(dups_total / df.shape[0] * 100, 1)}% of dataset).\n\n"
            f"💡 Go to **Delete & Dedupe** page to remove them."
        )

    if any(w in q for w in ["stat", "إحصاء", "average", "متوسط", "mean"]):
        if not num_cols:
            return "⚠️ No numeric columns found."
        lines = ["**Numeric Statistics:**\n"]
        for col in num_cols[:6]:
            s = df[col].dropna()
            if len(s):
                lines.append(f"**{col}:** mean={s.mean():.2f} | min={s.min():.2f} | max={s.max():.2f}")
        return "\n".join(lines)

    return (
        "🤖 **Demo Mode** — Ask me:\n\n"
        "- `summary` — Dataset overview\n"
        "- `nulls` — Missing values analysis\n"
        "- `duplicates` — Duplicate rows\n"
        "- `stats` — Numeric statistics\n\n"
        "For full AI, connect an engine in the sidebar 🔑"
    )


def render(df: pd.DataFrame) -> None:
    st.markdown(
        section_header("🤖", "AI Data Assistant", "DataBridge Intelligence"),
        unsafe_allow_html=True,
    )

    mode      = st.session_state.get("ai_mode", "demo")
    ai_engine = st.session_state.get("ai_engine", None)

    banner_map = {
        "anthropic": '<div class="info-box">✅ <b style="color:#6bffb8">Claude AI connected</b> — Anthropic Cloud</div>',
        "ollama":    '<div class="info-box">🟢 <b style="color:#6bffb8">Ollama connected</b> — local engine</div>',
        "gemini":    '<div class="info-box">🔴 <b style="color:#6bffb8">Google Gemini connected</b> — 🛡️ PII masking active</div>',
        "demo":      '<div class="info-box">🟡 <b style="color:#ffb86b">Demo Mode</b> — no API key. Choose engine in sidebar.</div>',
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
                with st.spinner("Analysing..."):
                    reply = ai_engine.process_task(df_for_ai, user_text, history)
            except Exception as exc:
                reply = f"⚠️ Error: {exc}"
        else:
            reply = _demo_response(df, user_text)

        st.session_state.ai_messages.append({"role": "assistant", "content": reply})

    # ── Suggested prompts ──
    suggestions = ["Dataset summary", "Missing values", "Duplicates", "Statistics"]
    s_cols = st.columns(4)
    for i, sug in enumerate(suggestions):
        with s_cols[i]:
            if st.button(sug, key=f"sug_{i}", use_container_width=True):
                handle_send(sug)
                st.rerun()

    st.markdown("---")

    # ── Chat history ──
    for msg in st.session_state.ai_messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="ai-message-user">{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            content = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", msg["content"]).replace("\n", "<br>")
            st.markdown(
                f'<div class="ai-message-bot">🤖 {content}</div>',
                unsafe_allow_html=True,
            )

    # ── Input ──
    ai_c1, ai_c2 = st.columns([5, 1])
    with ai_c1:
        user_input = st.text_input(
            "", placeholder="Ask anything about your data...",
            key="ai_input", label_visibility="collapsed",
        )
    with ai_c2:
        if st.button("Send →", use_container_width=True) and user_input.strip():
            handle_send(user_input.strip())
            st.rerun()

    if st.session_state.ai_messages:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️ Clear Chat"):
            st.session_state.ai_messages = []
            st.rerun()
