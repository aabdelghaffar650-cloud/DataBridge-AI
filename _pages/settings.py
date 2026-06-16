# ════════════════════════════════════════════════════════
#  PAGE: Settings
# ════════════════════════════════════════════════════════
import streamlit as st

from core.auth import get_current_auth_info, render_change_password_form
<<<<<<< HEAD
=======
from core.i18n import LANGUAGE_OPTIONS
>>>>>>> c8e725118d9c65808b1a67b5349827ad4e22458d
from core.security import safe_html


def render(df):
    st.markdown(
        '<div class="section-header"><div class="icon">⚙</div><h2>Settings</h2><div class="count">Account & Preferences</div></div>',
        unsafe_allow_html=True,
    )

    info = get_current_auth_info()

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="label">User</div>
                <div class="value" style="font-size:1.05rem;">{safe_html(info['username'])}</div>
                <div class="sub">local account</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="label">Password Hash</div>
                <div class="value" style="font-size:1.05rem;">{safe_html(info['hash_type'])}</div>
                <div class="sub">stored securely</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        mode = "Environment" if info["env_managed"] else "Local File"
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="label">Auth Mode</div>
                <div class="value" style="font-size:1.05rem;">{safe_html(mode)}</div>
                <div class="sub">credential source</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    left, right = st.columns([1.1, 0.9])
    with left:
        st.markdown('<div class="settings-panel">', unsafe_allow_html=True)
        render_change_password_form(prefix="settings")
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
<<<<<<< HEAD
=======
        st.markdown("#### Preferences")
        st.selectbox(
            "Language",
            list(LANGUAGE_OPTIONS.keys()),
            format_func=lambda code: LANGUAGE_OPTIONS[code],
            key="language",
        )

>>>>>>> c8e725118d9c65808b1a67b5349827ad4e22458d
        st.markdown("#### Credential Storage")
        st.code(info["auth_file"], language="text")
        st.caption(
            "After changing the password, DataBridge AI stores a salted PBKDF2-SHA256 hash in this local file. The password itself is never saved."
        )
