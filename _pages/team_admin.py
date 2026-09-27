# ════════════════════════════════════════════════════════
# DataBridge AI — Stage 20 Team Administration
# ════════════════════════════════════════════════════════
from __future__ import annotations

import time
from typing import Any

import pandas as pd
import streamlit as st

from core.security import safe_error_message, safe_html
from core.session import append_audit_event
from core.team_access import (
    PERM_TEAM_MANAGE,
    ROLE_LABELS,
    VALID_ROLES,
    identity_from_session,
    require_permission,
)
from core.team_store import TeamAuthError, get_team_auth_store
from ui.cards import section_header
from ui.streamlit_compat import safe_dataframe


def _actor() -> tuple[str, str]:
    identity = identity_from_session(st.session_state)
    return identity.user_id, identity.username


def _user_rows() -> list[dict[str, Any]]:
    rows = []
    now = int(time.time())
    for user in get_team_auth_store().list_users():
        if not user.active:
            status = "Disabled"
        elif user.locked_until > now:
            status = "Locked"
        elif user.must_change_password:
            status = "Password change required"
        else:
            status = "Active"
        rows.append(
            {
                "Username": user.username,
                "Display Name": user.display_name,
                "Role": user.role_label,
                "Status": status,
                "Revision": user.revision,
                "Last Login": user.last_login_at or "—",
                "User ID": user.user_id,
            }
        )
    return rows


def render(df=None) -> None:
    try:
        require_permission(PERM_TEAM_MANAGE, session_state=st.session_state)
    except Exception:
        st.error("Admin permission is required to manage team accounts.")
        return

    st.markdown(
        section_header("👥", "Team Administration", "Stage 20 · roles, users, access control"),
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="info-box">Accounts are local to this DataBridge AI installation. '
        'Passwords are stored only as salted PBKDF2 verifiers. Role changes and account '
        'deactivation are revalidated on every Streamlit interaction.</div>',
        unsafe_allow_html=True,
    )

    with st.expander("Role responsibilities", expanded=False):
        safe_dataframe(
            pd.DataFrame(
                [
                    {"Role": "Viewer", "Data": "Read / visualize", "ML": "No training", "Governance": "View only", "Administration": "None"},
                    {"Role": "Analyst", "Data": "Import / clean / export", "ML": "Predict / monitor", "Governance": "No approval", "Administration": "None"},
                    {"Role": "Data Scientist", "Data": "Transform", "ML": "Train / explain / retrain", "Governance": "Submit Challenger", "Administration": "None"},
                    {"Role": "Manager / Approver", "Data": "Read / export", "ML": "Predict / monitor", "Governance": "Approve / reject / promote", "Administration": "Audit / schedules"},
                    {"Role": "Admin", "Data": "Full", "ML": "Full", "Governance": "Full", "Administration": "Users / secrets / signing / API / registry"},
                ]
            ),
            width="stretch",
            hide_index=True,
        )

    store = get_team_auth_store()
    users = store.list_users()
    active = sum(1 for user in users if user.active)
    admins = sum(1 for user in users if user.active and user.role == "admin")
    locked = sum(1 for user in users if user.locked_until > int(time.time()))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Users", len(users))
    c2.metric("Active", active)
    c3.metric("Active Admins", admins)
    c4.metric("Locked", locked)

    tabs = st.tabs(["Users", "Create User", "Security Events"])

    with tabs[0]:
        rows = _user_rows()
        if rows:
            safe_dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, height=300)
        else:
            st.info("No team users found.")
            return

        labels = {f"{user.username} · {user.role_label} · {'Active' if user.active else 'Disabled'}": user for user in users}
        selected_label = st.selectbox("Manage account", list(labels), key="team_manage_user")
        selected = labels[selected_label]
        actor_id, actor_name = _actor()

        left, right = st.columns(2)
        with left:
            display_name = st.text_input(
                "Display name",
                value=selected.display_name,
                key=f"team_display_{selected.user_id}_{selected.revision}",
            )
            role_options = list(VALID_ROLES)
            role = st.selectbox(
                "Role",
                role_options,
                index=role_options.index(selected.role),
                format_func=lambda value: ROLE_LABELS.get(value, value),
                key=f"team_role_{selected.user_id}_{selected.revision}",
            )
            active_value = st.checkbox(
                "Account active",
                value=selected.active,
                key=f"team_active_{selected.user_id}_{selected.revision}",
            )
            if st.button("Apply account changes", type="primary", width="stretch", key=f"team_update_{selected.user_id}"):
                try:
                    updated = store.update_user(
                        selected.user_id,
                        display_name=display_name,
                        role=role,
                        active=active_value,
                        actor_user_id=actor_id,
                        actor_username=actor_name,
                    )
                    append_audit_event(
                        {
                            "event": "team_user_updated",
                            "action": "Team account role/status updated",
                            "target_user": updated.username,
                            "target_role": updated.role,
                            "active": updated.active,
                        }
                    )
                    st.success("Account updated. Existing sessions will be revalidated on their next interaction.")
                    st.rerun()
                except Exception as exc:
                    st.error(safe_error_message(exc))

            if selected.locked_until > int(time.time()):
                if st.button("Unlock account", width="stretch", key=f"team_unlock_{selected.user_id}"):
                    try:
                        store.unlock_user(selected.user_id, actor_user_id=actor_id, actor_username=actor_name)
                        append_audit_event(
                            {
                                "event": "team_user_unlocked",
                                "action": "Team account unlocked by Admin",
                                "target_user": selected.username,
                            }
                        )
                        st.success("Account unlocked.")
                        st.rerun()
                    except Exception as exc:
                        st.error(safe_error_message(exc))

        with right:
            st.markdown("#### Reset password")
            st.caption("The reset password is temporary. The user must change it immediately after the next sign-in.")
            reset_password = st.text_input(
                "Temporary password",
                type="password",
                key=f"team_reset_password_{selected.user_id}",
            )
            reset_confirm = st.text_input(
                "Confirm temporary password",
                type="password",
                key=f"team_reset_confirm_{selected.user_id}",
            )
            confirm_user = st.text_input(
                f"Type username to confirm: {selected.username}",
                key=f"team_reset_username_{selected.user_id}",
            )
            if st.button(
                "Reset password",
                width="stretch",
                disabled=confirm_user.strip() != selected.username or not reset_password,
                key=f"team_reset_{selected.user_id}",
            ):
                if reset_password != reset_confirm:
                    st.error("Temporary passwords do not match.")
                else:
                    try:
                        store.admin_reset_password(
                            selected.user_id,
                            reset_password,
                            actor_user_id=actor_id,
                            actor_username=actor_name,
                        )
                        append_audit_event(
                            {
                                "event": "team_password_reset",
                                "action": "Team password reset by Admin",
                                "target_user": selected.username,
                            }
                        )
                        st.success("Temporary password saved. The user must change it on next sign-in.")
                        st.rerun()
                    except Exception as exc:
                        st.error(safe_error_message(exc))

    with tabs[1]:
        st.markdown("#### Create team account")
        with st.form("team_create_user_form", clear_on_submit=True):
            username = st.text_input("Username")
            display_name = st.text_input("Display name")
            role = st.selectbox(
                "Role",
                list(VALID_ROLES),
                index=list(VALID_ROLES).index("analyst"),
                format_func=lambda value: ROLE_LABELS.get(value, value),
            )
            password = st.text_input("Temporary password", type="password")
            confirm = st.text_input("Confirm temporary password", type="password")
            force_change = st.checkbox("Require password change at first sign-in", value=True)
            submitted = st.form_submit_button("Create user", width="stretch")
        if submitted:
            if password != confirm:
                st.error("Temporary passwords do not match.")
            else:
                actor_id, actor_name = _actor()
                try:
                    user = store.create_user(
                        username=username,
                        display_name=display_name or username,
                        role=role,
                        password=password,
                        must_change_password=force_change,
                        actor_user_id=actor_id,
                        actor_username=actor_name,
                    )
                    append_audit_event(
                        {
                            "event": "team_user_created",
                            "action": "Team account created",
                            "target_user": user.username,
                            "target_role": user.role,
                            "must_change_password": user.must_change_password,
                        }
                    )
                    st.success(f"Created {safe_html(user.username)} as {safe_html(user.role_label)}.")
                    st.rerun()
                except TeamAuthError as exc:
                    st.error(str(exc))

    with tabs[2]:
        events = store.security_events(limit=300)
        if not events:
            st.caption("No team security events recorded yet.")
        else:
            display = []
            user_map = {user.user_id: user.username for user in store.list_users()}
            for event in events:
                display.append(
                    {
                        "Timestamp": event.get("timestamp", ""),
                        "Actor": event.get("actor", "") or "system",
                        "Action": event.get("action", ""),
                        "Target": user_map.get(str(event.get("target_user_id", "")), str(event.get("target_user_id", ""))[:12]),
                        "Details": str(event.get("details", {}))[:240],
                    }
                )
            safe_dataframe(pd.DataFrame(display), width="stretch", hide_index=True, height=360)
            st.caption("Password values and password hashes are never stored in this event table.")
