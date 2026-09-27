# ════════════════════════════════════════════════════════
#  DataBridge AI — Settings
#  Stage 12: account, credentials, audit, model-key lifecycle
# ════════════════════════════════════════════════════════
from __future__ import annotations

from datetime import datetime, timezone

import streamlit as st
from ui.streamlit_compat import safe_dataframe

from core.audit import audit_csv_bytes, audit_jsonl_bytes, load_persistent_audit_events
from core.auth import get_current_auth_info, render_change_password_form
from core.credential_store import get_credential_store
from core.security import safe_error_message, safe_html
from core.session import append_audit_event
from core.team_access import (
    PERM_AUDIT_VIEW,
    PERM_SECRETS_MANAGE,
    PERM_SIGNING_KEY_MANAGE,
    PERM_TEAM_MANAGE,
    has_permission,
)
from modules.model_package import (
    ModelPackageError,
    create_encrypted_signing_key_backup,
    restore_encrypted_signing_key_backup,
    rotate_signing_key,
    signing_key_info,
)


def _render_account_security() -> None:
    info = get_current_auth_info()
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f'<div class="metric-card"><div class="label">User</div>'
            f'<div class="value" style="font-size:1.05rem;">{safe_html(info["username"])}</div>'
            '<div class="sub">local account</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="metric-card"><div class="label">Password Hash</div>'
            f'<div class="value" style="font-size:1.05rem;">{safe_html(info["hash_type"])}</div>'
            '<div class="sub">salted local verifier</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        mode = "Environment" if info["env_managed"] else "Team SQLite"
        st.markdown(
            f'<div class="metric-card"><div class="label">Auth Mode</div>'
            f'<div class="value" style="font-size:1.05rem;">{safe_html(mode)}</div>'
            f'<div class="sub">{safe_html(info.get("role_label", ""))}</div></div>',
            unsafe_allow_html=True,
        )

    left, right = st.columns([1.1, 0.9])
    with left:
        st.markdown('<div class="settings-panel">', unsafe_allow_html=True)
        render_change_password_form(prefix="settings")
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown("#### Authentication storage")
        st.code(info["auth_file"], language="text")
        st.caption(
            "Stage 20 stores one salted PBKDF2 verifier per team user. Password plaintext is never written to disk."
        )
        if has_permission(PERM_TEAM_MANAGE, session_state=st.session_state):
            if st.button("Open Team Administration", width="stretch", key="settings_open_team_admin"):
                st.session_state.current_page = "team_admin"
                st.rerun()


def _render_api_credentials() -> None:
    st.markdown("### OS Credential Manager")
    store = get_credential_store()
    backend = store.backend_name()
    st.caption(f"Active backend: {backend}")
    rows = []
    for secret_name, provider in (
        ("anthropic_api_key", "Anthropic"),
        ("gemini_api_key", "Gemini"),
    ):
        status = store.status(secret_name)
        rows.append((secret_name, provider, status))

    for secret_name, provider, status in rows:
        c1, c2, c3 = st.columns([1.2, 1.2, 0.8])
        c1.markdown(f"**{provider}**")
        c2.caption(
            f"{'Configured' if status.configured else 'Not configured'} · {status.source}"
        )
        with c3:
            if st.button(
                "Delete",
                key=f"settings_delete_{secret_name}",
                width="stretch",
                disabled=not status.configured or status.source.startswith("environment"),
            ):
                try:
                    store.delete_secret(secret_name)
                    append_audit_event(
                        {
                            "event": "credential_deleted",
                            "action": "Saved AI credential deleted from OS credential manager",
                            "provider": provider,
                        }
                    )
                    st.success(f"{provider} credential deleted.")
                    st.rerun()
                except Exception as exc:
                    st.error(safe_error_message(exc))
    st.info("Add or replace API keys from the AI Engine section in the sidebar. Saved values are never displayed back to the app.")


def _render_signing_key_management() -> None:
    st.markdown("### Model-package signing key")
    try:
        info = signing_key_info()
    except Exception as exc:
        st.error(f"Signing key unavailable: {safe_error_message(exc)}")
        return

    c1, c2 = st.columns(2)
    c1.metric("Active Key ID", info["key_id"])
    c2.metric("Key Size", f"{info['size'] * 8} bit")
    st.code(info["path"], language="text")
    st.warning(
        "Keep an encrypted backup. Losing this key makes existing .dbmlpkg files unverifiable. Rotating it intentionally invalidates old packages unless the archived key is restored."
    )

    with st.expander("Create encrypted backup", expanded=False):
        passphrase = st.text_input(
            "Backup passphrase",
            type="password",
            key="signing_backup_passphrase",
            help="Minimum 12 characters. This passphrase is not stored.",
        )
        confirm = st.text_input(
            "Confirm backup passphrase",
            type="password",
            key="signing_backup_confirm",
        )
        if st.button("Create encrypted key backup", key="create_key_backup"):
            if passphrase != confirm:
                st.error("Backup passphrases do not match.")
            else:
                try:
                    backup = create_encrypted_signing_key_backup(passphrase)
                    st.session_state.stage12_key_backup_bytes = backup
                    append_audit_event(
                        {
                            "event": "signing_key_backup_created",
                            "action": "Encrypted model signing-key backup created",
                            "key_id": info["key_id"],
                        }
                    )
                    st.success("Encrypted backup created in memory. Download it now.")
                except Exception as exc:
                    st.error(safe_error_message(exc))
        backup_bytes = st.session_state.get("stage12_key_backup_bytes")
        if backup_bytes:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
            st.download_button(
                "Download encrypted .dbkey backup",
                data=backup_bytes,
                file_name=f"databridge_signing_key_{info['key_id']}_{stamp}.dbkey",
                mime="application/json",
                width="stretch",
                key="download_key_backup",
            )

    with st.expander("Restore encrypted backup", expanded=False):
        uploaded = st.file_uploader(
            "Encrypted .dbkey file",
            type=["dbkey", "json"],
            key="restore_signing_key_file",
        )
        restore_passphrase = st.text_input(
            "Backup passphrase",
            type="password",
            key="restore_key_passphrase",
        )
        overwrite = st.checkbox(
            "Overwrite the active signing key after verified decryption.",
            key="restore_key_overwrite",
        )
        confirm_id = st.text_input(
            "Type the current Key ID to confirm overwrite",
            key="restore_key_confirm_id",
            disabled=not overwrite,
        )
        if st.button(
            "Verify and restore backup",
            key="restore_signing_key",
            disabled=uploaded is None or not restore_passphrase,
        ):
            if overwrite and confirm_id.strip() != info["key_id"]:
                st.error("Current Key ID confirmation does not match.")
            else:
                try:
                    result = restore_encrypted_signing_key_backup(
                        uploaded.getvalue(),
                        restore_passphrase,
                        overwrite=overwrite,
                    )
                    append_audit_event(
                        {
                            "event": "signing_key_restored",
                            "action": "Encrypted model signing-key backup restored",
                            "key_id": result.get("key_id"),
                            "restored": result.get("restored"),
                        }
                    )
                    st.success("Signing key verified and restored safely.")
                    st.session_state.loaded_model_package = None
                    st.rerun()
                except ModelPackageError as exc:
                    st.error(safe_error_message(exc))

    with st.expander("Rotate signing key — destructive", expanded=False):
        st.error("Rotation changes the signer. Existing packages will fail verification until the archived old key is restored.")
        rotation_id = st.text_input(
            "Type the active Key ID",
            key="rotation_confirm_id",
        )
        rotation_check = st.checkbox(
            "I created and downloaded an encrypted backup and accept the package compatibility impact.",
            key="rotation_confirm_checkbox",
        )
        if st.button(
            "Rotate signing key",
            type="primary",
            key="rotate_signing_key",
            disabled=not rotation_check or rotation_id.strip() != info["key_id"],
        ):
            try:
                result = rotate_signing_key(rotation_id)
                append_audit_event(
                    {
                        "event": "signing_key_rotated",
                        "action": "Model signing key rotated with archived prior key",
                        "old_key_id": result["old_key_id"],
                        "new_key_id": result["new_key_id"],
                    }
                )
                st.session_state.model_package_bytes = None
                st.session_state.model_package_manifest = {}
                st.session_state.loaded_model_package = None
                st.success("Signing key rotated. The previous key was archived locally.")
                st.rerun()
            except Exception as exc:
                st.error(safe_error_message(exc))


def _render_audit_export() -> None:
    st.markdown("### Sanitized audit log")
    persistent = load_persistent_audit_events(limit=5000)
    session_events = list(st.session_state.get("dataset_audit_log", []) or [])
    events = persistent or session_events
    st.caption(
        f"Exportable events: {len(events)}. Secret-like fields and token patterns are redacted before storage and export."
    )
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Download audit JSONL",
            data=audit_jsonl_bytes(events),
            file_name="databridge_audit.jsonl",
            mime="application/x-ndjson",
            width="stretch",
            disabled=not events,
        )
    with c2:
        st.download_button(
            "Download audit CSV",
            data=audit_csv_bytes(events),
            file_name="databridge_audit.csv",
            mime="text/csv",
            width="stretch",
            disabled=not events,
        )
    if events:
        safe_dataframe(events[-50:], width="stretch", height=300)


def render(df):
    st.markdown(
        '<div class="section-header"><div class="icon">⚙</div><h2>Settings</h2>'
        '<div class="count">Security & Preferences</div></div>',
        unsafe_allow_html=True,
    )

    sections = [("Account", _render_account_security)]
    if has_permission(PERM_SECRETS_MANAGE, session_state=st.session_state):
        sections.append(("AI Credentials", _render_api_credentials))
    if has_permission(PERM_SIGNING_KEY_MANAGE, session_state=st.session_state):
        sections.append(("Model Signing Key", _render_signing_key_management))
    if has_permission(PERM_AUDIT_VIEW, session_state=st.session_state):
        sections.append(("Audit Export", _render_audit_export))

    tabs = st.tabs([label for label, _ in sections])
    for tab, (_, renderer) in zip(tabs, sections):
        with tab:
            renderer()
