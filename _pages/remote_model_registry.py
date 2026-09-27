# ════════════════════════════════════════════════════════
# DataBridge AI — Remote Model Registry
# Stage 19: shared trusted packages + governance fast-forward sync
# ════════════════════════════════════════════════════════
from __future__ import annotations

import pandas as pd
import streamlit as st
from ui.streamlit_compat import safe_dataframe

from core.security import safe_html
from core.session import append_audit_event
from modules.model_governance import list_governance_families
from modules.model_package import get_or_create_signing_key, signing_key_id
from modules.remote_model_registry import (
    RemoteRegistryError,
    client_from_profile,
    create_remote_registry_client_profile,
    create_remote_registry_server_config,
    generate_remote_registry_token,
    list_remote_registry_client_profiles,
    list_remote_registry_server_configs,
    pull_family_from_remote,
    push_family_to_remote,
    read_remote_registry_process_status,
    remote_registry_token_status,
    save_remote_registry_token,
    start_remote_registry_process,
    stop_remote_registry_process,
)
from ui.cards import section_header


def _trust_key_id() -> str:
    return signing_key_id(get_or_create_signing_key())


def _render_token() -> None:
    st.markdown("### Registry authentication")
    status = remote_registry_token_status()
    if status.get("configured"):
        st.success(
            f"Remote registry Bearer token is stored in the OS credential backend ({status.get('source')})."
        )
    else:
        st.warning("No remote registry Bearer token is configured on this machine.")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Generate new registry token", width="stretch", key="remote_registry_generate_token"):
            try:
                token = generate_remote_registry_token()
                st.session_state["remote_registry_new_token"] = token
                append_audit_event(
                    {
                        "event": "remote_registry_token_generated",
                        "action": "Remote registry token generated in OS credential storage",
                    }
                )
                st.rerun()
            except Exception as exc:
                st.error(f"Token generation failed safely: {exc}")
    with c2:
        supplied = st.text_input(
            "Use an existing shared registry token",
            type="password",
            key="remote_registry_existing_token",
            help="Paste the token issued by the central registry administrator. It is stored in the OS credential manager, not in project files.",
        )
        if st.button("Save shared token", width="stretch", key="remote_registry_save_token"):
            try:
                save_remote_registry_token(supplied)
                append_audit_event(
                    {
                        "event": "remote_registry_token_saved",
                        "action": "Existing remote registry token saved in OS credential storage",
                    }
                )
                st.success("Shared token saved securely.")
                st.rerun()
            except Exception as exc:
                st.error(f"Token save failed safely: {exc}")

    generated = st.session_state.pop("remote_registry_new_token", "")
    if generated:
        st.warning(
            "Copy this token now to trusted registry clients. It is shown once in this session and will not be written to a file."
        )
        st.code(generated, language="text")


def _render_server() -> None:
    st.markdown("### Host the central registry")
    st.caption(
        "Loopback HTTP is allowed for local testing. Any LAN/WAN bind requires explicit approval and TLS. "
        "The central registry verifies every package and governance envelope using the shared model trust key."
    )

    with st.expander("Create registry server configuration", expanded=not bool(list_remote_registry_server_configs())):
        name = st.text_input("Server name", value="DataBridge Central Registry", key="rr_server_name")
        remote = st.checkbox("Allow network clients", value=False, key="rr_server_remote")
        host_default = "0.0.0.0" if remote else "127.0.0.1"
        host = st.text_input("Bind address", value=host_default, key="rr_server_host")
        port = st.number_input("Port", min_value=1024, max_value=65535, value=8890, step=1, key="rr_server_port")
        cert = key = ""
        if remote:
            cert = st.text_input("TLS certificate absolute path", key="rr_server_cert")
            key = st.text_input("TLS private key absolute path", type="password", key="rr_server_key")
        if st.button("Create signed server config", width="stretch", key="rr_create_server"):
            try:
                cfg = create_remote_registry_server_config(
                    name=name,
                    host=host,
                    port=int(port),
                    allow_remote=remote,
                    tls_cert_path=cert,
                    tls_key_path=key,
                )
                append_audit_event(
                    {
                        "event": "remote_registry_server_config_created",
                        "action": "Signed remote registry server configuration created",
                        "config_id": cfg.get("config_id"),
                        "network_exposed": bool(cfg.get("allow_remote")),
                    }
                )
                st.success("Server configuration created.")
                st.rerun()
            except Exception as exc:
                st.error(f"Server configuration failed safely: {exc}")

    configs = list_remote_registry_server_configs()
    if not configs:
        st.info("No server configuration exists yet.")
        return
    labels = {f"{row['name']} · {row['scheme']}://{row['host']}:{row['port']}": row for row in configs}
    selected = labels[st.selectbox("Server configuration", list(labels), key="rr_server_select")]
    status = read_remote_registry_process_status(str(selected["config_id"]))
    tone = "#6bff8e" if status.running else "#ffb86b"
    st.markdown(
        f"<div class='info-box' style='border-left:4px solid {tone};'>"
        f"<b>{'Running' if status.running else 'Stopped'}</b> · "
        f"{safe_html(status.base_url)} · signer <code>{safe_html(_trust_key_id())}</code>"
        "</div>",
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Start registry server", disabled=status.running, width="stretch", key="rr_server_start"):
            try:
                started = start_remote_registry_process(str(selected["config_id"]))
                append_audit_event(
                    {
                        "event": "remote_registry_server_started",
                        "action": "Remote model registry server started",
                        "config_id": started.config_id,
                    }
                )
                st.rerun()
            except Exception as exc:
                st.error(f"Registry start failed safely: {exc}")
    with c2:
        if st.button("Stop registry server", disabled=not status.running, width="stretch", key="rr_server_stop"):
            try:
                stopped = stop_remote_registry_process(str(selected["config_id"]))
                append_audit_event(
                    {
                        "event": "remote_registry_server_stopped",
                        "action": "Remote model registry server stopped",
                        "config_id": stopped.config_id,
                    }
                )
                st.rerun()
            except Exception as exc:
                st.error(f"Registry stop failed safely: {exc}")


def _render_client() -> None:
    st.markdown("### Connect to a central registry")
    st.caption(
        "Every trusted node must use the same Stage 12 model signing key. The Bearer token authenticates transport; "
        "the signing key authenticates model packages and Champion/Challenger governance."
    )
    with st.expander("Add remote registry profile", expanded=not bool(list_remote_registry_client_profiles())):
        name = st.text_input("Profile name", value="Central Registry", key="rr_profile_name")
        base_url = st.text_input("Registry URL", value="http://127.0.0.1:8890", key="rr_profile_url")
        ca_path = st.text_input(
            "Custom CA certificate path (optional)",
            key="rr_profile_ca",
            help="For an internal HTTPS registry signed by a private CA. TLS verification can never be disabled.",
        )
        if st.button("Save signed client profile", width="stretch", key="rr_profile_create"):
            try:
                profile = create_remote_registry_client_profile(
                    name=name,
                    base_url=base_url,
                    ca_cert_path=ca_path,
                )
                append_audit_event(
                    {
                        "event": "remote_registry_profile_created",
                        "action": "Signed remote registry client profile created",
                        "profile_id": profile.get("profile_id"),
                    }
                )
                st.success("Client profile saved.")
                st.rerun()
            except Exception as exc:
                st.error(f"Profile creation failed safely: {exc}")

    profiles = list_remote_registry_client_profiles()
    if not profiles:
        st.info("No remote registry client profile exists yet.")
        return
    labels = {f"{row['name']} · {row['base_url']}": row for row in profiles}
    profile = labels[st.selectbox("Remote registry", list(labels), key="rr_profile_select")]
    profile_id = str(profile["profile_id"])

    client = None
    if st.button("Verify connection & shared trust", width="stretch", key="rr_health"):
        try:
            client = client_from_profile(profile_id)
            health = client.health()
            st.success(
                f"Connected securely. Shared signer key: {health.get('signer_key_id')}"
            )
        except Exception as exc:
            st.error(f"Connection verification failed safely: {exc}")

    st.markdown("#### Push local family")
    local_families = list_governance_families()
    if not local_families:
        st.info("No local governed model families are available to push.")
    else:
        local_labels = {
            f"{row['family_name']} · rev {row['revision']} · Champion {row.get('champion_id') or 'None'}": row
            for row in local_families
        }
        local = local_labels[st.selectbox("Local family", list(local_labels), key="rr_push_family")]
        st.warning(
            "Push uses optimistic concurrency. If another trusted node changed the central family, the push is rejected until you pull and resolve the conflict."
        )
        if st.button("Push family to remote", width="stretch", key="rr_push_btn"):
            try:
                result = push_family_to_remote(profile_id, str(local["family_id"]))
                append_audit_event(
                    {
                        "event": "remote_registry_family_pushed",
                        "action": "Governed model family synchronized to remote registry",
                        "family_id": result.family_id,
                        "revision": result.governance_revision,
                        "packages_transferred": result.packages_transferred,
                    }
                )
                st.success(
                    f"{result.message} Revision {result.governance_revision}; uploaded {result.packages_transferred} package(s)."
                )
            except Exception as exc:
                st.error(f"Remote push stopped safely: {exc}")

    st.markdown("#### Pull remote family")
    try:
        client = client or client_from_profile(profile_id)
        remote_families = client.list_families()
    except Exception as exc:
        st.info(f"Remote families unavailable until the connection is verified: {exc}")
        return
    if not remote_families:
        st.info("The central registry has no governed families yet.")
        return
    remote_table = pd.DataFrame(
        [
            {
                "Family": row.get("family_name"),
                "Task": row.get("task"),
                "Target": row.get("target"),
                "Revision": row.get("revision"),
                "Champion": row.get("champion_id") or "—",
            }
            for row in remote_families
        ]
    )
    safe_dataframe(remote_table, width="stretch", hide_index=True)
    remote_labels = {
        f"{row['family_name']} · rev {row['revision']} · Champion {row.get('champion_id') or 'None'}": row
        for row in remote_families
    }
    remote = remote_labels[st.selectbox("Remote family", list(remote_labels), key="rr_pull_family")]
    if st.button("Pull family safely", width="stretch", key="rr_pull_btn"):
        try:
            result = pull_family_from_remote(profile_id, str(remote["family_id"]))
            append_audit_event(
                {
                    "event": "remote_registry_family_pulled",
                    "action": "Governed model family fast-forwarded from remote registry",
                    "family_id": result.family_id,
                    "revision": result.governance_revision,
                    "packages_transferred": result.packages_transferred,
                }
            )
            st.success(
                f"{result.message} Revision {result.governance_revision}; downloaded {result.packages_transferred} package(s)."
            )
            st.rerun()
        except Exception as exc:
            st.error(f"Remote pull stopped safely: {exc}")


def render(df: pd.DataFrame) -> None:
    st.markdown(
        section_header("🌐", "Remote Model Registry", "Stage 19 · trusted multi-device model lifecycle"),
        unsafe_allow_html=True,
    )
    st.markdown(
        f"<div class='info-box'><b>Shared trust contract:</b> remote synchronization accepts only signed <code>.dbmlpkg</code> artifacts "
        f"and authenticated governance histories. This installation trust key is <code>{safe_html(_trust_key_id())}</code>. "
        "Use the encrypted signing-key backup from Stage 12 to provision the same trust key on every registry node. "
        "Concurrent governance histories are never silently overwritten.</div>",
        unsafe_allow_html=True,
    )
    _render_token()
    st.markdown("---")
    host_tab, sync_tab = st.tabs(["Host Central Registry", "Connect & Synchronize"])
    with host_tab:
        _render_server()
    with sync_tab:
        _render_client()
