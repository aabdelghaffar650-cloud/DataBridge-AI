# ════════════════════════════════════════════════════════
# DataBridge AI — Deployment API
# Stage 18: governed Champion-only prediction endpoint
# ════════════════════════════════════════════════════════
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st
from ui.streamlit_compat import safe_dataframe

from core.security import safe_error_message, safe_html
from core.session import append_audit_event
from modules.deployment_api import (
    DEFAULT_MAX_BODY_BYTES,
    DEFAULT_MAX_ROWS,
    DEFAULT_PORT,
    DEFAULT_RATE_LIMIT_PER_MINUTE,
    DeploymentAPIError,
    build_api_contract,
    create_deployment_config,
    delete_deployment_config,
    deployment_token_status,
    generate_deployment_token,
    list_deployment_configs,
    read_deployment_status,
    start_deployment_process,
    stop_deployment_process,
)
from modules.model_governance import list_governance_families
from ui.cards import section_header


def _render_token() -> None:
    st.markdown("### API authentication")
    status = deployment_token_status()
    configured = bool(status.get("configured"))
    if configured:
        st.success(f"Bearer token configured securely · {status.get('source', 'credential manager')}")
        confirm = st.checkbox(
            "I understand that rotating the token immediately invalidates existing API clients",
            key="deployment_token_rotate_confirm",
        )
        if st.button("Rotate deployment token", disabled=not confirm, key="deployment_token_rotate"):
            try:
                token = generate_deployment_token()
                st.session_state["deployment_api_new_token"] = token
                append_audit_event({"event": "deployment_api_token_rotated", "action": "Deployment API bearer token rotated"})
                st.rerun()
            except Exception as exc:
                st.error(f"Token rotation failed safely: {safe_error_message(exc)}")
    else:
        st.warning("No deployment Bearer token is configured. The API cannot start until one is generated.")
        if st.button("Generate deployment token", type="primary", key="deployment_token_generate"):
            try:
                token = generate_deployment_token()
                st.session_state["deployment_api_new_token"] = token
                append_audit_event({"event": "deployment_api_token_generated", "action": "Deployment API bearer token generated"})
                st.rerun()
            except Exception as exc:
                st.error(f"Token generation failed safely: {safe_error_message(exc)}")

    token_once = st.session_state.pop("deployment_api_new_token", None)
    if token_once:
        st.warning("Copy this token now. DataBridge AI will not display the stored token again.")
        st.code(token_once, language="text")


def _family_choices() -> tuple[list[dict], list[str]]:
    families = [row for row in list_governance_families() if row.get("champion_id")]
    labels = [f"{row['family_name']} · {row['task']} → {row['target']}" for row in families]
    return families, labels


def _render_create_config() -> None:
    st.markdown("### Create deployment")
    families, labels = _family_choices()
    if not families:
        st.warning("Promote a model to Champion first. Stage 18 serves governed Champions only.")
        return
    family_label = st.selectbox("Governed model family", labels, key="deployment_family")
    family = families[labels.index(family_label)]

    left, right = st.columns(2)
    with left:
        name = st.text_input("Deployment name", value=f"{family['family_name']} API", key="deployment_name")
        port = st.number_input("Port", min_value=1024, max_value=65535, value=DEFAULT_PORT, step=1, key="deployment_port")
        max_rows = st.number_input(
            "Maximum rows per request",
            min_value=1,
            max_value=50_000,
            value=DEFAULT_MAX_ROWS,
            step=100,
            key="deployment_max_rows",
        )
        max_body_mb = st.number_input(
            "Maximum JSON body (MB)",
            min_value=1,
            max_value=32,
            value=DEFAULT_MAX_BODY_BYTES // (1024 * 1024),
            step=1,
            key="deployment_body_mb",
        )
        rate = st.number_input(
            "Requests per minute / client IP",
            min_value=1,
            max_value=5000,
            value=DEFAULT_RATE_LIMIT_PER_MINUTE,
            step=10,
            key="deployment_rate",
        )
    with right:
        remote = st.checkbox(
            "Expose to another machine / network",
            value=False,
            key="deployment_remote",
            help="Loopback is the safest default. Remote binding is blocked unless TLS is configured.",
        )
        host = "127.0.0.1"
        cert = ""
        key_path = ""
        if remote:
            st.error("Remote API requires HTTPS. Plain HTTP outside loopback is blocked.")
            host = st.text_input("Bind IPv4 address", value="0.0.0.0", key="deployment_host")
            cert = st.text_input("TLS certificate absolute path", key="deployment_tls_cert")
            key_path = st.text_input("TLS private-key absolute path", type="password", key="deployment_tls_key")
        else:
            st.info("Local-only mode binds to 127.0.0.1 and does not expose the API to the LAN.")
        origins_text = st.text_area(
            "Allowed browser origins (optional, one per line)",
            placeholder="https://internal.example.com",
            key="deployment_origins",
            help="Leave blank for non-browser clients. Wildcard CORS is intentionally blocked.",
        )
        origins = [line.strip() for line in origins_text.splitlines() if line.strip()]

    st.caption(
        "The deployment follows future approved Champion promotions automatically. Candidate, Challenger, Archived, and Rejected models are never served."
    )
    if st.button("Save signed deployment configuration", type="primary", width="stretch", key="deployment_save"):
        try:
            config = create_deployment_config(
                name=name,
                family_id=str(family["family_id"]),
                host=host,
                port=int(port),
                allow_remote=remote,
                tls_cert_path=cert,
                tls_key_path=key_path,
                cors_origins=origins,
                max_rows=int(max_rows),
                max_body_bytes=int(max_body_mb) * 1024 * 1024,
                rate_limit_per_minute=int(rate),
            )
            append_audit_event(
                {
                    "event": "deployment_api_config_saved",
                    "action": "Signed governed deployment configuration saved",
                    "config_id": config["config_id"],
                    "family_id": family["family_id"],
                }
            )
            st.success("Deployment configuration saved and authenticated.")
            st.rerun()
        except Exception as exc:
            st.error(f"Deployment configuration was blocked safely: {safe_error_message(exc)}")


def _render_existing() -> None:
    st.markdown("### Deployment services")
    configs = list_deployment_configs()
    if not configs:
        st.info("No deployment configuration has been created yet.")
        return

    for config in configs:
        config_id = str(config["config_id"])
        status = read_deployment_status(config_id)
        bind = config.get("bind") or {}
        tls = config.get("tls") or {}
        contract = build_api_contract(config)
        with st.expander(f"{'🟢' if status.running else '⚪'} {config.get('name')} · {config_id}", expanded=status.running):
            st.markdown(
                f"<div class='info-box'><b>Family:</b> {safe_html(config.get('family_id'))}<br>"
                f"<b>Endpoint:</b> <code>{safe_html(contract['base_url'])}</code><br>"
                f"<b>Mode:</b> {'HTTPS remote-capable' if tls.get('enabled') else 'HTTP loopback-only'} · "
                f"Champion-only · follows approved promotions</div>",
                unsafe_allow_html=True,
            )
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                if st.button("Start", key=f"deployment_start_{config_id}", disabled=status.running, width="stretch"):
                    try:
                        start_deployment_process(config_id)
                        st.success("Deployment API started.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Start failed safely: {safe_error_message(exc)}")
            with c2:
                if st.button("Stop", key=f"deployment_stop_{config_id}", disabled=not status.running, width="stretch"):
                    try:
                        stop_deployment_process(config_id)
                        st.success("Deployment API stopped.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Stop failed safely: {safe_error_message(exc)}")
            with c3:
                if st.button("Refresh", key=f"deployment_refresh_{config_id}", width="stretch"):
                    st.rerun()
            with c4:
                confirm = st.checkbox("Confirm delete", key=f"deployment_delete_confirm_{config_id}")
                if st.button("Delete", key=f"deployment_delete_{config_id}", disabled=not confirm or status.running, width="stretch"):
                    try:
                        delete_deployment_config(config_id)
                        st.success("Deployment configuration deleted.")
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Delete failed safely: {safe_error_message(exc)}")

            if status.running:
                st.success(f"Running · PID {status.pid} · {status.base_url}")
            elif status.last_error:
                st.warning(f"Last startup error: {safe_html(status.last_error)}")

            st.markdown("#### API contract")
            safe_dataframe(pd.DataFrame(contract["endpoints"]), width="stretch", hide_index=True)
            st.markdown(
                "Request body for validation/prediction: `records` is an array of flat JSON objects. "
                "`/v1/predict` can also receive `include_probabilities: false`. Input rows are never written to the audit log."
            )
            st.code(
                json.dumps(
                    {
                        "records": [{"feature_1": "value", "feature_2": 123.45}],
                        "include_probabilities": True,
                    },
                    indent=2,
                ),
                language="json",
            )


def render(df: pd.DataFrame) -> None:
    st.markdown(
        section_header("🚀", "Deployment API", "Stage 18 · governed Champion predictions"),
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="info-box"><b>Production safety contract:</b> the service resolves the current authenticated '
        '<b>Champion</b> at request time, verifies its signed package, requires a high-entropy Bearer token, limits request size/rate, '
        'does not log source records, and blocks non-loopback HTTP. Remote exposure requires explicit approval plus TLS.</div>',
        unsafe_allow_html=True,
    )
    _render_token()
    st.markdown("---")
    _render_existing()
    st.markdown("---")
    _render_create_config()
