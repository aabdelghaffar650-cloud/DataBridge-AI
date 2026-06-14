# ════════════════════════════════════════════════════════
#  DataBridge AI — Local Login Gate + First-run Setup
# ════════════════════════════════════════════════════════
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

from config.settings import DEFAULT_LOGIN_USERNAME, LOGIN_REQUIRED
from core.i18n import LANGUAGE_OPTIONS, t

from pathlib import Path as _DBPath
import os as _db_os

def _databridge_user_data_dir() -> _DBPath:
    """Writable per-user DataBridge AI directory.

    Never write auth/settings into Program Files. Installed desktop apps should
    store mutable user data under LOCALAPPDATA on Windows.
    """
    base = (
        _db_os.environ.get("DATABRIDGE_USER_DATA_DIR")
        or _db_os.environ.get("LOCALAPPDATA")
        or _db_os.environ.get("APPDATA")
        or str(_DBPath.home())
    )
    path = _DBPath(base) / "DataBridgeAI"
    path.mkdir(parents=True, exist_ok=True)
    return path

def _databridge_auth_file() -> _DBPath:
    return _databridge_user_data_dir() / "auth.json"


PBKDF2_ITERATIONS = 260_000
AUTH_VERSION = 2
MIN_PASSWORD_LENGTH = 8


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _auth_file_path() -> Path:
    custom = os.getenv("DATABRIDGE_AUTH_FILE", "").strip()
    if custom:
        return Path(custom).expanduser().resolve()
    return _project_root() / ".databridge" / "auth.json"


def _load_auth_file() -> dict[str, Any]:
    path = _auth_file_path()
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_auth_file(username: str, password_hash: str) -> None:
    path = _auth_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": AUTH_VERSION,
        "username": username.strip() or DEFAULT_LOGIN_USERNAME,
        "password_hash": password_hash,
        "updated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    tmp_path = path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def legacy_sha256(password: str) -> str:
    """Backward compatibility only for users who already created legacy auth files."""
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def hash_password(password: str) -> str:
    """Return a salted PBKDF2 password hash suitable for local app authentication."""
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), PBKDF2_ITERATIONS
    ).hex()
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt}${digest}"


def verify_password(password: str, stored_hash: str) -> bool:
    if not stored_hash:
        return False

    if stored_hash.startswith("pbkdf2_sha256$"):
        try:
            _, iterations, salt_hex, expected = stored_hash.split("$", 3)
            digest = hashlib.pbkdf2_hmac(
                "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), int(iterations)
            ).hex()
            return hmac.compare_digest(digest, expected)
        except Exception:
            return False

    # Legacy fallback only for existing local auth files. New installs never ship a default hash.
    return hmac.compare_digest(legacy_sha256(password), stored_hash)


def _env_password_is_forced() -> bool:
    return bool(os.getenv("DATABRIDGE_PASSWORD_HASH") or os.getenv("DATABRIDGE_PASSWORD"))


def _env_auth_is_available() -> bool:
    return bool(os.getenv("DATABRIDGE_PASSWORD_HASH") or os.getenv("DATABRIDGE_PASSWORD"))


def needs_first_run_setup() -> bool:
    """True when there are no local credentials and no env-managed credentials."""
    return LOGIN_REQUIRED and not _env_auth_is_available() and not _load_auth_file()


def _expected_username() -> str:
    if os.getenv("DATABRIDGE_USERNAME"):
        return os.getenv("DATABRIDGE_USERNAME", DEFAULT_LOGIN_USERNAME).strip()

    local = _load_auth_file()
    return str(local.get("username") or DEFAULT_LOGIN_USERNAME).strip()


def _expected_password_hash() -> str:
    if os.getenv("DATABRIDGE_PASSWORD_HASH"):
        return os.getenv("DATABRIDGE_PASSWORD_HASH", "")
    if os.getenv("DATABRIDGE_PASSWORD"):
        # Env password is not stored; it is converted to a fresh PBKDF2 hash for verification.
        return hash_password(os.getenv("DATABRIDGE_PASSWORD", ""))

    local = _load_auth_file()
    return str(local.get("password_hash") or "")


def get_current_auth_info() -> dict[str, Any]:
    local = _load_auth_file()
    password_hash = _expected_password_hash()
    hash_type = "PBKDF2-SHA256" if password_hash.startswith("pbkdf2_sha256$") else "Legacy SHA-256" if password_hash else "Not configured"
    return {
        "username": _expected_username(),
        "auth_file": str(_auth_file_path()),
        "local_config_exists": bool(local),
        "env_managed": _env_password_is_forced() or bool(os.getenv("DATABRIDGE_USERNAME")),
        "hash_type": hash_type,
        "first_run_setup_required": needs_first_run_setup(),
    }


def check_credentials(username: str, password: str) -> bool:
    return hmac.compare_digest(username.strip(), _expected_username()) and verify_password(
        password, _expected_password_hash()
    )


def create_initial_admin(username: str, new_password: str, confirm_password: str) -> tuple[bool, str]:
    if _env_auth_is_available():
        return False, "Environment-managed credentials are already configured."
    if _load_auth_file():
        return False, "Local admin account already exists."

    username = (username or DEFAULT_LOGIN_USERNAME).strip() or DEFAULT_LOGIN_USERNAME
    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if len(new_password) < MIN_PASSWORD_LENGTH:
        return False, f"Password must be at least {MIN_PASSWORD_LENGTH} characters."
    if new_password != confirm_password:
        return False, "Password and confirmation do not match."

    _save_auth_file(username, hash_password(new_password))
    st.session_state.is_authenticated = True
    st.session_state.current_user = username
    return True, "Admin account created successfully."


def change_password(
    current_password: str,
    new_password: str,
    confirm_password: str,
    username: str | None = None,
) -> tuple[bool, str]:
    """Update local credentials. Returns (ok, message)."""
    if needs_first_run_setup():
        return False, "Create the first admin account before changing the password."
    if _env_password_is_forced():
        return False, "Password is controlled by environment variables. Remove DATABRIDGE_PASSWORD or DATABRIDGE_PASSWORD_HASH to change it from the app."

    current_user = _expected_username()
    new_username = (username or current_user).strip() or current_user

    if not verify_password(current_password, _expected_password_hash()):
        return False, "Current password is incorrect."
    if len(new_password) < MIN_PASSWORD_LENGTH:
        return False, f"New password must be at least {MIN_PASSWORD_LENGTH} characters."
    if new_password != confirm_password:
        return False, "New password and confirmation do not match."
    if new_password == current_password:
        return False, "New password must be different from the current password."

    _save_auth_file(new_username, hash_password(new_password))
    st.session_state.current_user = new_username
    return True, f"Password updated successfully. Credentials saved to {_auth_file_path()}"


def logout() -> None:
    st.session_state.is_authenticated = False
    st.session_state.current_user = ""


def render_change_password_form(prefix: str = "account") -> None:
    """Reusable password-change form for sidebar and Settings page."""
    info = get_current_auth_info()
    st.markdown("<div class='settings-card-title'>Account Security</div>", unsafe_allow_html=True)
    st.caption(f"Current user: {info['username']} · Hash: {info['hash_type']}")

    if info["first_run_setup_required"]:
        st.info("No local admin account exists yet. The first-run setup screen will create one.")
        return
    if info["env_managed"]:
        st.warning("Login is managed by environment variables, so password changes are disabled inside the app.")

    with st.form(f"{prefix}_change_password_form", clear_on_submit=True):
        username = st.text_input("Username", value=info["username"], disabled=info["env_managed"])
        current = st.text_input("Current password", type="password", disabled=info["env_managed"])
        new = st.text_input("New password", type="password", disabled=info["env_managed"])
        confirm = st.text_input("Confirm new password", type="password", disabled=info["env_managed"])
        submitted = st.form_submit_button("Update password", use_container_width=True, disabled=info["env_managed"])

    if submitted:
        ok, msg = change_password(current, new, confirm, username=username)
        if ok:
            st.success(msg)
        else:
            st.error(msg)


def _render_first_run_setup() -> None:
    top_c1, top_c2, top_c3 = st.columns([1, 1.2, 1])
    with top_c2:
        st.selectbox(
            "🌐 Language / اللغة",
            list(LANGUAGE_OPTIONS.keys()),
            format_func=lambda code: LANGUAGE_OPTIONS[code],
            key="language",
            label_visibility="visible",
        )
        st.markdown(
            """
            <div class="login-card">
              <div class="login-logo">🌉</div>
              <h2>First-time setup</h2>
              <p>Create the local admin account. No default password is shipped with DataBridge AI.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.form("first_run_setup_form", clear_on_submit=False):
            username = st.text_input("Admin username", value=DEFAULT_LOGIN_USERNAME)
            password = st.text_input("Create password", type="password")
            confirm = st.text_input("Confirm password", type="password")
            submitted = st.form_submit_button("Create admin account", use_container_width=True)
        if submitted:
            ok, msg = create_initial_admin(username, password, confirm)
            if ok:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)


def _sync_secrets_to_env() -> None:
    """Bridge Streamlit Cloud Secrets into os.environ for credential overrides.

    Lets a deployment configure DATABRIDGE_USERNAME / DATABRIDGE_PASSWORD (or
    DATABRIDGE_PASSWORD_HASH) via the Streamlit Cloud "Secrets" panel without any
    code change, so a public demo account works without the first-run setup gate.
    """
    for key in ("DATABRIDGE_USERNAME", "DATABRIDGE_PASSWORD", "DATABRIDGE_PASSWORD_HASH"):
        if not os.environ.get(key):
            try:
                if key in st.secrets:
                    os.environ[key] = str(st.secrets[key])
            except Exception:
                pass


def render_login_gate() -> bool:
    """Render setup/login screen if needed. Returns True when the app can continue."""
    _sync_secrets_to_env()

    if not LOGIN_REQUIRED:
        st.session_state.is_authenticated = True
        return True

    if st.session_state.get("is_authenticated", False):
        return True

    if needs_first_run_setup():
        _render_first_run_setup()
        return False

    top_c1, top_c2, top_c3 = st.columns([1, 1.2, 1])
    with top_c2:
        st.selectbox(
            "🌐 Language / اللغة",
            list(LANGUAGE_OPTIONS.keys()),
            format_func=lambda code: LANGUAGE_OPTIONS[code],
            key="language",
            label_visibility="visible",
        )

        st.markdown(
            f"""
            <div class="login-card">
              <div class="login-logo">🌉</div>
              <h2>{t('login_title')}</h2>
              <p>{t('login_subtitle')}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.form("login_form", clear_on_submit=False):
            username = st.text_input(t("username"), value=_expected_username())
            password = st.text_input(t("password"), type="password")
            submitted = st.form_submit_button(t("sign_in"), use_container_width=True)

        if submitted:
            if check_credentials(username, password):
                st.session_state.is_authenticated = True
                st.session_state.current_user = username.strip()
                st.rerun()
            else:
                st.error(t("wrong_login"))

    return False
