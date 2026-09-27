# ════════════════════════════════════════════════════════
# DataBridge AI — Authentication + Stage 20 Team Identity
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
from core.team_access import ROLE_ADMIN, permissions_for_role
from core.team_store import (
    MIN_PASSWORD_LENGTH,
    AuthenticationResult,
    TeamAuthError,
    TeamUser,
    get_team_auth_store,
)
from core.user_paths import auth_file as user_auth_file
from core.user_paths import team_auth_db

PBKDF2_ITERATIONS = 600_000
MIN_ENV_PBKDF2_ITERATIONS = 260_000
AUTH_VERSION = 2


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _auth_file_path() -> Path:
    """Legacy Stage 1-19 verifier path retained only for safe migration."""
    custom = os.getenv("DATABRIDGE_AUTH_FILE", "").strip()
    if custom:
        return Path(custom).expanduser().resolve()
    return user_auth_file()


def _legacy_auth_file_path() -> Path:
    return _project_root() / ".databridge" / "auth.json"


def _migrate_legacy_auth_file() -> None:
    """Move/remove a pre-Stage-12 auth file from the project tree safely."""
    target = _auth_file_path()
    legacy = _legacy_auth_file_path()
    if not legacy.exists() or target == legacy:
        return
    try:
        raw = legacy.read_bytes()
        legacy_data = json.loads(raw.decode("utf-8"))
        if not isinstance(legacy_data, dict) or not str(legacy_data.get("password_hash") or "").strip():
            return
        if target.exists():
            target_data = json.loads(target.read_text(encoding="utf-8"))
            if not isinstance(target_data, dict) or not str(target_data.get("password_hash") or "").strip():
                return
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            temp = target.with_name(target.name + ".migration.tmp")
            temp.write_bytes(raw)
            try:
                os.chmod(temp, 0o600)
            except OSError:
                pass
            os.replace(temp, target)
            try:
                os.chmod(target, 0o600)
            except OSError:
                pass
        legacy.unlink()
        try:
            legacy.parent.rmdir()
        except OSError:
            pass
    except Exception:
        return


def _load_auth_file() -> dict[str, Any]:
    _migrate_legacy_auth_file()
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
    """Legacy helper retained for compatibility; Stage 20 writes team_auth.db."""
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
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def hash_password(password: str) -> str:
    """Backward-compatible PBKDF2 helper for environment/legacy credentials."""
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), PBKDF2_ITERATIONS
    ).hex()
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt}${digest}"


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify environment-managed hashes using PBKDF2 only.

    Legacy unsalted SHA-256 remains available exclusively inside TeamAuthStore for
    one-time migration and immediate rehash. It is not a valid persistent
    environment-managed authentication format.
    """
    if not str(stored_hash or "").startswith("pbkdf2_sha256$"):
        return False
    try:
        _, iterations, salt_hex, expected = str(stored_hash).split("$", 3)
        parsed_iterations = int(iterations)
        if parsed_iterations < MIN_ENV_PBKDF2_ITERATIONS:
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), parsed_iterations
        ).hex()
        return hmac.compare_digest(digest, expected)
    except Exception:
        return False


def _env_password_is_forced() -> bool:
    return bool(os.getenv("DATABRIDGE_PASSWORD_HASH") or os.getenv("DATABRIDGE_PASSWORD"))


def _env_auth_is_available() -> bool:
    return bool(os.getenv("DATABRIDGE_PASSWORD_HASH") or os.getenv("DATABRIDGE_PASSWORD"))


def _expected_username() -> str:
    if os.getenv("DATABRIDGE_USERNAME"):
        return os.getenv("DATABRIDGE_USERNAME", DEFAULT_LOGIN_USERNAME).strip()
    return DEFAULT_LOGIN_USERNAME


def _expected_password_hash() -> str:
    if os.getenv("DATABRIDGE_PASSWORD_HASH"):
        return os.getenv("DATABRIDGE_PASSWORD_HASH", "")
    if os.getenv("DATABRIDGE_PASSWORD"):
        return hash_password(os.getenv("DATABRIDGE_PASSWORD", ""))
    local = _load_auth_file()
    return str(local.get("password_hash") or "")


def _audit(event: dict[str, Any]) -> None:
    try:
        from core.session import append_audit_event
        append_audit_event(event)
    except Exception:
        pass


def _ensure_team_store():
    store = get_team_auth_store()
    if _env_auth_is_available() or store.count_users() > 0:
        return store
    legacy = _load_auth_file()
    username = str(legacy.get("username") or DEFAULT_LOGIN_USERNAME).strip()
    password_hash = str(legacy.get("password_hash") or "").strip()
    if password_hash:
        try:
            store.bootstrap_legacy_admin(username, password_hash)
            # Default per-user auth.json is retired only after successful import.
            legacy_path = _auth_file_path()
            if legacy_path == user_auth_file() and legacy_path.exists():
                legacy_path.unlink()
        except Exception:
            # Fail closed: keep the old credential so the account is recoverable.
            pass
    return store


def team_mode_enabled() -> bool:
    return not _env_auth_is_available()


def needs_first_run_setup() -> bool:
    if not LOGIN_REQUIRED or _env_auth_is_available():
        return False
    return _ensure_team_store().count_users() == 0


def _set_team_identity(user: TeamUser) -> None:
    st.session_state.is_authenticated = True
    st.session_state.current_user_id = user.user_id
    st.session_state.current_user = user.username
    st.session_state.current_display_name = user.display_name
    st.session_state.current_role = user.role
    st.session_state.current_permissions = sorted(permissions_for_role(user.role))
    st.session_state.current_user_revision = int(user.revision)
    st.session_state.current_user_must_change_password = bool(user.must_change_password)
    st.session_state.current_user_env_managed = False


def _set_env_identity(username: str) -> None:
    st.session_state.is_authenticated = True
    st.session_state.current_user_id = f"env:{username}"
    st.session_state.current_user = username
    st.session_state.current_display_name = username
    st.session_state.current_role = ROLE_ADMIN
    st.session_state.current_permissions = sorted(permissions_for_role(ROLE_ADMIN))
    st.session_state.current_user_revision = 1
    st.session_state.current_user_must_change_password = False
    st.session_state.current_user_env_managed = True


def revalidate_current_session() -> bool:
    if not st.session_state.get("is_authenticated", False):
        return False
    if bool(st.session_state.get("current_user_env_managed", False)):
        if not _env_auth_is_available():
            logout(reason="Environment-managed authentication was removed.")
            return False
        return True
    user_id = str(st.session_state.get("current_user_id") or "")
    if not user_id:
        logout(reason="Session identity is incomplete.")
        return False
    user = _ensure_team_store().get_user(user_id)
    if user is None or not user.active:
        logout(reason="Account was disabled or removed.")
        return False
    _set_team_identity(user)
    return True


def get_current_auth_info() -> dict[str, Any]:
    if _env_auth_is_available():
        username = _expected_username()
        return {
            "username": username,
            "display_name": username,
            "role": ROLE_ADMIN,
            "role_label": "Admin",
            "auth_file": "Environment variables",
            "team_db": str(team_auth_db()),
            "local_config_exists": False,
            "env_managed": True,
            "hash_type": "Environment-managed",
            "first_run_setup_required": False,
            "team_mode": False,
        }
    store = _ensure_team_store()
    current = None
    user_id = str(st.session_state.get("current_user_id") or "")
    if user_id:
        current = store.get_user(user_id)
    return {
        "username": current.username if current else str(st.session_state.get("current_user") or DEFAULT_LOGIN_USERNAME),
        "display_name": current.display_name if current else str(st.session_state.get("current_display_name") or ""),
        "role": current.role if current else str(st.session_state.get("current_role") or ROLE_ADMIN),
        "role_label": current.role_label if current else "Admin",
        "auth_file": str(store.path),
        "team_db": str(store.path),
        "local_config_exists": store.count_users() > 0,
        "env_managed": False,
        "hash_type": "PBKDF2-SHA256 per user",
        "first_run_setup_required": store.count_users() == 0,
        "team_mode": True,
        "user_count": store.count_users(),
    }


def authenticate_user(username: str, password: str) -> AuthenticationResult:
    if _env_auth_is_available():
        expected_username = _expected_username()
        expected_hash = _expected_password_hash()
        ok = hmac.compare_digest(username.strip(), expected_username) and verify_password(password, expected_hash)
        if ok:
            _set_env_identity(expected_username)
            return AuthenticationResult(True, "ok", None)
        return AuthenticationResult(False, "invalid", None)
    result = _ensure_team_store().authenticate(username, password)
    if result.ok and result.user is not None:
        _set_team_identity(result.user)
    return result


def check_credentials(username: str, password: str) -> bool:
    return bool(authenticate_user(username, password).ok)


def create_initial_admin(username: str, new_password: str, confirm_password: str) -> tuple[bool, str]:
    if _env_auth_is_available():
        return False, "Environment-managed credentials are already configured."
    if _ensure_team_store().count_users() > 0:
        return False, "Local Admin account already exists."
    if new_password != confirm_password:
        return False, "Password and confirmation do not match."
    try:
        user = _ensure_team_store().create_initial_admin(username, new_password, display_name=username)
    except TeamAuthError as exc:
        return False, str(exc)
    _set_team_identity(user)
    _audit({"event": "team_admin_bootstrapped", "action": "Initial team Admin account created", "actor": user.username, "actor_role": user.role})
    return True, "Admin account created successfully. Team mode is ready."


def change_password(
    current_password: str,
    new_password: str,
    confirm_password: str,
    username: str | None = None,
) -> tuple[bool, str]:
    if needs_first_run_setup():
        return False, "Create the first Admin account before changing the password."
    if _env_password_is_forced():
        return False, "Password is controlled by environment variables."
    if new_password != confirm_password:
        return False, "New password and confirmation do not match."
    user_id = str(st.session_state.get("current_user_id") or "")
    if not user_id:
        return False, "No authenticated local account is available."
    try:
        user = _ensure_team_store().change_own_password(user_id, current_password, new_password)
    except TeamAuthError as exc:
        return False, str(exc)
    _set_team_identity(user)
    _audit({"event": "password_changed", "action": "User changed their own password"})
    return True, "Password updated successfully."


def logout(*, reason: str = "") -> None:
    """End the identity session and purge all user-scoped in-memory data.

    Stage 20 must never let the next person using the same browser session inherit
    the previous user's dataset, model package, prediction input or temporary AI
    credential widgets.
    """
    username = str(st.session_state.get("current_user") or "")
    role = str(st.session_state.get("current_role") or "")
    if username:
        _audit({"event": "logout", "action": "User signed out", "actor": username, "actor_role": role, "reason": reason})
    manager = st.session_state.get("history_manager")
    try:
        if manager is not None and hasattr(manager, "close"):
            manager.close()
    except Exception:
        pass
    try:
        st.session_state.clear()
    except Exception:
        for key in list(st.session_state.keys()):
            try:
                del st.session_state[key]
            except Exception:
                pass
    st.session_state["is_authenticated"] = False
    st.session_state["current_user_id"] = ""
    st.session_state["current_user"] = ""
    st.session_state["current_role"] = ""


def render_change_password_form(prefix: str = "account", *, mandatory: bool = False) -> None:
    info = get_current_auth_info()
    st.markdown("<div class='settings-card-title'>Account Security</div>", unsafe_allow_html=True)
    st.caption(f"Current user: {info['username']} · Role: {info.get('role_label', 'Admin')} · Hash: {info['hash_type']}")
    if info["first_run_setup_required"]:
        st.info("No local Admin account exists yet. The first-run setup screen will create one.")
        return
    if info["env_managed"]:
        st.warning("Login is managed by environment variables, so password changes are disabled inside the app.")
        return

    with st.form(f"{prefix}_change_password_form", clear_on_submit=True):
        st.text_input("Username", value=info["username"], disabled=True)
        current = st.text_input("Current password", type="password")
        new = st.text_input("New password", type="password")
        confirm = st.text_input("Confirm new password", type="password")
        submitted = st.form_submit_button("Update password", width="stretch")
    if submitted:
        ok, msg = change_password(current, new, confirm)
        if ok:
            st.success(msg)
            if mandatory:
                st.rerun()
        else:
            st.error(msg)


def _render_first_run_setup() -> None:
    top_c1, top_c2, top_c3 = st.columns([1, 1.2, 1])
    with top_c2:
        st.markdown(
            """
            <div class="login-card">
              <div class="login-logo">🌉</div>
              <h2>First-time setup</h2>
              <p>Create the first Admin account. Stage 20 stores team identities in a local per-user SQLite security database.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.form("first_run_setup_form", clear_on_submit=False):
            username = st.text_input("Admin username", value=DEFAULT_LOGIN_USERNAME)
            password = st.text_input("Create password", type="password")
            confirm = st.text_input("Confirm password", type="password")
            submitted = st.form_submit_button("Create Admin account", width="stretch")
        if submitted:
            ok, msg = create_initial_admin(username, password, confirm)
            if ok:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)


def _render_mandatory_password_change() -> None:
    top_c1, top_c2, top_c3 = st.columns([1, 1.2, 1])
    with top_c2:
        st.warning("Your Administrator reset this account password. Change the temporary password before using DataBridge AI.")
        render_change_password_form(prefix="mandatory", mandatory=True)
        if st.button("Sign out", width="stretch", key="mandatory_logout"):
            logout()
            st.rerun()


def render_login_gate() -> bool:
    """Render setup/login screen and revalidate active team sessions every rerun."""
    if not LOGIN_REQUIRED:
        _set_env_identity("local-admin")
        return True

    if st.session_state.get("is_authenticated", False):
        if revalidate_current_session():
            if st.session_state.get("current_user_must_change_password", False):
                _render_mandatory_password_change()
                return False
            return True

    if needs_first_run_setup():
        _render_first_run_setup()
        return False

    top_c1, top_c2, top_c3 = st.columns([1, 1.2, 1])
    with top_c2:
        st.markdown(
            """
            <div class="login-card">
              <div class="login-logo">🌉</div>
              <h2>Sign in to DataBridge AI</h2>
              <p>Stage 20 team access uses per-user roles and revokes disabled sessions on the next interaction.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign in", width="stretch")
        if submitted:
            result = authenticate_user(username, password)
            if result.ok:
                _audit({"event": "login", "action": "User signed in", "actor": username})
                st.rerun()
            else:
                st.error("Invalid username or password, or the account is temporarily unavailable.")
    return False
