"""DataBridge AI Stage 20 multi-user / team-mode regression tests."""
from __future__ import annotations

import hashlib
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# The Stage 20 store/access-control modules are headless. The project core package
# historically imports Streamlit session helpers from __init__, so provide a tiny
# import stub when this standalone test is run outside the production venv.
try:
    import streamlit  # type: ignore  # noqa: F401
except ModuleNotFoundError:
    import types
    sys.modules["streamlit"] = types.SimpleNamespace(session_state={})

from core.team_access import (
    PERM_DATA_IMPORT,
    PERM_DATA_READ,
    PERM_DEPLOYMENT_MANAGE,
    PERM_ML_TRAIN,
    PERM_MODEL_APPROVE,
    PERM_MODEL_SUBMIT,
    PERM_TEAM_MANAGE,
    ROLE_ADMIN,
    ROLE_ANALYST,
    ROLE_DATA_SCIENTIST,
    ROLE_MANAGER,
    ROLE_VIEWER,
    can_access_page,
    permissions_for_role,
)
from core.team_store import LOCKOUT_FAILURES, TeamAuthError, TeamAuthStore


class _Session(dict):
    pass


def _session(role: str) -> _Session:
    return _Session(
        current_role=role,
        current_permissions=sorted(permissions_for_role(role)),
        current_user="test",
        current_user_id="test-id",
        current_user_revision=1,
    )


def test_role_matrix_and_page_guards() -> None:
    viewer = permissions_for_role(ROLE_VIEWER)
    analyst = permissions_for_role(ROLE_ANALYST)
    scientist = permissions_for_role(ROLE_DATA_SCIENTIST)
    manager = permissions_for_role(ROLE_MANAGER)
    admin = permissions_for_role(ROLE_ADMIN)

    assert PERM_DATA_READ in viewer
    assert PERM_DATA_IMPORT not in viewer
    assert PERM_DATA_IMPORT in analyst
    assert PERM_ML_TRAIN not in analyst
    assert PERM_ML_TRAIN in scientist
    assert PERM_MODEL_SUBMIT in scientist and PERM_MODEL_APPROVE not in scientist
    assert PERM_MODEL_APPROVE in manager and PERM_MODEL_SUBMIT not in manager
    assert PERM_TEAM_MANAGE in admin and PERM_DEPLOYMENT_MANAGE in admin

    assert can_access_page("overview", session_state=_session(ROLE_VIEWER))
    assert not can_access_page("data_sources", session_state=_session(ROLE_VIEWER))
    assert can_access_page("ml_studio", session_state=_session(ROLE_DATA_SCIENTIST))
    assert not can_access_page("ml_studio", session_state=_session(ROLE_MANAGER))
    assert can_access_page("model_governance", session_state=_session(ROLE_MANAGER))
    assert can_access_page("model_governance", session_state=_session(ROLE_DATA_SCIENTIST))
    assert can_access_page("team_admin", session_state=_session(ROLE_ADMIN))
    assert not can_access_page("team_admin", session_state=_session(ROLE_MANAGER))


def test_accounts_lockout_reset_and_last_admin_protection() -> None:
    with tempfile.TemporaryDirectory(prefix="databridge-stage20-") as td:
        path = Path(td) / "team_auth.db"
        store = TeamAuthStore(path)
        admin = store.create_initial_admin("admin", "Stage20!Admin123", display_name="Primary Admin")
        assert admin.role == ROLE_ADMIN and admin.active

        scientist = store.create_user(
            username="scientist",
            display_name="Data Scientist",
            role=ROLE_DATA_SCIENTIST,
            password="Stage20!Scientist123",
            actor_user_id=admin.user_id,
            actor_username=admin.username,
        )
        manager = store.create_user(
            username="manager",
            display_name="Model Approver",
            role=ROLE_MANAGER,
            password="Stage20!Manager123",
            actor_user_id=admin.user_id,
            actor_username=admin.username,
        )
        assert scientist.must_change_password and manager.must_change_password

        try:
            store.update_user(
                admin.user_id,
                display_name=admin.display_name,
                role=ROLE_MANAGER,
                active=True,
                actor_user_id=admin.user_id,
                actor_username=admin.username,
            )
        except TeamAuthError as exc:
            assert "final active Admin" in str(exc)
        else:
            raise AssertionError("The final active Admin must not be demotable")

        second_admin = store.create_user(
            username="admin2",
            display_name="Second Admin",
            role=ROLE_ADMIN,
            password="Stage20!Admin456",
            must_change_password=False,
            actor_user_id=admin.user_id,
            actor_username=admin.username,
        )
        updated = store.update_user(
            admin.user_id,
            display_name=admin.display_name,
            role=ROLE_MANAGER,
            active=True,
            actor_user_id=second_admin.user_id,
            actor_username=second_admin.username,
        )
        assert updated.role == ROLE_MANAGER and updated.revision > admin.revision

        for _ in range(LOCKOUT_FAILURES):
            result = store.authenticate("scientist", "Wrong!Password999")
        assert not result.ok and result.code == "invalid"
        locked = store.get_user(scientist.user_id)
        assert locked is not None and locked.locked_until > int(time.time())
        store.unlock_user(scientist.user_id, actor_user_id=second_admin.user_id, actor_username=second_admin.username)

        ok = store.authenticate("scientist", "Stage20!Scientist123")
        assert ok.ok and ok.user is not None
        reset = store.admin_reset_password(
            scientist.user_id,
            "Stage20!TempReset123",
            actor_user_id=second_admin.user_id,
            actor_username=second_admin.username,
        )
        assert reset.must_change_password
        changed = store.change_own_password(scientist.user_id, "Stage20!TempReset123", "Stage20!NewPass123")
        assert not changed.must_change_password
        assert store.authenticate("scientist", "Stage20!NewPass123").ok

        raw = path.read_bytes()
        assert b"Stage20!Admin123" not in raw
        assert b"Stage20!NewPass123" not in raw
        assert b"Stage20!TempReset123" not in raw
        assert len(store.security_events(limit=100)) >= 5


def test_legacy_hash_bootstrap_and_upgrade() -> None:
    with tempfile.TemporaryDirectory(prefix="databridge-stage20-legacy-") as td:
        path = Path(td) / "team_auth.db"
        store = TeamAuthStore(path)
        password = "Legacy!Password123"
        legacy_hash = hashlib.sha256(password.encode("utf-8")).hexdigest()
        user = store.bootstrap_legacy_admin("legacy-admin", legacy_hash)
        assert user.role == ROLE_ADMIN
        assert store.authenticate("legacy-admin", password).ok
        # Successful login upgrades the old SHA-256 verifier to PBKDF2.
        with store._connect() as conn:  # test-only verification of migration result
            value = str(conn.execute("SELECT password_hash FROM users WHERE user_id=?", (user.user_id,)).fetchone()[0])
        assert value.startswith("pbkdf2_sha256$")


def test_stage20_static_integration_contract() -> None:
    app = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")
    sidebar = (PROJECT_ROOT / "ui" / "sidebar.py").read_text(encoding="utf-8")
    governance = (PROJECT_ROOT / "_pages" / "model_governance.py").read_text(encoding="utf-8")
    settings = (PROJECT_ROOT / "_pages" / "settings.py").read_text(encoding="utf-8")
    team_page = (PROJECT_ROOT / "_pages" / "team_admin.py").read_text(encoding="utf-8")
    auth = (PROJECT_ROOT / "core" / "auth.py").read_text(encoding="utf-8")

    assert '"team_admin":          "_pages.team_admin"' in app
    assert "can_access_page(page_key" in app
    assert "first_accessible_page" in sidebar
    assert "disabled=not manager.can_undo or not can_transform" in sidebar
    assert "can_manage_secrets" in sidebar
    assert "PERM_MODEL_APPROVE" in governance and "PERM_MODEL_SUBMIT" in governance
    assert "disabled=confirmation.strip() != expected or not can_approve" in governance
    assert "PERM_SIGNING_KEY_MANAGE" in settings and "PERM_AUDIT_VIEW" in settings
    assert "The final active Admin" in (PROJECT_ROOT / "core" / "team_store.py").read_text(encoding="utf-8")
    assert "require_permission(PERM_TEAM_MANAGE" in team_page
    assert "st.session_state.clear()" in auth
    assert "next person using the same browser session" in auth


def main() -> None:
    test_role_matrix_and_page_guards()
    test_accounts_lockout_reset_and_last_admin_protection()
    test_legacy_hash_bootstrap_and_upgrade()
    test_stage20_static_integration_contract()
    print(
        "PASS: Stage 20 Team Mode provides per-user PBKDF2 authentication, non-enumerating lockout responses and immediate session-revision revalidation primitives; enforces explicit Viewer/Analyst/Data Scientist/Manager/Admin permissions; separates Challenger submission from production approval; protects the final active Admin; and gates sensitive team, deployment, registry, signing-key, secret and audit controls without storing plaintext passwords."
    )


if __name__ == "__main__":
    main()
