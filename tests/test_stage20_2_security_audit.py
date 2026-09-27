"""Stage 20.2 regression tests for security-audit authentication hardening."""
from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import streamlit  # type: ignore  # noqa: F401
except ModuleNotFoundError:
    import types
    sys.modules["streamlit"] = types.SimpleNamespace(session_state={})

import core.auth as auth
import core.team_store as team_store_module
from core.team_access import ROLE_ANALYST
from core.team_store import (
    LOCKOUT_FAILURES,
    PBKDF2_ITERATIONS,
    TeamAuthError,
    TeamAuthStore,
    hash_password,
)


def _stored_hash(store: TeamAuthStore, user_id: str) -> str:
    with store._connect() as conn:  # test-only inspection
        row = conn.execute("SELECT password_hash FROM users WHERE user_id=?", (user_id,)).fetchone()
    assert row is not None
    return str(row[0])


def _old_pbkdf2(password: str, iterations: int = 260_000) -> str:
    salt = bytes.fromhex("00112233445566778899aabbccddeeff")
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations).hex()
    return f"pbkdf2_sha256${iterations}${salt.hex()}${digest}"


def test_current_password_work_factor_and_upgrade_paths() -> None:
    assert PBKDF2_ITERATIONS >= 600_000
    assert auth.PBKDF2_ITERATIONS >= 600_000

    current = hash_password("Stage20!CurrentHash123")
    assert current.startswith(f"pbkdf2_sha256${PBKDF2_ITERATIONS}$")

    with tempfile.TemporaryDirectory(prefix="databridge-stage20-2-upgrade-") as td:
        store = TeamAuthStore(Path(td) / "team_auth.db")

        legacy_password = "Legacy!Stage20Hash123"
        legacy_hash = hashlib.sha256(legacy_password.encode("utf-8")).hexdigest()
        legacy = store.bootstrap_legacy_admin("legacy-admin", legacy_hash)
        assert store.authenticate(legacy.username, legacy_password).ok
        assert _stored_hash(store, legacy.user_id).startswith(f"pbkdf2_sha256${PBKDF2_ITERATIONS}$")

        # A previously valid PBKDF2 verifier with the old Stage 20 cost is also
        # rehashed to the current work factor after the next successful login.
        old_password = "OldPbkdf2!Stage20_123"
        with store._connect() as conn:
            conn.execute(
                "UPDATE users SET password_hash=? WHERE user_id=?",
                (_old_pbkdf2(old_password), legacy.user_id),
            )
        assert store.authenticate(legacy.username, old_password).ok
        assert _stored_hash(store, legacy.user_id).startswith(f"pbkdf2_sha256${PBKDF2_ITERATIONS}$")


def test_legacy_sha256_is_not_valid_for_environment_managed_auth() -> None:
    password = "EnvManaged!Stage20_123"
    legacy_hash = hashlib.sha256(password.encode("utf-8")).hexdigest()
    assert not auth.verify_password(password, legacy_hash)

    current = auth.hash_password(password)
    assert auth.verify_password(password, current)

    # Existing environment-managed PBKDF2 hashes remain compatible across the
    # hotfix; newly generated hashes use the stronger current work factor.
    assert auth.verify_password(password, _old_pbkdf2(password))


def test_lockout_does_not_expose_account_existence_and_work_is_equalized() -> None:
    with tempfile.TemporaryDirectory(prefix="databridge-stage20-2-enum-") as td:
        store = TeamAuthStore(Path(td) / "team_auth.db")
        admin = store.create_initial_admin("admin", "Stage20!AdminEnum123")
        user = store.create_user(
            username="analyst",
            display_name="Analyst",
            role=ROLE_ANALYST,
            password="Stage20!AnalystEnum123",
            actor_user_id=admin.user_id,
            actor_username=admin.username,
        )

        calls: list[int] = []
        original_burn = team_store_module._burn_password_cost

        def record_burn(password: str, iterations: int) -> None:
            calls.append(int(iterations))

        team_store_module._burn_password_cost = record_burn
        try:
            missing = store.authenticate("definitely-missing", "Wrong!Password123")
            assert not missing.ok and missing.code == "invalid"
            assert calls[-1] == PBKDF2_ITERATIONS
        finally:
            team_store_module._burn_password_cost = original_burn

        for _ in range(LOCKOUT_FAILURES):
            result = store.authenticate(user.username, "Wrong!Password123")
        assert not result.ok and result.code == "invalid"
        locked = store.get_user(user.user_id)
        assert locked is not None and locked.locked_until > int(time.time())

        calls = []
        team_store_module._burn_password_cost = record_burn
        try:
            locked_result = store.authenticate(user.username, "Stage20!AnalystEnum123")
            assert not locked_result.ok and locked_result.code == "invalid"
            assert calls[-1] == PBKDF2_ITERATIONS
        finally:
            team_store_module._burn_password_cost = original_burn


def test_team_admin_mutations_authorize_persisted_actor_identity() -> None:
    with tempfile.TemporaryDirectory(prefix="databridge-stage20-2-rbac-") as td:
        store = TeamAuthStore(Path(td) / "team_auth.db")
        admin = store.create_initial_admin("admin", "Stage20!AdminRbac123")
        analyst = store.create_user(
            username="analyst",
            display_name="Analyst",
            role=ROLE_ANALYST,
            password="Stage20!AnalystRbac123",
            actor_user_id=admin.user_id,
            actor_username=admin.username,
        )

        try:
            store.create_user(
                username="forbidden-user",
                display_name="Forbidden",
                role=ROLE_ANALYST,
                password="Stage20!Forbidden123",
                actor_user_id=analyst.user_id,
                actor_username=analyst.username,
            )
        except TeamAuthError as exc:
            assert "not authorized" in str(exc)
        else:
            raise AssertionError("A non-Admin persisted actor must not create team accounts")

        try:
            store.update_user(
                analyst.user_id,
                display_name="Escalated",
                role="admin",
                active=True,
                actor_user_id=analyst.user_id,
                actor_username=analyst.username,
            )
        except TeamAuthError as exc:
            assert "not authorized" in str(exc)
        else:
            raise AssertionError("A non-Admin persisted actor must not escalate roles")

        try:
            store.admin_reset_password(
                analyst.user_id,
                "Stage20!ResetBlocked123",
                actor_user_id=admin.user_id,
                actor_username="spoofed-admin-name",
            )
        except TeamAuthError as exc:
            assert "does not match" in str(exc)
        else:
            raise AssertionError("Actor username spoofing must be rejected")

        # The valid persisted Admin identity remains authorized.
        updated = store.update_user(
            analyst.user_id,
            display_name="Analyst Updated",
            role=ROLE_ANALYST,
            active=True,
            actor_user_id=admin.user_id,
            actor_username=admin.username,
        )
        assert updated.display_name == "Analyst Updated"


def test_model_package_trust_boundary_is_documented_precisely() -> None:
    source = (PROJECT_ROOT / "modules" / "model_package.py").read_text(encoding="utf-8")
    assert "trust domain, not a unique installation" in source
    assert "different DataBridge AI model trust key" in source
    assert "If that signing key is compromised" in source
    assert "not a sandbox" in source
    assert "different DataBridge AI installation. It was not deserialized." not in source


def test_release_contract_includes_stage20_2() -> None:
    import json

    spec = json.loads((PROJECT_ROOT / "release" / "release_spec.json").read_text(encoding="utf-8"))
    assert "tests/test_stage20_2_security_audit.py" in spec["stage_tests"]
    auth_source = (PROJECT_ROOT / "core" / "auth.py").read_text(encoding="utf-8")
    store_source = (PROJECT_ROOT / "core" / "team_store.py").read_text(encoding="utf-8")
    assert "legacy_sha256" not in auth_source
    assert "20_000" not in store_source
    assert "return AuthenticationResult(False, \"locked\")" not in store_source
    assert "Invalid username or password, or the account is temporarily unavailable." in auth_source


def main() -> None:
    test_current_password_work_factor_and_upgrade_paths()
    test_legacy_sha256_is_not_valid_for_environment_managed_auth()
    test_lockout_does_not_expose_account_existence_and_work_is_equalized()
    test_team_admin_mutations_authorize_persisted_actor_identity()
    test_model_package_trust_boundary_is_documented_precisely()
    test_release_contract_includes_stage20_2()
    print(
        "PASS: Stage 20.2 raises the current PBKDF2 work factor, confines legacy SHA-256 to one-time Team Store migration, upgrades older verifiers after successful login, removes username/lockout disclosure, equalizes blocked authentication work, authorizes Team Administration mutations against the persisted actor identity, and documents the signed-pickle trust boundary accurately."
    )


if __name__ == "__main__":
    main()
