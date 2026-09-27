"""Local team identity store for DataBridge AI Stage 20.

The store uses SQLite with parameterized statements, per-user salted PBKDF2
verifiers, account lockout, revision-based session revalidation and fail-closed
protection for the final active Admin account.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import sqlite3
import time
from contextlib import contextmanager
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from core.team_access import (
    PERM_TEAM_MANAGE,
    ROLE_ADMIN,
    ROLE_LABELS,
    VALID_ROLES,
    normalize_role,
    permissions_for_role,
)
from core.user_paths import team_auth_db

PBKDF2_ITERATIONS = 600_000
MIN_PASSWORD_LENGTH = 10
MAX_USERNAME_LENGTH = 64
MAX_DISPLAY_NAME_LENGTH = 120
LOCKOUT_FAILURES = 5
LOCKOUT_SECONDS = 5 * 60
SCHEMA_VERSION = 1


class TeamAuthError(RuntimeError):
    pass


@dataclass(frozen=True)
class TeamUser:
    user_id: str
    username: str
    display_name: str
    role: str
    active: bool
    must_change_password: bool
    revision: int
    created_at: str
    updated_at: str
    last_login_at: str
    locked_until: int

    @property
    def role_label(self) -> str:
        return ROLE_LABELS.get(self.role, self.role)


@dataclass(frozen=True)
class AuthenticationResult:
    ok: bool
    code: str
    user: TeamUser | None = None


def _utc_text() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _username_key(username: str) -> str:
    return str(username or "").strip().casefold()


def validate_username(username: str) -> str:
    clean = str(username or "").strip()
    if len(clean) < 3 or len(clean) > MAX_USERNAME_LENGTH:
        raise TeamAuthError(f"Username must be 3-{MAX_USERNAME_LENGTH} characters.")
    if any(ord(ch) < 32 for ch in clean) or any(ch in "\r\n\t/\\" for ch in clean):
        raise TeamAuthError("Username contains unsupported characters.")
    return clean


def validate_password(password: str) -> None:
    value = str(password or "")
    if len(value) < MIN_PASSWORD_LENGTH:
        raise TeamAuthError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters.")
    classes = sum(
        [
            any(ch.islower() for ch in value),
            any(ch.isupper() for ch in value),
            any(ch.isdigit() for ch in value),
            any(not ch.isalnum() for ch in value),
        ]
    )
    if classes < 3:
        raise TeamAuthError("Password must use at least three of: uppercase, lowercase, number, symbol.")


def _hash_password_unchecked(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", str(password).encode("utf-8"), bytes.fromhex(salt), PBKDF2_ITERATIONS
    ).hex()
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt}${digest}"


def hash_password(password: str) -> str:
    validate_password(password)
    return _hash_password_unchecked(password)


def _pbkdf2_iterations(stored_hash: str) -> int:
    value = str(stored_hash or "")
    if not value.startswith("pbkdf2_sha256$"):
        return 0
    try:
        _, iterations, salt_hex, expected = value.split("$", 3)
        parsed = int(iterations)
        if parsed < 1 or len(bytes.fromhex(salt_hex)) < 16 or len(expected) != 64:
            return 0
        return parsed
    except Exception:
        return 0


def password_hash_needs_upgrade(stored_hash: str) -> bool:
    iterations = _pbkdf2_iterations(stored_hash)
    return iterations < PBKDF2_ITERATIONS


def verify_password(password: str, stored_hash: str) -> bool:
    value = str(stored_hash or "")
    if not value.startswith("pbkdf2_sha256$"):
        # Legacy unsalted SHA-256 is accepted only inside the Team Store migration
        # path. A successful login immediately replaces it with the current PBKDF2
        # verifier; environment-managed authentication does not use this fallback.
        if len(value) == 64:
            candidate = hashlib.sha256(str(password).encode("utf-8")).hexdigest()
            return hmac.compare_digest(candidate, value)
        return False
    try:
        _, iterations, salt_hex, expected = value.split("$", 3)
        digest = hashlib.pbkdf2_hmac(
            "sha256", str(password).encode("utf-8"), bytes.fromhex(salt_hex), int(iterations)
        ).hex()
        return hmac.compare_digest(digest, expected)
    except Exception:
        return False


def _burn_password_cost(password: str, iterations: int) -> None:
    """Spend deterministic PBKDF2 work without creating a reusable verifier."""
    work = max(0, int(iterations))
    if work:
        hashlib.pbkdf2_hmac(
            "sha256",
            str(password).encode("utf-8"),
            b"DataBridgeAI-auth-equalizer-v1",
            work,
        )


def _equalize_failed_authentication(password: str, stored_hash: str = "") -> None:
    """Bring failed/blocked authentication paths up to the current work factor."""
    spent = _pbkdf2_iterations(stored_hash)
    _burn_password_cost(password, max(0, PBKDF2_ITERATIONS - spent))


class TeamAuthStore:
    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path is not None else team_auth_db()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connect(self):
        """Open a short-lived SQLite connection and always close it.

        ``sqlite3.Connection`` implements transaction handling in its context
        manager, but it does *not* close the underlying database handle on
        ``__exit__``.  That behavior is easy to miss and is particularly
        visible on Windows, where an otherwise finished test/process can keep
        ``team_auth.db`` locked.  This wrapper preserves commit/rollback
        semantics and guarantees ``close()`` in ``finally``.
        """
        connection = sqlite3.connect(str(self.path), timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT NOT NULL,
                    username_key TEXT NOT NULL UNIQUE,
                    display_name TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 1,
                    must_change_password INTEGER NOT NULL DEFAULT 0,
                    failed_attempts INTEGER NOT NULL DEFAULT 0,
                    locked_until INTEGER NOT NULL DEFAULT 0,
                    revision INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_login_at TEXT NOT NULL DEFAULT ''
                );
                CREATE TABLE IF NOT EXISTS security_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    actor_user_id TEXT NOT NULL DEFAULT '',
                    actor_username TEXT NOT NULL DEFAULT '',
                    action TEXT NOT NULL,
                    target_user_id TEXT NOT NULL DEFAULT '',
                    details_json TEXT NOT NULL DEFAULT '{}'
                );
                """
            )
            conn.execute(
                "INSERT INTO meta(key, value) VALUES('schema_version', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(SCHEMA_VERSION),),
            )
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass

    @staticmethod
    def _row_to_user(row: sqlite3.Row | None) -> TeamUser | None:
        if row is None:
            return None
        return TeamUser(
            user_id=str(row["user_id"]),
            username=str(row["username"]),
            display_name=str(row["display_name"]),
            role=str(row["role"]),
            active=bool(row["active"]),
            must_change_password=bool(row["must_change_password"]),
            revision=int(row["revision"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            last_login_at=str(row["last_login_at"]),
            locked_until=int(row["locked_until"] or 0),
        )

    def _require_actor_permission(
        self,
        conn: sqlite3.Connection,
        *,
        actor_user_id: str,
        actor_username: str,
        permission: str,
    ) -> None:
        """Authorize a mutating team operation against the persisted actor identity."""
        actor_id = str(actor_user_id or "").strip()
        actor_name = str(actor_username or "").strip()
        if not actor_id or not actor_name:
            raise TeamAuthError("Authenticated team administrator identity is required.")
        row = conn.execute(
            "SELECT username, role, active FROM users WHERE user_id=?",
            (actor_id,),
        ).fetchone()
        if row is None or not bool(row["active"]):
            raise TeamAuthError("The acting account is unavailable or is not authorized.")
        if _username_key(str(row["username"])) != _username_key(actor_name):
            raise TeamAuthError("The acting account identity does not match the authenticated user.")
        try:
            allowed = str(permission) in permissions_for_role(str(row["role"]))
        except Exception:
            allowed = False
        if not allowed:
            raise TeamAuthError("The acting account is not authorized for team administration.")

    def _event(
        self,
        conn: sqlite3.Connection,
        action: str,
        *,
        actor_user_id: str = "",
        actor_username: str = "",
        target_user_id: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        payload = json.dumps(details or {}, ensure_ascii=False, sort_keys=True)[:4000]
        conn.execute(
            "INSERT INTO security_events(timestamp, actor_user_id, actor_username, action, target_user_id, details_json) "
            "VALUES(?,?,?,?,?,?)",
            (_utc_text(), actor_user_id[:80], actor_username[:80], action[:120], target_user_id[:80], payload),
        )

    def count_users(self, *, active_only: bool = False) -> int:
        query = "SELECT COUNT(*) FROM users" + (" WHERE active=1" if active_only else "")
        with self._connect() as conn:
            return int(conn.execute(query).fetchone()[0])

    def count_active_admins(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM users WHERE active=1 AND role=?", (ROLE_ADMIN,)).fetchone()[0])

    def list_users(self) -> list[TeamUser]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM users ORDER BY CASE role WHEN 'admin' THEN 0 WHEN 'manager' THEN 1 WHEN 'data_scientist' THEN 2 WHEN 'analyst' THEN 3 ELSE 4 END, username_key"
            ).fetchall()
        return [self._row_to_user(row) for row in rows if row is not None]

    def get_user_by_username(self, username: str) -> TeamUser | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE username_key=?", (_username_key(username),)).fetchone()
        return self._row_to_user(row)

    def get_user(self, user_id: str) -> TeamUser | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE user_id=?", (str(user_id),)).fetchone()
        return self._row_to_user(row)

    def bootstrap_legacy_admin(self, username: str, password_hash: str) -> TeamUser:
        if self.count_users() != 0:
            user = self.get_user_by_username(username)
            if user is None:
                raise TeamAuthError("Team store already contains accounts; legacy bootstrap was refused.")
            return user
        clean_username = validate_username(username)
        if not str(password_hash or "").strip():
            raise TeamAuthError("Legacy credential has no password verifier.")
        now = _utc_text()
        user_id = uuid.uuid4().hex
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "INSERT INTO users(user_id, username, username_key, display_name, password_hash, role, active, must_change_password, revision, created_at, updated_at) "
                "VALUES(?,?,?,?,?,?,1,0,1,?,?)",
                (user_id, clean_username, _username_key(clean_username), clean_username, str(password_hash), ROLE_ADMIN, now, now),
            )
            self._event(conn, "legacy_admin_migrated", target_user_id=user_id, details={"role": ROLE_ADMIN})
            conn.commit()
        user = self.get_user(user_id)
        if user is None:
            raise TeamAuthError("Could not read migrated Admin account.")
        return user

    def create_initial_admin(self, username: str, password: str, *, display_name: str = "") -> TeamUser:
        if self.count_users() != 0:
            raise TeamAuthError("The first Admin account already exists.")
        return self.create_user(
            username=username,
            display_name=display_name or username,
            role=ROLE_ADMIN,
            password=password,
            must_change_password=False,
            actor_user_id="system",
            actor_username="system",
            allow_when_empty=True,
        )

    def create_user(
        self,
        *,
        username: str,
        display_name: str,
        role: str,
        password: str,
        must_change_password: bool = True,
        actor_user_id: str = "",
        actor_username: str = "",
        allow_when_empty: bool = False,
    ) -> TeamUser:
        clean_username = validate_username(username)
        clean_role = normalize_role(role)
        clean_display = str(display_name or clean_username).strip()[:MAX_DISPLAY_NAME_LENGTH] or clean_username
        validate_password(password)
        if not allow_when_empty and self.count_users() == 0:
            raise TeamAuthError("Create the initial Admin through first-run setup.")
        now = _utc_text()
        user_id = uuid.uuid4().hex
        try:
            with self._connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                initial_system_bootstrap = bool(
                    allow_when_empty
                    and actor_user_id == "system"
                    and int(conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]) == 0
                )
                if not initial_system_bootstrap:
                    self._require_actor_permission(
                        conn,
                        actor_user_id=actor_user_id,
                        actor_username=actor_username,
                        permission=PERM_TEAM_MANAGE,
                    )
                password_hash = _hash_password_unchecked(password)
                conn.execute(
                    "INSERT INTO users(user_id, username, username_key, display_name, password_hash, role, active, must_change_password, revision, created_at, updated_at) "
                    "VALUES(?,?,?,?,?,?,1,?,1,?,?)",
                    (
                        user_id,
                        clean_username,
                        _username_key(clean_username),
                        clean_display,
                        password_hash,
                        clean_role,
                        1 if must_change_password else 0,
                        now,
                        now,
                    ),
                )
                self._event(
                    conn,
                    "user_created",
                    actor_user_id=actor_user_id,
                    actor_username=actor_username,
                    target_user_id=user_id,
                    details={"username": clean_username, "role": clean_role, "must_change_password": bool(must_change_password)},
                )
                conn.commit()
        except sqlite3.IntegrityError as exc:
            raise TeamAuthError("Username already exists.") from exc
        user = self.get_user(user_id)
        if user is None:
            raise TeamAuthError("Created account could not be read back safely.")
        return user

    def authenticate(self, username: str, password: str) -> AuthenticationResult:
        key = _username_key(username)
        now_epoch = int(time.time())
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM users WHERE username_key=?", (key,)).fetchone()
            if row is None:
                _equalize_failed_authentication(password)
                conn.commit()
                return AuthenticationResult(False, "invalid")
            user = self._row_to_user(row)
            assert user is not None
            stored_hash = str(row["password_hash"] or "")
            if not user.active:
                _equalize_failed_authentication(password)
                conn.commit()
                return AuthenticationResult(False, "invalid")
            if user.locked_until > now_epoch:
                _equalize_failed_authentication(password)
                conn.commit()
                return AuthenticationResult(False, "invalid")
            if not verify_password(password, stored_hash):
                _equalize_failed_authentication(password, stored_hash)
                failures = int(row["failed_attempts"] or 0) + 1
                locked_until = 0
                if failures >= LOCKOUT_FAILURES:
                    locked_until = now_epoch + LOCKOUT_SECONDS
                    failures = 0
                conn.execute(
                    "UPDATE users SET failed_attempts=?, locked_until=?, updated_at=? WHERE user_id=?",
                    (failures, locked_until, _utc_text(), user.user_id),
                )
                self._event(conn, "login_failed", target_user_id=user.user_id, details={"locked": bool(locked_until)})
                conn.commit()
                # Public authentication responses never disclose whether an account
                # exists, is disabled, or is currently locked. Admins can inspect the
                # persisted lock state from Team Administration.
                return AuthenticationResult(False, "invalid")

            # Upgrade legacy SHA-256 and older PBKDF2 work factors immediately after
            # a successful login. The plaintext password is already available here,
            # so no reversible credential material is persisted.
            new_hash = stored_hash
            if password_hash_needs_upgrade(stored_hash):
                new_hash = _hash_password_unchecked(password)
            now_text = _utc_text()
            conn.execute(
                "UPDATE users SET password_hash=?, failed_attempts=0, locked_until=0, last_login_at=?, updated_at=? WHERE user_id=?",
                (new_hash, now_text, now_text, user.user_id),
            )
            self._event(conn, "login_success", target_user_id=user.user_id)
            conn.commit()
        return AuthenticationResult(True, "ok", self.get_user(user.user_id))

    def change_own_password(self, user_id: str, current_password: str, new_password: str) -> TeamUser:
        validate_password(new_password)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT * FROM users WHERE user_id=?", (str(user_id),)).fetchone()
            if row is None or not bool(row["active"]):
                raise TeamAuthError("Account is unavailable.")
            if not verify_password(current_password, str(row["password_hash"])):
                raise TeamAuthError("Current password is incorrect.")
            if verify_password(new_password, str(row["password_hash"])):
                raise TeamAuthError("New password must be different from the current password.")
            now = _utc_text()
            conn.execute(
                "UPDATE users SET password_hash=?, must_change_password=0, failed_attempts=0, locked_until=0, revision=revision+1, updated_at=? WHERE user_id=?",
                (hash_password(new_password), now, str(user_id)),
            )
            self._event(conn, "password_changed", actor_user_id=str(user_id), target_user_id=str(user_id))
            conn.commit()
        user = self.get_user(user_id)
        if user is None:
            raise TeamAuthError("Account disappeared during password update.")
        return user

    def admin_reset_password(
        self,
        user_id: str,
        new_password: str,
        *,
        actor_user_id: str,
        actor_username: str,
    ) -> TeamUser:
        validate_password(new_password)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._require_actor_permission(
                conn,
                actor_user_id=actor_user_id,
                actor_username=actor_username,
                permission=PERM_TEAM_MANAGE,
            )
            row = conn.execute("SELECT * FROM users WHERE user_id=?", (str(user_id),)).fetchone()
            if row is None:
                raise TeamAuthError("Account not found.")
            now = _utc_text()
            conn.execute(
                "UPDATE users SET password_hash=?, must_change_password=1, failed_attempts=0, locked_until=0, revision=revision+1, updated_at=? WHERE user_id=?",
                (hash_password(new_password), now, str(user_id)),
            )
            self._event(
                conn,
                "password_reset_by_admin",
                actor_user_id=actor_user_id,
                actor_username=actor_username,
                target_user_id=str(user_id),
            )
            conn.commit()
        user = self.get_user(user_id)
        if user is None:
            raise TeamAuthError("Account not found after password reset.")
        return user

    def update_user(
        self,
        user_id: str,
        *,
        display_name: str,
        role: str,
        active: bool,
        actor_user_id: str,
        actor_username: str,
    ) -> TeamUser:
        clean_role = normalize_role(role)
        clean_display = str(display_name or "").strip()[:MAX_DISPLAY_NAME_LENGTH]
        if not clean_display:
            raise TeamAuthError("Display name is required.")
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._require_actor_permission(
                conn,
                actor_user_id=actor_user_id,
                actor_username=actor_username,
                permission=PERM_TEAM_MANAGE,
            )
            row = conn.execute("SELECT * FROM users WHERE user_id=?", (str(user_id),)).fetchone()
            if row is None:
                raise TeamAuthError("Account not found.")
            was_admin = str(row["role"]) == ROLE_ADMIN and bool(row["active"])
            remains_admin = clean_role == ROLE_ADMIN and bool(active)
            if was_admin and not remains_admin:
                other_admins = int(
                    conn.execute(
                        "SELECT COUNT(*) FROM users WHERE active=1 AND role=? AND user_id<>?",
                        (ROLE_ADMIN, str(user_id)),
                    ).fetchone()[0]
                )
                if other_admins < 1:
                    raise TeamAuthError("The final active Admin cannot be demoted or deactivated.")
            now = _utc_text()
            conn.execute(
                "UPDATE users SET display_name=?, role=?, active=?, revision=revision+1, updated_at=?, locked_until=CASE WHEN ?=1 THEN locked_until ELSE 0 END WHERE user_id=?",
                (clean_display, clean_role, 1 if active else 0, now, 1 if active else 0, str(user_id)),
            )
            self._event(
                conn,
                "user_updated",
                actor_user_id=actor_user_id,
                actor_username=actor_username,
                target_user_id=str(user_id),
                details={"role": clean_role, "active": bool(active)},
            )
            conn.commit()
        user = self.get_user(user_id)
        if user is None:
            raise TeamAuthError("Account not found after update.")
        return user

    def unlock_user(self, user_id: str, *, actor_user_id: str, actor_username: str) -> TeamUser:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            self._require_actor_permission(
                conn,
                actor_user_id=actor_user_id,
                actor_username=actor_username,
                permission=PERM_TEAM_MANAGE,
            )
            if conn.execute("SELECT 1 FROM users WHERE user_id=?", (str(user_id),)).fetchone() is None:
                raise TeamAuthError("Account not found.")
            conn.execute(
                "UPDATE users SET failed_attempts=0, locked_until=0, revision=revision+1, updated_at=? WHERE user_id=?",
                (_utc_text(), str(user_id)),
            )
            self._event(
                conn,
                "account_unlocked",
                actor_user_id=actor_user_id,
                actor_username=actor_username,
                target_user_id=str(user_id),
            )
            conn.commit()
        user = self.get_user(user_id)
        assert user is not None
        return user

    def security_events(self, *, limit: int = 200) -> list[dict[str, Any]]:
        safe_limit = max(1, min(int(limit), 1000))
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT timestamp, actor_username, action, target_user_id, details_json FROM security_events ORDER BY event_id DESC LIMIT ?",
                (safe_limit,),
            ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            try:
                details = json.loads(str(row["details_json"] or "{}"))
            except Exception:
                details = {}
            result.append(
                {
                    "timestamp": str(row["timestamp"]),
                    "actor": str(row["actor_username"]),
                    "action": str(row["action"]),
                    "target_user_id": str(row["target_user_id"]),
                    "details": details,
                }
            )
        return result


def get_team_auth_store(path: Path | str | None = None) -> TeamAuthStore:
    return TeamAuthStore(path)
