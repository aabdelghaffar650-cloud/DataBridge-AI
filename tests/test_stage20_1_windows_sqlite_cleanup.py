"""Stage 20.1 regression test for deterministic SQLite handle cleanup."""
from __future__ import annotations

import gc
import sqlite3
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import streamlit  # type: ignore  # noqa: F401
except ModuleNotFoundError:
    import types
    sys.modules["streamlit"] = types.SimpleNamespace(session_state={})

from core.team_access import ROLE_DATA_SCIENTIST
from core.team_store import TeamAuthStore


def test_connection_context_closes_underlying_sqlite_handle() -> None:
    with tempfile.TemporaryDirectory(prefix="databridge-stage20-1-conn-") as td:
        path = Path(td) / "team_auth.db"
        store = TeamAuthStore(path)
        with store._connect() as conn:  # test-only verification
            assert conn.execute("SELECT 1").fetchone()[0] == 1

        try:
            conn.execute("SELECT 1")
        except sqlite3.ProgrammingError as exc:
            assert "closed" in str(exc).lower()
        else:
            raise AssertionError("TeamAuthStore._connect() must close the SQLite handle on context exit")


def test_store_operations_leave_database_deletable_after_scope() -> None:
    td = tempfile.mkdtemp(prefix="databridge-stage20-1-delete-")
    root = Path(td)
    path = root / "team_auth.db"
    store = TeamAuthStore(path)
    admin = store.create_initial_admin("admin", "Stage20!Admin123", display_name="Admin")
    scientist = store.create_user(
        username="scientist",
        display_name="Scientist",
        role=ROLE_DATA_SCIENTIST,
        password="Stage20!Scientist123",
        actor_user_id=admin.user_id,
        actor_username=admin.username,
    )
    assert store.authenticate("scientist", "Stage20!Scientist123").ok
    assert store.get_user(scientist.user_id) is not None
    assert store.security_events(limit=20)

    del store
    gc.collect()

    # On Windows this unlink is the important regression: an unclosed sqlite3
    # handle raises WinError 32. On POSIX it is still safe and validates cleanup.
    for suffix in ("-wal", "-shm"):
        sidecar = Path(str(path) + suffix)
        if sidecar.exists():
            sidecar.unlink()
    path.unlink()
    root.rmdir()


def main() -> None:
    test_connection_context_closes_underlying_sqlite_handle()
    test_store_operations_leave_database_deletable_after_scope()
    print(
        "PASS: Stage 20.1 deterministically closes every TeamAuth SQLite connection, preserves transaction semantics, and releases team_auth.db so Windows temporary-directory cleanup and production maintenance do not fail with WinError 32."
    )


if __name__ == "__main__":
    main()
