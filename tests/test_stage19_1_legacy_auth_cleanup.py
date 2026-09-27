"""Stage 19.1: legacy project credential cleanup regression tests."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.release_gate import cleanup_legacy_project_auth


def _payload(username: str, password_hash: str) -> dict[str, object]:
    return {
        "version": 2,
        "username": username,
        "password_hash": password_hash,
        "updated_at": "2026-08-09T00:00:00Z",
    }


def test_migrates_legacy_auth_and_removes_project_secret() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "project"
        legacy = root / ".databridge" / "auth.json"
        target = Path(td) / "user-data" / "auth.json"
        legacy.parent.mkdir(parents=True)
        expected = _payload("admin", "pbkdf2_sha256$260000$abc$def")
        legacy.write_text(json.dumps(expected), encoding="utf-8")

        result = cleanup_legacy_project_auth(root, target=target)
        assert "migrated project auth" in result
        assert not legacy.exists()
        assert json.loads(target.read_text(encoding="utf-8")) == expected


def test_existing_valid_per_user_auth_wins_and_stale_legacy_is_removed() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "project"
        legacy = root / ".databridge" / "auth.json"
        target = Path(td) / "user-data" / "auth.json"
        legacy.parent.mkdir(parents=True)
        target.parent.mkdir(parents=True)
        legacy.write_text(json.dumps(_payload("old", "pbkdf2_sha256$260000$old$old")), encoding="utf-8")
        current = _payload("current", "pbkdf2_sha256$260000$new$new")
        target.write_text(json.dumps(current), encoding="utf-8")

        result = cleanup_legacy_project_auth(root, target=target)
        assert "removed stale project auth" in result
        assert not legacy.exists()
        assert json.loads(target.read_text(encoding="utf-8")) == current


def test_invalid_legacy_auth_is_never_deleted() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "project"
        legacy = root / ".databridge" / "auth.json"
        target = Path(td) / "user-data" / "auth.json"
        legacy.parent.mkdir(parents=True)
        legacy.write_text("not-json", encoding="utf-8")

        try:
            cleanup_legacy_project_auth(root, target=target)
        except AssertionError:
            pass
        else:
            raise AssertionError("Invalid legacy credentials must fail closed")
        assert legacy.exists()
        assert not target.exists()


def test_invalid_existing_target_blocks_cleanup() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td) / "project"
        legacy = root / ".databridge" / "auth.json"
        target = Path(td) / "user-data" / "auth.json"
        legacy.parent.mkdir(parents=True)
        target.parent.mkdir(parents=True)
        legacy.write_text(json.dumps(_payload("old", "pbkdf2_sha256$260000$old$old")), encoding="utf-8")
        target.write_text("{}", encoding="utf-8")

        try:
            cleanup_legacy_project_auth(root, target=target)
        except AssertionError:
            pass
        else:
            raise AssertionError("Invalid active credentials must block destructive cleanup")
        assert legacy.exists()
        assert target.exists()


def main() -> None:
    test_migrates_legacy_auth_and_removes_project_secret()
    test_existing_valid_per_user_auth_wins_and_stale_legacy_is_removed()
    test_invalid_legacy_auth_is_never_deleted()
    test_invalid_existing_target_blocks_cleanup()
    print(
        "PASS: Stage 19.1 migrates a valid legacy project-local auth.json to the per-user credential location, removes stale project copies only after validating the active credential, and fails closed without deleting invalid credentials."
    )


if __name__ == "__main__":
    main()
