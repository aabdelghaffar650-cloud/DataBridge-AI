"""DataBridge AI Stage 4 database-safety verification.

Run from project root:
    python tests/test_stage4_database_safety.py
"""
from __future__ import annotations

import io
import os
import sqlite3
import sys
import tempfile
import types
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class AttrDict(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = value


fake_streamlit = types.ModuleType("streamlit")
fake_streamlit.session_state = AttrDict()


def _cache_data(*args, **kwargs):
    if args and callable(args[0]) and len(args) == 1 and not kwargs:
        return args[0]
    return lambda func: func


fake_streamlit.cache_data = _cache_data
sys.modules.setdefault("streamlit", fake_streamlit)

from modules.import_engine import (  # noqa: E402
    RAW_DATAFRAME_REPORT_KEY,
    _database_url_details,
    _sanitise_database_exception,
    read_sqlalchemy_query,
    smart_parse_sqlite,
    validate_readonly_select_query,
)


class UploadedBytes(io.BytesIO):
    def __init__(self, payload: bytes, name: str):
        super().__init__(payload)
        self.name = name
        self.size = len(payload)


def _build_sqlite_file(row_count: int = 125) -> str:
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    con = sqlite3.connect(path)
    try:
        con.execute("CREATE TABLE records (id INTEGER PRIMARY KEY, label TEXT)")
        con.executemany(
            "INSERT INTO records(id, label) VALUES (?, ?)",
            [(i, f"row-{i}") for i in range(row_count)],
        )
        con.commit()
    finally:
        con.close()
    return path


def _assert_blocked(query: str) -> None:
    try:
        validate_readonly_select_query(query)
    except ValueError:
        return
    raise AssertionError(f"Unsafe query was not blocked: {query}")


def test_sql_lexer_allows_literals_comments_and_ctes() -> None:
    allowed = [
        "SELECT 1",
        "SELECT 'delete; update; drop' AS note; -- harmless comment",
        "WITH source AS (SELECT 1 AS value) SELECT * FROM source",
        "SELECT replace(label, 'row', 'item') FROM records",
        'SELECT "update" AS quoted_identifier',
        "SELECT $$delete; update$$ AS quoted_text",
    ]
    for query in allowed:
        cleaned = validate_readonly_select_query(query)
        assert cleaned.lower().startswith(("select", "with"))
        assert "harmless comment" not in cleaned


def test_unsafe_sql_is_blocked() -> None:
    blocked = [
        "SELECT 1; SELECT 2",
        "SELECT * INTO copied_table FROM records",
        "WITH changed AS (DELETE FROM records RETURNING *) SELECT * FROM changed",
        "SELECT pg_read_file('/etc/passwd')",
        "SELECT load_file('/etc/passwd')",
        "SELECT sleep(60)",
        "SELECT * FROM records FOR UPDATE",
        "PRAGMA query_only = OFF",
        "UPDATE records SET label = 'changed'",
        "COPY records TO '/tmp/export.csv'",
    ]
    for query in blocked:
        _assert_blocked(query)


def test_sqlalchemy_sqlite_is_readonly_and_bounded_before_pandas() -> None:
    path = _build_sqlite_file(125)
    url = f"sqlite:///{path}"
    try:
        df, report = read_sqlalchemy_query(
            url,
            "SELECT * FROM records ORDER BY id",
            row_limit=40,
            query_timeout_seconds=10,
        )
        assert tuple(df.shape) == (40, 2)
        assert int(df.iloc[0]["id"]) == 0
        assert int(df.iloc[-1]["id"]) == 39
        assert report["sql_rows_truncated"] is True
        assert report["sql_rows_limit"] == 40
        assert report["database_dialect"] == "sqlite"
        assert report[RAW_DATAFRAME_REPORT_KEY].equals(df)
        assert any(
            "verified read-only" in step.get("action", "").lower()
            for step in report["cleaning_steps"]
        )

        # The source database remains unchanged.
        con = sqlite3.connect(path)
        try:
            count = con.execute("SELECT COUNT(*) FROM records").fetchone()[0]
        finally:
            con.close()
        assert count == 125
    finally:
        os.remove(path)


def test_uploaded_sqlite_uses_readonly_mode_and_capped_discovery() -> None:
    path = _build_sqlite_file(90)
    try:
        payload = Path(path).read_bytes()
    finally:
        os.remove(path)

    uploaded = UploadedBytes(payload, "safe_source.sqlite")
    df, report = smart_parse_sqlite(
        uploaded,
        row_limit=25,
        query_timeout_seconds=10,
    )
    assert tuple(df.shape) == (25, 2)
    assert report["sql_rows_truncated"] is True
    assert report["sql_rows_limit"] == 25
    table_info = report["tables_found"][0]
    assert table_info["rows"] == 26
    assert table_info["rows_capped"] is True


def test_connection_password_is_redacted() -> None:
    secret = "VerySecretPassword123"
    connection_url = f"postgresql+psycopg2://analyst:{secret}@localhost:5432/reporting"
    url, dialect, safe_url = _database_url_details(connection_url)
    assert dialect == "postgresql"
    assert secret not in safe_url
    assert "***" in safe_url

    message = _sanitise_database_exception(
        RuntimeError(f"Failed URL={connection_url}; password={secret}"),
        connection_url=connection_url,
        safe_url=safe_url,
        password=url.password,
    )
    assert secret not in message
    assert connection_url not in message


def main() -> None:
    test_sql_lexer_allows_literals_comments_and_ctes()
    test_unsafe_sql_is_blocked()
    test_sqlalchemy_sqlite_is_readonly_and_bounded_before_pandas()
    test_uploaded_sqlite_uses_readonly_mode_and_capped_discovery()
    test_connection_password_is_redacted()
    print(
        "PASS: Stage 4 read-only SQL validation, bounded loading, timeout controls, SQLite safety, and secret redaction are functioning."
    )


if __name__ == "__main__":
    main()
