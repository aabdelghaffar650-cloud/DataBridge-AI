# ════════════════════════════════════════════════════════
#  DataBridge AI — Smart Import Engine
#  CSV · Excel · JSON · Parquet · SQLite · SQLAlchemy sources
# ════════════════════════════════════════════════════════
from __future__ import annotations

import io
import json
import os
import re
import sqlite3
import tempfile
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional

import pandas as pd

from config.constants import BOOL_MAP
from core.security import validate_uploaded_file


SUPPORTED_FILE_TYPES = [
    "csv", "xlsx", "xls", "json", "jsonl", "ndjson", "parquet", "db", "sqlite", "sqlite3"
]


# ════════════════════════════════════════════════════════
#  Shared helpers
# ════════════════════════════════════════════════════════
def detect_header_row(df_raw: pd.DataFrame, max_scan: int = 15) -> int:
    """Return the index of the row most likely to be the header."""
    best_row, best_score = 0, -1.0
    for i in range(min(max_scan, len(df_raw))):
        row = df_raw.iloc[i]
        non_null = row.notna().sum()
        if non_null == 0:
            continue
        str_ratio = sum(1 for v in row if isinstance(v, str) and v.strip()) / non_null
        numeric_ratio = sum(1 for v in row if isinstance(v, (int, float)) and not pd.isna(v)) / non_null
        unique_ratio = len({str(v) for v in row if pd.notna(v)}) / non_null
        score = (str_ratio * 2) + (unique_ratio * 1.5) - (numeric_ratio * 2) + (non_null / max(df_raw.shape[1], 1))
        if score > best_score:
            best_score, best_row = score, i
    return best_row


def _base_report(source_type: str, source_name: str = "") -> Dict[str, Any]:
    return {
        "source_type": source_type,
        "source_name": source_name,
        "sheets_found": [],
        "sheet_selected": "",
        "header_row": 0,
        "tables_found": [],
        "table_selected": "",
        "query_used": "",
        "cleaning_steps": [],
    }


def _clean_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Apply generic cleaning steps.
    Returns (cleaned_df, report_steps).
    """
    steps: List[Dict[str, Any]] = []

    def _step(action: str, detail: str, count: int = 0, severity: str = "info"):
        steps.append({"action": action, "detail": detail, "count": int(count), "severity": severity})

    # Normalize duplicate/blank column names safely
    seen: Dict[str, int] = {}
    cleaned_cols: List[str] = []
    for idx, col in enumerate(df.columns):
        name = str(col).strip()
        if not name or name.lower() in {"nan", "none", "unnamed: 0"}:
            name = f"column_{idx + 1}"
        base = name
        seen[base] = seen.get(base, 0) + 1
        if seen[base] > 1:
            name = f"{base}_{seen[base]}"
        cleaned_cols.append(name)
    df.columns = cleaned_cols

    # ── Step 1: Strip whitespace & convert fake nulls ─────────────────
    obj_cols = df.select_dtypes(include="object").columns
    NULL_STRINGS = {"nan", "none", "null", "na", "n/a", "", "–", "-", "--", "?", "<na>"}
    null_str_total = 0
    for col in obj_cols:
        null_mask = df[col].isna()
        cleaned = df[col].astype(str).str.strip()
        is_null_str = cleaned.str.lower().isin(NULL_STRINGS)
        new_nulls = max(int(is_null_str.sum()) - int(null_mask.sum()), 0)
        null_str_total += new_nulls
        df[col] = cleaned.where(~is_null_str, other=float("nan"))
        df.loc[null_mask, col] = float("nan")

    if len(obj_cols) > 0:
        _step(
            "✂️ Whitespace stripped",
            f"{len(obj_cols)} text columns — leading/trailing spaces removed",
            len(obj_cols),
            "info",
        )

    if null_str_total > 0:
        _step(
            "🔍 Fake nulls → real NaN",
            f'{null_str_total} cells containing text like "nan", "N/A", "none", "null" were converted to actual NaN',
            null_str_total,
            "warning",
        )

    # ── Step 2: Standardise boolean responses ─────────────────────────
    bool_converted = 0
    for col in df.select_dtypes(include="object").columns:
        mapped = df[col].str.lower().map(BOOL_MAP)
        coverage = mapped.notna().sum() / max(df[col].notna().sum(), 1)
        if coverage > 0.7:
            df[col] = df[col].str.lower().map(BOOL_MAP).fillna(df[col])
            bool_converted += 1
    if bool_converted:
        _step(
            "🔄 Boolean values standardised",
            f"{bool_converted} columns: yes/no/نعم/لا/Y/N/True/False → unified Yes / No",
            bool_converted,
            "info",
        )

    # ── Step 3: Auto-detect & coerce date columns ─────────────────────
    date_converted = 0
    date_cols_found: List[str] = []
    for col in df.select_dtypes(include="object").columns:
        sample = df[col].dropna().head(30)
        if sample.empty:
            continue
        date_ratio = pd.to_datetime(sample, errors="coerce", dayfirst=True).notna().sum() / max(len(sample), 1)
        if date_ratio > 0.65:
            df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=True)
            date_converted += 1
            date_cols_found.append(col)
    if date_converted:
        _step(
            "📅 Dates auto-converted",
            f"{date_converted} column(s) detected as datetime and converted: {', '.join(date_cols_found)}",
            date_converted,
            "info",
        )

    # ── Step 4: Remove fully-empty rows & columns ─────────────────────
    before_rows = len(df)
    before_cols = df.shape[1]
    empty_row_indices = df.index[df.isnull().all(axis=1)].tolist()
    empty_col_names = [c for c in df.columns if df[c].isnull().all()]

    df.dropna(how="all", inplace=True)
    df.dropna(axis=1, how="all", inplace=True)
    df.reset_index(drop=True, inplace=True)

    dropped_rows = before_rows - len(df)
    dropped_cols = before_cols - df.shape[1]

    if dropped_rows:
        sample_indices = empty_row_indices[:5]
        more = f" (and {len(empty_row_indices) - 5} more)" if len(empty_row_indices) > 5 else ""
        _step(
            "🗑️ Fully-empty rows removed",
            f"{dropped_rows} rows had NO data in any column — removed automatically. Original row positions: {sample_indices}{more}",
            dropped_rows,
            "removed",
        )

    if dropped_cols:
        _step(
            "🗑️ Fully-empty columns removed",
            f"{dropped_cols} column(s) were completely empty and removed: {', '.join(map(str, empty_col_names))}",
            dropped_cols,
            "removed",
        )

    return df, steps


def _flatten_if_needed(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str]]:
    """
    Flatten nested JSON-like dict/list columns directly without a JSON round-trip.

    Why: the old approach used df.to_json(..., default_handler=str) which silently
    coerces datetime and numeric values to strings, corrupting column types.
    Direct per-column expansion preserves all non-nested dtypes exactly.
    """
    nested_cols = [
        c for c in df.columns
        if df[c].map(lambda x: isinstance(x, dict)).any()  # lists stay as-is; only dicts expand cleanly
    ]
    if not nested_cols:
        return df, None

    flattened_names: List[str] = []
    for col in nested_cols:
        try:
            mask = df[col].map(lambda x: isinstance(x, dict))
            expanded = pd.json_normalize(df.loc[mask, col].tolist(), sep=".")
            expanded.index = df.index[mask]
            # Prefix new columns to avoid name collisions
            expanded.columns = [f"{col}.{sub}" for sub in expanded.columns]
            df = df.drop(columns=[col]).join(expanded, how="left")
            flattened_names.append(col)
        except Exception:
            # If a single column fails, skip it silently — don't break the whole import
            continue

    if not flattened_names:
        return df, None

    note = (
        f"Nested dict columns expanded into dot-notated sub-columns: "
        f"{', '.join(flattened_names[:8])}"
        + (f" (+{len(flattened_names) - 8} more)" if len(flattened_names) > 8 else "")
    )
    return df, note


def _quote_sqlite_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


# _uploaded_to_tempfile removed — smart_parse_sqlite now uses TemporaryDirectory
# for guaranteed cleanup. If you need a temp file elsewhere use:
#   with tempfile.TemporaryDirectory() as d: path = os.path.join(d, "file.ext")



DESTRUCTIVE_SQL_PATTERN = re.compile(
    r"\b(insert|update|delete|drop|alter|truncate|create|replace|merge|grant|revoke|vacuum|attach|detach|pragma|copy|call|execute)\b",
    flags=re.IGNORECASE,
)


def validate_readonly_select_query(query: str) -> str:
    """Allow SELECT/CTE read queries only and block multi-statements or destructive keywords."""
    sql = (query or "").strip()
    if not sql:
        raise ValueError("SQL query is required.")
    if ";" in sql.rstrip(";") or sql.count(";") > 1:
        raise ValueError("Multiple SQL statements are not allowed.")
    sql = sql.rstrip(";").strip()
    if not re.match(r"^(select|with)\b", sql, flags=re.IGNORECASE):
        raise ValueError("Only SELECT queries are allowed.")
    if DESTRUCTIVE_SQL_PATTERN.search(sql):
        raise ValueError("Query contains blocked SQL keywords. Only read-only SELECT queries are allowed.")
    return sql

# ════════════════════════════════════════════════════════
#  File importers
# ════════════════════════════════════════════════════════
def smart_parse_excel(uploaded_file) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    validate_uploaded_file(uploaded_file)
    report = _base_report("Excel", getattr(uploaded_file, "name", ""))

    xl = pd.ExcelFile(uploaded_file)
    report["sheets_found"] = xl.sheet_names

    best_sheet, best_size = xl.sheet_names[0], 0
    for sh in xl.sheet_names:
        raw = xl.parse(sh, header=None)
        size = raw.shape[0] * raw.shape[1]
        if size > best_size:
            best_size, best_sheet = size, sh
    report["sheet_selected"] = best_sheet

    raw_df = xl.parse(best_sheet, header=None)
    h_row = detect_header_row(raw_df)
    report["header_row"] = h_row

    df = xl.parse(best_sheet, header=h_row)
    df, steps = _clean_dataframe(df)
    report["cleaning_steps"] = steps
    return df, report


def smart_parse_csv(uploaded_file) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    validate_uploaded_file(uploaded_file)
    report = _base_report("CSV", getattr(uploaded_file, "name", ""))
    report["sheets_found"] = ["CSV"]
    report["sheet_selected"] = "CSV"

    # Try UTF-8 first, then common fallbacks for Arabic/Windows files.
    last_error: Optional[Exception] = None
    for encoding in ("utf-8-sig", "utf-8", "cp1256", "latin1"):
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding=encoding)
            report["encoding"] = encoding
            break
        except Exception as exc:
            last_error = exc
    else:
        raise ValueError(f"Could not read CSV file. Last error: {last_error}")

    df, steps = _clean_dataframe(df)
    report["cleaning_steps"] = steps
    return df, report


def smart_parse_json(uploaded_file, selected_key: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Parse a JSON / JSONL / NDJSON file into a DataFrame.

    Args:
        uploaded_file: Streamlit UploadedFile or file-like object.
        selected_key:  When the JSON root is a dict with multiple arrays, the caller
                       can pass the desired key to skip auto-selection.  If None and
                       multiple arrays exist, the report will contain
                       ``json_array_candidates`` so the UI can prompt the user.
    """
    validate_uploaded_file(uploaded_file)
    report = _base_report("JSON", getattr(uploaded_file, "name", ""))
    suffix = Path(uploaded_file.name).suffix.lower()

    uploaded_file.seek(0)
    raw = uploaded_file.read()
    text = raw.decode("utf-8-sig") if isinstance(raw, (bytes, bytearray)) else str(raw)
    uploaded_file.seek(0)

    if suffix in {".jsonl", ".ndjson"}:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
        df = pd.json_normalize(records, sep=".")
        report["sheet_selected"] = "JSON Lines"
    else:
        data = json.loads(text)
        if isinstance(data, dict):
            list_candidates = {k: v for k, v in data.items() if isinstance(v, list)}
            if list_candidates:
                if selected_key is not None:
                    # Caller already chose — validate and use directly
                    if selected_key not in list_candidates:
                        raise ValueError(f"Key '{selected_key}' not found in JSON object. Available: {list(list_candidates)}")
                    chosen_key = selected_key
                elif len(list_candidates) > 1:
                    # Expose ambiguity to the caller so the UI can ask the user.
                    sorted_keys = sorted(list_candidates, key=lambda k: len(list_candidates[k]), reverse=True)
                    report["json_array_candidates"] = [
                        {"key": k, "length": len(list_candidates[k])} for k in sorted_keys
                    ]
                    # Default to the largest, but caller should confirm via selected_key.
                    chosen_key = sorted_keys[0]
                else:
                    chosen_key = next(iter(list_candidates))
                df = pd.json_normalize(list_candidates[chosen_key], sep=".")
                report["sheet_selected"] = chosen_key
            else:
                df = pd.json_normalize(data, sep=".")
                report["sheet_selected"] = "root object"
        elif isinstance(data, list):
            df = pd.json_normalize(data, sep=".")
            report["sheet_selected"] = "root array"
        else:
            raise ValueError("Unsupported JSON structure. Expected an object, an array, or JSON Lines.")

    df, flatten_note = _flatten_if_needed(df)
    df, steps = _clean_dataframe(df)
    if flatten_note:
        steps.insert(0, {"action": "🧬 JSON flattened", "detail": flatten_note, "count": len(df.columns), "severity": "info"})
    report["cleaning_steps"] = steps
    return df, report


def smart_parse_parquet(uploaded_file) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    validate_uploaded_file(uploaded_file)
    report = _base_report("Parquet", getattr(uploaded_file, "name", ""))
    uploaded_file.seek(0)
    df = pd.read_parquet(uploaded_file)
    report["sheet_selected"] = "Parquet dataset"
    df, steps = _clean_dataframe(df)
    report["cleaning_steps"] = steps
    return df, report


def list_sqlite_tables_from_path(db_path: str) -> List[Dict[str, Any]]:
    """Return SQLite tables/views with row counts. Does not execute user SQL."""
    con = sqlite3.connect(db_path)
    try:
        objects = pd.read_sql_query(
            """
            SELECT name, type
            FROM sqlite_master
            WHERE type IN ('table', 'view')
              AND name NOT LIKE 'sqlite_%'
            ORDER BY type, name
            """,
            con,
        )
        result: List[Dict[str, Any]] = []
        for _, row in objects.iterrows():
            name = str(row["name"])
            obj_type = str(row["type"])
            try:
                count_df = pd.read_sql_query(f"SELECT COUNT(*) AS n FROM {_quote_sqlite_identifier(name)}", con)
                n = int(count_df.loc[0, "n"])
            except Exception:
                n = None
            result.append({"name": name, "type": obj_type, "rows": n})
        return result
    finally:
        con.close()


def smart_parse_sqlite(uploaded_file, table: Optional[str] = None, query: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Parse uploaded SQLite database files (.db/.sqlite/.sqlite3).
    Default behavior: loads the largest table/view by row count.
    Optional table/query are intended for trusted local use inside the app UI.

    Uses TemporaryDirectory (not NamedTemporaryFile) so cleanup is guaranteed
    even on Windows and on unexpected exceptions — no leftover temp files.
    """
    validate_uploaded_file(uploaded_file)
    suffix = Path(uploaded_file.name).suffix.lower() or ".db"
    report = _base_report("SQLite", getattr(uploaded_file, "name", ""))

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = os.path.join(tmp_dir, f"upload{suffix}")
        uploaded_file.seek(0)
        with open(tmp_path, "wb") as fh:
            fh.write(uploaded_file.read())
        uploaded_file.seek(0)

        tables = list_sqlite_tables_from_path(tmp_path)
        if not tables:
            raise ValueError("No tables or views found in this SQLite database.")
        report["tables_found"] = tables

        con = sqlite3.connect(tmp_path)
        try:
            if query and query.strip():
                sql = validate_readonly_select_query(query)
                df = pd.read_sql_query(sql, con)
                report["query_used"] = sql
                report["table_selected"] = "Custom SELECT query"
            else:
                if table is None:
                    best = max(tables, key=lambda t: (t.get("rows") if t.get("rows") is not None else -1))
                    table = best["name"]
                valid_names = {t["name"] for t in tables}
                if table not in valid_names:
                    raise ValueError(f"Table not found in SQLite database: {table}")
                df = pd.read_sql_query(f"SELECT * FROM {_quote_sqlite_identifier(table)}", con)
                report["table_selected"] = table
        finally:
            con.close()
        # TemporaryDirectory context exits here — tmp_path deleted automatically

    df, steps = _clean_dataframe(df)
    steps.insert(
        0,
        {
            "action": "🗄️ SQLite table loaded",
            "detail": f"Loaded {report['table_selected']} from SQLite database. Tables/views found: {len(tables)}",
            "count": len(df),
            "severity": "info",
        },
    )
    report["cleaning_steps"] = steps
    return df, report


def smart_parse_file(uploaded_file) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Dispatch an uploaded file to the correct parser."""
    if uploaded_file is None:
        raise ValueError("No file uploaded.")

    name = getattr(uploaded_file, "name", "").lower()
    suffix = Path(name).suffix.lower()

    if suffix == ".csv":
        return smart_parse_csv(uploaded_file)
    if suffix in {".xlsx", ".xls"}:
        return smart_parse_excel(uploaded_file)
    if suffix in {".json", ".jsonl", ".ndjson"}:
        return smart_parse_json(uploaded_file)
    if suffix == ".parquet":
        return smart_parse_parquet(uploaded_file)
    if suffix in {".db", ".sqlite", ".sqlite3"}:
        return smart_parse_sqlite(uploaded_file)
    if suffix == ".sql":
        raise ValueError(
            "Raw .sql dump files are not executed for safety. Import a SQLite .db file or use the Database Connector with a SELECT query."
        )
    raise ValueError(f"Unsupported file type: {suffix or 'unknown'}")


# ════════════════════════════════════════════════════════
#  Database connector helpers
# ════════════════════════════════════════════════════════
MAX_SQL_ROWS = 200_000  # Hard ceiling to prevent accidental full-table loads into RAM


def read_sqlalchemy_query(connection_url: str, query: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Read a SELECT query from any SQLAlchemy-supported database.

    Enforces a hard row ceiling of MAX_SQL_ROWS to prevent accidental full-table
    loads. If the result is truncated, a warning step is added to the report so
    the user sees it in the Import Report UI.
    """
    if not connection_url or not connection_url.strip():
        raise ValueError("Connection URL is required.")
    sql = validate_readonly_select_query(query)

    try:
        from sqlalchemy import create_engine, text
    except Exception as exc:
        raise ImportError("SQLAlchemy is required for database connections. Run: pip install sqlalchemy") from exc

    report = _base_report("Database", "SQLAlchemy connection")
    safe_url = re.sub(r":[^:@/]+@", ":***@", connection_url.strip())
    report["source_name"] = safe_url
    report["query_used"] = sql

    engine = create_engine(connection_url.strip())
    with engine.connect() as conn:
        df = pd.read_sql_query(text(sql), conn)

    was_truncated = len(df) >= MAX_SQL_ROWS
    if was_truncated:
        df = df.head(MAX_SQL_ROWS)
        report["sql_rows_truncated"] = True
        report["sql_rows_limit"] = MAX_SQL_ROWS

    df, steps = _clean_dataframe(df)

    load_detail = f"Loaded {len(df):,} rows from database using a read-only SELECT query."
    steps.insert(
        0,
        {
            "action": "🔌 Database query loaded",
            "detail": load_detail,
            "count": len(df),
            "severity": "info",
        },
    )

    if was_truncated:
        steps.insert(
            1,
            {
                "action": "⚠️ Result truncated",
                "detail": (
                    f"Query returned ≥ {MAX_SQL_ROWS:,} rows. "
                    f"Only the first {MAX_SQL_ROWS:,} rows were loaded to protect memory. "
                    "Add a LIMIT clause to your query to control the result size."
                ),
                "count": MAX_SQL_ROWS,
                "severity": "warning",
            },
        )

    report["cleaning_steps"] = steps
    return df, report
