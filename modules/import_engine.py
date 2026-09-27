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
import time
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional

import pandas as pd

from config.constants import BOOL_MAP
from core.security import validate_uploaded_file


SUPPORTED_FILE_TYPES = [
    "csv", "xlsx", "xls", "json", "jsonl", "ndjson", "parquet", "db", "sqlite", "sqlite3"
]

# Private transfer key consumed and removed by core.dataset.activate_dataset.
# It never remains inside the public import report stored in session state.
RAW_DATAFRAME_REPORT_KEY = "_databridge_raw_dataframe"


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



FAKE_NULL_TOKENS = {
    "", "nan", "none", "null", "na", "n/a", "–", "-", "--", "?", "<na>",
    "missing", "not available",
}
IMPORT_REVIEW_VERSION = 1
PROPOSED_IMPORT_ACTIONS_KEY = "proposed_import_actions"
COMPLETED_IMPORT_SIGNATURES_KEY = "completed_import_action_signatures"


def _normalise_working_columns(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Apply only the minimum structural column-name safeguards required by the UI.

    No cell values, rows, data types, or source columns are removed here. The
    protected raw snapshot keeps the exact parsed source schema.
    """
    working = df.copy(deep=True)
    seen: Dict[str, int] = {}
    cleaned_cols: List[str] = []
    changed: List[Tuple[str, str]] = []

    for idx, col in enumerate(working.columns):
        original = str(col)
        name = original.strip()
        if not name or name.lower() in {"nan", "none", "unnamed: 0"}:
            name = f"column_{idx + 1}"

        base = name
        seen[base] = seen.get(base, 0) + 1
        if seen[base] > 1:
            name = f"{base}_{seen[base]}"

        cleaned_cols.append(name)
        if name != original:
            changed.append((original, name))

    working.columns = cleaned_cols
    steps: List[Dict[str, Any]] = []
    if changed:
        sample = ", ".join(
            f"{before!r} → {after!r}" for before, after in changed[:6]
        )
        if len(changed) > 6:
            sample += f" (+{len(changed) - 6} more)"
        steps.append(
            {
                "action": "🧱 Column names made structurally safe",
                "detail": (
                    f"{len(changed)} blank, padded, or duplicate column name(s) "
                    f"were normalised so the application can address columns safely. "
                    f"Source values and rows were not changed. {sample}"
                ),
                "count": len(changed),
                "severity": "info",
                "operation": "normalise_column_names",
                "approved": True,
            }
        )
    return working, steps


def _is_text_like(series: pd.Series) -> bool:
    return bool(
        pd.api.types.is_object_dtype(series.dtype)
        or pd.api.types.is_string_dtype(series.dtype)
        or isinstance(series.dtype, pd.CategoricalDtype)
    )


def _strip_scalar(value: Any) -> Any:
    return value.strip() if isinstance(value, str) else value


def _normalised_token(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    return value.strip().lower()


def _is_fake_null_scalar(value: Any) -> bool:
    token = _normalised_token(value)
    return token in FAKE_NULL_TOKENS if token is not None else False


def _parse_datetime_series(series: pd.Series) -> pd.Series:
    """Parse mixed date formats without changing the supplied Series."""
    try:
        return pd.to_datetime(
            series,
            errors="coerce",
            dayfirst=True,
            format="mixed",
        )
    except TypeError:
        return pd.to_datetime(series, errors="coerce", dayfirst=True)


def _safe_preview_value(value: Any, max_len: int = 80) -> str:
    if value is None:
        return "NaN"
    try:
        if not isinstance(value, (dict, list, tuple, set)) and pd.isna(value):
            return "NaN"
    except Exception:
        pass
    text = str(value).replace("\n", " ").replace("\r", " ")
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


def _preview_pairs(
    before: pd.Series,
    after: pd.Series,
    mask: pd.Series,
    limit: int = 3,
) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    positions = list(mask[mask].index[:limit])
    for idx in positions:
        rows.append(
            {
                "before": _safe_preview_value(before.loc[idx]),
                "after": _safe_preview_value(after.loc[idx]),
            }
        )
    return rows


def _unique_column_name(columns: List[Any], desired: str) -> str:
    existing = {str(c) for c in columns}
    if desired not in existing:
        return desired
    i = 2
    while f"{desired}_{i}" in existing:
        i += 1
    return f"{desired}_{i}"


def _action_identity(
    operation: str,
    column: str = "",
    target_column: str = "",
) -> Tuple[str, str]:
    import hashlib

    signature = f"{operation}|{column}"
    digest = hashlib.sha256(
        f"{operation}|{column}|{target_column}".encode("utf-8")
    ).hexdigest()[:16]
    return f"import_{digest}", signature


def _date_name_hint(column: str) -> bool:
    name = str(column).strip().lower()
    hints = (
        "date", "datetime", "timestamp", "time", "day", "month", "year",
        "تاريخ", "وقت", "يوم", "شهر", "سنة",
    )
    return any(hint in name for hint in hints)


def _looks_like_year_codes(non_null: pd.Series) -> bool:
    text = non_null.astype(str).str.strip()
    if text.empty or not text.str.fullmatch(r"\d{4}").all():
        return False
    years = pd.to_numeric(text, errors="coerce")
    return bool(years.between(1800, 2200).all())


def build_import_action_proposals(
    df: pd.DataFrame,
    *,
    excluded_signatures: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Inspect data and return reviewable recommendations without mutating anything.

    Every recommendation is opt-in. Date and boolean recognition create new
    derived columns rather than overwriting the source column.
    """
    excluded = set(excluded_signatures or [])
    proposals: List[Dict[str, Any]] = []

    def add(action: Dict[str, Any]) -> None:
        if action["signature"] not in excluded:
            proposals.append(action)

    for col in list(df.columns):
        series = df[col]
        if not _is_text_like(series):
            continue

        stripped = series.map(_strip_scalar)
        strip_mask = series.notna() & stripped.ne(series)
        strip_count = int(strip_mask.sum())
        if strip_count:
            action_id, signature = _action_identity("strip_whitespace", str(col))
            add(
                {
                    "id": action_id,
                    "signature": signature,
                    "operation": "strip_whitespace",
                    "title": f"Strip leading/trailing whitespace — {col}",
                    "column": str(col),
                    "target_column": str(col),
                    "count": strip_count,
                    "risk": "low",
                    "destructive": False,
                    "default_selected": False,
                    "detail": (
                        f"{strip_count:,} text cell(s) contain leading or trailing whitespace. "
                        "Only those surrounding spaces will be removed after approval."
                    ),
                    "preview": _preview_pairs(series, stripped, strip_mask),
                }
            )

        fake_mask = series.map(_is_fake_null_scalar).fillna(False).astype(bool)
        fake_count = int(fake_mask.sum())
        if fake_count:
            after_null = series.copy(deep=True)
            after_null.loc[fake_mask] = float("nan")
            action_id, signature = _action_identity("convert_fake_nulls", str(col))
            add(
                {
                    "id": action_id,
                    "signature": signature,
                    "operation": "convert_fake_nulls",
                    "title": f"Convert explicit fake-null tokens — {col}",
                    "column": str(col),
                    "target_column": str(col),
                    "count": fake_count,
                    "risk": "medium",
                    "destructive": False,
                    "default_selected": False,
                    "detail": (
                        f"{fake_count:,} cell(s) exactly match known placeholders such as "
                        "N/A, none, null, -, or ?. Conversion is never automatic."
                    ),
                    "preview": _preview_pairs(series, after_null, fake_mask),
                }
            )

        eligible_mask = series.notna() & ~fake_mask
        eligible = series.loc[eligible_mask]
        if eligible.empty:
            continue
        normalised = eligible.map(_normalised_token)

        mapped = normalised.map(BOOL_MAP)
        mapped_count = int(mapped.notna().sum())
        coverage = mapped_count / max(len(eligible), 1)
        if len(eligible) >= 2 and coverage >= 0.95 and mapped_count:
            target = _unique_column_name(list(df.columns), f"{col}__boolean")
            full_after = pd.Series(index=series.index, dtype="object")
            full_after.loc[eligible.index] = mapped
            bool_mask = pd.Series(False, index=series.index)
            bool_mask.loc[eligible.index] = mapped.notna().values
            action_id, signature = _action_identity("derive_boolean", str(col), target)
            add(
                {
                    "id": action_id,
                    "signature": signature,
                    "operation": "derive_boolean",
                    "title": f"Create reviewed boolean feature — {target}",
                    "column": str(col),
                    "target_column": target,
                    "count": mapped_count,
                    "risk": "low",
                    "destructive": False,
                    "default_selected": False,
                    "detail": (
                        f"{mapped_count:,} of {len(eligible):,} non-placeholder values "
                        f"({coverage * 100:.1f}%) match approved boolean tokens. "
                        f"The source column stays unchanged; a new column named "
                        f"'{target}' will be created."
                    ),
                    "preview": _preview_pairs(series, full_after, bool_mask),
                }
            )

        text_values = eligible.astype(str).str.strip()
        if len(eligible) >= 3 and not _looks_like_year_codes(eligible):
            parsed = _parse_datetime_series(text_values)
            valid_count = int(parsed.notna().sum())
            date_ratio = valid_count / max(len(eligible), 1)
            explicit_pattern_ratio = float(
                text_values.str.contains(
                    r"(?:\d{1,4}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,4}|"
                    r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b)",
                    case=False,
                    regex=True,
                    na=False,
                ).mean()
            )
            if date_ratio >= 0.90 and (
                _date_name_hint(str(col)) or explicit_pattern_ratio >= 0.80
            ):
                target = _unique_column_name(list(df.columns), f"{col}__date")
                full_after = pd.Series(
                    pd.NaT,
                    index=series.index,
                    dtype="datetime64[ns]",
                )
                full_after.loc[eligible.index] = parsed.values
                date_mask = pd.Series(False, index=series.index)
                date_mask.loc[eligible.index] = parsed.notna().values
                invalid_count = len(eligible) - valid_count
                action_id, signature = _action_identity(
                    "derive_datetime",
                    str(col),
                    target,
                )
                add(
                    {
                        "id": action_id,
                        "signature": signature,
                        "operation": "derive_datetime",
                        "title": f"Create reviewed datetime feature — {target}",
                        "column": str(col),
                        "target_column": target,
                        "count": valid_count,
                        "risk": "medium" if invalid_count else "low",
                        "destructive": False,
                        "default_selected": False,
                        "detail": (
                            f"{valid_count:,} of {len(eligible):,} values "
                            f"({date_ratio * 100:.1f}%) parse as dates across the full column. "
                            f"{invalid_count:,} value(s) would become NaT in the new derived "
                            f"column. The source column remains unchanged."
                        ),
                        "preview": _preview_pairs(series, full_after, date_mask),
                        "invalid_count": int(invalid_count),
                    }
                )

    empty_rows_mask = df.isna().all(axis=1)
    empty_row_count = int(empty_rows_mask.sum())
    if empty_row_count:
        action_id, signature = _action_identity("drop_empty_rows")
        add(
            {
                "id": action_id,
                "signature": signature,
                "operation": "drop_empty_rows",
                "title": "Remove fully empty rows",
                "column": "",
                "target_column": "",
                "count": empty_row_count,
                "risk": "high",
                "destructive": True,
                "default_selected": False,
                "detail": (
                    f"{empty_row_count:,} row(s) contain no value in any column. "
                    "They will only be removed after explicit approval and are protected by Undo."
                ),
                "preview": [],
                "row_indices": [
                    int(i) if isinstance(i, int) else str(i)
                    for i in df.index[empty_rows_mask][:10]
                ],
            }
        )

    empty_columns = [str(c) for c in df.columns if df[c].isna().all()]
    if empty_columns:
        action_id, signature = _action_identity("drop_empty_columns")
        add(
            {
                "id": action_id,
                "signature": signature,
                "operation": "drop_empty_columns",
                "title": "Remove fully empty columns",
                "column": "",
                "target_column": "",
                "count": len(empty_columns),
                "risk": "high",
                "destructive": True,
                "default_selected": False,
                "detail": (
                    f"{len(empty_columns):,} column(s) are completely empty: "
                    f"{', '.join(empty_columns[:8])}"
                    + (
                        f" (+{len(empty_columns) - 8} more)"
                        if len(empty_columns) > 8
                        else ""
                    )
                    + ". They will only be removed after explicit approval and are protected by Undo."
                ),
                "preview": [],
                "columns": empty_columns,
            }
        )

    risk_order = {"high": 0, "medium": 1, "low": 2}
    proposals.sort(
        key=lambda item: (
            risk_order.get(str(item.get("risk")), 9),
            str(item.get("column", "")),
            str(item.get("operation", "")),
        )
    )
    return proposals


def apply_import_actions(
    df: pd.DataFrame,
    proposals: List[Dict[str, Any]],
    selected_action_ids: List[str],
) -> Tuple[pd.DataFrame, List[Dict[str, Any]], List[str]]:
    """
    Apply only actions that were generated by the review engine and approved.

    The operation is performed on a deep copy. Unknown or stale action IDs are
    rejected. Derived date/boolean actions never overwrite source columns.
    """
    selected = list(dict.fromkeys(selected_action_ids or []))
    proposal_by_id = {
        str(action.get("id")): action
        for action in proposals
        if isinstance(action, dict) and action.get("id")
    }
    unknown = [action_id for action_id in selected if action_id not in proposal_by_id]
    if unknown:
        raise ValueError("The import review contains unknown or stale action IDs.")

    result = df.copy(deep=True)
    applied_steps: List[Dict[str, Any]] = []
    completed_signatures: List[str] = []

    # Deletions run first against the exact reviewed fingerprint. This prevents
    # a selected fake-null conversion from making additional rows/columns empty
    # and silently expanding the deletion beyond the reviewed count.
    operation_order = {
        "drop_empty_rows": 5,
        "drop_empty_columns": 6,
        "strip_whitespace": 10,
        "convert_fake_nulls": 20,
        "derive_boolean": 30,
        "derive_datetime": 40,
    }
    chosen = [proposal_by_id[action_id] for action_id in selected]
    chosen.sort(
        key=lambda action: operation_order.get(
            str(action.get("operation")),
            999,
        )
    )

    for action in chosen:
        operation = str(action.get("operation", ""))
        column = str(action.get("column", ""))
        target = str(action.get("target_column", ""))
        affected = 0
        severity = "removed" if action.get("destructive") else "info"

        if operation == "strip_whitespace":
            if column not in result.columns:
                raise ValueError(f"Column no longer exists: {column}")
            before = result[column]
            after = before.map(_strip_scalar)
            affected = int((before.notna() & after.ne(before)).sum())
            result[column] = after

        elif operation == "convert_fake_nulls":
            if column not in result.columns:
                raise ValueError(f"Column no longer exists: {column}")
            mask = result[column].map(_is_fake_null_scalar).fillna(False).astype(bool)
            affected = int(mask.sum())
            result.loc[mask, column] = float("nan")

        elif operation == "derive_boolean":
            if column not in result.columns:
                raise ValueError(f"Column no longer exists: {column}")
            if target in result.columns:
                raise ValueError(f"Target column already exists: {target}")
            source = result[column]
            fake_mask = source.map(_is_fake_null_scalar).fillna(False).astype(bool)
            normalised = source.map(_normalised_token)
            mapped = normalised.map(BOOL_MAP)
            eligible_count = int((source.notna() & ~fake_mask).sum())
            affected = int(mapped.notna().sum())
            coverage = affected / max(eligible_count, 1)
            if eligible_count and coverage < 0.95:
                raise ValueError(
                    f"Boolean proposal for '{column}' is stale; re-scan the current data."
                )
            result[target] = pd.Series(mapped, index=result.index, dtype="object")

        elif operation == "derive_datetime":
            if column not in result.columns:
                raise ValueError(f"Column no longer exists: {column}")
            if target in result.columns:
                raise ValueError(f"Target column already exists: {target}")
            source = result[column]
            fake_mask = source.map(_is_fake_null_scalar).fillna(False).astype(bool)
            eligible_mask = source.notna() & ~fake_mask
            parsed = pd.Series(
                pd.NaT,
                index=result.index,
                dtype="datetime64[ns]",
            )
            if int(eligible_mask.sum()):
                converted = _parse_datetime_series(
                    source.loc[eligible_mask].astype(str).str.strip()
                )
                parsed.loc[eligible_mask] = converted.values
            affected = int(parsed.notna().sum())
            coverage = affected / max(int(eligible_mask.sum()), 1)
            if int(eligible_mask.sum()) and coverage < 0.90:
                raise ValueError(
                    f"Date proposal for '{column}' is stale; re-scan the current data."
                )
            result[target] = parsed

        elif operation == "drop_empty_rows":
            mask = result.isna().all(axis=1)
            affected = int(mask.sum())
            result = result.loc[~mask].reset_index(drop=True)

        elif operation == "drop_empty_columns":
            requested = [str(c) for c in action.get("columns", [])]
            safe_to_drop = [
                c
                for c in requested
                if c in result.columns and bool(result[c].isna().all())
            ]
            affected = len(safe_to_drop)
            if safe_to_drop:
                result = result.drop(columns=safe_to_drop)

        else:
            raise ValueError(
                f"Unsupported import review operation: {operation}"
            )

        completed_signatures.append(str(action.get("signature", "")))
        applied_steps.append(
            {
                "action": f"✅ Approved: {action.get('title', operation)}",
                "detail": (
                    "Applied after explicit user approval. "
                    f"Operation: {operation}. Affected: {affected:,}."
                ),
                "count": int(affected),
                "severity": severity,
                "operation": operation,
                "approved": True,
                "signature": str(action.get("signature", "")),
            }
        )

    return result, applied_steps, completed_signatures


def prepare_safe_working_dataframe(
    source_df: pd.DataFrame,
    *,
    excluded_signatures: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Build the initial working DataFrame under the Stage 3 import policy.

    Only structural column-name safeguards are automatic. All value changes,
    type conversions, and row/column deletion are returned as opt-in proposals.
    """
    if not isinstance(source_df, pd.DataFrame):
        raise TypeError("Import parser did not produce a pandas DataFrame.")

    working, structural_steps = _normalise_working_columns(source_df)
    proposals = build_import_action_proposals(
        working,
        excluded_signatures=excluded_signatures,
    )
    return working, structural_steps, proposals


def _finalize_import_dataframe(
    source_df: pd.DataFrame,
    report: Dict[str, Any],
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Preserve the parsed source and create a non-destructive working copy.

    Stage 3 policy:
    - no automatic cell-value conversion;
    - no automatic date/boolean coercion;
    - no automatic row or column deletion;
    - only structurally necessary column-name normalisation is automatic;
    - every recommended transformation is reviewable and opt-in.
    """
    if not isinstance(source_df, pd.DataFrame):
        raise TypeError("Import parser did not produce a pandas DataFrame.")

    raw_snapshot = source_df.copy(deep=True)
    working_df, structural_steps, proposals = prepare_safe_working_dataframe(
        source_df
    )

    report[RAW_DATAFRAME_REPORT_KEY] = raw_snapshot
    report["raw_shape"] = tuple(raw_snapshot.shape)
    report["initial_working_shape"] = tuple(working_df.shape)
    report["safe_import_policy"] = "review_required"
    report["safe_import_version"] = IMPORT_REVIEW_VERSION
    report["automatic_value_changes"] = 0
    report["automatic_rows_removed"] = 0
    report["automatic_columns_removed"] = 0
    report[PROPOSED_IMPORT_ACTIONS_KEY] = proposals
    report[COMPLETED_IMPORT_SIGNATURES_KEY] = []
    report["import_review_status"] = "pending" if proposals else "clean"
    report["cleaning_steps"] = structural_steps

    return working_df, structural_steps


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



MAX_SQL_QUERY_CHARS = 100_000
MAX_SQL_ROWS = 200_000
DEFAULT_SQL_ROWS = 50_000
DEFAULT_SQL_QUERY_TIMEOUT_SECONDS = 30
MAX_SQL_QUERY_TIMEOUT_SECONDS = 120
SQL_CONNECT_TIMEOUT_SECONDS = 10
SQL_FETCH_CHUNK_ROWS = 5_000
SQLITE_DISCOVERY_PROBE_ROWS = 10_000

SUPPORTED_DATABASE_DIALECTS = {"sqlite", "postgresql", "mysql", "mariadb"}
SUPPORTED_DATABASE_DRIVERS = {
    "sqlite": {"pysqlite"},
    "postgresql": {"psycopg2"},
    "mysql": {"pymysql"},
    "mariadb": {"pymysql"},
}

_BLOCKED_SQL_TOKENS = {
    "insert", "update", "delete", "drop", "alter", "truncate", "create",
    "merge", "grant", "revoke", "vacuum", "attach", "detach", "pragma",
    "copy", "call", "execute", "exec", "do", "reindex", "refresh",
    "lock", "unlock", "set", "use",
}

_BLOCKED_SQL_FUNCTIONS = {
    # File-system / extension access
    "pg_read_file", "pg_read_binary_file", "pg_ls_dir", "pg_stat_file",
    "lo_import", "lo_export", "load_file", "load_extension", "readfile",
    "writefile", "openrowset", "opendatasource", "xp_cmdshell",
    # External connections / commands
    "dblink", "dblink_exec", "sys_eval", "sys_exec",
    # Administrative / session / sequence side effects
    "pg_terminate_backend", "pg_cancel_backend", "pg_reload_conf",
    "pg_rotate_logfile", "pg_create_restore_point", "pg_switch_wal",
    "pg_promote", "pg_notify", "set_config", "nextval", "setval",
    "lo_unlink", "pg_advisory_lock", "pg_advisory_xact_lock",
    "pg_try_advisory_lock", "pg_try_advisory_xact_lock",
    "get_lock", "release_lock", "master_pos_wait",
    # Deliberate resource exhaustion
    "sleep", "pg_sleep", "benchmark",
}


def _strip_sql_comments_and_build_analysis(query: str) -> Tuple[str, str]:
    """
    Remove SQL comments while preserving quoted content for execution.

    A second string masks literals and quoted identifiers so security checks only
    inspect executable SQL tokens. PostgreSQL dollar-quoted strings, MySQL hash
    comments, escaped quotes, and bracket identifiers are handled explicitly.
    """
    executable: List[str] = []
    analysis: List[str] = []
    i = 0
    n = len(query)

    def append_quoted(segment: str) -> None:
        executable.append(segment)
        analysis.append(" " * len(segment))

    while i < n:
        ch = query[i]
        nxt = query[i + 1] if i + 1 < n else ""

        # Line comments: SQL standard / MySQL.
        if ch == "-" and nxt == "-":
            i += 2
            while i < n and query[i] not in "\r\n":
                i += 1
            executable.append(" ")
            analysis.append(" ")
            continue
        if ch == "#":
            i += 1
            while i < n and query[i] not in "\r\n":
                i += 1
            executable.append(" ")
            analysis.append(" ")
            continue

        # Block comments. Nested comments are intentionally rejected by treating
        # the first closing marker as the end; unsupported/unclosed input fails.
        if ch == "/" and nxt == "*":
            end = query.find("*/", i + 2)
            if end < 0:
                raise ValueError("Unterminated SQL block comment.")
            segment = query[i : end + 2]
            executable.append("\n" * segment.count("\n") or " ")
            analysis.append("\n" * segment.count("\n") or " ")
            i = end + 2
            continue

        # PostgreSQL dollar-quoted strings: $$...$$ or $tag$...$tag$.
        if ch == "$":
            match = re.match(r"\$[A-Za-z_][A-Za-z0-9_]*\$|\$\$", query[i:])
            if match:
                delimiter = match.group(0)
                end = query.find(delimiter, i + len(delimiter))
                if end < 0:
                    raise ValueError("Unterminated SQL dollar-quoted string.")
                segment = query[i : end + len(delimiter)]
                append_quoted(segment)
                i = end + len(delimiter)
                continue

        # Single/double/backtick quoted strings or identifiers.
        if ch in {"'", '"', "`"}:
            quote = ch
            j = i + 1
            while j < n:
                if query[j] == "\\" and j + 1 < n:
                    j += 2
                    continue
                if query[j] == quote:
                    if j + 1 < n and query[j + 1] == quote:
                        j += 2
                        continue
                    j += 1
                    break
                j += 1
            else:
                raise ValueError("Unterminated quoted SQL value or identifier.")
            append_quoted(query[i:j])
            i = j
            continue

        # SQL Server-style bracket identifiers are masked even though SQL Server
        # is not an enabled connector dialect.
        if ch == "[":
            j = i + 1
            while j < n:
                if query[j] == "]":
                    if j + 1 < n and query[j + 1] == "]":
                        j += 2
                        continue
                    j += 1
                    break
                j += 1
            else:
                raise ValueError("Unterminated bracketed SQL identifier.")
            append_quoted(query[i:j])
            i = j
            continue

        executable.append(ch)
        analysis.append(ch)
        i += 1

    return "".join(executable), "".join(analysis)


def _sql_word_tokens(analysis_sql: str) -> List[Tuple[str, int, int]]:
    return [
        (match.group(0).lower(), match.start(), match.end())
        for match in re.finditer(r"[A-Za-z_][A-Za-z0-9_$]*", analysis_sql)
    ]


def _token_is_function(analysis_sql: str, token_end: int) -> bool:
    tail = analysis_sql[token_end:]
    match = re.match(r"\s*\(", tail)
    return bool(match)


def validate_readonly_select_query(query: str) -> str:
    """
    Validate one read-only SELECT/CTE statement.

    Security is layered: lexical validation blocks multi-statements, data-changing
    CTEs, SELECT INTO, locking reads, and known side-effect functions. The database
    connector then enforces a verified read-only session and a bounded result set.
    """
    sql_input = (query or "").strip()
    if not sql_input:
        raise ValueError("SQL query is required.")
    if len(sql_input) > MAX_SQL_QUERY_CHARS:
        raise ValueError(
            f"SQL query is too long. Maximum allowed: {MAX_SQL_QUERY_CHARS:,} characters."
        )

    executable_sql, analysis_sql = _strip_sql_comments_and_build_analysis(sql_input)

    # Permit one optional trailing semicolon only. Semicolons inside strings and
    # comments are masked by the scanner and therefore do not trigger this check.
    semicolon_positions = [
        i for i, char in enumerate(analysis_sql) if char == ";"
    ]
    if semicolon_positions:
        if len(semicolon_positions) != 1:
            raise ValueError("Multiple SQL statements are not allowed.")
        pos = semicolon_positions[0]
        if analysis_sql[pos + 1 :].strip():
            raise ValueError("Multiple SQL statements are not allowed.")
        executable_sql = executable_sql[:pos].rstrip()
        analysis_sql = analysis_sql[:pos]

    tokens = _sql_word_tokens(analysis_sql)
    if not tokens:
        raise ValueError("SQL query does not contain an executable SELECT statement.")

    first = tokens[0][0]
    if first not in {"select", "with"}:
        raise ValueError("Only SELECT queries and SELECT-based CTEs are allowed.")
    if first == "with" and not any(token == "select" for token, _, _ in tokens):
        raise ValueError("A WITH query must end in a SELECT statement.")

    token_words = [token for token, _, _ in tokens]
    for token, _start, end in tokens:
        # REPLACE(...) is a common read-only string function. REPLACE as a SQL
        # statement is still blocked because the first token cannot be REPLACE.
        if token == "replace" and _token_is_function(analysis_sql, end):
            continue
        if token in _BLOCKED_SQL_TOKENS:
            raise ValueError(
                f"Blocked SQL token '{token}'. Only read-only SELECT expressions are allowed."
            )
        if token == "into":
            raise ValueError("SELECT INTO / INTO OUTFILE operations are not allowed.")
        if token in _BLOCKED_SQL_FUNCTIONS and _token_is_function(analysis_sql, end):
            raise ValueError(
                f"Blocked SQL function '{token}' because it can access external resources or exhaust the server."
            )

    # Locking reads can affect concurrency and are not appropriate for analytics.
    joined = " ".join(token_words)
    if re.search(r"\bfor\s+(update|share)\b", joined):
        raise ValueError("Locking SELECT queries (FOR UPDATE/FOR SHARE) are not allowed.")
    if re.search(r"\block\s+in\s+share\s+mode\b", joined):
        raise ValueError("Locking SELECT queries are not allowed.")

    return executable_sql.strip()


def _validate_row_limit(row_limit: int) -> int:
    try:
        value = int(row_limit)
    except Exception as exc:
        raise ValueError("Row limit must be an integer.") from exc
    if value < 1 or value > MAX_SQL_ROWS:
        raise ValueError(f"Row limit must be between 1 and {MAX_SQL_ROWS:,}.")
    return value


def _validate_query_timeout(query_timeout_seconds: int) -> int:
    try:
        value = int(query_timeout_seconds)
    except Exception as exc:
        raise ValueError("Query timeout must be an integer number of seconds.") from exc
    if value < 1 or value > MAX_SQL_QUERY_TIMEOUT_SECONDS:
        raise ValueError(
            f"Query timeout must be between 1 and {MAX_SQL_QUERY_TIMEOUT_SECONDS} seconds."
        )
    return value


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
    df, steps = _finalize_import_dataframe(df, report)
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

    df, steps = _finalize_import_dataframe(df, report)
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
    df, steps = _finalize_import_dataframe(df, report)
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
    df, steps = _finalize_import_dataframe(df, report)
    report["cleaning_steps"] = steps
    return df, report


def _open_sqlite_readonly(db_path: str, timeout_seconds: int = 10) -> sqlite3.Connection:
    """Open a SQLite database in OS-level read-only mode and verify query_only."""
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    con = sqlite3.connect(
        uri,
        uri=True,
        timeout=max(1, int(timeout_seconds)),
        check_same_thread=False,
    )
    con.execute("PRAGMA query_only = ON")
    try:
        con.execute("PRAGMA trusted_schema = OFF")
    except sqlite3.DatabaseError:
        pass
    query_only = int(con.execute("PRAGMA query_only").fetchone()[0])
    if query_only != 1:
        con.close()
        raise PermissionError("Could not enforce SQLite read-only query mode.")
    return con


def _install_sqlite_timeout(con: sqlite3.Connection, timeout_seconds: int) -> None:
    deadline = time.monotonic() + timeout_seconds

    def abort_when_expired() -> int:
        return 1 if time.monotonic() >= deadline else 0

    con.set_progress_handler(abort_when_expired, 10_000)


def _execute_sqlite_bounded_select(
    con: sqlite3.Connection,
    sql: str,
    *,
    row_limit: int,
    query_timeout_seconds: int,
) -> Tuple[pd.DataFrame, bool]:
    """Execute a SQLite SELECT with a server-side LIMIT and bounded fetching."""
    row_limit = _validate_row_limit(row_limit)
    timeout = _validate_query_timeout(query_timeout_seconds)
    limited_sql = (
        "SELECT * FROM (\n"
        + sql
        + f"\n) AS databridge_secure_source LIMIT {row_limit + 1}"
    )
    cursor = con.cursor()
    _install_sqlite_timeout(con, timeout)
    try:
        cursor.execute(limited_sql)
        columns = [item[0] for item in (cursor.description or [])]
        rows: List[tuple] = []
        remaining = row_limit + 1
        while remaining > 0:
            batch = cursor.fetchmany(min(SQL_FETCH_CHUNK_ROWS, remaining))
            if not batch:
                break
            rows.extend(batch)
            remaining -= len(batch)
    except sqlite3.OperationalError as exc:
        if "interrupted" in str(exc).lower():
            raise TimeoutError(
                f"SQLite query exceeded the {timeout}-second safety timeout."
            ) from exc
        raise
    finally:
        con.set_progress_handler(None, 0)
        cursor.close()

    truncated = len(rows) > row_limit
    if truncated:
        rows = rows[:row_limit]
    return pd.DataFrame.from_records(rows, columns=columns), truncated


def _probe_sqlite_object_rows(
    con: sqlite3.Connection,
    name: str,
    *,
    probe_limit: int,
    timeout_seconds: int,
) -> Tuple[Optional[int], bool]:
    """Count at most probe_limit rows without a full COUNT(*) table scan."""
    cursor = con.cursor()
    _install_sqlite_timeout(con, timeout_seconds)
    count = 0
    try:
        cursor.execute(
            f"SELECT 1 FROM {_quote_sqlite_identifier(name)} LIMIT {int(probe_limit)}"
        )
        while count < probe_limit:
            batch = cursor.fetchmany(min(SQL_FETCH_CHUNK_ROWS, probe_limit - count))
            if not batch:
                break
            count += len(batch)
        return count, count >= probe_limit
    except (sqlite3.DatabaseError, TimeoutError):
        return None, False
    finally:
        con.set_progress_handler(None, 0)
        cursor.close()


def list_sqlite_tables_from_path(
    db_path: str,
    *,
    probe_limit: int = SQLITE_DISCOVERY_PROBE_ROWS,
    query_timeout_seconds: int = 5,
) -> List[Dict[str, Any]]:
    """
    Return SQLite tables/views using capped probes instead of full COUNT(*) scans.

    ``rows`` is exact only when ``rows_capped`` is false. Large objects are
    reported as at least ``probe_limit`` rows, which is sufficient for safe
    default selection without scanning every table in full.
    """
    probe_limit = max(1, min(int(probe_limit), MAX_SQL_ROWS + 1))
    timeout = max(1, min(int(query_timeout_seconds), MAX_SQL_QUERY_TIMEOUT_SECONDS))
    con = _open_sqlite_readonly(db_path, timeout_seconds=timeout)
    try:
        objects = con.execute(
            """
            SELECT name, type
            FROM sqlite_master
            WHERE type IN ('table', 'view')
              AND name NOT LIKE 'sqlite_%'
            ORDER BY CASE type WHEN 'table' THEN 0 ELSE 1 END, name
            """
        ).fetchall()
        result: List[Dict[str, Any]] = []
        per_object_timeout = max(1, min(timeout, 3))
        for name_value, type_value in objects:
            name = str(name_value)
            obj_type = str(type_value)
            rows, rows_capped = _probe_sqlite_object_rows(
                con,
                name,
                probe_limit=probe_limit,
                timeout_seconds=per_object_timeout,
            )
            result.append(
                {
                    "name": name,
                    "type": obj_type,
                    "rows": rows,
                    "rows_capped": rows_capped,
                }
            )
        return result
    finally:
        con.close()


def smart_parse_sqlite(
    uploaded_file,
    table: Optional[str] = None,
    query: Optional[str] = None,
    *,
    row_limit: int = MAX_SQL_ROWS,
    query_timeout_seconds: int = DEFAULT_SQL_QUERY_TIMEOUT_SECONDS,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Parse an uploaded SQLite database using a read-only connection.

    Table discovery never performs an unbounded COUNT(*), and data loading is
    capped before rows reach pandas. Custom SQL is validated as one read-only
    SELECT/CTE statement.
    """
    validate_uploaded_file(uploaded_file)
    row_limit = _validate_row_limit(row_limit)
    timeout = _validate_query_timeout(query_timeout_seconds)
    suffix = Path(uploaded_file.name).suffix.lower() or ".db"
    report = _base_report("SQLite", getattr(uploaded_file, "name", ""))

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = os.path.join(tmp_dir, f"upload{suffix}")
        uploaded_file.seek(0)
        with open(tmp_path, "wb") as fh:
            fh.write(uploaded_file.read())
        uploaded_file.seek(0)

        tables = list_sqlite_tables_from_path(
            tmp_path,
            probe_limit=min(SQLITE_DISCOVERY_PROBE_ROWS, row_limit + 1),
            query_timeout_seconds=min(timeout, 10),
        )
        if not tables:
            raise ValueError("No tables or views found in this SQLite database.")
        report["tables_found"] = tables

        con = _open_sqlite_readonly(tmp_path, timeout_seconds=min(timeout, 30))
        try:
            if query and query.strip():
                sql = validate_readonly_select_query(query)
                report["query_used"] = sql
                report["table_selected"] = "Custom SELECT query"
            else:
                if table is None:
                    # Prefer the largest safely probed object; tables win ties.
                    best = max(
                        tables,
                        key=lambda item: (
                            item.get("rows") if item.get("rows") is not None else -1,
                            1 if item.get("type") == "table" else 0,
                        ),
                    )
                    table = str(best["name"])
                valid_names = {str(item["name"]) for item in tables}
                if table not in valid_names:
                    raise ValueError(f"Table not found in SQLite database: {table}")
                sql = f"SELECT * FROM {_quote_sqlite_identifier(table)}"
                report["table_selected"] = table

            df, was_truncated = _execute_sqlite_bounded_select(
                con,
                sql,
                row_limit=row_limit,
                query_timeout_seconds=timeout,
            )
        finally:
            con.close()

    if was_truncated:
        report["sql_rows_truncated"] = True
        report["sql_rows_limit"] = row_limit

    df, steps = _finalize_import_dataframe(df, report)
    steps.insert(
        0,
        {
            "action": "🗄️ SQLite source loaded read-only",
            "detail": (
                f"Loaded {len(df):,} row(s) from {report['table_selected']} using "
                f"SQLite query-only mode. Objects found: {len(tables)}."
            ),
            "count": len(df),
            "severity": "info",
        },
    )
    if was_truncated:
        steps.insert(
            1,
            {
                "action": "⚠️ SQLite result capped",
                "detail": (
                    f"The source contained more than {row_limit:,} matching rows. "
                    "Only the approved row limit was loaded; the full result never entered memory."
                ),
                "count": row_limit,
                "severity": "warning",
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
def _database_url_details(connection_url: str):
    try:
        from sqlalchemy.engine import make_url
    except Exception as exc:
        raise ImportError(
            "SQLAlchemy is required for database connections. Run: pip install sqlalchemy"
        ) from exc

    try:
        url = make_url((connection_url or "").strip())
    except Exception as exc:
        raise ValueError("Invalid SQLAlchemy connection URL.") from exc

    dialect = url.get_backend_name().lower()
    if dialect not in SUPPORTED_DATABASE_DIALECTS:
        supported = ", ".join(sorted(SUPPORTED_DATABASE_DIALECTS))
        raise ValueError(
            f"Unsupported database dialect '{dialect}'. Secure connector supports: {supported}."
        )

    driver = url.get_driver_name().lower()
    allowed_drivers = SUPPORTED_DATABASE_DRIVERS[dialect]
    if driver not in allowed_drivers:
        expected = ", ".join(sorted(allowed_drivers))
        raise ValueError(
            f"Unsupported driver '{driver}' for {dialect}. "
            f"Use the tested secure driver: {expected}."
        )

    safe_url = url.render_as_string(hide_password=True)
    return url, dialect, safe_url


def _sanitise_database_exception(
    exc: Exception,
    *,
    connection_url: str,
    safe_url: str,
    password: Optional[str],
) -> str:
    message = str(exc) or exc.__class__.__name__
    if connection_url:
        message = message.replace(connection_url, safe_url)
    if password:
        message = message.replace(str(password), "***")
    # SQLAlchemy exceptions often echo the full SQL and bound parameters. They
    # may contain sensitive literals, so only the driver-level error is retained.
    for marker in ("[SQL:", "[parameters:"):
        if marker in message:
            message = message.split(marker, 1)[0] + "[SQL details redacted]"
            break
    # Keep UI errors useful without exposing a huge driver trace or query body.
    message = re.sub(r"\s+", " ", message).strip()
    return message[:700]


def _secure_engine_kwargs(dialect: str, query_timeout_seconds: int) -> Dict[str, Any]:
    timeout = _validate_query_timeout(query_timeout_seconds)
    timeout_ms = timeout * 1000
    kwargs: Dict[str, Any] = {
        "pool_pre_ping": True,
        "pool_recycle": 300,
    }

    if dialect == "postgresql":
        kwargs["connect_args"] = {
            "connect_timeout": SQL_CONNECT_TIMEOUT_SECONDS,
            "options": (
                f"-c statement_timeout={timeout_ms} "
                "-c default_transaction_read_only=on"
            ),
        }
    elif dialect in {"mysql", "mariadb"}:
        kwargs["connect_args"] = {
            "connect_timeout": SQL_CONNECT_TIMEOUT_SECONDS,
            "read_timeout": timeout,
            "write_timeout": timeout,
        }
    elif dialect == "sqlite":
        kwargs["connect_args"] = {
            "timeout": min(timeout, 30),
            "check_same_thread": False,
        }
    return kwargs


def _truthy_database_flag(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "on", "true", "yes"}


def _configure_verified_readonly_connection(
    conn,
    *,
    dialect: str,
    query_timeout_seconds: int,
) -> str:
    """Enable and verify a read-only session for every supported dialect."""
    timeout = _validate_query_timeout(query_timeout_seconds)
    timeout_ms = timeout * 1000

    if dialect == "sqlite":
        conn.exec_driver_sql("PRAGMA query_only = ON")
        try:
            conn.exec_driver_sql("PRAGMA trusted_schema = OFF")
        except Exception:
            pass
        value = conn.exec_driver_sql("PRAGMA query_only").scalar()
        if not _truthy_database_flag(value):
            raise PermissionError("Could not verify SQLite query-only mode.")
        raw = getattr(getattr(conn, "connection", None), "driver_connection", None)
        if raw is not None and hasattr(raw, "set_progress_handler"):
            deadline = time.monotonic() + timeout

            def abort_when_expired() -> int:
                return 1 if time.monotonic() >= deadline else 0

            raw.set_progress_handler(abort_when_expired, 10_000)
        return "SQLite PRAGMA query_only=ON"

    if dialect == "postgresql":
        # default_transaction_read_only and statement_timeout were applied during
        # connection creation; verify instead of trusting configuration text.
        readonly = conn.exec_driver_sql("SHOW transaction_read_only").scalar()
        if not _truthy_database_flag(readonly):
            raise PermissionError("PostgreSQL connection is not read-only.")
        return f"PostgreSQL read-only session; {timeout}s statement timeout"

    if dialect in {"mysql", "mariadb"}:
        conn.exec_driver_sql("SET SESSION TRANSACTION READ ONLY")
        if dialect == "mysql":
            try:
                conn.exec_driver_sql(f"SET SESSION MAX_EXECUTION_TIME = {timeout_ms}")
            except Exception:
                pass
        else:
            try:
                conn.exec_driver_sql(f"SET SESSION max_statement_time = {timeout}")
            except Exception:
                pass
        # Commit SET statements so the next transaction inherits read-only mode.
        try:
            conn.commit()
        except Exception:
            pass

        readonly = None
        for probe in (
            "SELECT @@session.transaction_read_only",
            "SELECT @@session.tx_read_only",
        ):
            try:
                readonly = conn.exec_driver_sql(probe).scalar()
                break
            except Exception:
                continue
        if not _truthy_database_flag(readonly):
            raise PermissionError(
                "Could not verify a read-only MySQL/MariaDB session. Use a dedicated read-only account."
            )
        return f"{dialect.title()} read-only session; {timeout}s driver timeout"

    raise ValueError(f"Unsupported secure database dialect: {dialect}")


def _execute_sqlalchemy_bounded_select(
    conn,
    sql: str,
    *,
    row_limit: int,
) -> Tuple[pd.DataFrame, bool]:
    """Run the validated query with a database-side cap and streamed fetching."""
    from sqlalchemy import text

    row_limit = _validate_row_limit(row_limit)
    limited_sql = (
        "SELECT * FROM (\n"
        + sql
        + f"\n) AS databridge_secure_source LIMIT {row_limit + 1}"
    )
    stream_conn = conn.execution_options(
        stream_results=True,
        max_row_buffer=min(SQL_FETCH_CHUNK_ROWS, row_limit + 1),
    )
    result = stream_conn.execute(text(limited_sql))
    try:
        columns = list(result.keys())
        rows: List[tuple] = []
        remaining = row_limit + 1
        while remaining > 0:
            batch = result.fetchmany(min(SQL_FETCH_CHUNK_ROWS, remaining))
            if not batch:
                break
            rows.extend(tuple(row) for row in batch)
            remaining -= len(batch)
    finally:
        result.close()

    truncated = len(rows) > row_limit
    if truncated:
        rows = rows[:row_limit]
    return pd.DataFrame.from_records(rows, columns=columns), truncated


def read_sqlalchemy_query(
    connection_url: str,
    query: str,
    *,
    row_limit: int = MAX_SQL_ROWS,
    query_timeout_seconds: int = DEFAULT_SQL_QUERY_TIMEOUT_SECONDS,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Read one validated SELECT query through a verified read-only connection.

    The result is capped inside SQL at ``row_limit + 1`` and fetched in bounded
    chunks. This prevents the previous failure mode where an unlimited result was
    fully loaded into RAM before ``head()`` was applied.
    """
    if not connection_url or not connection_url.strip():
        raise ValueError("Connection URL is required.")
    row_limit = _validate_row_limit(row_limit)
    timeout = _validate_query_timeout(query_timeout_seconds)
    sql = validate_readonly_select_query(query)

    try:
        from sqlalchemy import create_engine
    except Exception as exc:
        raise ImportError(
            "SQLAlchemy is required for database connections. Run: pip install sqlalchemy"
        ) from exc

    url, dialect, safe_url = _database_url_details(connection_url)
    report = _base_report("Database", "SQLAlchemy connection")
    report["source_name"] = safe_url
    report["query_used"] = sql
    report["sql_rows_limit"] = row_limit
    report["sql_query_timeout_seconds"] = timeout
    report["database_dialect"] = dialect

    engine = None
    try:
        engine = create_engine(
            url,
            **_secure_engine_kwargs(dialect, timeout),
        )
        with engine.connect() as conn:
            readonly_detail = _configure_verified_readonly_connection(
                conn,
                dialect=dialect,
                query_timeout_seconds=timeout,
            )
            df, was_truncated = _execute_sqlalchemy_bounded_select(
                conn,
                sql,
                row_limit=row_limit,
            )
    except Exception as exc:
        safe_message = _sanitise_database_exception(
            exc,
            connection_url=connection_url.strip(),
            safe_url=safe_url,
            password=url.password,
        )
        if isinstance(exc, (ValueError, PermissionError, TimeoutError)):
            raise type(exc)(safe_message) from exc
        raise RuntimeError(f"Database query failed safely: {safe_message}") from exc
    finally:
        if engine is not None:
            engine.dispose()

    if was_truncated:
        report["sql_rows_truncated"] = True

    df, steps = _finalize_import_dataframe(df, report)
    steps.insert(
        0,
        {
            "action": "🔒 Database query loaded in verified read-only mode",
            "detail": (
                f"Loaded {len(df):,} row(s) from {dialect}. {readonly_detail}. "
                f"Result cap: {row_limit:,} rows."
            ),
            "count": len(df),
            "severity": "info",
        },
    )

    if was_truncated:
        steps.insert(
            1,
            {
                "action": "⚠️ Result capped before pandas",
                "detail": (
                    f"The query matched more than {row_limit:,} rows. "
                    f"Only the first {row_limit:,} rows were requested and fetched; "
                    "the unlimited result was never loaded into application memory."
                ),
                "count": row_limit,
                "severity": "warning",
            },
        )

    report["cleaning_steps"] = steps
    return df, report
