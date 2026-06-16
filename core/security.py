# ════════════════════════════════════════════════════════
#  DataBridge AI — Security Utilities
# ════════════════════════════════════════════════════════
from __future__ import annotations

import html
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from config.constants import MAX_UPLOAD_SIZE_MB, PII_PATTERNS

logger = logging.getLogger(__name__)

TEXT_MIME_PREFIXES = ("text/",)

ALLOWED_EXTENSIONS = {
    ".csv", ".xlsx", ".xls", ".json", ".jsonl", ".ndjson",
    ".parquet", ".db", ".sqlite", ".sqlite3",
}


def safe_html(value: Any) -> str:
    """Escape any value for safe HTML rendering."""
    return html.escape("" if value is None else str(value))


def _read_file_head(uploaded_file, size: int = 8192) -> bytes:
    """Read a small prefix without changing the caller's file pointer."""
    try:
        pos = uploaded_file.tell()
    except Exception:
        pos = 0
    try:
        uploaded_file.seek(0)
        head = uploaded_file.read(size)
        if isinstance(head, str):
            head = head.encode("utf-8", errors="ignore")
        return bytes(head or b"")
    finally:
        try:
            uploaded_file.seek(pos)
        except Exception:
            try:
                uploaded_file.seek(0)
            except Exception:
                pass


def _detect_mime_with_python_magic(head: bytes) -> str:
    """Best-effort MIME detection using python-magic when available."""
    try:
        import magic  # type: ignore
        return str(magic.from_buffer(head, mime=True) or "").lower()
    except Exception:
        return ""


def _looks_like_text(head: bytes) -> bool:
    if not head:
        return False
    sample = head[:4096]
    if b"\x00" in sample:
        return False
    try:
        sample.decode("utf-8")
        return True
    except UnicodeDecodeError:
        try:
            sample.decode("cp1256")
            return True
        except UnicodeDecodeError:
            try:
                sample.decode("latin1")
                return True
            except UnicodeDecodeError:
                return False


def _matches_extension_signature(ext: str, head: bytes, mime: str) -> bool:
    """Validate file content against expected extension family."""
    ext = ext.lower()
    mime = (mime or "").lower()

    if ext == ".csv":
        return mime.startswith(TEXT_MIME_PREFIXES) or mime in {"application/csv", "text/csv", "application/vnd.ms-excel", ""} or _looks_like_text(head)

    if ext in {".json", ".jsonl", ".ndjson"}:
        stripped = head.lstrip()
        return (
            mime in {"application/json", "application/x-ndjson", ""}
            or mime.startswith(TEXT_MIME_PREFIXES)
            or stripped.startswith((b"{", b"["))
        )

    if ext == ".xlsx":
        # xlsx is a ZIP container.
        return head.startswith(b"PK\x03\x04") or "zip" in mime or "officedocument" in mime

    if ext == ".xls":
        # Legacy Excel OLE compound document.
        return head.startswith(b"\xD0\xCF\x11\xE0") or "excel" in mime or "cdf" in mime or "msword" in mime

    if ext == ".parquet":
        return head.startswith(b"PAR1") or "parquet" in mime

    if ext in {".db", ".sqlite", ".sqlite3"}:
        return head.startswith(b"SQLite format 3\x00") or "sqlite" in mime or mime == "application/vnd.sqlite3"

    return False


def validate_uploaded_file(uploaded_file) -> None:
    """
    Validate file size, extension, and content signature/MIME.

    Uses python-magic when installed, with a pure-Python signature fallback so the app
    does not break on Windows machines where libmagic is unavailable.
    """
    if uploaded_file is None:
        raise ValueError("No file uploaded.")

    size_mb = uploaded_file.size / (1024 * 1024)
    if size_mb > MAX_UPLOAD_SIZE_MB:
        raise ValueError(
            f"File too large ({size_mb:.1f} MB). Maximum allowed: {MAX_UPLOAD_SIZE_MB} MB."
        )

    name = getattr(uploaded_file, "name", "") or ""
    ext = Path(name).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file extension: {ext or 'unknown'}")

    head = _read_file_head(uploaded_file)
    if not head:
        raise ValueError("Uploaded file is empty or unreadable.")

    mime = _detect_mime_with_python_magic(head)
    if not _matches_extension_signature(ext, head, mime):
        detected = mime or "unknown"
        raise ValueError(
            f"File content does not match extension {ext}. Detected MIME/signature: {detected}."
        )


def anonymise_df_for_ai(df: pd.DataFrame) -> pd.DataFrame:
    """Mask columns that match PII patterns before sending to cloud AI."""
    df_anon = df.copy()
    for col in df_anon.columns:
        col_l = str(col).lower()
        if any(p in col_l for p in PII_PATTERNS):
            df_anon[col] = "***REDACTED***"
            logger.info("PII column masked: %s", col)
    return df_anon
