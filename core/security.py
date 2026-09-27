# ════════════════════════════════════════════════════════
#  DataBridge AI — Security Utilities
#  Stage 12: upload validation, PII controls, endpoint trust, redaction
# ════════════════════════════════════════════════════════
from __future__ import annotations

import html
import ipaddress
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping
from urllib.parse import urlsplit

import pandas as pd

from config.constants import MAX_UPLOAD_SIZE_MB, PII_PATTERNS

logger = logging.getLogger(__name__)

TEXT_MIME_PREFIXES = ("text/",)

ALLOWED_EXTENSIONS = {
    ".csv", ".xlsx", ".xls", ".json", ".jsonl", ".ndjson",
    ".parquet", ".db", ".sqlite", ".sqlite3",
}

EMAIL_PATTERN = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
PHONE_PATTERN = re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{6,}\d)(?!\d)")
CARD_PATTERN = re.compile(r"(?<!\d)(?:\d[ -]*?){13,19}(?!\d)")
IPV4_PATTERN = re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)")
API_KEY_PATTERNS = (
    re.compile(r"(?i)\bsk-ant-[A-Za-z0-9_-]{10,}\b"),
    re.compile(r"\bAIza[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+\-/=]{8,}"),
    re.compile(r"(?i)((?:api[_ -]?key|token|secret|password)\s*[:=]\s*)[^\s,;]+"),
)
URL_CREDENTIAL_PATTERN = re.compile(
    r"(?i)([a-z][a-z0-9+.-]*://)([^/@:\s]+):([^/@\s]+)@"
)


@dataclass(frozen=True)
class PIIMaskReport:
    masked_columns: tuple[str, ...]
    value_redactions: int
    sample_rows: int


def safe_html(value: Any) -> str:
    """Escape any value before inserting it into ``unsafe_allow_html`` markup."""
    return html.escape("" if value is None else str(value), quote=True)


def redact_sensitive_text(value: Any, extra_secrets: Iterable[str] = ()) -> str:
    """Remove common API-key, bearer-token and URL-credential patterns."""
    text = "" if value is None else str(value)
    for secret in extra_secrets:
        secret_text = str(secret or "")
        if secret_text:
            text = text.replace(secret_text, "***REDACTED***")
    text = URL_CREDENTIAL_PATTERN.sub(r"\1***:***@", text)
    for pattern in API_KEY_PATTERNS:
        if pattern.groups:
            text = pattern.sub(r"\1***REDACTED***", text)
        else:
            text = pattern.sub("***REDACTED***", text)
    return text


def safe_error_message(error: BaseException | str, extra_secrets: Iterable[str] = ()) -> str:
    """Return a bounded, redacted error suitable for the UI or logs."""
    text = redact_sensitive_text(error, extra_secrets=extra_secrets)
    text = re.sub(r"[\r\n\t]+", " ", text).strip()
    return (text[:600] + "…") if len(text) > 600 else (text or "Operation failed safely.")


def _read_file_head(uploaded_file, size: int = 8192) -> bytes:
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
    for encoding in ("utf-8", "cp1256", "latin1"):
        try:
            sample.decode(encoding)
            return True
        except UnicodeDecodeError:
            continue
    return False


def _matches_extension_signature(ext: str, head: bytes, mime: str) -> bool:
    ext = ext.lower()
    mime = (mime or "").lower()
    if ext == ".csv":
        return mime.startswith(TEXT_MIME_PREFIXES) or mime in {
            "application/csv", "text/csv", "application/vnd.ms-excel", ""
        } or _looks_like_text(head)
    if ext in {".json", ".jsonl", ".ndjson"}:
        stripped = head.lstrip()
        return (
            mime in {"application/json", "application/x-ndjson", ""}
            or mime.startswith(TEXT_MIME_PREFIXES)
            or stripped.startswith((b"{", b"["))
        )
    if ext == ".xlsx":
        return head.startswith(b"PK\x03\x04") or "zip" in mime or "officedocument" in mime
    if ext == ".xls":
        return head.startswith(b"\xD0\xCF\x11\xE0") or "excel" in mime or "cdf" in mime or "msword" in mime
    if ext == ".parquet":
        return head.startswith(b"PAR1") or "parquet" in mime
    if ext in {".db", ".sqlite", ".sqlite3"}:
        return head.startswith(b"SQLite format 3\x00") or "sqlite" in mime or mime == "application/vnd.sqlite3"
    return False


def validate_uploaded_file(uploaded_file) -> None:
    """Validate file size, extension and content signature/MIME."""
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
        raise ValueError(
            f"File content does not match extension {ext}. Detected MIME/signature: {mime or 'unknown'}."
        )


def _column_name_is_pii(column: Any) -> bool:
    folded = str(column).casefold().replace("_", " ").replace("-", " ")
    compact = re.sub(r"\s+", " ", folded).strip()
    tokens = set(re.findall(r"[\w\u0600-\u06ff]+", compact))
    for pattern in PII_PATTERNS:
        p = str(pattern).casefold().strip()
        if not p:
            continue
        if " " in p and p in compact:
            return True
        if p in tokens:
            return True
    return False


def _semantic_type_is_pii(profile: Mapping[str, Any] | None) -> bool:
    if not profile:
        return False
    semantic = str(
        profile.get("effective_semantic_type")
        or profile.get("semantic_type")
        or profile.get("business_role")
        or ""
    ).casefold()
    return semantic in {
        "email", "phone", "phone / contact", "name", "identifier", "id / identifier"
    }


def _luhn_valid(number: str) -> bool:
    digits = [int(ch) for ch in number if ch.isdigit()]
    if len(digits) < 13 or len(digits) > 19:
        return False
    checksum = 0
    parity = len(digits) % 2
    for index, digit in enumerate(digits):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


def _mask_text_value(value: Any) -> tuple[Any, int]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return value, 0
    text = str(value)
    count = 0

    def replace_email(match: re.Match[str]) -> str:
        nonlocal count
        count += 1
        return "***EMAIL***"

    def replace_phone(match: re.Match[str]) -> str:
        nonlocal count
        digits = re.sub(r"\D", "", match.group(0))
        if 7 <= len(digits) <= 15:
            count += 1
            return "***PHONE***"
        return match.group(0)

    def replace_card(match: re.Match[str]) -> str:
        nonlocal count
        if _luhn_valid(match.group(0)):
            count += 1
            return "***CARD***"
        return match.group(0)

    def replace_ip(match: re.Match[str]) -> str:
        nonlocal count
        try:
            ipaddress.ip_address(match.group(0))
        except ValueError:
            return match.group(0)
        count += 1
        return "***IP***"

    masked = EMAIL_PATTERN.sub(replace_email, text)
    masked = PHONE_PATTERN.sub(replace_phone, masked)
    masked = CARD_PATTERN.sub(replace_card, masked)
    masked = IPV4_PATTERN.sub(replace_ip, masked)
    return masked, count


def detect_pii_columns(
    df: pd.DataFrame,
    semantic_profiles: Mapping[str, Mapping[str, Any]] | None = None,
    *,
    sample_size: int = 100,
) -> list[str]:
    """Conservatively identify columns that should be fully masked."""
    profiles = semantic_profiles or {}
    detected: list[str] = []
    for column in df.columns:
        if _column_name_is_pii(column) or _semantic_type_is_pii(profiles.get(str(column))):
            detected.append(str(column))
            continue
        series = df[column].dropna().head(max(1, int(sample_size)))
        if series.empty:
            continue
        if not (
            pd.api.types.is_object_dtype(series.dtype)
            or pd.api.types.is_string_dtype(series.dtype)
        ):
            continue
        text = series.astype("string[python]")
        email_ratio = text.str.contains(EMAIL_PATTERN, regex=True, na=False).mean()
        if float(email_ratio) >= 0.20:
            detected.append(str(column))
            continue
        phone_hits = 0
        for value in text.tolist():
            if any(7 <= len(re.sub(r"\D", "", m.group(0))) <= 15 for m in PHONE_PATTERN.finditer(str(value))):
                phone_hits += 1
        if phone_hits / max(len(text), 1) >= 0.40:
            detected.append(str(column))
    return list(dict.fromkeys(detected))


def anonymise_df_for_ai(
    df: pd.DataFrame,
    semantic_profiles: Mapping[str, Mapping[str, Any]] | None = None,
    *,
    return_report: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, PIIMaskReport]:
    """Return an independent DataFrame with PII columns and inline values masked."""
    masked = df.copy(deep=True)
    pii_columns = detect_pii_columns(masked, semantic_profiles)
    redactions = 0
    for column in masked.columns:
        column_name = str(column)
        if column_name in pii_columns:
            non_null = int(masked[column].notna().sum())
            masked[column] = masked[column].where(masked[column].isna(), "***REDACTED***")
            redactions += non_null
            continue
        if not (
            pd.api.types.is_object_dtype(masked[column].dtype)
            or pd.api.types.is_string_dtype(masked[column].dtype)
        ):
            continue
        values = []
        for value in masked[column].tolist():
            replaced, count = _mask_text_value(value)
            values.append(replaced)
            redactions += count
        masked[column] = values
    report = PIIMaskReport(
        masked_columns=tuple(pii_columns),
        value_redactions=int(redactions),
        sample_rows=int(len(masked)),
    )
    if return_report:
        return masked, report
    return masked


def is_loopback_url(url: str) -> bool:
    """Return True only for explicit loopback HTTP(S) endpoints."""
    try:
        parsed = urlsplit(str(url or "").strip())
    except Exception:
        return False
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False
    if parsed.username or parsed.password:
        return False
    host = parsed.hostname.casefold().rstrip(".")
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def validate_http_endpoint(url: str) -> str:
    """Validate an HTTP endpoint and remove trailing slash."""
    cleaned = str(url or "").strip().rstrip("/")
    try:
        parsed = urlsplit(cleaned)
    except Exception as exc:
        raise ValueError("The endpoint URL is invalid.") from exc
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Only http:// or https:// endpoints are allowed.")
    if not parsed.hostname:
        raise ValueError("The endpoint must include a host name.")
    if parsed.username or parsed.password:
        raise ValueError("Credentials must not be embedded in an endpoint URL.")
    if parsed.fragment:
        raise ValueError("Endpoint URL fragments are not allowed.")
    return cleaned
