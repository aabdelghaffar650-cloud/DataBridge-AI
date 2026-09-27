# ════════════════════════════════════════════════════════
#  DataBridge AI — Safe History Manager (Undo / Redo)
#  Stage 2: full snapshots, disk spill, atomic restore
# ════════════════════════════════════════════════════════
from __future__ import annotations

import copy
import gzip
import logging
import os
import pickle
import shutil
import tempfile
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, Optional

import pandas as pd

from config.constants import (
    HISTORY_MEMORY_SNAPSHOT_MB,
    MAX_HISTORY,
    MAX_HISTORY_DISK_MB,
)

logger = logging.getLogger(__name__)


class HistoryError(RuntimeError):
    """Raised when a safe history snapshot cannot be created or restored."""


@dataclass
class SnapshotRef:
    """Reference to one complete, immutable dataset snapshot."""

    snapshot_id: str
    action: str
    created_at: float
    rows: int
    columns: int
    fingerprint: str
    storage: str
    size_bytes: int
    payload: Optional[bytes] = field(default=None, repr=False)
    path: Optional[str] = None

    @property
    def shape(self) -> tuple[int, int]:
        return self.rows, self.columns


@dataclass
class HistoryRestore:
    """Loaded snapshot returned by undo or redo."""

    dataframe: pd.DataFrame
    context: Dict[str, Any]
    action: str
    snapshot_id: str
    fingerprint: str


class SmartHistoryManager:
    """
    Full-fidelity, bounded undo/redo manager.

    Small snapshots are stored as compressed bytes in memory. Larger snapshots
    spill to a private temporary directory. A snapshot is never truncated: if
    the full state cannot be stored, the operation is blocked with HistoryError.
    """

    FORMAT_VERSION = 2

    def __init__(
        self,
        max_history: int = MAX_HISTORY,
        memory_snapshot_mb: float = HISTORY_MEMORY_SNAPSHOT_MB,
        max_disk_mb: float = MAX_HISTORY_DISK_MB,
        temp_root: str | os.PathLike[str] | None = None,
    ) -> None:
        self.max_history = max(1, int(max_history))
        self.memory_snapshot_bytes = max(0, int(float(memory_snapshot_mb) * 1024 * 1024))
        self.max_disk_bytes = max(1, int(float(max_disk_mb) * 1024 * 1024))
        self.history: Deque[SnapshotRef] = deque()
        self.redo_stack: list[SnapshotRef] = []

        root = Path(temp_root).expanduser().resolve() if temp_root else None
        if root is not None:
            root.mkdir(parents=True, exist_ok=True)
            self._temp_dir_obj = None
            self._temp_dir = Path(tempfile.mkdtemp(prefix="databridge_history_", dir=str(root)))
        else:
            self._temp_dir_obj = tempfile.TemporaryDirectory(prefix="databridge_history_")
            self._temp_dir = Path(self._temp_dir_obj.name)

        self.last_error = ""

    # ── serialization ──────────────────────────────────────────────────────
    @staticmethod
    def _estimate_memory(df: pd.DataFrame) -> int:
        try:
            return int(df.memory_usage(index=True, deep=True).sum())
        except Exception:
            return 0

    @staticmethod
    def _safe_context(context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            return copy.deepcopy(dict(context or {}))
        except Exception as exc:
            raise HistoryError(f"Could not copy dataset context safely: {exc}") from exc

    def _build_envelope(
        self,
        df: pd.DataFrame,
        context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not isinstance(df, pd.DataFrame):
            raise HistoryError("History can only store pandas DataFrames.")
        return {
            "format_version": self.FORMAT_VERSION,
            "dataframe": df,
            "context": self._safe_context(context),
        }

    @staticmethod
    def _compress_to_bytes(envelope: Dict[str, Any]) -> bytes:
        try:
            raw = pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL)
            return gzip.compress(raw, compresslevel=3)
        except Exception as exc:
            raise HistoryError(f"Could not serialize the complete dataset snapshot: {exc}") from exc

    @staticmethod
    def _read_compressed_bytes(payload: bytes) -> Dict[str, Any]:
        try:
            return pickle.loads(gzip.decompress(payload))
        except Exception as exc:
            raise HistoryError(f"History snapshot is unreadable or corrupted: {exc}") from exc

    def _disk_usage_bytes(self) -> int:
        total = 0
        for ref in self._all_refs():
            if ref.storage == "disk":
                total += int(ref.size_bytes)
        return total

    def _write_disk_snapshot(self, envelope: Dict[str, Any], snapshot_id: str) -> tuple[str, int]:
        target = self._temp_dir / f"{snapshot_id}.pkl.gz"
        temp_target = self._temp_dir / f".{snapshot_id}.tmp"
        try:
            with gzip.open(temp_target, "wb", compresslevel=3) as handle:
                pickle.dump(envelope, handle, protocol=pickle.HIGHEST_PROTOCOL)
            size = int(temp_target.stat().st_size)
            if size > self.max_disk_bytes:
                raise HistoryError(
                    "The complete undo snapshot exceeds the configured history disk limit "
                    f"({size / (1024 * 1024):.1f} MB > {self.max_disk_bytes / (1024 * 1024):.1f} MB). "
                    "The change was blocked to prevent partial or unsafe history."
                )
            os.replace(temp_target, target)
            return str(target), size
        except HistoryError:
            temp_target.unlink(missing_ok=True)
            target.unlink(missing_ok=True)
            raise
        except Exception as exc:
            temp_target.unlink(missing_ok=True)
            target.unlink(missing_ok=True)
            raise HistoryError(f"Could not write the complete undo snapshot: {exc}") from exc

    def _create_snapshot(
        self,
        df: pd.DataFrame,
        *,
        action: str,
        fingerprint: str,
        context: Optional[Dict[str, Any]],
    ) -> SnapshotRef:
        snapshot_id = uuid.uuid4().hex
        action_label = (action or "Dataset change").strip()[:240]
        envelope = self._build_envelope(df, context)
        estimate = self._estimate_memory(df)

        if estimate <= self.memory_snapshot_bytes:
            payload = self._compress_to_bytes(envelope)
            return SnapshotRef(
                snapshot_id=snapshot_id,
                action=action_label,
                created_at=time.time(),
                rows=int(df.shape[0]),
                columns=int(df.shape[1]),
                fingerprint=fingerprint,
                storage="memory",
                size_bytes=len(payload),
                payload=payload,
            )

        path, size = self._write_disk_snapshot(envelope, snapshot_id)
        return SnapshotRef(
            snapshot_id=snapshot_id,
            action=action_label,
            created_at=time.time(),
            rows=int(df.shape[0]),
            columns=int(df.shape[1]),
            fingerprint=fingerprint,
            storage="disk",
            size_bytes=size,
            path=path,
        )

    def _load_snapshot(self, ref: SnapshotRef) -> HistoryRestore:
        try:
            if ref.storage == "memory":
                if ref.payload is None:
                    raise HistoryError("The in-memory history snapshot has no payload.")
                envelope = self._read_compressed_bytes(ref.payload)
            elif ref.storage == "disk":
                if not ref.path or not Path(ref.path).is_file():
                    raise HistoryError("The disk history snapshot is missing.")
                with gzip.open(ref.path, "rb") as handle:
                    envelope = pickle.load(handle)
            else:
                raise HistoryError(f"Unknown history storage type: {ref.storage}")

            if envelope.get("format_version") != self.FORMAT_VERSION:
                raise HistoryError("The history snapshot format is not supported.")
            df = envelope.get("dataframe")
            if not isinstance(df, pd.DataFrame):
                raise HistoryError("The history snapshot does not contain a valid DataFrame.")
            context = envelope.get("context") or {}
            if not isinstance(context, dict):
                context = {}

            return HistoryRestore(
                dataframe=df,
                context=context,
                action=ref.action,
                snapshot_id=ref.snapshot_id,
                fingerprint=ref.fingerprint,
            )
        except HistoryError:
            raise
        except Exception as exc:
            raise HistoryError(f"Could not restore history snapshot: {exc}") from exc

    # ── lifecycle / trimming ───────────────────────────────────────────────
    def _all_refs(self) -> Iterable[SnapshotRef]:
        yield from self.history
        yield from self.redo_stack

    @staticmethod
    def _delete_ref(ref: SnapshotRef) -> None:
        if ref.storage == "disk" and ref.path:
            try:
                Path(ref.path).unlink(missing_ok=True)
            except Exception:
                logger.warning("Could not delete history snapshot %s", ref.path, exc_info=True)
        ref.payload = None
        ref.path = None

    def _clear_stack(self, refs: Iterable[SnapshotRef]) -> None:
        for ref in list(refs):
            self._delete_ref(ref)

    def _trim_history_count(self) -> None:
        while len(self.history) > self.max_history:
            self._delete_ref(self.history.popleft())

    def _trim_disk_budget(self, protected_ids: Optional[set[str]] = None) -> None:
        """Evict the farthest history points without ever truncating a snapshot."""
        protected = protected_ids or set()
        while self._disk_usage_bytes() > self.max_disk_bytes:
            candidate = None
            candidate_stack = None

            # Oldest undo states are the least valuable. Keep protected states.
            for ref in self.history:
                if ref.snapshot_id not in protected:
                    candidate = ref
                    candidate_stack = self.history
                    break

            # Then evict the farthest redo state (index 0); the next redo is last.
            if candidate is None:
                for ref in self.redo_stack:
                    if ref.snapshot_id not in protected:
                        candidate = ref
                        candidate_stack = self.redo_stack
                        break

            if candidate is None:
                raise HistoryError(
                    "History storage is full. The complete snapshot could not be retained safely."
                )

            candidate_stack.remove(candidate)
            self._delete_ref(candidate)

    def clear(self) -> None:
        self._clear_stack(self.history)
        self._clear_stack(self.redo_stack)
        self.history.clear()
        self.redo_stack.clear()
        self.last_error = ""

    def close(self) -> None:
        self.clear()
        try:
            if self._temp_dir_obj is not None:
                self._temp_dir_obj.cleanup()
            else:
                shutil.rmtree(self._temp_dir, ignore_errors=True)
        except Exception:
            logger.warning("Could not clean history temporary directory", exc_info=True)

    def __del__(self) -> None:  # pragma: no cover - interpreter shutdown is nondeterministic
        try:
            self.close()
        except Exception:
            pass

    # ── public operations ──────────────────────────────────────────────────
    def push(
        self,
        df: pd.DataFrame,
        *,
        action: str = "Dataset change",
        fingerprint: str = "",
        context: Optional[Dict[str, Any]] = None,
        clear_redo: bool = True,
    ) -> str:
        """Store the complete current state before a mutation."""
        try:
            ref = self._create_snapshot(
                df,
                action=action,
                fingerprint=fingerprint,
                context=context,
            )
            self.history.append(ref)
            if clear_redo:
                self._clear_stack(self.redo_stack)
                self.redo_stack.clear()
            self._trim_history_count()
            self._trim_disk_budget({ref.snapshot_id})
            self.last_error = ""
            return ref.snapshot_id
        except Exception as exc:
            self.last_error = str(exc)
            if isinstance(exc, HistoryError):
                raise
            raise HistoryError(str(exc)) from exc


    def commit_push(self, snapshot_id: str) -> bool:
        """Commit a pending mutation and invalidate the old redo branch."""
        if not snapshot_id or not any(ref.snapshot_id == snapshot_id for ref in self.history):
            return False
        self._clear_stack(self.redo_stack)
        self.redo_stack.clear()
        return True

    def cancel_push(self, snapshot_id: str) -> bool:
        """Remove a just-created history point when the attempted mutation was a no-op."""
        if not snapshot_id:
            return False
        for index in range(len(self.history) - 1, -1, -1):
            ref = self.history[index]
            if ref.snapshot_id == snapshot_id:
                del self.history[index]
                self._delete_ref(ref)
                return True
        return False

    def undo(
        self,
        current_df: pd.DataFrame,
        *,
        current_context: Optional[Dict[str, Any]] = None,
        current_fingerprint: str = "",
    ) -> Optional[HistoryRestore]:
        if not self.history:
            return None

        target_ref = self.history[-1]
        target = self._load_snapshot(target_ref)
        # The current state becomes the redo target for the action being undone.
        current_ref = self._create_snapshot(
            current_df,
            action=target_ref.action,
            fingerprint=current_fingerprint,
            context=current_context,
        )

        self.history.pop()
        self.redo_stack.append(current_ref)
        self._delete_ref(target_ref)
        self._trim_disk_budget({current_ref.snapshot_id})
        return target

    def redo(
        self,
        current_df: pd.DataFrame,
        *,
        current_context: Optional[Dict[str, Any]] = None,
        current_fingerprint: str = "",
    ) -> Optional[HistoryRestore]:
        if not self.redo_stack:
            return None

        target_ref = self.redo_stack[-1]
        target = self._load_snapshot(target_ref)
        # The current state becomes the undo target for the action being redone.
        current_ref = self._create_snapshot(
            current_df,
            action=target_ref.action,
            fingerprint=current_fingerprint,
            context=current_context,
        )

        self.redo_stack.pop()
        self.history.append(current_ref)
        self._delete_ref(target_ref)
        self._trim_history_count()
        self._trim_disk_budget({current_ref.snapshot_id})
        return target

    # ── presentation helpers ───────────────────────────────────────────────
    @property
    def can_undo(self) -> bool:
        return bool(self.history)

    @property
    def can_redo(self) -> bool:
        return bool(self.redo_stack)

    @property
    def undo_count(self) -> int:
        return len(self.history)

    @property
    def redo_count(self) -> int:
        return len(self.redo_stack)

    @property
    def next_undo_action(self) -> str:
        return self.history[-1].action if self.history else ""

    @property
    def next_redo_action(self) -> str:
        return self.redo_stack[-1].action if self.redo_stack else ""

    def metadata(self) -> list[Dict[str, Any]]:
        return [
            {
                "snapshot_id": ref.snapshot_id,
                "action": ref.action,
                "created_at": ref.created_at,
                "rows": ref.rows,
                "columns": ref.columns,
                "fingerprint": ref.fingerprint,
                "storage": ref.storage,
                "size_bytes": ref.size_bytes,
            }
            for ref in self.history
        ]

    def storage_summary(self) -> Dict[str, Any]:
        memory_bytes = sum(ref.size_bytes for ref in self._all_refs() if ref.storage == "memory")
        disk_bytes = sum(ref.size_bytes for ref in self._all_refs() if ref.storage == "disk")
        return {
            "undo_count": self.undo_count,
            "redo_count": self.redo_count,
            "memory_bytes": int(memory_bytes),
            "disk_bytes": int(disk_bytes),
            "temp_dir": str(self._temp_dir),
        }
