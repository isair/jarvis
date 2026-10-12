"""
Dictation history — persists transcription results to a local JSON file.

Privacy-first: all data stays on disk, never leaves the machine.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime

from filelock import FileLock
from pathlib import Path
from typing import Any, Dict, List, Optional


def _validate_entries(entries: Any) -> None:
    """Require records their existing consumers can read without coercion."""
    if not isinstance(entries, list):
        raise ValueError("Dictation history must contain a JSON list")
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Dictation history entries must be objects")
        if not isinstance(entry.get("id"), str) or not entry["id"]:
            raise ValueError("Dictation history entries require a non-empty text ID")
        if not isinstance(entry.get("text"), str):
            raise ValueError("Dictation history entries require text")
        entry["id"].encode("utf-8")
        entry["text"].encode("utf-8")
        for field in ("timestamp", "duration"):
            value = entry.get(field, 0)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("Dictation history metadata must be finite numbers")
        datetime.fromtimestamp(entry.get("timestamp", 0))


def _default_history_path() -> Path:
    """Return the default path for dictation history storage."""
    base = Path.home() / ".local" / "share" / "jarvis"
    base.mkdir(parents=True, exist_ok=True)
    return base / "dictation_history.json"


class HistoryUpdateError(RuntimeError):
    """A requested history mutation could not be published."""


class DictationHistory:
    """Thread-safe, file-backed dictation history.

    Each entry is a dict with keys:
        id       – unique identifier (UUID4 hex)
        text     – transcribed text
        timestamp – epoch seconds (float)
        duration – recording duration in seconds (float)
    """

    def __init__(self, path: Optional[Path] = None, max_entries: int = 500) -> None:
        self._path = path or _default_history_path()
        self._max_entries = max_entries
        self._lock = threading.Lock()
        self._file_lock = FileLock(str(self._path) + ".lock", timeout=2, mode=0o600,
                                   fallback_to_soft=False, preserve_lock_file=True)
        self._entries: List[Dict[str, Any]] = []
        self.reload_from_disk()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, text: str, duration: float = 0.0) -> Dict[str, Any]:
        """Append a new dictation entry and persist. Returns the new entry."""
        entry: Dict[str, Any] = {
            "id": uuid.uuid4().hex,
            "text": text,
            "timestamp": time.time(),
            "duration": round(duration, 1),
        }
        with self._write_transaction(wait_for_writer=True):
            self._entries.append(entry)
            # Trim oldest entries if over limit
            if len(self._entries) > self._max_entries:
                self._entries = self._entries[-self._max_entries:]
            self._save()
        return entry

    def get_all(self) -> List[Dict[str, Any]]:
        """Return all entries, newest first."""
        with self._lock:
            return list(reversed(self._entries))

    def delete(self, entry_id: str, *, wait_for_writer: bool = True) -> bool:
        """Delete an entry by ID. Returns True if found and removed."""
        with self._write_transaction(wait_for_writer=wait_for_writer):
            before = len(self._entries)
            self._entries = [e for e in self._entries if e["id"] != entry_id]
            if len(self._entries) < before:
                self._save()
                return True
            return False

    def clear(self, *, wait_for_writer: bool = True) -> None:
        """Delete all entries."""
        with self._write_transaction(wait_for_writer=wait_for_writer):
            self._entries = []
            self._save()

    def reload_from_disk(self) -> None:
        """Re-read entries from the JSON file (thread-safe).

        Useful for external consumers (e.g. the desktop app) that need to
        pick up changes written by another process.
        """
        with self._lock:
            try:
                self._entries = self._read_entries()
            except Exception as exc:
                from jarvis.debug import debug_log
                debug_log(f"dictation history refresh failed: {type(exc).__name__}", "dictation")

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._entries)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _read_entries(self) -> List[Dict[str, Any]]:
        """Read the current JSON list; only a missing file represents a new store."""
        try:
            stream = self._path.open("r", encoding="utf-8")
        except FileNotFoundError:
            return []
        with stream:
            data = json.load(stream)
        _validate_entries(data)
        return data

    @contextmanager
    def _write_transaction(self, *, wait_for_writer: bool):
        """Serialise a complete mutation, or immediately reject foreground contention."""
        try:
            timeout = self._file_lock.timeout if wait_for_writer else 0
            with self._file_lock.acquire(timeout=timeout):
                if not self._lock.acquire(blocking=wait_for_writer):
                    raise HistoryUpdateError("Dictation history is busy")
                try:
                    current = self._read_entries()
                    self._entries = list(current)
                    try:
                        yield
                    except Exception:
                        self._entries = current
                        raise
                finally:
                    self._lock.release()
        except Exception as exc:
            from jarvis.debug import debug_log
            debug_log(f"dictation history update failed: {type(exc).__name__}", "dictation")
            raise HistoryUpdateError("Unable to save dictation history. Please try again.") from exc

    def _save(self) -> None:
        _validate_entries(self._entries)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self._path.parent,
                prefix=f".{self._path.name}.", suffix=".tmp", delete=False,
            ) as stream:
                temporary = Path(stream.name)
                json.dump(self._entries, stream, ensure_ascii=False, indent=2)
            os.replace(temporary, self._path)
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
