"""
Dictation history — persists transcription results to a local JSON file.

Privacy-first: all data stays on disk, never leaves the machine.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional


def _default_history_path() -> Path:
    """Return the default path for dictation history storage."""
    base = Path.home() / ".local" / "share" / "jarvis"
    base.mkdir(parents=True, exist_ok=True)
    return base / "dictation_history.json"


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
        self._entries: List[Dict[str, Any]] = self._load()

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
        with self._lock:
            # Re-read from disk to pick up external changes (e.g. deletions
            # made by the desktop app while the daemon runs in a subprocess).
            self._entries = self._load()
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

    def delete(self, entry_id: str) -> bool:
        """Delete an entry by ID. Returns True if found and removed."""
        with self._lock:
            before = len(self._entries)
            self._entries = [e for e in self._entries if e["id"] != entry_id]
            if len(self._entries) < before:
                self._save()
                return True
            return False

    def clear(self) -> None:
        """Delete all entries."""
        with self._lock:
            self._entries = []
            self._save()

    def reload_from_disk(self) -> None:
        """Re-read entries from the JSON file (thread-safe).

        Useful for external consumers (e.g. the desktop app) that need to
        pick up changes written by another process.
        """
        with self._lock:
            self._entries = self._load()

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._entries)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> List[Dict[str, Any]]:
        try:
            if self._path.exists():
                with self._path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
        except Exception:
            pass
        return []

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("w", encoding="utf-8") as f:
                json.dump(self._entries, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            from jarvis.debug import debug_log
            debug_log(f"failed to save dictation history: {exc}", "dictation")
