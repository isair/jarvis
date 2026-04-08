"""
Undo Registry – session-scoped stack of reversible operations.

Tracks completed tool executions that can be reversed within a configurable
time window.  The registry supports:

- Undoing the last action ("undo that")
- Undoing the last N actions in reverse chronological order
- Undoing a specific action by its step ID
- Automatic expiry after a configurable window (default 5 minutes)

Design principles:
- In-memory only; clears on daemon restart
- Thread-safe via a simple lock
- Capped at MAX_ENTRIES (default 20) entries to bound memory use
- Tools declare their own undo inverse; the registry just stores and replays
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .debug import debug_log

# Maximum number of undo entries to keep in the stack at one time.
MAX_ENTRIES: int = 20

# Default expiry window in seconds (5 minutes).
DEFAULT_EXPIRY_SECONDS: float = 300.0


@dataclass
class UndoEntry:
    """
    A single reversible operation.

    Attributes:
        step_id:      Unique identifier matching the TaskStep that created it.
        description:  Human-readable description, e.g. "deleted shopping_list.txt".
        tool_name:    The tool that was executed (used for audit / display).
        tool_args:    Arguments originally passed to the tool.
        undo_tool:    Tool to call to reverse the operation.
        undo_args:    Arguments to pass to undo_tool.
        snapshot:     Optional captured state before execution (e.g. file
                      contents before overwrite) needed to reconstruct the
                      original state.
        created_at:   Unix timestamp when the entry was created.
        expires_at:   Unix timestamp after which the entry is considered expired.
    """
    step_id: str
    description: str
    tool_name: str
    tool_args: Dict[str, Any]
    undo_tool: str
    undo_args: Dict[str, Any]
    snapshot: Optional[Any] = None
    created_at: float = field(default_factory=time.time)
    expires_at: float = field(default_factory=lambda: time.time() + DEFAULT_EXPIRY_SECONDS)

    @property
    def is_expired(self) -> bool:
        """Return True when the undo window has closed."""
        return time.time() > self.expires_at

    @property
    def age_seconds(self) -> float:
        """Seconds elapsed since this entry was created."""
        return time.time() - self.created_at


class UndoRegistry:
    """
    Session-scoped stack of reversible operations.

    The stack is ordered oldest-first; ``pop_last(n)`` returns entries in
    reverse chronological order (most-recent first) ready for sequential
    execution.
    """

    def __init__(self, max_entries: int = MAX_ENTRIES) -> None:
        self._entries: List[UndoEntry] = []
        self._max_entries = max_entries
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def push(self, entry: UndoEntry) -> None:
        """
        Add a new reversible entry to the stack.

        Expired entries and entries beyond the cap are pruned before
        the new entry is added.
        """
        with self._lock:
            # Purge expired entries first
            self._entries = [e for e in self._entries if not e.is_expired]
            self._entries.append(entry)
            # Cap the stack
            if len(self._entries) > self._max_entries:
                dropped = len(self._entries) - self._max_entries
                self._entries = self._entries[dropped:]
                debug_log(
                    f"undo stack capped — dropped {dropped} oldest entries",
                    "undo",
                )
            debug_log(
                f"undo entry pushed: {entry.description!r} "
                f"(step_id={entry.step_id}, expires_in="
                f"{entry.expires_at - time.time():.0f}s)",
                "undo",
            )

    def pop_last(self, n: int = 1) -> List[UndoEntry]:
        """
        Remove and return the last ``n`` non-expired entries.

        The returned list is ordered most-recent first so callers can
        execute reversals in reverse chronological order by iterating
        the list in order.

        Expired entries encountered during the scan are silently dropped
        without counting towards ``n``.
        """
        with self._lock:
            result: List[UndoEntry] = []
            surviving: List[UndoEntry] = []
            # Walk from newest to oldest
            for entry in reversed(self._entries):
                if entry.is_expired:
                    debug_log(
                        f"undo entry expired and dropped: {entry.description!r}",
                        "undo",
                    )
                    continue
                if len(result) < n:
                    result.append(entry)
                else:
                    surviving.insert(0, entry)

            # Rebuild stack: surviving entries in original order
            # (oldest first, without the ones we just popped)
            popped_ids = {e.step_id for e in result}
            self._entries = [e for e in self._entries if e.step_id not in popped_ids and not e.is_expired]

            for entry in result:
                debug_log(f"undo entry popped: {entry.description!r}", "undo")
            return result  # most-recent first

    def pop_by_id(self, step_id: str) -> Optional[UndoEntry]:
        """
        Remove and return the entry with the given step_id.

        Returns ``None`` if no matching entry exists or if the entry has
        expired.
        """
        with self._lock:
            for i, entry in enumerate(self._entries):
                if entry.step_id == step_id:
                    if entry.is_expired:
                        debug_log(
                            f"undo entry {step_id!r} found but expired", "undo"
                        )
                        self._entries.pop(i)
                        return None
                    self._entries.pop(i)
                    debug_log(
                        f"undo entry popped by id: {entry.description!r}", "undo"
                    )
                    return entry
            debug_log(f"undo entry not found: {step_id!r}", "undo")
            return None

    def clear(self) -> None:
        """Remove all entries from the stack."""
        with self._lock:
            count = len(self._entries)
            self._entries.clear()
            debug_log(f"undo stack cleared ({count} entries removed)", "undo")

    # ------------------------------------------------------------------
    # Read-only
    # ------------------------------------------------------------------

    def peek_all(self) -> List[UndoEntry]:
        """
        Return a snapshot of all non-expired entries, oldest first.

        Does not remove any entries.
        """
        with self._lock:
            return [e for e in self._entries if not e.is_expired]

    @property
    def size(self) -> int:
        """Number of non-expired entries currently in the stack."""
        with self._lock:
            return sum(1 for e in self._entries if not e.is_expired)

    def __len__(self) -> int:
        return self.size


# ---------------------------------------------------------------------------
# Module-level singleton – one registry per daemon session
# ---------------------------------------------------------------------------

_registry = UndoRegistry()


def get_undo_registry() -> UndoRegistry:
    """Return the singleton undo registry."""
    return _registry


def push_undo(entry: UndoEntry) -> None:
    """Convenience wrapper: push an entry onto the singleton registry."""
    _registry.push(entry)


def pop_last_undo(n: int = 1) -> List[UndoEntry]:
    """Convenience wrapper: pop and return the last ``n`` entries."""
    return _registry.pop_last(n)


def pop_undo_by_id(step_id: str) -> Optional[UndoEntry]:
    """Convenience wrapper: pop a specific entry by step_id."""
    return _registry.pop_by_id(step_id)
