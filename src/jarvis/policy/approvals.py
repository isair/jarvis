"""
Durable approval grants.

When the policy engine decides that ``approval_required=True``, the caller
obtains consent from the user.  That consent is stored as a
:class:`ScopedGrant` so that repeated requests for the same scope do not
prompt the user again during the session (or across sessions if persisted
to SQLite).

Grant scopes
------------
A grant covers a ``(tool_name, operation, path_prefix)`` triple where any
component may be ``"*"`` (wildcard).  Scopes are intentionally coarse so
that operators understand what they approved.

Storage
-------
The :class:`ApprovalStore` can operate in two modes:

* ``memory`` – grants are kept only for the lifetime of the process.
* ``sqlite`` – grants are written to the audit database and survive restarts.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..debug import debug_log


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ScopedGrant:
    """A single approval grant covering a specific (tool, operation, path) scope."""

    tool_name: str
    """e.g. ``"localFiles"`` or ``"*"`` for all tools."""

    operation: str
    """e.g. ``"write"`` or ``"*"`` for all operations."""

    path_prefix: str = "*"
    """Canonical path prefix (absolute) or ``"*"`` for all paths."""

    granted_at: float = field(default_factory=time.time)
    """UNIX timestamp when the grant was given."""

    expires_at: Optional[float] = None
    """Optional UNIX timestamp after which the grant is no longer valid."""

    granted_by: str = "user"
    """Identity of the approver — always ``"user"`` in the current model."""

    def is_valid(self) -> bool:
        """Return True when this grant has not expired."""
        if self.expires_at is not None and time.time() > self.expires_at:
            return False
        return True

    def matches(self, tool_name: str, operation: str, path: str = "") -> bool:
        """
        Return True when this grant covers the requested (tool, operation, path).

        Wildcard ``"*"`` matches any value.
        """
        if not self.is_valid():
            return False

        tool_ok = self.tool_name == "*" or self.tool_name == tool_name
        op_ok = self.operation == "*" or self.operation == operation
        if not self.path_prefix or self.path_prefix == "*":
            path_ok = True
        else:
            path_ok = path.startswith(self.path_prefix)

        return tool_ok and op_ok and path_ok


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class ApprovalStore:
    """
    Thread-safe store for :class:`ScopedGrant` objects.

    Pass ``db_path`` to enable SQLite persistence.  When omitted the store
    operates in in-memory mode and grants are lost on process exit.
    """

    _CREATE_TABLE = """
    CREATE TABLE IF NOT EXISTS approval_grants (
        id            INTEGER PRIMARY KEY,
        tool_name     TEXT NOT NULL,
        operation     TEXT NOT NULL,
        path_prefix   TEXT NOT NULL DEFAULT '*',
        granted_at    REAL NOT NULL,
        expires_at    REAL,
        granted_by    TEXT NOT NULL DEFAULT 'user'
    );
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        self._lock = threading.Lock()
        self._memory: List[ScopedGrant] = []
        self._db_path = db_path
        self._db_conn: Optional[sqlite3.Connection] = None

        if db_path:
            try:
                self._db_conn = sqlite3.connect(db_path, check_same_thread=False)
                self._db_conn.execute(self._CREATE_TABLE)
                self._db_conn.commit()
                self._load_from_db()
                debug_log(f"ApprovalStore: loaded {len(self._memory)} grants from {db_path}", "policy")
            except Exception as exc:
                debug_log(f"ApprovalStore: SQLite init failed, falling back to memory: {exc}", "policy")
                self._db_conn = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def grant(
        self,
        tool_name: str,
        operation: str,
        path_prefix: str = "*",
        expires_at: Optional[float] = None,
        granted_by: str = "user",
    ) -> ScopedGrant:
        """
        Record a new approval grant.

        Returns the created :class:`ScopedGrant`.
        """
        g = ScopedGrant(
            tool_name=tool_name,
            operation=operation,
            path_prefix=path_prefix,
            granted_at=time.time(),
            expires_at=expires_at,
            granted_by=granted_by,
        )
        with self._lock:
            self._memory.append(g)
            if self._db_conn:
                try:
                    self._db_conn.execute(
                        "INSERT INTO approval_grants "
                        "(tool_name, operation, path_prefix, granted_at, expires_at, granted_by) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (g.tool_name, g.operation, g.path_prefix,
                         g.granted_at, g.expires_at, g.granted_by),
                    )
                    self._db_conn.commit()
                except Exception as exc:
                    debug_log(f"ApprovalStore: failed to persist grant: {exc}", "policy")
        debug_log(
            f"ApprovalStore: grant recorded — tool={tool_name} op={operation} path={path_prefix}",
            "policy",
        )
        return g

    def is_granted(self, tool_name: str, operation: str, path: str = "") -> bool:
        """
        Return True when a valid un-expired grant covers this request.
        """
        with self._lock:
            return any(g.matches(tool_name, operation, path) for g in self._memory)

    def revoke_all(self) -> int:
        """Remove all grants.  Returns the number of grants removed."""
        with self._lock:
            count = len(self._memory)
            self._memory.clear()
            if self._db_conn:
                try:
                    self._db_conn.execute("DELETE FROM approval_grants")
                    self._db_conn.commit()
                except Exception:
                    pass
        debug_log(f"ApprovalStore: revoked {count} grants", "policy")
        return count

    def list_grants(self) -> List[ScopedGrant]:
        """Return a snapshot of all current (including expired) grants."""
        with self._lock:
            return list(self._memory)

    def prune_expired(self) -> int:
        """Remove expired grants from memory and the database."""
        now = time.time()
        with self._lock:
            before = len(self._memory)
            self._memory = [g for g in self._memory if g.is_valid()]
            pruned = before - len(self._memory)
            if pruned and self._db_conn:
                try:
                    self._db_conn.execute(
                        "DELETE FROM approval_grants WHERE expires_at IS NOT NULL AND expires_at < ?",
                        (now,),
                    )
                    self._db_conn.commit()
                except Exception:
                    pass
        if pruned:
            debug_log(f"ApprovalStore: pruned {pruned} expired grants", "policy")
        return pruned

    def close(self) -> None:
        """Close the SQLite connection if open."""
        if self._db_conn:
            try:
                self._db_conn.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_from_db(self) -> None:
        """Load persisted grants from the database into memory."""
        if not self._db_conn:
            return
        try:
            rows = self._db_conn.execute(
                "SELECT tool_name, operation, path_prefix, granted_at, expires_at, granted_by "
                "FROM approval_grants"
            ).fetchall()
            for row in rows:
                self._memory.append(ScopedGrant(
                    tool_name=row[0],
                    operation=row[1],
                    path_prefix=row[2],
                    granted_at=row[3],
                    expires_at=row[4],
                    granted_by=row[5],
                ))
        except Exception as exc:
            debug_log(f"ApprovalStore: failed to load grants: {exc}", "policy")
