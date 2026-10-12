"""
🧠 Knowledge Graph

A self-organising node graph that stores the assistant's accumulated world
knowledge — anything learned during conversations that it wouldn't already know.
Three fast-access entry points (recent nodes, top nodes, root node) ensure the
most relevant knowledge is always reachable without exhaustive search.

See graph.spec.md for the full specification.
"""

from __future__ import annotations

import re
import sqlite3
import threading
import unicodedata
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Optional, Sequence, TYPE_CHECKING

from ..debug import debug_log

if TYPE_CHECKING:
    from .imports import MemoryImportRecord, ImportResult


# ── Mutation listeners ─────────────────────────────────────────────────────
#
# Lightweight observer registry so consumers (e.g. DialogueMemory's warm
# profile cache) can invalidate derived state when a node is created,
# updated, or deleted. The listener receives the action name, node id, and
# the FIXED_BRANCH ancestor (e.g. ``"user"``, ``"directives"``, ``"world"``)
# so it can scope its reaction. Failures in listeners are logged and
# swallowed so they cannot break a write.

_MUTATION_LISTENERS: "list[Callable[..., None]]" = []


def register_graph_mutation_listener(cb: Callable[..., None]) -> None:
    """Register a callback invoked after every node mutation.

    The callback is invoked with keyword arguments ``action``, ``node_id``,
    and ``branch`` where ``branch`` is the id of the FIXED_BRANCH ancestor
    (or the node id itself when the node is a fixed branch), or ``None``
    when the branch cannot be resolved (e.g. root mutations).
    """
    if cb not in _MUTATION_LISTENERS:
        _MUTATION_LISTENERS.append(cb)


def unregister_graph_mutation_listener(cb: Callable[..., None]) -> None:
    """Remove a previously registered mutation listener (idempotent)."""
    try:
        _MUTATION_LISTENERS.remove(cb)
    except ValueError:
        pass


def _notify_graph_mutation(action: str, node_id: str, branch: Optional[str]) -> None:
    for cb in list(_MUTATION_LISTENERS):
        try:
            cb(action=action, node_id=node_id, branch=branch)
        except Exception as exc:
            debug_log(f"graph mutation listener failed (non-fatal): {exc}", "memory")


# ── Fact normalisation ─────────────────────────────────────────────────────
#
# Used for dedupe comparisons. Locale-safe — the user base includes
# non-Latin scripts (e.g. Turkish, where ``"İ".lower()`` returns ``"i"``
# but Turkish lowercase is ``"ı"``), so we use ``unicodedata.NFKC`` plus
# ``str.casefold`` rather than ``str.lower``. ``casefold`` also folds
# German ß to ss, and NFKC collapses visually identical code points.

_WS_RE = re.compile(r"\s+")


def normalise_fact(text: str) -> str:
    """Lowercase (Unicode-aware) + collapse all whitespace, including
    newlines, into single spaces for fuzzy equality. ``_WS_RE`` matches
    ``\\s+``, so any newline embedded in an extracted fact collapses to
    a space on the candidate side, keeping the dedupe key well-formed
    even if the extractor accidentally emits a multi-line statement."""
    folded = unicodedata.normalize("NFKC", text).casefold()
    return _WS_RE.sub(" ", folded.strip())


# ── Configuration defaults ──────────────────────────────────────────────────

SPLIT_THRESHOLD = 1500       # tokens — when to split a node into children
MERGE_THRESHOLD = 200        # tokens — when to collapse sparse children back
RECENT_NODES_COUNT = 10      # number of recently-accessed nodes to track
TOP_NODES_COUNT = 15         # most-accessed nodes to surface
MAX_TRAVERSAL_DEPTH = 8      # safety limit on graph traversal
SUMMARY_MAX_LENGTH = 300     # max characters for a node description
DECAY_HALF_LIFE_DAYS = 14    # days until a node's access score halves


# ── Fixed top-level branches ────────────────────────────────────────────────
#
# The root is seeded with three fixed children on first run. The graph
# is still self-organising below these — auto-split/merge runs within
# each branch — but the top level is purpose-shaped, not content-shaped,
# so the extractor can route each new fact into the right semantic slot.
#
# - USER: everything about the person the assistant serves (identity,
#   tastes, preferences, plans, opinions). Warm-loaded into the system
#   prompt on every turn.
# - DIRECTIVES: imperatives the user issued at the assistant about its
#   own behaviour ("be concise", "use British English", "stop apologising").
#   Verbatim rules, never summarised. Warm-loaded on every turn.
# - WORLD: external facts with attribution (current graph content —
#   films, businesses, recipes, techniques). Unbounded. Not warm-loaded;
#   retrieved on demand via searchMemory.
#
# The IDs are stable strings so re-opening an existing graph is
# idempotent — no duplicate branches get seeded if the store already
# has them.

BRANCH_USER = "user"
BRANCH_DIRECTIVES = "directives"
BRANCH_WORLD = "world"

FIXED_BRANCHES: tuple[tuple[str, str, str], ...] = (
    (
        BRANCH_USER,
        "User",
        "Everything about the user: identity, location, relationships, "
        "tastes, preferences, history, plans, opinions. Always injected "
        "into the system prompt.",
    ),
    (
        BRANCH_DIRECTIVES,
        "Directives",
        "Imperatives the user issued at the assistant about its own "
        "behaviour — tone, verbosity, language, style rules. Verbatim, "
        "never summarised. Always injected into the system prompt.",
    ),
    (
        BRANCH_WORLD,
        "World",
        "External facts the assistant has learned and wants to carry "
        "forward: films, businesses, recipes, techniques, events. "
        "Retrieved on demand via searchMemory.",
    ),
)

FIXED_BRANCH_IDS: frozenset[str] = frozenset(bid for bid, _, _ in FIXED_BRANCHES)


# ── SQL helpers ────────────────────────────────────────────────────────────

def _decay_score_sql(half_life_days: int = DECAY_HALF_LIFE_DAYS) -> str:
    """Return a SQL expression that computes a time-decayed access score.

    Uses hyperbolic decay: access_count / (1 + age_days / half_life).
    A node accessed 100 times 14 days ago scores the same as one
    accessed 50 times today (with default half-life of 14 days).

    The raw access_count is never modified — decay is computed at query time
    so no data is lost and the half-life can be changed freely.
    """
    return (
        f"(access_count * 1.0 / "
        f"(1.0 + MAX(0, julianday('now') - julianday(last_accessed)) / {half_life_days}.0))"
    )


# ── Data model ──────────────────────────────────────────────────────────────

@dataclass
class MemoryNode:
    """A single node in the memory graph."""
    id: str
    name: str
    description: str
    data: str = ""
    parent_id: Optional[str] = None
    access_count: int = 0
    last_accessed: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    data_token_count: int = 0

    def to_dict(self) -> dict:
        """Serialise to a dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "data": self.data,
            "parent_id": self.parent_id,
            "access_count": self.access_count,
            "last_accessed": self.last_accessed,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "data_token_count": self.data_token_count,
        }


def _estimate_tokens(text: str) -> int:
    """Rough token estimate — ~4 chars per token for English text."""
    if not text:
        return 0
    return max(1, len(text) // 4)


# ── Schema ──────────────────────────────────────────────────────────────────

_GRAPH_SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS memory_nodes (
    id               TEXT PRIMARY KEY,
    name             TEXT NOT NULL,
    description      TEXT NOT NULL,
    data             TEXT NOT NULL DEFAULT '',
    parent_id        TEXT REFERENCES memory_nodes(id) ON DELETE SET NULL,
    access_count     INTEGER NOT NULL DEFAULT 0,
    last_accessed    TEXT NOT NULL,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    data_token_count INTEGER NOT NULL DEFAULT 0,
    CHECK(parent_id IS NULL OR parent_id != id)
);

CREATE INDEX IF NOT EXISTS idx_nodes_parent ON memory_nodes(parent_id);
CREATE INDEX IF NOT EXISTS idx_nodes_last_accessed ON memory_nodes(last_accessed DESC);
CREATE INDEX IF NOT EXISTS idx_nodes_access_count ON memory_nodes(access_count DESC);

CREATE TABLE IF NOT EXISTS memory_imports (
    id TEXT PRIMARY KEY,
    branch TEXT NOT NULL,
    content_key TEXT NOT NULL,
    node_id TEXT REFERENCES memory_nodes(id) ON DELETE SET NULL,
    provider TEXT NOT NULL,
    source_id TEXT NOT NULL,
    source_digest TEXT NOT NULL,
    source_label TEXT NOT NULL,
    source_scope TEXT NOT NULL,
    source_date TEXT,
    source_metadata TEXT NOT NULL,
    original_text TEXT NOT NULL,
    reviewed_text TEXT NOT NULL,
    imported_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_imports_content ON memory_imports(branch, content_key);
CREATE INDEX IF NOT EXISTS idx_imports_node ON memory_imports(node_id);

CREATE TABLE IF NOT EXISTS memory_import_node_sources (
    import_id TEXT NOT NULL REFERENCES memory_imports(id) ON DELETE CASCADE,
    node_id TEXT NOT NULL REFERENCES memory_nodes(id) ON DELETE CASCADE,
    current_binding INTEGER NOT NULL CHECK(current_binding IN (0, 1)),
    PRIMARY KEY(import_id, node_id)
);
CREATE INDEX IF NOT EXISTS idx_import_sources_node ON memory_import_node_sources(node_id);
CREATE TRIGGER IF NOT EXISTS import_source_data_write
AFTER UPDATE OF data ON memory_nodes BEGIN
    UPDATE memory_import_node_sources SET current_binding = 0 WHERE node_id = NEW.id;
END;

CREATE TABLE IF NOT EXISTS memory_graph_revisions (
    name TEXT PRIMARY KEY,
    revision INTEGER NOT NULL DEFAULT 0
);
"""


_WARM_PROFILE_TRIGGERS_SQL = "\n".join(
    f"""CREATE TRIGGER IF NOT EXISTS warm_profile_{name}
    {event} ON memory_nodes
    WHEN EXISTS (
        WITH RECURSIVE ancestry(id, parent_id) AS (
            SELECT id, parent_id FROM memory_nodes WHERE id = {row}.id
            UNION
            SELECT parent.id, parent.parent_id FROM memory_nodes AS parent
            JOIN ancestry ON parent.id = ancestry.parent_id
        )
        SELECT 1 FROM ancestry WHERE id IN ('{BRANCH_USER}', '{BRANCH_DIRECTIVES}')
    )
    BEGIN
        UPDATE memory_graph_revisions SET revision = revision + 1 WHERE name = 'warm_profile';
    END;"""
    for name, event, row in (
        ('insert_old', 'BEFORE INSERT', 'NEW'),
        ('insert', 'AFTER INSERT', 'NEW'),
        ('delete', 'BEFORE DELETE', 'OLD'),
        ('update_old', 'BEFORE UPDATE OF id, name, description, data, parent_id', 'OLD'),
        ('update_new', 'AFTER UPDATE OF id, name, description, data, parent_id', 'NEW'),
    )
)


# ── Graph Memory Store ──────────────────────────────────────────────────────

class GraphMemoryStore:
    """
    Self-organising node graph for persistent memory.

    Backed by SQLite with thread-safe access. Provides three entry points
    for fast retrieval: recent nodes, top nodes, and the root node.
    """

    def __init__(self, db_path: str, *, read_only: bool = False) -> None:
        from pathlib import Path
        path = Path(db_path)
        if not read_only:
            path.parent.mkdir(parents=True, exist_ok=True)

        self.db_path = db_path
        target = path.resolve().as_uri() + '?mode=ro' if read_only else db_path
        self.conn = sqlite3.connect(target, uri=read_only, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        if not read_only:
            try:
                self._init_schema()
                self._ensure_root()
            except BaseException:
                self.conn.close()
                debug_log("graph bootstrap failed: connection released", "memory")
                raise

    # ── Schema & bootstrap ──────────────────────────────────────────────

    def _init_schema(self) -> None:
        with self._lock:
            self.conn.execute("PRAGMA foreign_keys = ON")
            self.conn.executescript(_GRAPH_SCHEMA_SQL + _WARM_PROFILE_TRIGGERS_SQL)
            if self.conn.execute(
                "SELECT 1 FROM memory_graph_revisions WHERE name = 'warm_profile'"
            ).fetchone() is None:
                self.conn.execute(
                    "INSERT OR IGNORE INTO memory_graph_revisions(name, revision) VALUES('warm_profile', 0)"
                )
            self.conn.commit()

    def _ensure_root(self) -> None:
        """Create the root node and the three fixed top-level branches
        (User / Directives / World) if they don't exist.

        Idempotent: each branch has a stable string id, so re-opening an
        existing graph never duplicates them. Branches are also created
        on first boot for existing graphs that predate the taxonomy —
        this is the migration path.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            row = self.conn.execute(
                "SELECT id FROM memory_nodes WHERE parent_id IS NULL LIMIT 1"
            ).fetchone()
            if row is None:
                self.conn.execute(
                    """INSERT INTO memory_nodes
                       (id, name, description, data, parent_id,
                        access_count, last_accessed, created_at, updated_at,
                        data_token_count)
                       VALUES (?, ?, ?, ?, NULL, 0, ?, ?, ?, 0)""",
                    ("root", "Root", "Top-level memory node — contains all knowledge domains.", "", now, now, now),
                )
                self.conn.commit()
                debug_log("Created root memory node", "memory")

            branch_ids = tuple(branch_id for branch_id, _, _ in FIXED_BRANCHES)
            placeholders = ','.join('?' for _ in branch_ids)
            existing_ids = {row['id'] for row in self.conn.execute(
                f"SELECT id FROM memory_nodes WHERE id IN ({placeholders})",
                branch_ids,
            )}
            # Existing branches require no writer lock. The insertion guard
            # also prevents duplicate writes when another owner seeds first.
            for branch_id, name, description in FIXED_BRANCHES:
                if branch_id in existing_ids:
                    continue
                self.conn.execute(
                    """INSERT OR IGNORE INTO memory_nodes
                       (id, name, description, data, parent_id,
                        access_count, last_accessed, created_at, updated_at,
                        data_token_count)
                       SELECT ?, ?, ?, '', 'root', 0, ?, ?, ?, 0
                       WHERE NOT EXISTS (SELECT 1 FROM memory_nodes WHERE id = ?)""",
                    (branch_id, name, description, now, now, now, branch_id),
                )
            self.conn.commit()

    def has_unclassified_root_content(self) -> bool:
        """Report root-level content without changing saved memory.

        Root facts and children outside the fixed branches can be manually
        created or imported. Their shape does not establish their age or
        classification, so they remain available for explicit organisation.
        """
        with self._lock:
            root_row = self.conn.execute(
                "SELECT data FROM memory_nodes WHERE id = 'root'"
            ).fetchone()
            root_has_data = bool(root_row and (root_row["data"] or "").strip())
            root_child = self.conn.execute(
                "SELECT id FROM memory_nodes "
                "WHERE parent_id = 'root' AND id NOT IN ({}) LIMIT 1".format(
                    ",".join("?" * len(FIXED_BRANCH_IDS))
                ),
                tuple(FIXED_BRANCH_IDS),
            ).fetchone()
            return root_has_data or root_child is not None

    # ── CRUD ────────────────────────────────────────────────────────────

    def get_warm_profile_revision(self) -> Optional[int]:
        """Read the profile revision, or None when its metadata is absent."""
        with self._lock:
            try:
                row = self.conn.execute(
                    "SELECT revision FROM memory_graph_revisions WHERE name = 'warm_profile'"
                ).fetchone()
            except sqlite3.OperationalError:
                table = self.conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' "
                    "AND name = 'memory_graph_revisions' COLLATE NOCASE"
                ).fetchone()
                if table is not None:
                    raise
                return None
        return row["revision"] if row is not None else None

    def get_node(self, node_id: str) -> Optional[MemoryNode]:
        """Fetch a single node by ID."""
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM memory_nodes WHERE id = ?", (node_id,)
            ).fetchone()
            if row is None:
                return None
            return self._row_to_node(row)

    def get_children(self, node_id: str) -> list[MemoryNode]:
        """Get all direct children of a node, ordered by decayed access score."""
        score = _decay_score_sql()
        with self._lock:
            rows = self.conn.execute(
                f"SELECT * FROM memory_nodes WHERE parent_id = ? ORDER BY {score} DESC",
                (node_id,),
            ).fetchall()
            return [self._row_to_node(r) for r in rows]

    def get_root(self) -> MemoryNode:
        """Return the root node."""
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM memory_nodes WHERE parent_id IS NULL LIMIT 1"
            ).fetchone()
            return self._row_to_node(row)

    def _resolve_branch(self, node_id: Optional[str]) -> Optional[str]:
        """Walk parents from ``node_id`` up to find the FIXED_BRANCH id it
        belongs to (or itself, if the node IS a fixed branch). Returns
        ``None`` for the root or when the node cannot be located.

        Capped at ``MAX_TRAVERSAL_DEPTH`` so a corrupt parent cycle cannot
        spin the loop. SQLite reads only — safe to call from write paths.
        """
        if not node_id or node_id == "root":
            return None
        if node_id in FIXED_BRANCH_IDS:
            return node_id
        current = node_id
        for _ in range(MAX_TRAVERSAL_DEPTH):
            row = self.conn.execute(
                "SELECT parent_id FROM memory_nodes WHERE id = ?", (current,)
            ).fetchone()
            if row is None:
                return None
            parent = row["parent_id"]
            if parent is None or parent == "root":
                return None
            if parent in FIXED_BRANCH_IDS:
                return parent
            current = parent
        return None

    def create_node(
        self,
        name: str,
        description: str,
        data: str = "",
        parent_id: Optional[str] = None,
    ) -> MemoryNode:
        """Create a new node and return it.

        Raises ValueError if parent_id references a non-existent node.
        """
        if parent_id is not None:
            parent = self.get_node(parent_id)
            if parent is None:
                raise ValueError(f"Parent node '{parent_id}' does not exist")

        with self._lock, self.conn:
            node = self._insert_node(name, description, data, parent_id)
            branch = self._resolve_branch(parent_id)

        debug_log(f"Created memory node '{name}' ({node.id[:8]})", "memory")
        _notify_graph_mutation("create", node.id, branch)
        return node

    def _insert_node(
        self, name: str, description: str, data: str, parent_id: Optional[str],
    ) -> MemoryNode:
        """Insert a new node inside the caller's locked write transaction."""
        description = description[:SUMMARY_MAX_LENGTH]
        node_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        token_count = _estimate_tokens(data)
        self.conn.execute(
            """INSERT INTO memory_nodes
               (id, name, description, data, parent_id,
                access_count, last_accessed, created_at, updated_at,
                data_token_count)
               VALUES (?, ?, ?, ?, ?, 0, ?, ?, ?, ?)""",
            (node_id, name, description, data, parent_id, now, now, now, token_count),
        )
        return MemoryNode(
            id=node_id, name=name, description=description, data=data,
            parent_id=parent_id, access_count=0, last_accessed=now,
            created_at=now, updated_at=now, data_token_count=token_count,
        )

    def import_memories(self, records: Sequence[MemoryImportRecord]) -> ImportResult:
        """Commit reviewed nodes and durable source records together."""
        from .imports import ImportResult, _digest
        if not records or any(record.branch not in FIXED_BRANCH_IDS or
                              not isinstance(record.reviewed_text, str) or
                              not record.reviewed_text.strip() for record in records):
            raise ValueError('Invalid reviewed memory batch')
        created, node_ids = [], []
        imported_at = datetime.now().astimezone().isoformat()
        with self._lock, self.conn:
            self.conn.execute('BEGIN IMMEDIATE')
            known = {}
            for branch in {record.branch for record in records}:
                rows = self.conn.execute('''WITH RECURSIVE subtree(id) AS (
                    SELECT id FROM memory_nodes WHERE id = ?
                    UNION SELECT child.id FROM memory_nodes child
                    JOIN subtree ON child.parent_id = subtree.id
                ) SELECT id, data FROM memory_nodes WHERE id IN (SELECT id FROM subtree)
                  AND NOT EXISTS (SELECT 1 FROM memory_import_node_sources source
                                  WHERE source.node_id = memory_nodes.id)''', (branch,))
                for row in rows:
                    for text in (row['data'], *row['data'].splitlines()):
                        if key := normalise_fact(text):
                            known.setdefault((branch, key), row['id'])
            for record in records:
                key = normalise_fact(record.reviewed_text)
                identity = _digest('\0'.join((record.source_id, record.source_digest,
                                   record.item_id, record.branch, key, record.source_date or '')))
                previous = self.conn.execute('SELECT node_id FROM memory_imports WHERE id = ?', (identity,)).fetchone()
                if previous is not None:
                    node_ids.append(previous['node_id'])
                    continue
                shared = self.conn.execute('SELECT node_id FROM memory_imports WHERE branch = ? AND content_key = ? AND source_date IS ? ORDER BY imported_at LIMIT 1',
                                           (record.branch, key, record.source_date)).fetchone()
                if shared is not None:
                    node_id = shared['node_id']
                elif record.source_date is None and (record.branch, key) in known:
                    node_id = known[(record.branch, key)]
                else:
                    node = self._insert_node(f'{record.provider} memory {record.item_id[:8]}',
                           f'Reviewed import from {record.provider}. Scope: {record.source_scope}.',
                           record.reviewed_text, record.branch)
                    created.append((node, record.branch))
                    node_id = node.id
                    if record.source_date is None:
                        known[(record.branch, key)] = node_id
                self.conn.execute('''INSERT INTO memory_imports
                    (id, branch, content_key, node_id, provider, source_id, source_digest,
                     source_label, source_scope, source_date, source_metadata, original_text, reviewed_text, imported_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                    (identity, record.branch, key, node_id, record.provider, record.source_id,
                     record.source_digest, record.source_label, record.source_scope,
                     record.source_date, record.source_metadata, record.original_text, record.reviewed_text, imported_at))
                if node_id is not None:
                    self.conn.execute('INSERT INTO memory_import_node_sources VALUES (?, ?, 1)', (identity, node_id))
                    if self._resolve_branch(node_id) in (BRANCH_USER, BRANCH_DIRECTIVES):
                        self.conn.execute("UPDATE memory_graph_revisions SET revision = revision + 1 WHERE name = 'warm_profile'")
                node_ids.append(node_id)
        for node, branch in created:
            _notify_graph_mutation('create', node.id, branch)
        return ImportResult(len(created), len(records) - len(created), tuple(node_ids))

    def get_import_provenance(self, node_id: Optional[str] = None, *, limit: int = 100) -> list[dict]:
        """Read import sources, including ancestor sources after a split."""
        import json
        limit = max(1, min(limit, 500))
        with self._lock:
            if node_id is None:
                rows = self.conn.execute('SELECT * FROM memory_imports ORDER BY imported_at DESC, id LIMIT ?', (limit,))
            else:
                rows = self.conn.execute('''SELECT source.*, link.current_binding
                    FROM memory_imports source JOIN memory_import_node_sources link
                    ON source.id = link.import_id WHERE link.node_id = ?
                    ORDER BY source.imported_at DESC, source.id LIMIT ?''', (node_id, limit))
            records = [dict(row) for row in rows]
            for record in records:
                record['source_metadata'] = json.loads(record['source_metadata'])
            return records

    def get_context_data(self, node: MemoryNode, *, max_chars: int, suffix: str = '') -> str:
        """Render bounded recall data with its original source dates."""
        from .imports import render_memory_context
        try:
            records = self.get_import_provenance(node.id, limit=3)
        except sqlite3.OperationalError:
            # Persisted graphs can be read before their writable schema bootstrap.
            if self.conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'memory_imports'").fetchone():
                raise
            records = []
        if records:
            return render_memory_context(node.data, self._resolve_branch(node.id) or BRANCH_WORLD,
                                         records, max_chars=max_chars)
        text = node.data.strip()
        return text if len(text) <= max_chars else text[:max(0, max_chars-len(suffix))].rstrip()+suffix

    def split_node(
        self, source: MemoryNode, *, children: list[tuple[str, str, str]], summary: str,
    ) -> bool:
        """Commit children and the parent summary only from current source input."""
        with self._lock, self.conn:
            self.conn.execute("BEGIN IMMEDIATE")
            current = self.get_node(source.id)
            if current is None or (current.name, current.description, current.data) != (
                source.name, source.description, source.data,
            ):
                debug_log(f"graph split: rejected stale input for {source.id}", "memory")
                return False
            created = [self._insert_node(name, description, data, source.id)
                       for name, description, data in children]
            for child in created:
                self.conn.execute('''INSERT INTO memory_import_node_sources
                    SELECT import_id, ?, current_binding FROM memory_import_node_sources
                    WHERE node_id = ?''', (child.id, source.id))
            current.data = ""
            current.data_token_count = 0
            current.description = summary[:SUMMARY_MAX_LENGTH]
            current.updated_at = datetime.now(timezone.utc).isoformat()
            self._persist_node(current, write_data=True)
            branch = self._resolve_branch(source.id)

        for child in created:
            _notify_graph_mutation("create", child.id, branch)
        _notify_graph_mutation("update", source.id, branch)
        return True

    def update_node(
        self,
        node_id: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        data: Optional[str] = None,
        expected_data: Optional[str] = None,
        parent_id: Optional[str] = ...,  # type: ignore[assignment]
    ) -> Optional[MemoryNode]:
        """Return the committed node, or None if absent, stale or invalid.

        expected_data conditionally accepts a write only while its source
        data matches the current row inside the write transaction.
        """
        with self._lock, self.conn:
            self.conn.execute("BEGIN IMMEDIATE")
            node = self.get_node(node_id)
            if node is None:
                return None
            if expected_data is not None and node.data != expected_data:
                debug_log(f"graph update: rejected stale data for {node_id}", "memory")
                return None
            if parent_id is not ... and parent_id != node.parent_id:
                if not self._valid_parent(node_id, parent_id):
                    debug_log(f"graph update: rejected invalid parent for {node_id}", "memory")
                    return None

            if name is not None:
                node.name = name
            if description is not None:
                node.description = description[:SUMMARY_MAX_LENGTH]
            if data is not None:
                node.data = data
                node.data_token_count = _estimate_tokens(data)
            if parent_id is not ...:
                node.parent_id = parent_id
            node.updated_at = datetime.now(timezone.utc).isoformat()
            self._persist_node(node, write_data=data is not None)
            branch = self._resolve_branch(node_id)

        _notify_graph_mutation("update", node_id, branch)
        return node

    def _valid_parent(self, node_id: str, parent_id: Optional[str]) -> bool:
        """Check parent ancestry inside the caller's locked write transaction."""
        if node_id == "root" or node_id in FIXED_BRANCH_IDS:
            return False
        visited = {node_id}
        current_id = parent_id
        while current_id is not None:
            if current_id in visited:
                return False
            visited.add(current_id)
            current = self.get_node(current_id)
            if current is None:
                return False
            current_id = current.parent_id
        return True

    def _persist_node(self, node: MemoryNode, *, write_data: bool) -> None:
        """Write a node snapshot inside the caller's locked write transaction."""
        fields = ['name', 'description', 'parent_id', 'updated_at']
        if write_data:
            fields.extend(('data', 'data_token_count'))
        assignments = ', '.join(f'{field} = ?' for field in fields)
        self.conn.execute(f'UPDATE memory_nodes SET {assignments} WHERE id = ?',
                          (*(getattr(node, field) for field in fields), node.id))

    def delete_node(self, node_id: str) -> bool:
        """Delete a node. Children are orphaned (parent_id set to NULL by FK).

        Root and the seeded fixed branches (see ``FIXED_BRANCHES``) are
        non-deletable — the warm profile and extractor routing rely on
        their stable presence (graph.spec.md §"Fixed Top-Level Branches").
        """
        if node_id == "root" or node_id in FIXED_BRANCH_IDS:
            return False
        # Resolve branch BEFORE the delete so listeners get a meaningful
        # branch attribution even though the row is about to vanish.
        branch = self._resolve_branch(node_id)
        with self._lock, self.conn:
            cur = self.conn.execute(
                "DELETE FROM memory_nodes WHERE id = ?", (node_id,)
            )
            deleted = cur.rowcount > 0
        if deleted:
            _notify_graph_mutation("delete", node_id, branch)
        return deleted

    def get_branch_fact_keys(self, branch_id: str) -> set[str]:
        """Snapshot folded fact lines from a fixed branch and all descendants."""
        if branch_id not in FIXED_BRANCH_IDS:
            return set()
        with self._lock:
            rows = self.conn.execute(
                """WITH RECURSIVE branch_nodes(id) AS (
                    SELECT id FROM memory_nodes WHERE id = ?
                    UNION
                    SELECT child.id FROM memory_nodes AS child
                    JOIN branch_nodes AS parent ON child.parent_id = parent.id
                )
                SELECT data FROM memory_nodes
                WHERE id IN (SELECT id FROM branch_nodes)""",
                (branch_id,),
            ).fetchall()
        return {key for row in rows for line in row["data"].splitlines()
                if (key := normalise_fact(line))}

    def append_to_node(self, node_id: str, text: str) -> Optional[MemoryNode]:
        """Append text and return the committed node, or None if absent."""
        with self._lock, self.conn:
            self.conn.execute("BEGIN IMMEDIATE")
            node = self.get_node(node_id)
            if node is None:
                return None

            separator = "\n" if node.data else ""
            node.data += separator + text
            node.data_token_count = _estimate_tokens(node.data)
            node.updated_at = datetime.now(timezone.utc).isoformat()
            self._persist_node(node, write_data=True)
            branch = self._resolve_branch(node_id)

        _notify_graph_mutation("update", node_id, branch)
        return node

    def touch_node(self, node_id: str) -> None:
        """Track access without waiting for a writer or interrupting retrieval."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            if self.conn.in_transaction:
                debug_log("Graph access tracking skipped: caller transaction active", "memory")
                return
            wait_ms = self.conn.execute("PRAGMA busy_timeout").fetchone()[0]
            try:
                self.conn.execute("PRAGMA busy_timeout = 0")
                with self.conn:
                    self.conn.execute(
                        """UPDATE memory_nodes
                           SET access_count = access_count + 1, last_accessed = ?
                           WHERE id = ?""",
                        (now, node_id),
                    )
            except sqlite3.DatabaseError as exc:
                debug_log(f"Graph access tracking skipped: {exc}", "memory")
            finally:
                self.conn.execute(f"PRAGMA busy_timeout = {int(wait_ms)}")

    # ── Entry points ────────────────────────────────────────────────────

    def get_recent_nodes(self, limit: int = RECENT_NODES_COUNT) -> list[MemoryNode]:
        """Get the most recently accessed nodes."""
        with self._lock:
            rows = self.conn.execute(
                """SELECT * FROM memory_nodes
                   WHERE id != 'root'
                   ORDER BY last_accessed DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()
            return [self._row_to_node(r) for r in rows]

    def get_top_nodes(self, limit: int = TOP_NODES_COUNT) -> list[MemoryNode]:
        """Get nodes with the highest time-decayed access score.

        Uses hyperbolic decay so frequently accessed nodes that haven't
        been touched in a while naturally fall off without needing a hard
        window cutoff.
        """
        score = _decay_score_sql()
        with self._lock:
            rows = self.conn.execute(
                f"""SELECT * FROM memory_nodes
                   WHERE id != 'root'
                   ORDER BY {score} DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()
            return [self._row_to_node(r) for r in rows]

    # ── Tree queries ────────────────────────────────────────────────────

    def get_subtree(self, node_id: str, max_depth: int = 3) -> dict:
        """
        Return a nested dict representing the subtree rooted at node_id.

        Each dict has keys: node (MemoryNode.to_dict()) and children (list of subtrees).
        Useful for the tree sidebar in the UI.
        """
        node = self.get_node(node_id)
        if node is None:
            return {}

        def _build(nid: str, depth: int) -> dict:
            n = self.get_node(nid)
            if n is None:
                return {}
            children = []
            if depth < max_depth:
                for child in self.get_children(nid):
                    children.append(_build(child.id, depth + 1))
            return {"node": n.to_dict(), "children": children}

        return _build(node_id, 0)

    def get_ancestors(self, node_id: str) -> list[MemoryNode]:
        """Return the path from root to this node (inclusive), root first."""
        ancestors: list[MemoryNode] = []
        visited: set[str] = set()
        current = self.get_node(node_id)
        while current is not None:
            if current.id in visited or len(ancestors) > MAX_TRAVERSAL_DEPTH:
                debug_log(f"Cycle or depth limit hit in get_ancestors for {node_id}", "memory")
                break
            visited.add(current.id)
            ancestors.append(current)
            if current.parent_id is None:
                break
            current = self.get_node(current.parent_id)
        ancestors.reverse()
        return ancestors

    def get_all_nodes(self) -> list[MemoryNode]:
        """Return all nodes — use with care on large graphs."""
        score = _decay_score_sql()
        with self._lock:
            rows = self.conn.execute(
                f"SELECT * FROM memory_nodes ORDER BY {score} DESC"
            ).fetchall()
            return [self._row_to_node(r) for r in rows]

    def get_node_count(self) -> int:
        """Return total number of nodes in the graph."""
        with self._lock:
            row = self.conn.execute("SELECT COUNT(*) as cnt FROM memory_nodes").fetchone()
            return row["cnt"]

    def get_total_tokens(self) -> int:
        """Return total data tokens across all nodes. Zero means no knowledge stored."""
        with self._lock:
            row = self.conn.execute(
                "SELECT COALESCE(SUM(data_token_count), 0) as total FROM memory_nodes"
            ).fetchone()
            return int(row["total"])

    # ── Search ─────────────────────────────────────────────────────────

    def search_nodes(self, query: str, limit: int = 10) -> list[MemoryNode]:
        """Search nodes by keyword match across name, description, and data.

        Uses case-insensitive LIKE matching on each keyword (split by whitespace).
        Scoring weights: name/description matches are worth 3× data matches, so
        specific nodes about a topic rank above broad category nodes that merely
        contain the keyword somewhere in their data blob.
        Excludes the root node from results and touches matched nodes.
        """
        keywords = [k.strip() for k in query.split() if k.strip()]
        if not keywords:
            return []

        # Build a score expression: name/description matches worth 3, data worth 1
        score_parts: list[str] = []
        params: list[str] = []
        for kw in keywords:
            # Escape LIKE wildcards so literal %, _, \ are matched exactly
            escaped = kw.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            pattern = f"%{escaped}%"
            score_parts.append(
                "(CASE WHEN name LIKE ? ESCAPE '\\' THEN 3 ELSE 0 END"
                " + CASE WHEN description LIKE ? ESCAPE '\\' THEN 3 ELSE 0 END"
                " + CASE WHEN data LIKE ? ESCAPE '\\' THEN 1 ELSE 0 END)"
            )
            params.extend([pattern, pattern, pattern])

        score_expr = " + ".join(score_parts)
        # Use a subquery to avoid duplicating the score expression (and its bindings)
        sql = f"""
            SELECT * FROM (
                SELECT *, ({score_expr}) AS relevance
                FROM memory_nodes
                WHERE id != 'root'
            ) WHERE relevance > 0
            ORDER BY relevance DESC, {_decay_score_sql()} DESC
            LIMIT ?
        """
        params.append(str(limit))

        with self._lock:
            rows = self.conn.execute(sql, params).fetchall()
            nodes = [self._row_to_node(r) for r in rows]

        # Touch matched nodes (updates access tracking)
        for node in nodes:
            self.touch_node(node.id)

        debug_log(f"Graph search for '{query}' found {len(nodes)} nodes", "memory")
        return nodes

    def find_node_by_name(self, name: str, parent_id: Optional[str] = None) -> Optional[MemoryNode]:
        """Find a node by exact name match (case-insensitive), optionally under a specific parent."""
        with self._lock:
            if parent_id is not None:
                row = self.conn.execute(
                    "SELECT * FROM memory_nodes WHERE LOWER(name) = LOWER(?) AND parent_id = ? LIMIT 1",
                    (name, parent_id),
                ).fetchone()
            else:
                row = self.conn.execute(
                    "SELECT * FROM memory_nodes WHERE LOWER(name) = LOWER(?) AND id != 'root' LIMIT 1",
                    (name,),
                ).fetchone()
            if row is None:
                return None
            return self._row_to_node(row)

    # ── Graph edges for visualisation ───────────────────────────────────

    def get_graph_data(self, root_id: str = "root", max_depth: int = 4) -> dict:
        """
        Return nodes and edges suitable for graph visualisation.

        Returns:
            {"nodes": [...], "edges": [...]}
            Each node: {id, name, description, data_token_count, access_count,
                        last_accessed, parent_id, has_children, depth}
            Each edge: {source, target}
        """
        nodes_out: list[dict] = []
        edges_out: list[dict] = []
        visited: set[str] = set()

        def _walk(nid: str, depth: int) -> None:
            if nid in visited or depth > max_depth:
                return
            visited.add(nid)

            node = self.get_node(nid)
            if node is None:
                return

            children = self.get_children(nid)
            nodes_out.append({
                "id": node.id,
                "name": node.name,
                "description": node.description,
                "data_token_count": node.data_token_count,
                "access_count": node.access_count,
                "last_accessed": node.last_accessed,
                "parent_id": node.parent_id,
                "has_children": len(children) > 0,
                "depth": depth,
            })

            for child in children:
                edges_out.append({"source": nid, "target": child.id})
                _walk(child.id, depth + 1)

        _walk(root_id, 0)
        return {"nodes": nodes_out, "edges": edges_out}

    # ── Internal helpers ────────────────────────────────────────────────

    @staticmethod
    def _row_to_node(row: sqlite3.Row) -> MemoryNode:
        return MemoryNode(
            id=row["id"],
            name=row["name"],
            description=row["description"],
            data=row["data"],
            parent_id=row["parent_id"],
            access_count=row["access_count"],
            last_accessed=row["last_accessed"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            data_token_count=row["data_token_count"],
        )

    def close(self) -> None:
        """Close the database connection."""
        try:
            with self._lock:
                self.conn.close()
        except Exception:
            pass
