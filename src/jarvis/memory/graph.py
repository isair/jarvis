"""
🧠 Node Graph Memory System (v2)

A self-organising node graph that dynamically structures memories by topic
relevance. Three fast-access entry points (recent nodes, top nodes, root node)
ensure the most relevant memories are always reachable without exhaustive search.

Current: CRUD operations, access tracking, tree/graph queries, UI explorer.
Future: LLM-powered auto-split when data exceeds SPLIT_THRESHOLD, summary
cascade upward on writes, auto-merge for sparse subtrees, and periodic
housekeeping. See graph.spec.md for the full roadmap.
"""

from __future__ import annotations

import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from ..debug import debug_log


# ── Configuration defaults ──────────────────────────────────────────────────

SPLIT_THRESHOLD = 1500       # tokens — when to split a node into children
MERGE_THRESHOLD = 200        # tokens — when to collapse sparse children back
RECENT_NODES_COUNT = 10      # number of recently-accessed nodes to track
TOP_NODES_COUNT = 15         # most-accessed nodes to surface
TOP_NODES_WINDOW_DAYS = 30   # time window for top-nodes ranking
MAX_TRAVERSAL_DEPTH = 8      # safety limit on graph traversal
SUMMARY_MAX_LENGTH = 300     # max characters for a node description


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
"""


# ── Graph Memory Store ──────────────────────────────────────────────────────

class GraphMemoryStore:
    """
    Self-organising node graph for persistent memory.

    Backed by SQLite with thread-safe access. Provides three entry points
    for fast retrieval: recent nodes, top nodes, and the root node.
    """

    def __init__(self, db_path: str) -> None:
        from pathlib import Path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._init_schema()
        self._ensure_root()

    # ── Schema & bootstrap ──────────────────────────────────────────────

    def _init_schema(self) -> None:
        with self._lock:
            self.conn.execute("PRAGMA foreign_keys = ON")
            self.conn.executescript(_GRAPH_SCHEMA_SQL)
            self.conn.commit()

    def _ensure_root(self) -> None:
        """Create the root node if it doesn't exist."""
        with self._lock:
            row = self.conn.execute(
                "SELECT id FROM memory_nodes WHERE parent_id IS NULL LIMIT 1"
            ).fetchone()
            if row is None:
                now = datetime.now(timezone.utc).isoformat()
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

    # ── CRUD ────────────────────────────────────────────────────────────

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
        """Get all direct children of a node, ordered by access_count desc."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM memory_nodes WHERE parent_id = ? ORDER BY access_count DESC",
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

        node_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        token_count = _estimate_tokens(data)

        with self._lock:
            self.conn.execute(
                """INSERT INTO memory_nodes
                   (id, name, description, data, parent_id,
                    access_count, last_accessed, created_at, updated_at,
                    data_token_count)
                   VALUES (?, ?, ?, ?, ?, 0, ?, ?, ?, ?)""",
                (node_id, name, description, data, parent_id, now, now, now, token_count),
            )
            self.conn.commit()

        debug_log(f"Created memory node '{name}' ({node_id[:8]})", "memory")
        return MemoryNode(
            id=node_id,
            name=name,
            description=description,
            data=data,
            parent_id=parent_id,
            access_count=0,
            last_accessed=now,
            created_at=now,
            updated_at=now,
            data_token_count=token_count,
        )

    def update_node(
        self,
        node_id: str,
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        data: Optional[str] = None,
        parent_id: Optional[str] = ...,  # type: ignore[assignment]
    ) -> Optional[MemoryNode]:
        """Update fields on an existing node. Returns the updated node."""
        node = self.get_node(node_id)
        if node is None:
            return None

        now = datetime.now(timezone.utc).isoformat()
        if name is not None:
            node.name = name
        if description is not None:
            node.description = description
        if data is not None:
            node.data = data
            node.data_token_count = _estimate_tokens(data)
        if parent_id is not ...:
            node.parent_id = parent_id
        node.updated_at = now

        with self._lock:
            self.conn.execute(
                """UPDATE memory_nodes
                   SET name = ?, description = ?, data = ?, parent_id = ?,
                       updated_at = ?, data_token_count = ?
                   WHERE id = ?""",
                (node.name, node.description, node.data, node.parent_id,
                 node.updated_at, node.data_token_count, node_id),
            )
            self.conn.commit()

        return node

    def delete_node(self, node_id: str) -> bool:
        """Delete a node. Children are orphaned (parent_id set to NULL by FK)."""
        if node_id == "root":
            return False
        with self._lock:
            cur = self.conn.execute(
                "DELETE FROM memory_nodes WHERE id = ?", (node_id,)
            )
            self.conn.commit()
            return cur.rowcount > 0

    def touch_node(self, node_id: str) -> None:
        """Increment access_count and update last_accessed."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self.conn.execute(
                """UPDATE memory_nodes
                   SET access_count = access_count + 1, last_accessed = ?
                   WHERE id = ?""",
                (now, node_id),
            )
            self.conn.commit()

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

    def get_top_nodes(
        self,
        limit: int = TOP_NODES_COUNT,
        window_days: int = TOP_NODES_WINDOW_DAYS,
    ) -> list[MemoryNode]:
        """Get the most frequently accessed nodes within a time window."""
        from datetime import timedelta
        cutoff = (datetime.now(timezone.utc) - timedelta(days=window_days)).isoformat()
        with self._lock:
            rows = self.conn.execute(
                """SELECT * FROM memory_nodes
                   WHERE id != 'root' AND last_accessed >= ?
                   ORDER BY access_count DESC
                   LIMIT ?""",
                (cutoff, limit),
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
        with self._lock:
            rows = self.conn.execute(
                "SELECT * FROM memory_nodes ORDER BY access_count DESC"
            ).fetchall()
            return [self._row_to_node(r) for r in rows]

    def get_node_count(self) -> int:
        """Return total number of nodes in the graph."""
        with self._lock:
            row = self.conn.execute("SELECT COUNT(*) as cnt FROM memory_nodes").fetchone()
            return row["cnt"]

    # ── Search ─────────────────────────────────────────────────────────

    def search_nodes(self, query: str, limit: int = 10) -> list[MemoryNode]:
        """Search nodes by keyword match across name, description, and data.

        Uses case-insensitive LIKE matching on each keyword (split by whitespace).
        Nodes matching more keywords rank higher; ties broken by access_count.
        Excludes the root node from results and touches matched nodes.
        """
        keywords = [k.strip() for k in query.split() if k.strip()]
        if not keywords:
            return []

        # Build a score expression: +1 per keyword per field matched
        score_parts: list[str] = []
        params: list[str] = []
        for kw in keywords:
            pattern = f"%{kw}%"
            score_parts.append(
                "(CASE WHEN name LIKE ? THEN 1 ELSE 0 END"
                " + CASE WHEN description LIKE ? THEN 1 ELSE 0 END"
                " + CASE WHEN data LIKE ? THEN 1 ELSE 0 END)"
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
            ORDER BY relevance DESC, access_count DESC
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
