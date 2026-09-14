"""Offline indexing and semantic search for explicitly configured documents."""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from ..debug import debug_log
from ..llm import get_embedding_backend
from ..utils.vector_store import get_best_vector_store


_ALLOWED_EXTENSIONS = {".txt", ".md"}
_VECTOR_ID_BASE = -1_000_000_000
_MAX_CHUNK_LINES = 40
_MAX_CHUNK_CHARS = 2_000


@dataclass(frozen=True)
class DocumentSearchHit:
    path: str
    start_line: int
    end_line: int
    text: str
    distance: float


class DocumentIndex:
    """Maintain an incremental local-document index in SQLite and vectors."""

    def __init__(
        self,
        db: Any,
        cfg: Any,
        *,
        embedding_backend: Any = None,
        vector_store: Any = None,
    ) -> None:
        self.db = db
        self.cfg = cfg
        self.embedding_backend = embedding_backend or get_embedding_backend(cfg)
        db_path = getattr(db, "db_path", ":memory:")
        self.vector_store = vector_store or get_best_vector_store(db_path)
        self._lock = threading.RLock()
        conn = getattr(db, "conn", None)
        self._conn = conn if isinstance(conn, sqlite3.Connection) else sqlite3.connect(":memory:")
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS document_files (
                    path TEXT PRIMARY KEY,
                    mtime_ns INTEGER NOT NULL,
                    size INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id INTEGER PRIMARY KEY,
                    path TEXT NOT NULL REFERENCES document_files(path) ON DELETE CASCADE,
                    start_line INTEGER NOT NULL,
                    end_line INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    vector_id INTEGER NOT NULL UNIQUE
                );
                """
            )
            self._conn.commit()

    @staticmethod
    def _under_root(path: Path, root: Path) -> bool:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            return False

    def _roots(self) -> list[Path]:
        raw_paths = getattr(self.cfg, "document_search_paths", [])
        if not isinstance(raw_paths, list):
            return []
        roots: list[Path] = []
        for raw in raw_paths:
            if not isinstance(raw, str) or not raw.strip():
                continue
            try:
                root = Path(raw).expanduser().resolve()
                if root.is_dir():
                    roots.append(root)
                else:
                    debug_log(f"Skipping document root that is not a folder: {raw}", "documents")
            except OSError as exc:
                debug_log(f"Skipping invalid document root {raw}: {exc}", "documents")
        return roots

    def _iter_allowed_files(self, roots: Iterable[Path]) -> Iterable[Path]:
        seen: set[Path] = set()
        for root in roots:
            try:
                candidates = root.rglob("*")
            except OSError as exc:
                debug_log(f"Could not scan document root {root}: {exc}", "documents")
                continue
            for candidate in candidates:
                if candidate.suffix.lower() not in _ALLOWED_EXTENSIONS:
                    continue
                try:
                    resolved = candidate.resolve()
                    if resolved in seen or not self._under_root(resolved, root):
                        debug_log(f"Skipping document outside allowed roots: {candidate}", "documents")
                        continue
                    if not resolved.is_file():
                        continue
                    seen.add(resolved)
                    yield resolved
                except OSError as exc:
                    debug_log(f"Skipping unreadable document {candidate}: {exc}", "documents")

    @staticmethod
    def _chunks(text: str) -> list[tuple[int, int, str]]:
        lines = text.splitlines()
        chunks: list[tuple[int, int, str]] = []
        current: list[str] = []
        start = 1
        for line_number, line in enumerate(lines, start=1):
            if current and (
                len(current) >= _MAX_CHUNK_LINES
                or len("\n".join(current)) + len(line) + 1 > _MAX_CHUNK_CHARS
            ):
                chunks.append((start, line_number - 1, "\n".join(current)))
                current = []
                start = line_number
            current.append(line)
        if current:
            chunks.append((start, len(lines), "\n".join(current)))
        return [(a, b, value) for a, b, value in chunks if value.strip()]

    def _remove_file(self, path: str) -> None:
        rows = self._conn.execute(
            "SELECT vector_id FROM document_chunks WHERE path = ?", (path,)
        ).fetchall()
        for row in rows:
            self.vector_store.delete_vector(int(row["vector_id"]))
        self._conn.execute("DELETE FROM document_chunks WHERE path = ?", (path,))
        self._conn.execute("DELETE FROM document_files WHERE path = ?", (path,))

    def _index_file(self, path: Path, mtime_ns: int, size: int) -> None:
        existing = self._conn.execute(
            "SELECT mtime_ns, size FROM document_files WHERE path = ?", (str(path),)
        ).fetchone()
        if existing and existing["mtime_ns"] == mtime_ns and existing["size"] == size:
            return
        self._remove_file(str(path))
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            debug_log(f"Could not read document {path}: {exc}", "documents")
            return
        self._conn.execute(
            "INSERT INTO document_files(path, mtime_ns, size) VALUES (?, ?, ?)",
            (str(path), mtime_ns, size),
        )
        for start_line, end_line, chunk in self._chunks(text):
            vector = self.embedding_backend.embed(
                chunk,
                getattr(self.cfg, "embedding_model", ""),
                timeout_sec=float(getattr(self.cfg, "llm_embedding_timeout_sec", 15.0)),
            )
            if vector is None:
                debug_log(f"Embedding failed for document chunk: {path}", "documents")
                continue
            cursor = self._conn.execute(
                """
                INSERT INTO document_chunks(path, start_line, end_line, text, vector_id)
                VALUES (?, ?, ?, ?, ?)
                """,
                (str(path), start_line, end_line, chunk, 0),
            )
            chunk_id = int(cursor.lastrowid)
            vector_id = _VECTOR_ID_BASE - chunk_id
            self._conn.execute(
                "UPDATE document_chunks SET vector_id = ? WHERE id = ?",
                (vector_id, chunk_id),
            )
            self.vector_store.add_vector(vector_id, vector)

    def refresh(self) -> None:
        """Scan configured roots and synchronise changed, new, and deleted files."""
        if not bool(getattr(self.cfg, "document_search_enabled", False)):
            return
        roots = self._roots()
        if not roots:
            return
        debug_log("Starting local document index refresh", "documents")
        scanned: set[str] = set()
        with self._lock:
            for path in self._iter_allowed_files(roots):
                try:
                    stat = path.stat()
                    scanned.add(str(path))
                    self._index_file(path, stat.st_mtime_ns, stat.st_size)
                except OSError as exc:
                    debug_log(f"Skipping document {path}: {exc}", "documents")
            known = {
                str(row["path"])
                for row in self._conn.execute("SELECT path FROM document_files")
            }
            for path in known - scanned:
                self._remove_file(path)
            if known - scanned and hasattr(self.vector_store, "rebuild"):
                self.vector_store.rebuild()
            self._conn.commit()
        debug_log("Finished local document index refresh", "documents")

    def search(self, query: str, top_k: int = 5) -> list[DocumentSearchHit]:
        """Return the closest indexed chunks for ``query``."""
        with self._lock:
            count = self._conn.execute("SELECT COUNT(*) FROM document_chunks").fetchone()[0]
            if not count:
                debug_log("Local document search found no indexed chunks", "documents")
                return []
            vector = self.embedding_backend.embed(
                query,
                getattr(self.cfg, "embedding_model", ""),
                timeout_sec=float(getattr(self.cfg, "llm_embedding_timeout_sec", 15.0)),
            )
            if vector is None:
                debug_log("Local document query embedding failed", "documents")
                return []
            results: list[DocumentSearchHit] = []
            for vector_id, distance in self.vector_store.search(vector, top_k=max(1, int(top_k))):
                row = self._conn.execute(
                    """
                    SELECT path, start_line, end_line, text
                    FROM document_chunks WHERE vector_id = ?
                    """,
                    (int(vector_id),),
                ).fetchone()
                if row:
                    results.append(DocumentSearchHit(
                        path=row["path"],
                        start_line=int(row["start_line"]),
                        end_line=int(row["end_line"]),
                        text=row["text"],
                        distance=float(distance),
                    ))
            return results
