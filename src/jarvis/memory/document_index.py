"""Isolated, atomic semantic storage for explicitly authorised documents."""
from __future__ import annotations

import hashlib
import heapq
import json
import os
import sqlite3
import stat
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..debug import debug_log
from ..llm import get_embedding_backend
from ..utils.embedding_vector import normalise_embedding

_ALLOWED_EXTENSIONS = {".txt", ".md"}
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
    """Own document metadata, text and embeddings in one SQLite transaction."""

    def __init__(self, db: Any, cfg: Any, *, embedding_backend: Any = None) -> None:
        self.cfg = cfg
        self.embedding_backend = embedding_backend or get_embedding_backend(cfg)
        diary_path = str(db.db_path)
        self.path = ':memory:' if diary_path == ':memory:' else diary_path + '.documents.sqlite3'
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, timeout=1)
        try:
            self._conn.row_factory = sqlite3.Row
            self._conn.execute('PRAGMA foreign_keys = ON')
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS document_files (
                    path TEXT PRIMARY KEY, identity TEXT NOT NULL,
                    embedding_key TEXT NOT NULL, dimension INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id INTEGER PRIMARY KEY,
                    path TEXT NOT NULL REFERENCES document_files(path) ON DELETE CASCADE,
                    start_line INTEGER NOT NULL, end_line INTEGER NOT NULL,
                    text TEXT NOT NULL, vector BLOB NOT NULL
                );
                CREATE INDEX IF NOT EXISTS document_chunks_path ON document_chunks(path);
            """)
        except BaseException:
            self._conn.close()
            raise

    def close(self) -> None:
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _embedding_key(self) -> str:
        backend = self.embedding_backend
        identity = [type(backend).__module__, type(backend).__qualname__,
                    getattr(backend, 'base_url', ''), self.cfg.embedding_model]
        return hashlib.sha256(json.dumps(identity).encode()).hexdigest()

    @staticmethod
    def _under_root(path: Path, root: Path) -> bool:
        return path.is_relative_to(root)

    def _roots(self) -> list[Path]:
        if not self.cfg.document_search_enabled:
            return []
        roots = []
        for raw in self.cfg.document_search_paths:
            try:
                root = Path(raw).expanduser().resolve(strict=True)
                if root.is_dir() and root not in roots:
                    roots.append(root)
            except (OSError, ValueError, RuntimeError):
                debug_log('Skipping an unavailable document root', 'documents')
        return roots

    @staticmethod
    def _identity(value) -> str:
        return json.dumps([value.st_dev, value.st_ino, value.st_size,
                           value.st_mtime_ns, value.st_ctime_ns])

    def _authorised_identity(self, path: Path, roots: list[Path]) -> str | None:
        try:
            resolved = path.resolve(strict=True)
            if resolved != path or not any(self._under_root(resolved, root) for root in roots):
                return None
            value = path.stat()
            if stat.S_ISREG(value.st_mode):
                return self._identity(value)
        except (OSError, ValueError, RuntimeError):
            pass
        return None

    def _iter_allowed_files(self, roots: Iterable[Path]) -> Iterable[Path]:
        seen = set()
        for root in roots:
            for candidate in root.rglob('*'):
                if candidate.suffix.lower() not in _ALLOWED_EXTENSIONS:
                    continue
                try:
                    resolved = candidate.resolve(strict=True)
                except (OSError, RuntimeError):
                    continue
                if resolved not in seen and self._authorised_identity(resolved, [root]):
                    seen.add(resolved)
                    yield resolved

    def _embed(self, text: str):
        vector = self.embedding_backend.embed(text, self.cfg.embedding_model,
                                             timeout_sec=self.cfg.llm_embedding_timeout_sec)
        return normalise_embedding(vector)

    @staticmethod
    def _chunks(text: str) -> list[tuple[int, int, str]]:
        lines = text.splitlines()
        chunks: list[tuple[int, int, str]] = []
        current: list[str] = []
        current_chars = 0
        start = 1
        for line_number, line in enumerate(lines, start=1):
            if current and (
                len(current) >= _MAX_CHUNK_LINES
                or current_chars + len(line) + 1 > _MAX_CHUNK_CHARS
            ):
                chunks.append((start, line_number - 1, "\n".join(current)))
                current = []
                current_chars = 0
            if not current:
                start = line_number
            if len(line) > _MAX_CHUNK_CHARS:
                full_parts = (len(line) - 1) // _MAX_CHUNK_CHARS
                for part in range(full_parts):
                    offset = part * _MAX_CHUNK_CHARS
                    chunks.append((line_number, line_number, line[offset:offset + _MAX_CHUNK_CHARS]))
                line = line[full_parts * _MAX_CHUNK_CHARS:]
            current_chars += len(line) + bool(current)
            current.append(line)
        if current:
            chunks.append((start, len(lines), "\n".join(current)))
        return [(a, b, value) for a, b, value in chunks if value.strip()]

    def _index_file(self, path: Path, roots: list[Path]) -> bool:
        identity = self._authorised_identity(path, roots)
        if identity is None:
            return False
        key = self._embedding_key()
        old = self._conn.execute('SELECT identity, embedding_key FROM document_files WHERE path=?',
                                 (str(path),)).fetchone()
        if old and old['identity'] == identity and old['embedding_key'] == key:
            return True
        try:
            fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
            with os.fdopen(fd, 'r', encoding='utf-8', errors='replace') as stream:
                if self._identity(os.fstat(stream.fileno())) != identity:
                    return False
                text = stream.read()
            prepared = []
            dimension = None
            for start, end, chunk in self._chunks(text):
                vector = self._embed(chunk)
                if dimension is not None and dimension != vector.size:
                    raise ValueError('Document embedding dimensions changed during refresh')
                dimension = vector.size
                prepared.append((str(path), start, end, chunk, vector.tobytes()))
            if self._authorised_identity(path, self._roots()) != identity:
                return False
            with self._conn:
                self._conn.execute('DELETE FROM document_files WHERE path=?', (str(path),))
                self._conn.execute('INSERT INTO document_files VALUES (?,?,?,?)',
                                   (str(path), identity, key, int(dimension or 0)))
                self._conn.executemany('INSERT INTO document_chunks(path,start_line,end_line,text,vector) '
                                       'VALUES (?,?,?,?,?)', prepared)
            return True
        except (OSError, ValueError, sqlite3.Error) as exc:
            debug_log(f'Document refresh did not publish {path}: {exc}', 'documents')
            return False

    def refresh(self) -> None:
        if not self.cfg.document_search_enabled:
            return
        with self._lock:
            roots = self._roots()
            for row in self._conn.execute('SELECT path,identity FROM document_files').fetchall():
                if self._authorised_identity(Path(row['path']), roots) != row['identity']:
                    with self._conn:
                        self._conn.execute('DELETE FROM document_files WHERE path=?', (row['path'],))
            for path in self._iter_allowed_files(roots):
                self._index_file(path, roots)

    def search(self, query: str, top_k: int = 5) -> list[DocumentSearchHit]:
        if not self.cfg.document_search_enabled:
            return []
        with self._lock:
            roots, key = self._roots(), self._embedding_key()
            if not roots:
                return []
            try:
                vector = self._embed(query)
            except ValueError:
                return []
            with self._conn:
                self._conn.execute('DELETE FROM document_files WHERE embedding_key=? AND dimension!=?',
                                   (key, int(vector.size)))
            candidates = []
            for file in self._conn.execute('SELECT * FROM document_files WHERE embedding_key=?', (key,)):
                path = Path(file['path'])
                if self._authorised_identity(path, roots) != file['identity']:
                    continue
                if file['dimension'] != vector.size:
                    continue
                for row in self._conn.execute('SELECT * FROM document_chunks WHERE path=?', (str(path),)):
                    stored = np.frombuffer(row['vector'], dtype=np.float32)
                    if stored.size != vector.size:
                        continue
                    distance = 1.0 - float(np.dot(vector, stored))
                    item = (-distance, row['id'], DocumentSearchHit(str(path), row['start_line'],
                                                                 row['end_line'], row['text'], distance))
                    heapq.heappush(candidates, item)
                    if len(candidates) > min(10, max(1, top_k)):
                        heapq.heappop(candidates)
            return [item[2] for item in sorted(candidates, key=lambda item: (-item[0], item[1]))]
