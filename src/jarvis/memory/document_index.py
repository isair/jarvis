"""Isolated, atomic semantic storage for explicitly authorised documents."""
from __future__ import annotations

import hashlib
import heapq
import json
import os
import sqlite3
import stat
import threading
import time
from dataclasses import dataclass, field
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
class IndexLimits:
    """Finite per-pass work, per-file allocation and persistent cache limits."""
    files: int = 64
    chunks: int = 128
    input_bytes: int = 1_048_576
    entries: int = 2_048
    seconds: float = 5.0
    file_bytes: int = 262_144
    cached_chunks: int = 1_024
    cached_files: int = 256
    dimension: int = 4_096
    depth: int = 32
    roots: int = 32

    def __post_init__(self):
        if any(value <= 0 for value in vars(self).values()):
            raise ValueError('Document index limits must be positive')


@dataclass
class IndexCoverage:
    files: int = 0
    chunks: int = 0
    input_bytes: int = 0
    entries: int = 0
    reasons: set[str] = field(default_factory=set)

    @property
    def complete(self) -> bool:
        return not self.reasons


class _Pass:
    def __init__(self, limits: IndexLimits, cancel_event):
        self.limits = limits
        self.cancel_event = cancel_event
        self.deadline = time.monotonic() + limits.seconds
        self.coverage = IndexCoverage()

    def stopped(self) -> bool:
        if self.cancel_event is not None and self.cancel_event.is_set():
            self.coverage.reasons.add('cancelled')
        if time.monotonic() >= self.deadline:
            self.coverage.reasons.add('seconds')
        return bool(self.coverage.reasons & {'cancelled', 'seconds'})

    def charge(self, resource: str, amount: int = 1) -> bool:
        if self.stopped():
            return False
        value = getattr(self.coverage, resource) + amount
        if value > getattr(self.limits, resource):
            self.coverage.reasons.add(resource)
            return False
        setattr(self.coverage, resource, value)
        return True


@dataclass(frozen=True)
class DocumentSearchHit:
    path: str
    start_line: int
    end_line: int
    text: str
    distance: float


class DocumentIndex:
    """Own document metadata, text and embeddings in one SQLite transaction."""

    def __init__(self, db: Any, cfg: Any, *, embedding_backend: Any = None,
                 limits: IndexLimits | None = None) -> None:
        self.cfg = cfg
        self.limits = limits or IndexLimits()
        self.coverage = IndexCoverage()
        self.embedding_backend = embedding_backend or get_embedding_backend(cfg)
        diary_path = str(db.db_path)
        self.path = ':memory:' if diary_path == ':memory:' else diary_path + '.documents.sqlite3'
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, timeout=1)
        try:
            self._conn.row_factory = sqlite3.Row
            self._conn.execute('PRAGMA foreign_keys = ON')
            self._conn.execute('PRAGMA max_page_count = 16384')
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
        for number, raw in enumerate(self.cfg.document_search_paths):
            if number >= self.limits.roots:
                self.coverage.reasons.add('roots')
                break
            try:
                root = Path(raw).expanduser().resolve(strict=True)
                if root.is_dir():
                    if root not in roots:
                        roots.append(root)
                else:
                    self.coverage.reasons.add('unavailable_root')
            except (OSError, ValueError, RuntimeError):
                self.coverage.reasons.add('unavailable_root')
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

    def _iter_allowed_files(self, roots: Iterable[Path], work: _Pass) -> Iterable[Path]:
        seen = set()
        for root in roots:
            if work.stopped():
                return
            stack = []
            try:
                stack.append(os.scandir(root))
                while stack and not work.stopped():
                    entry = next(stack[-1], None)
                    if entry is None:
                        stack.pop().close()
                        continue
                    if not work.charge('entries'):
                        return
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            if len(stack) >= self.limits.depth:
                                work.coverage.reasons.add('depth')
                            else:
                                stack.append(os.scandir(entry.path))
                        elif Path(entry.name).suffix.lower() in _ALLOWED_EXTENSIONS:
                            path = Path(entry.path).resolve(strict=True)
                            if path not in seen and self._authorised_identity(path, [root]):
                                seen.add(path)
                                yield path
                    except (OSError, RuntimeError):
                        work.coverage.reasons.add('unreadable')
            except OSError:
                work.coverage.reasons.add('unreadable')
            finally:
                for stream in stack:
                    stream.close()

    def _embed(self, text: str, work: _Pass):
        if work.stopped():
            raise ValueError('Document work was cancelled or reached its deadline')
        remaining = max(0.001, work.deadline - time.monotonic())
        vector = self.embedding_backend.embed(text, self.cfg.embedding_model,
                                             timeout_sec=min(self.cfg.llm_embedding_timeout_sec, remaining))
        if vector is None or len(vector) > self.limits.dimension:
            raise ValueError('Document embedding is unavailable or exceeds its dimension limit')
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

    def _index_file(self, path: Path, roots: list[Path], work: _Pass) -> bool:
        identity = self._authorised_identity(path, roots)
        if identity is None:
            return False
        key = self._embedding_key()
        old = self._conn.execute('SELECT identity, embedding_key FROM document_files WHERE path=?',
                                 (str(path),)).fetchone()
        if old and old['identity'] == identity and old['embedding_key'] == key:
            return True
        if not work.charge('files'):
            return False
        try:
            size = path.stat().st_size
            if size > self.limits.file_bytes:
                work.coverage.reasons.add('file_bytes')
                return False
            if not work.charge('input_bytes', size):
                return False
            fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
                         | getattr(os, 'O_NONBLOCK', 0))
            with os.fdopen(fd, 'rb') as stream:
                opened = os.fstat(stream.fileno())
                if not stat.S_ISREG(opened.st_mode) or self._identity(opened) != identity:
                    return False
                raw = stream.read(size + 1)
                if len(raw) != size:
                    work.coverage.reasons.add('changed')
                    return False
                text = raw.decode('utf-8', errors='replace')
            prepared = []
            dimension = None
            for start, end, chunk in self._chunks(text):
                if not work.charge('chunks'):
                    return False
                vector = self._embed(chunk, work)
                if dimension is not None and dimension != vector.size:
                    raise ValueError('Document embedding dimensions changed during refresh')
                dimension = vector.size
                prepared.append((str(path), start, end, chunk, vector.tobytes()))
            if work.stopped():
                return False
            if self._authorised_identity(path, self._roots()) != identity:
                work.coverage.reasons.add('changed')
                return False
            count = self._conn.execute('SELECT COUNT(*) FROM document_chunks WHERE path!=?',
                                       (str(path),)).fetchone()[0]
            if count + len(prepared) > self.limits.cached_chunks:
                work.coverage.reasons.add('cached_chunks')
                return False
            files = self._conn.execute('SELECT COUNT(*) FROM document_files WHERE path!=?',
                                       (str(path),)).fetchone()[0]
            if files >= self.limits.cached_files:
                work.coverage.reasons.add('cached_files')
                return False
            with self._conn:
                self._conn.execute('DELETE FROM document_files WHERE path=?', (str(path),))
                self._conn.execute('INSERT INTO document_files VALUES (?,?,?,?)',
                                   (str(path), identity, key, int(dimension or 0)))
                self._conn.executemany('INSERT INTO document_chunks(path,start_line,end_line,text,vector) '
                                       'VALUES (?,?,?,?,?)', prepared)
            return True
        except (OSError, ValueError, sqlite3.Error) as exc:
            work.coverage.reasons.add('failed')
            debug_log(f'Document refresh did not publish {path}: {exc}', 'documents')
            return False

    def refresh(self, *, cancel_event=None) -> IndexCoverage:
        work = _Pass(self.limits, cancel_event)
        self.coverage = work.coverage
        if not self.cfg.document_search_enabled:
            return self.coverage
        with self._lock:
            if work.stopped():
                return self.coverage
            roots = self._roots()
            if len(roots) > self.limits.roots:
                work.coverage.reasons.add('roots')
                roots = roots[:self.limits.roots]
            for row in self._conn.execute('SELECT path,identity FROM document_files').fetchall():
                if work.stopped():
                    return self.coverage
                if self._authorised_identity(Path(row['path']), roots) != row['identity']:
                    with self._conn:
                        self._conn.execute('DELETE FROM document_files WHERE path=?', (row['path'],))
            for path in self._iter_allowed_files(roots, work):
                self._index_file(path, roots, work)
                if work.stopped() or work.coverage.reasons & {'files', 'chunks', 'input_bytes'}:
                    break
            debug_log(f'Document refresh coverage: {self.coverage}', 'documents')
            return self.coverage

    def search(self, query: str, top_k: int = 5, *, cancel_event=None) -> list[DocumentSearchHit]:
        if not self.cfg.document_search_enabled:
            return []
        work = _Pass(self.limits, cancel_event)
        if len(query) > _MAX_CHUNK_CHARS:
            raise ValueError('Document search query exceeds its character limit')
        with self._lock:
            roots, key = self._roots(), self._embedding_key()
            if not roots:
                return []
            if not self._conn.execute('SELECT 1 FROM document_chunks LIMIT 1').fetchone():
                return []
            vector = self._embed(query, work)
            with self._conn:
                self._conn.execute('DELETE FROM document_files WHERE embedding_key=? AND dimension!=?',
                                   (key, int(vector.size)))
            candidates = []
            for file in self._conn.execute('SELECT * FROM document_files WHERE embedding_key=?', (key,)):
                if work.stopped():
                    self.coverage.reasons.update(work.coverage.reasons)
                    break
                path = Path(file['path'])
                if self._authorised_identity(path, roots) != file['identity']:
                    continue
                if file['dimension'] != vector.size:
                    continue
                for row in self._conn.execute('SELECT * FROM document_chunks WHERE path=?', (str(path),)):
                    if work.stopped():
                        self.coverage.reasons.update(work.coverage.reasons)
                        break
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
