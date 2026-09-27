"""Source-grounded, locally stored facts and bounded memory recall."""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from ..debug import debug_log
from ..utils.redact import scrub_secrets


_FACT_LISTENERS: list[Callable[..., None]] = []


def register_fact_mutation_listener(callback: Callable[..., None]) -> None:
    if callback not in _FACT_LISTENERS:
        _FACT_LISTENERS.append(callback)


def unregister_fact_mutation_listener(callback: Callable[..., None]) -> None:
    if callback in _FACT_LISTENERS:
        _FACT_LISTENERS.remove(callback)


def _notify(action: str, fact_id: int, kind: str, ownership: str) -> None:
    for callback in tuple(_FACT_LISTENERS):
        try:
            callback(action=action, fact_id=fact_id, kind=kind, ownership=ownership)
        except Exception as exc:
            debug_log(f"fact mutation listener failed: {exc}", "memory")


_SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS fact_sources (
    id INTEGER PRIMARY KEY,
    source_ref TEXT NOT NULL UNIQUE,
    source_type TEXT NOT NULL,
    source_app TEXT NOT NULL,
    source_role TEXT NOT NULL,
    source_channel TEXT NOT NULL,
    source_hash TEXT NOT NULL,
    evidence TEXT NOT NULL,
    observed_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS memory_facts (
    id INTEGER PRIMARY KEY,
    text TEXT NOT NULL,
    kind TEXT NOT NULL CHECK(kind IN ('user','directive','world')),
    owner TEXT NOT NULL CHECK(owner IN ('user','other','unknown')),
    subject TEXT NOT NULL DEFAULT '',
    predicate_key TEXT NOT NULL DEFAULT '',
    source_id INTEGER NOT NULL REFERENCES fact_sources(id),
    observed_at TEXT NOT NULL,
    valid_from TEXT NOT NULL,
    valid_to TEXT,
    status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active','superseded','retracted')),
    supersedes_id INTEGER REFERENCES memory_facts(id),
    embedding_json TEXT,
    created_at TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_facts_source_text ON memory_facts(source_id, text);
CREATE INDEX IF NOT EXISTS idx_facts_active ON memory_facts(status, observed_at DESC);
CREATE INDEX IF NOT EXISTS idx_facts_supersedes ON memory_facts(supersedes_id);
CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(text, subject, content='memory_facts', content_rowid='id', tokenize='unicode61');
CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON memory_facts BEGIN
  INSERT INTO facts_fts(rowid,text,subject) VALUES(new.id,new.text,new.subject);
END;
CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON memory_facts BEGIN
  INSERT INTO facts_fts(facts_fts,rowid,text,subject) VALUES('delete',old.id,old.text,old.subject);
END;
CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE ON memory_facts BEGIN
  INSERT INTO facts_fts(facts_fts,rowid,text,subject) VALUES('delete',old.id,old.text,old.subject);
  INSERT INTO facts_fts(rowid,text,subject) VALUES(new.id,new.text,new.subject);
END;
CREATE TABLE IF NOT EXISTS fact_events (
    id INTEGER PRIMARY KEY,
    fact_id INTEGER NOT NULL REFERENCES memory_facts(id),
    action TEXT NOT NULL,
    evidence TEXT NOT NULL,
    observed_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS fact_pending_batches (
    id INTEGER PRIMARY KEY,
    batch_ref TEXT NOT NULL UNIQUE,
    source_app TEXT NOT NULL,
    messages_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS fact_meta (key TEXT PRIMARY KEY, value INTEGER NOT NULL);
INSERT OR IGNORE INTO fact_meta(key,value) VALUES('revision',0);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _scrub(value: str) -> str:
    return re.sub(r"\s+", " ", scrub_secrets(value)).strip()


def _utc(value: str, *, end_of_day: bool = False) -> str:
    """Store and compare instants in one UTC representation."""
    if len(value) == 10:
        value += "T23:59:59.999999+00:00" if end_of_day else "T00:00:00+00:00"
    moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).isoformat()


def _fts_query(raw: str) -> str:
    words = re.findall(r"\w+", raw, flags=re.UNICODE)[:12]
    return " OR ".join(f'"{word}"' for word in words)


def _cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return -1.0
    product = sum(a * b for a, b in zip(left, right))
    norms = math.sqrt(sum(a * a for a in left) * sum(b * b for b in right))
    return product / norms if norms else -1.0


def _observed_bound(value: str | None, *, end: bool = False) -> str | None:
    if not value:
        return None
    return _utc(value, end_of_day=end)


class FactStore:
    """SQLite fact ledger. Retraction preserves an inspectable history."""

    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self.conn.executescript(_SCHEMA)
            self.conn.commit()

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    def _revision(self) -> None:
        self.conn.execute("UPDATE fact_meta SET value=value+1 WHERE key='revision'")

    def revision(self) -> int:
        with self._lock:
            return int(self.conn.execute("SELECT value FROM fact_meta WHERE key='revision'").fetchone()[0])

    def _row(self, row: sqlite3.Row | None) -> dict | None:
        if row is None:
            return None
        item = dict(row)
        item.pop("embedding_json", None)
        item["source"] = {
            key: item.pop(key) for key in (
                "source_ref", "source_type", "source_app", "source_role",
                "source_channel", "source_hash", "evidence",
            )
        }
        return item

    _SELECT = """SELECT f.*,s.source_ref,s.source_type,s.source_app,s.source_role,
        s.source_channel,s.source_hash,s.evidence FROM memory_facts f
        JOIN fact_sources s ON s.id=f.source_id"""

    def get_fact(self, fact_id: int) -> dict | None:
        with self._lock:
            return self._row(self.conn.execute(self._SELECT + " WHERE f.id=?", (fact_id,)).fetchone())

    def list_facts(self, *, status: str | None = "active", limit: int = 100,
                   offset: int = 0, subject: str | None = None) -> list[dict]:
        clauses, args = [], []
        if status is not None:
            clauses.append("f.status=?")
            args.append(status)
        if subject is not None:
            clauses.append("f.subject=?")
            args.append(subject)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        with self._lock:
            rows = self.conn.execute(self._SELECT + where + " ORDER BY f.observed_at DESC,f.id DESC LIMIT ? OFFSET ?",
                                     (*args, max(0, limit), max(0, offset))).fetchall()
            return [self._row(row) for row in rows]

    def add_fact(self, text: str, *, kind: str, owner: str, source_ref: str,
                 source_type: str, source_role: str, source_channel: str,
                 source_text: str, evidence: str, observed_at: str,
                 source_app: str = "jarvis", subject: str = "",
                 predicate_key: str = "", valid_from: str | None = None,
                 supersedes_id: int | None = None, embedding: list[float] | None = None) -> dict:
        text, evidence, source_text = _scrub(text), _scrub(evidence), _scrub(source_text)
        subject, predicate_key = _scrub(subject), _scrub(predicate_key)
        observed_at = _utc(observed_at)
        if not text or not evidence or evidence not in source_text:
            raise ValueError("A fact requires exact source evidence")
        if kind not in {"user", "directive", "world"} or owner not in {"user", "other", "unknown"}:
            raise ValueError("Invalid fact kind or owner")
        if source_type not in {"dialogue", "manual", "web", "tool"}:
            raise ValueError("Invalid source type")
        if (owner == "user" or kind == "directive") and (
            source_role != "user" or source_channel == "ambient"
        ):
            raise ValueError("User ownership and directives require addressed user speech")
        if kind == "directive" and owner != "user":
            raise ValueError("A directive requires confirmed user ownership")
        if kind == "directive" and source_type not in {"dialogue", "manual"}:
            raise ValueError("A directive requires a direct user source")
        if owner == "user" and source_type not in {"dialogue", "manual"}:
            raise ValueError("A user-owned fact requires a direct user source")
        valid_from = _utc(valid_from) if valid_from else observed_at
        if supersedes_id is not None:
            old = self.get_fact(supersedes_id)
            if old is None or old["status"] != "active" or old["kind"] != kind or old["owner"] != owner:
                raise ValueError("Supersession must target an active fact of the same ownership")
        digest = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
        source_ref = source_ref + ":" + hashlib.sha256(evidence.encode("utf-8")).hexdigest()[:12]
        with self._lock:
            try:
                with self.conn:
                    source = self.conn.execute("SELECT id,source_hash FROM fact_sources WHERE source_ref=?", (source_ref,)).fetchone()
                    if source is None:
                        source_id = self.conn.execute(
                            "INSERT INTO fact_sources(source_ref,source_type,source_app,source_role,source_channel,source_hash,evidence,observed_at) VALUES(?,?,?,?,?,?,?,?)",
                            (source_ref, source_type, source_app, source_role, source_channel, digest, evidence, observed_at),
                        ).lastrowid
                    else:
                        if source["source_hash"] != digest:
                            raise ValueError("Source reference points to different content")
                        source_id = source[0]
                    if supersedes_id is not None:
                        updated = self.conn.execute(
                            "UPDATE memory_facts SET status='superseded',valid_to=? WHERE id=? AND status='active' AND kind=? AND owner=?",
                            (valid_from, supersedes_id, kind, owner),
                        )
                        if updated.rowcount != 1:
                            raise ValueError("Fact was already corrected or retracted")
                        self.conn.execute("INSERT INTO fact_events(fact_id,action,evidence,observed_at) VALUES(?,?,?,?)",
                                          (supersedes_id, "superseded", evidence, observed_at))
                    fact_id = self.conn.execute(
                        """INSERT INTO memory_facts(text,kind,owner,subject,predicate_key,source_id,observed_at,valid_from,
                           supersedes_id,embedding_json,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)""",
                        (text, kind, owner, subject, predicate_key, source_id, observed_at, valid_from,
                         supersedes_id, json.dumps(embedding) if embedding else None, _now()),
                    ).lastrowid
                    self.conn.execute("INSERT INTO fact_events(fact_id,action,evidence,observed_at) VALUES(?,?,?,?)",
                                      (fact_id, "asserted", evidence, observed_at))
                    self._revision()
            except sqlite3.IntegrityError:
                row = self.conn.execute(self._SELECT + " WHERE s.source_ref=? AND f.text=?", (source_ref, text)).fetchone()
                if row is None:
                    raise
                return self._row(row)
        _notify("correct" if supersedes_id else "add", fact_id, kind, owner)
        return self.get_fact(fact_id)

    def correct_fact(self, fact_id: int, *, text: str, evidence: str, source_text: str,
                     source_ref: str, source_role: str = "user", source_channel: str = "text",
                     observed_at: str | None = None, valid_from: str | None = None,
                     embedding: list[float] | None = None) -> dict:
        old = self.get_fact(fact_id)
        if old is None:
            raise ValueError("Unknown fact")
        return self.add_fact(text, kind=old["kind"], owner=old["owner"], subject=old["subject"],
                             predicate_key=old["predicate_key"], source_ref=source_ref,
                             source_type="manual", source_role=source_role, source_channel=source_channel,
                             source_text=source_text, evidence=evidence, observed_at=observed_at or _now(),
                             valid_from=valid_from, supersedes_id=fact_id, embedding=embedding)

    def retract_fact(self, fact_id: int, *, evidence: str, observed_at: str | None = None) -> bool:
        if not evidence.strip():
            raise ValueError("Retraction requires a reason")
        evidence = _scrub(evidence)
        observed_at = _utc(observed_at) if observed_at else _now()
        with self._lock:
            row = self.conn.execute("SELECT kind,owner,status FROM memory_facts WHERE id=?", (fact_id,)).fetchone()
            if row is None or row["status"] != "active":
                return False
            with self.conn:
                updated = self.conn.execute("UPDATE memory_facts SET status='retracted',valid_to=? WHERE id=? AND status='active'",
                                            (observed_at, fact_id))
                if updated.rowcount != 1:
                    return False
                self.conn.execute("INSERT INTO fact_events(fact_id,action,evidence,observed_at) VALUES(?,?,?,?)",
                                  (fact_id, "retracted", evidence, observed_at))
                self._revision()
        _notify("retract", fact_id, row["kind"], row["owner"])
        return True

    def get_fact_history(self, fact_id: int) -> list[dict]:
        with self._lock:
            current = self.get_fact(fact_id)
            if current is None:
                return []
            ancestors = set()
            while current["supersedes_id"] and current["id"] not in ancestors:
                ancestors.add(current["id"])
                previous = self.get_fact(current["supersedes_id"])
                if previous is None:
                    break
                current = previous
            history = []
            seen = set()
            while current and current["id"] not in seen:
                history.append(current)
                seen.add(current["id"])
                next_row = self.conn.execute("SELECT id FROM memory_facts WHERE supersedes_id=? ORDER BY id LIMIT 1", (current["id"],)).fetchone()
                current = self.get_fact(next_row[0]) if next_row else None
            return history

    def get_fact_events(self, fact_id: int) -> list[dict]:
        """Return assertion, supersession and retraction evidence in order."""
        with self._lock:
            rows = self.conn.execute(
                "SELECT id,fact_id,action,evidence,observed_at FROM fact_events WHERE fact_id=? ORDER BY id",
                (fact_id,),
            ).fetchall()
            return [dict(row) for row in rows]

    def enqueue_batch(self, messages: list[dict], *, source_app: str, batch_ref: str) -> int:
        segments = []
        for message in messages:
            content = re.sub(r"\s+", " ", scrub_secrets(str(message.get("content", "")))).strip()
            for start in range(0, len(content), 4_000):
                segments.append({
                    "role": str(message.get("role", "unknown")),
                    "channel": str(message.get("channel", "unknown")),
                    "content": content[start:start + 4_000],
                    "ts": float(message.get("ts", 0.0)),
                })
        if not segments:
            raise ValueError("Empty fact extraction batch")
        batches = []
        current = []
        char_count = 0
        for segment in segments:
            if current and (len(current) >= 10 or char_count + len(segment["content"]) > 8_000):
                batches.append(current)
                current, char_count = [], 0
            current.append(segment)
            char_count += len(segment["content"])
        if current:
            batches.append(current)
        with self._lock, self.conn:
            first_id = None
            for index, batch in enumerate(batches):
                ref = f"{batch_ref}:{index}"
                self.conn.execute("INSERT OR IGNORE INTO fact_pending_batches(batch_ref,source_app,messages_json,created_at) VALUES(?,?,?,?)",
                                  (ref, source_app, json.dumps(batch, ensure_ascii=False), _now()))
                row = self.conn.execute("SELECT id FROM fact_pending_batches WHERE batch_ref=?", (ref,)).fetchone()
                if first_id is None:
                    first_id = int(row[0])
            debug_log(f"queued {len(segments)} redacted source segments in {len(batches)} fact batches", "memory")
            return first_id

    def pending_batches(self) -> list[dict]:
        with self._lock:
            return [dict(row) for row in self.conn.execute("SELECT * FROM fact_pending_batches ORDER BY id").fetchall()]

    def complete_batch(self, batch_id: int) -> None:
        with self._lock, self.conn:
            self.conn.execute("DELETE FROM fact_pending_batches WHERE id=?", (batch_id,))

    def search_facts(self, query: str, *, query_vector: list[float] | None = None,
                     from_time: str | None = None, to_time: str | None = None,
                     as_of: str | None = None, source_types: set[str] | None = None,
                     top_k: int = 20) -> list[dict]:
        clauses = ["f.status='active'" if as_of is None else "1=1"]
        args: list = []
        from_time = _observed_bound(from_time)
        to_time = _observed_bound(to_time, end=True)
        if from_time:
            clauses.append("f.observed_at>=?")
            args.append(from_time)
        if to_time:
            clauses.append("f.observed_at<=?")
            args.append(to_time)
        if as_of:
            as_of = _observed_bound(as_of, end=True)
            clauses.append("f.valid_from<=? AND (f.valid_to IS NULL OR f.valid_to>?)")
            args.extend((as_of, as_of))
        if source_types:
            clauses.append("s.source_type IN (" + ",".join("?" for _ in source_types) + ")")
            args.extend(sorted(source_types))
        where = " AND ".join(clauses)
        limit = max(50, top_k * 3)
        with self._lock:
            keyword = []
            safe = _fts_query(query)
            if safe:
                try:
                    keyword = self.conn.execute(
                        self._SELECT + " JOIN facts_fts ON facts_fts.rowid=f.id WHERE " + where +
                        " AND facts_fts MATCH ? ORDER BY bm25(facts_fts),f.id LIMIT ?", (*args, safe, limit),
                    ).fetchall()
                except sqlite3.OperationalError as exc:
                    debug_log(f"fact FTS failed: {exc}", "memory")
            vector = []
            if query_vector:
                candidates = self.conn.execute(self._SELECT + " WHERE " + where + " AND f.embedding_json IS NOT NULL", args).fetchall()
                vector = sorted(candidates, key=lambda row: -_cosine(json.loads(row["embedding_json"]), query_vector))[:limit]
            scores: dict[int, float] = {}
            for rows, weight in ((keyword, 0.5), (vector, 0.5)):
                for rank, row in enumerate(rows, 1):
                    scores[row["id"]] = scores.get(row["id"], 0.0) + weight / (60 + rank)
            if not scores and not safe:
                rows = self.conn.execute(self._SELECT + " WHERE " + where + " ORDER BY f.observed_at DESC,f.id DESC LIMIT ?", (*args, top_k)).fetchall()
                return [self._row(row) for row in rows]
            if not scores:
                return []
            ids = ",".join("?" for _ in scores)
            rows = self.conn.execute(self._SELECT + f" WHERE f.id IN ({ids})", tuple(scores)).fetchall()
            result = [self._row(row) for row in rows]
            result.sort(key=lambda row: (-scores[row["id"]], row["id"]))
            return result[:top_k]


def _whole_lines(lines: list[str], cap: int) -> str:
    chosen = []
    remaining = max(0, cap)
    for line in lines:
        if len(line) <= remaining:
            chosen.append(line)
            remaining -= len(line) + 1
    return "\n".join(chosen)


def build_fact_warm_profile(db_path: str, *, user_max_chars: int = 1200,
                            directives_max_chars: int = 600) -> dict[str, str]:
    store = FactStore(db_path)
    try:
        rows = store.list_facts(limit=500)
        user = [row["text"] for row in rows if row["kind"] == "user" and row["owner"] == "user"]
        directives = [row["text"] for row in rows if row["kind"] == "directive" and row["owner"] == "user"]
    finally:
        store.close()
    legacy = []
    try:
        from .graph import GraphMemoryStore
        graph = GraphMemoryStore(db_path)
        try:
            for branch in ("user", "directives", "legacy"):
                queue = [branch]
                seen = set()
                while queue:
                    node_id = queue.pop(0)
                    if node_id in seen:
                        continue
                    seen.add(node_id)
                    node = graph.get_node(node_id)
                    if node and node.data:
                        legacy.extend(line.strip() for line in node.data.splitlines() if line.strip())
                    queue.extend(child.id for child in graph.get_children(node_id))
        finally:
            graph.close()
    except Exception as exc:
        debug_log(f"legacy profile read failed: {exc}", "memory")
    return {
        "user": _whole_lines(user, user_max_chars),
        "directives": _whole_lines(directives, directives_max_chars),
        "legacy": "Unverified legacy memory (reference only):\n" + _whole_lines(legacy, user_max_chars)
        if legacy else "",
    }


def format_fact_warm_profile_block(profile: dict[str, str]) -> str:
    parts = []
    if profile.get("user"):
        parts.append("INFORMATION THE USER HAS SHARED IN PRIOR CONVERSATIONS\n" + profile["user"])
    if profile.get("directives"):
        parts.append("STANDING INSTRUCTIONS FROM THE USER\n" + profile["directives"])
    if profile.get("legacy"):
        parts.append(profile["legacy"])
    return "\n\n".join(parts)


def recall_evidence(db, cfg, query: str, search_params: dict, max_tokens: int,
                    *, timeout_sec: float = 10.0) -> str:
    """Rank source-labelled fact and diary candidates within one prompt budget."""
    mode = getattr(cfg, "memory_enrichment_source", "all")
    if mode not in {"all", "diary", "graph"}:
        return ""
    words = search_params.get("keywords") or [query]
    search_text = " ".join(str(word) for word in words)
    query_vec = None
    if getattr(cfg, "embedding_model", ""):
        try:
            from ..llm import get_embedding_backend
            query_vec = get_embedding_backend(cfg).embed(search_text, cfg.embedding_model, timeout_sec=timeout_sec)
        except Exception as exc:
            debug_log(f"fact recall embedding unavailable: {exc}", "memory")
    raw_source_types = search_params.get("source_types")
    source_types = set(raw_source_types) if isinstance(raw_source_types, (list, tuple, set)) else None
    facts = []
    if mode in {"all", "graph"} and (source_types is None or source_types & {"dialogue", "manual", "web", "tool"}):
        store = FactStore(db.db_path)
        try:
            facts = store.search_facts(search_text, query_vector=query_vec,
                                       from_time=search_params.get("from"), to_time=search_params.get("to"),
                                       as_of=search_params.get("as_of"), source_types=source_types, top_k=30)
        finally:
            store.close()
    diary = []
    if mode in {"all", "diary"} and (source_types is None or "diary" in source_types):
        diary = db.search_hybrid(search_text, json.dumps(query_vec) if query_vec else None, top_k=30)
        diary = [row for row in diary if (
            (not search_params.get("from") or row["text"][1:11] >= search_params["from"][:10]) and
            (not search_params.get("to") or row["text"][1:11] <= search_params["to"][:10])
        )]
    legacy = []
    if mode in {"all", "graph"} and (source_types is None or "legacy" in source_types):
        try:
            from .graph import GraphMemoryStore
            graph = GraphMemoryStore(db.db_path)
            try:
                for node in graph.search_nodes(search_text, limit=10):
                    if node.data:
                        ancestors = graph.get_ancestors(node.id)
                        path = " > ".join(parent.name for parent in ancestors)
                        legacy.append((node, path))
            finally:
                graph.close()
        except Exception as exc:
            debug_log(f"legacy graph recall failed: {exc}", "memory")
    candidates = []
    for rank, fact in enumerate(facts, 1):
        source = fact["source"]
        label = "User statement" if fact["owner"] == "user" else "Attributed claim"
        line = f"[{label}; {fact['observed_at'][:10]}; {source['source_type']}; evidence: {source['evidence']}] {fact['text']}"
        weight = 1.05 if fact["owner"] == "user" else 0.9
        candidates.append((weight / (60 + rank), line))
    for rank, row in enumerate(diary, 1):
        candidates.append((1.0 / (60 + rank), f"[Diary summary; reference only] {row['text']}"))
    legacy_terms = set(re.findall(r"\w+", search_text.casefold(), flags=re.UNICODE))
    for rank, (node, path) in enumerate(legacy, 1):
        matching_lines = []
        for index, line in enumerate(node.data.splitlines()):
            folded = line.casefold()
            line_terms = set(re.findall(r"\w+", folded, flags=re.UNICODE))
            overlap = sum(term in line_terms or (not term.isascii() and term in folded)
                          for term in legacy_terms)
            if line.strip() and overlap:
                matching_lines.append((overlap, index, line.strip()))
        matching_lines.sort(key=lambda item: (-item[0], item[1]))
        for _, _, line in matching_lines[:5]:
            candidates.append((0.85 / (60 + rank),
                               f"[Unverified legacy graph; {path}; last edited {node.updated_at[:10]}] {line}"))
    candidates.sort(key=lambda item: -item[0])
    limit = max(0, max_tokens) * 4
    lines = []
    for _, line in candidates:
        if len(line) <= limit:
            lines.append(line)
            limit -= len(line) + 1
    return "\n".join(lines)
