"""Bounded, consented source refreshes with atomic graph/version updates."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import uuid
from pathlib import Path

from .imports import (DESTINATIONS, MAX_ENTRIES,
                      MemoryImportError, MemoryImportRecord, MemorySource,
                      _digest, source_identity)
from ..debug import debug_log
from ..utils.redact import scrub_secrets

MAX_AUTO_SOURCES = 32


def node_fingerprint(node):
    """Exclude read/access bookkeeping, retain every user-editable field."""
    return _digest(json.dumps([node.name, node.description, node.data, node.parent_id,
                              node.created_at, node.updated_at], ensure_ascii=False))


@dataclass(frozen=True)
class SourceGrant:
    source_id: str
    source: MemorySource
    generation: int
    digest: str
    can_enable: bool
    state: str
    error: str


def _grant(row):
    source = json.loads(row['source_json'])
    source['path'] = Path(source['path']) if source['path'] else None
    return SourceGrant(row['source_id'], MemorySource(**source), row['generation'],
                       row['digest'], bool(row['can_enable']), row['state'], row['error'])


def load_sources(store, *, include_disabled=False):
    """Read active grants and a bounded recent history without scanning tombstones."""
    columns = 'source_id,source_json,state,generation,digest,can_enable,error'
    with store._lock:
        active = list(store.conn.execute(f"SELECT {columns} FROM memory_auto_sources WHERE state='enabled' ORDER BY rowid DESC LIMIT ?", (MAX_AUTO_SOURCES,)))
        if include_disabled:
            recent = store.conn.execute(f"SELECT {columns} FROM memory_auto_sources NOT INDEXED WHERE state<>'enabled' ORDER BY rowid DESC LIMIT ?", (MAX_AUTO_SOURCES,))
            active.extend(recent)
        return [_grant(row) for row in active]


def record_success(store, preview, choices, result):
    """Offer consent after a successful import; never widen a partial review."""
    identity = source_identity(preview.source)
    originals = {item.id: item.text for item in preview.items}
    complete = (len(choices) == len(originals) == len(result.node_ids) and
                {c.item_id for c in choices} == set(originals) and
                all(c.text == originals[c.item_id] for c in choices) and
                len({c.branch for c in choices}) == 1)
    source = asdict(preview.source)
    source['path'] = str(preview.source.path) if preview.source.path else None
    branch = choices[0].branch if complete else None
    bindings = [dict(text=c.text, node_id=n, date=c.source_date)
                for c, n in zip(choices, result.node_ids)] if complete else []
    with store._lock, store.conn:
        store.conn.execute('BEGIN IMMEDIATE')
        existing = store.conn.execute('SELECT * FROM memory_auto_sources WHERE source_id=?', (identity,)).fetchone()
        if existing is not None and existing['state'] not in ('pending', 'review'):
            return None

        complete = complete and all(store.conn.execute('SELECT 1 FROM memory_nodes WHERE id=? AND data=?', (b['node_id'], b['text'])).fetchone() is not None and store._resolve_branch(b['node_id']) == branch for b in bindings)
        store.conn.execute("""INSERT INTO memory_auto_sources
            (source_id,source_json,state,digest,branch,bindings,can_enable) VALUES (?,?,?,?,?,?,?)
            ON CONFLICT(source_id) DO UPDATE SET source_json=excluded.source_json,
            state=excluded.state,generation=memory_auto_sources.generation+1,
            digest=excluded.digest,branch=excluded.branch,bindings=excluded.bindings,
            can_enable=excluded.can_enable,error=''""",
            (identity, json.dumps(source), 'pending', preview.source_digest, branch,
             json.dumps(bindings), int(complete and preview.source.path is not None)))
        row = store.conn.execute('SELECT * FROM memory_auto_sources WHERE source_id=?', (identity,)).fetchone()
    return _grant(row)


def begin_setup(store, identity):
    """An explicit setup action invalidates consent before another full review."""
    with store._lock, store.conn:
        store.conn.execute("UPDATE memory_auto_sources SET state='pending',generation=generation+1 WHERE source_id=?", (identity,))


def decide(store, identity, enabled):
    """Increment the consent generation, invalidating already-running refreshes."""
    with store._lock, store.conn:
        store.conn.execute('BEGIN IMMEDIATE')
        row = store.conn.execute('SELECT * FROM memory_auto_sources WHERE source_id=?', (identity,)).fetchone()
        if row is None or (enabled and not row['can_enable']):
            raise MemoryImportError('Review the complete local file with one destination before enabling automatic import.')
        if enabled and row['state'] != 'enabled' and store.conn.execute("SELECT COUNT(*) FROM memory_auto_sources WHERE state='enabled'").fetchone()[0] >= MAX_AUTO_SOURCES:
            raise MemoryImportError(f'Automatic import is already watching {MAX_AUTO_SOURCES} sources. Disable another source before enabling this one.')
        state = 'enabled' if enabled else 'disabled'
        store.conn.execute("UPDATE memory_auto_sources SET state=?,generation=generation+1,error='' WHERE source_id=?", (state, identity))
        row = store.conn.execute('SELECT * FROM memory_auto_sources WHERE source_id=?', (identity,)).fetchone()
    debug_log(f'📥 Automatic memory import {state}', 'memory')
    return _grant(row)


def _pristine(store, identity, binding):
    node = store.get_node(binding['node_id'])
    if node is None:
        return False
    owned = store.conn.execute('SELECT * FROM memory_import_owned WHERE node_id=?', (node.id,)).fetchone()
    if owned is None or owned['source_id'] != identity or owned['fingerprint'] != node_fingerprint(node):
        return False
    if store.conn.execute('SELECT 1 FROM memory_nodes WHERE parent_id=? LIMIT 1', (node.id,)).fetchone():
        return False
    if store.conn.execute('SELECT 1 FROM memory_import_node_sources link JOIN memory_imports i ON i.id=link.import_id WHERE link.node_id=? AND i.source_id<>? LIMIT 1', (node.id, identity)).fetchone():
        return False
    return True


def sync_source(store, identity, generation, preview):
    """Commit a bounded source version, or preserve the graph for human review."""
    if source_identity(preview.source) != identity:
        raise MemoryImportError('The selected source does not match this automatic import.')
    mutations = []
    with store._lock, store.conn:
        # Consent is rechecked under the same writer lock as the memory changes.
        row = store.conn.execute('SELECT * FROM memory_auto_sources WHERE source_id=?', (identity,)).fetchone()
        if row is None or row['state'] != 'enabled' or row['generation'] != generation:
            return 'revoked'
        if row['digest'] == preview.source_digest:
            return 'unchanged'
        store.conn.execute('BEGIN IMMEDIATE')
        row = store.conn.execute('SELECT * FROM memory_auto_sources WHERE source_id=?', (identity,)).fetchone()
        if row['state'] != 'enabled' or row['generation'] != generation:
            return 'revoked'
        if row['digest'] == preview.source_digest:
            return 'unchanged'
        saved_source = _grant(row).source
        if preview.source != saved_source or row['branch'] not in DESTINATIONS or len(preview.items) > MAX_ENTRIES:
            raise MemoryImportError('Review this source again before automatic import.')
        previous = json.loads(row['bindings'])
        incoming = {item.text: item for item in preview.items}
        removed = [b for b in previous if b['text'] not in incoming]
        if any(not _pristine(store, identity, b) for b in removed):
            store.conn.execute("UPDATE memory_auto_sources SET state='review',generation=generation+1,error=? WHERE source_id=?",
                               ('Changes overlap memories edited, moved, shared or deleted in Jarvis. Review this file again.', identity))
            debug_log('📥 Automatic memory import paused for review', 'memory')
            return 'review'
        kept = [b for b in previous if b['text'] in incoming]
        for node_id in {b['node_id'] for b in removed}:
            store.conn.execute('INSERT OR IGNORE INTO memory_import_retired SELECT id FROM memory_imports WHERE node_id=?', (node_id,))
            store.conn.execute('DELETE FROM memory_nodes WHERE id=?', (node_id,))
            mutations.append(('delete', node_id, row['branch']))
        known = {b['text'] for b in kept}
        version = uuid.uuid4().hex
        records = [MemoryImportRecord(identity, preview.source_digest, version + item.id,
                    saved_source.provider, scrub_secrets(saved_source.label)[:256], saved_source.scope,
                    None, json.dumps(dict(preview.source_metadata)), item.text, item.text, row['branch'])
                   for item in preview.items if item.text not in known]
        if records:
            result, created = store._import_memories_locked(records, match_unimported=False)
            kept.extend(dict(text=record.reviewed_text, node_id=node_id, date=None)
                        for record, node_id in zip(records, result.node_ids))
            mutations.extend(('create', node.id, branch) for node, branch in created)
        store.conn.execute('UPDATE memory_auto_sources SET digest=?,bindings=?,error=? WHERE source_id=?',
                           (preview.source_digest, json.dumps(kept), '', identity))
    from .graph import _notify_graph_mutation
    for action, node_id, branch in mutations:
        _notify_graph_mutation(action, node_id, branch)
    debug_log('📥 Automatic memory import refreshed a selected source', 'memory')
    return 'updated'
