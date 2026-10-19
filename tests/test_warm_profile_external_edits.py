"""Committed graph edits from other connections reach the next reply context."""
import sqlite3

import pytest

from jarvis.memory.graph import GraphMemoryStore
from jarvis.reply import engine

pytestmark = pytest.mark.unit


@pytest.fixture
def reply_context(monkeypatch, tmp_path, mock_config, db, dialogue_memory):
    mock_config.db_path = str(tmp_path / 'graph.db')
    mock_config.llm_chat_model = 'gpt-oss:20b'
    mock_config.ollama_chat_model = mock_config.llm_chat_model
    mock_config.memory_enrichment_source = 'diary'
    mock_config.memory_digest_enabled = False
    monkeypatch.setattr(engine, 'plan_query', lambda **kwargs: ['Reply to the user.'])
    monkeypatch.setattr(engine, 'select_tools', lambda **kwargs: [])
    monkeypatch.setattr(engine, '_live_time_location_string', lambda cfg: '')

    def chat(cfg, messages, **kwargs):
        system = '\n'.join(message['content'] for message in messages if message['role'] == 'system')
        value = next((marker for marker in ('Beta', 'Alpha') if f'Cache-visible fact: {marker}' in system), 'Absent')
        return {'message': {'content': f'Supplied memory: {value}'}}

    monkeypatch.setattr(engine, 'chat_with_messages', chat)
    store = GraphMemoryStore(mock_config.db_path)
    try:
        yield store, lambda text='Check the stored fact.': engine.run_reply_engine(db, mock_config, None, text, dialogue_memory), mock_config
    finally:
        store.close()


@pytest.mark.parametrize('branch', ['user', 'directives'])
@pytest.mark.parametrize('change', ['update', 'delete', 'insert', 'move_out', 'move_in', 'replace_out'])
def test_external_edit_is_visible_in_same_conversation(reply_context, branch, change):
    store, reply, _ = reply_context
    node = store.create_node('Marker', 'Synthetic fact', data='Cache-visible fact: Alpha',
                             parent_id='world' if change in ('move_in', 'insert') else branch)
    assert reply() == f"Supplied memory: {'Absent' if change in ('move_in', 'insert') else 'Alpha'}"
    if change == 'update':
        store.conn.execute('UPDATE memory_nodes SET data = ? WHERE id = ?', ('Cache-visible fact: Beta', node.id))
        expected = 'Beta'
    elif change == 'insert':
        store.conn.execute('''INSERT INTO memory_nodes
            (id, name, description, data, parent_id, last_accessed, created_at, updated_at)
            SELECT id || '-copy', name, description, data, ?, last_accessed, created_at, updated_at
            FROM memory_nodes WHERE id = ?''', (branch, node.id))
        expected = 'Alpha'
    elif change == 'replace_out':
        store.conn.execute('''INSERT OR REPLACE INTO memory_nodes
            (id, name, description, data, parent_id, last_accessed, created_at, updated_at)
            SELECT id, name, description, data, 'world', last_accessed, created_at, updated_at
            FROM memory_nodes WHERE id = ?''', (node.id,))
        expected = 'Absent'
    elif change == 'delete':
        store.conn.execute('DELETE FROM memory_nodes WHERE id = ?', (node.id,))
        expected = 'Absent'
    else:
        parent = branch if change == 'move_in' else 'world'
        store.conn.execute('UPDATE memory_nodes SET parent_id = ? WHERE id = ?', (parent, node.id))
        expected = 'Alpha' if change == 'move_in' else 'Absent'
    store.conn.commit()
    assert reply() == f'Supplied memory: {expected}'


@pytest.mark.parametrize('world_edit', [False, True])
def test_unchanged_warm_profile_reuses_cache_when_traversal_is_unavailable(reply_context, monkeypatch, world_edit):
    store, reply, _ = reply_context
    store.create_node('Marker', 'Synthetic fact', data='Cache-visible fact: Alpha', parent_id='user')
    world = store.create_node('World marker', 'External fact', parent_id='world')
    assert reply() == 'Supplied memory: Alpha'
    if world_edit:
        store.conn.execute('UPDATE memory_nodes SET data = ? WHERE id = ?', ('Changed world fact', world.id))
        store.conn.commit()

    def unavailable(store):
        raise RuntimeError('Synthetic profile traversal unavailable')

    monkeypatch.setattr('jarvis.memory.graph_ops.build_warm_profile', unavailable)
    assert reply() == 'Supplied memory: Alpha'


def test_changed_profile_is_not_reused_when_rebuilding_fails(reply_context, monkeypatch):
    store, reply, _ = reply_context
    node = store.create_node('Marker', 'Synthetic fact', data='Cache-visible fact: Alpha', parent_id='user')
    assert reply() == 'Supplied memory: Alpha'
    store.conn.execute('UPDATE memory_nodes SET data = ? WHERE id = ?', ('Cache-visible fact: Beta', node.id))
    store.conn.commit()

    def unavailable(store):
        raise RuntimeError('Synthetic profile traversal unavailable')

    monkeypatch.setattr('jarvis.memory.graph_ops.build_warm_profile', unavailable)
    assert reply() == 'Supplied memory: Absent'


def test_switching_graph_files_cannot_reuse_another_profile(reply_context, tmp_path):
    store, reply, cfg = reply_context
    store.create_node('Marker', 'Synthetic fact', data='Cache-visible fact: Alpha', parent_id='user')
    assert reply() == 'Supplied memory: Alpha'
    other = GraphMemoryStore(str(tmp_path / 'other.db'))
    try:
        other.create_node('Marker', 'Synthetic fact', data='Cache-visible fact: Beta', parent_id='user')
        assert store.get_warm_profile_revision() == other.get_warm_profile_revision()
        cfg.db_path = other.db_path
        assert reply() == 'Supplied memory: Beta'
    finally:
        other.close()


@pytest.mark.parametrize('cached', [False, True])
def test_profile_remains_available_during_an_unrelated_writer(reply_context, cached):
    store, reply, _ = reply_context
    store.conn.execute('PRAGMA journal_mode=WAL')
    store.create_node('Marker', 'Synthetic fact', data='Cache-visible fact: Alpha', parent_id='user')
    world = store.create_node('World marker', 'External fact', parent_id='world')
    if cached:
        assert reply() == 'Supplied memory: Alpha'
    store.conn.execute('BEGIN IMMEDIATE')
    try:
        store.conn.execute('UPDATE memory_nodes SET data = ? WHERE id = ?', ('Uncommitted world fact', world.id))
        assert reply() == 'Supplied memory: Alpha'
    finally:
        store.conn.rollback()


@pytest.fixture
def graph_recall_context(reply_context, monkeypatch):
    store, reply, cfg = reply_context
    cfg.memory_enrichment_source = 'graph'
    store.create_node('Marker', 'Synthetic fact', data='Cache-visible fact: Alpha', parent_id='world')
    monkeypatch.setattr(engine, 'plan_query', lambda **kwargs: ["searchMemory topic='stored fact'", 'Reply to the user.'])
    monkeypatch.setattr(engine, 'extract_search_params_for_memory', lambda *args, **kwargs: {
        'keywords': ['cache-visible', 'fact'], 'questions': ['Which cache-visible fact was recorded?'],
    })
    return store, lambda: reply('Please check which cache-visible fact was recorded in saved graph memory.')



@pytest.mark.parametrize('path', ['normal', 'failure', 'short_question'])
def test_graph_reply_readers_close_after_every_retrieval_path(graph_recall_context, monkeypatch, path):
    store, reply = graph_recall_context
    opened = []
    store_type = type(store)

    def open_store(*args, **kwargs):
        reader = store_type(*args, **kwargs)
        opened.append(reader.conn)
        return reader

    monkeypatch.setattr('jarvis.memory.graph.GraphMemoryStore', open_store)
    if path == 'failure':
        def unavailable(*args):
            raise RuntimeError('Synthetic ancestry unavailable')
        monkeypatch.setattr(store_type, 'get_ancestors', unavailable)
    elif path == 'short_question':
        monkeypatch.setattr(engine, 'extract_search_params_for_memory', lambda *args, **kwargs: {
            'keywords': [], 'questions': ['what is the?'],
        })
    marker = store.find_node_by_name('Marker', parent_id='world')
    previous_access_count = marker.access_count
    assert reply() == f"Supplied memory: {'Alpha' if path == 'normal' else 'Absent'}"
    if path == 'normal':
        assert store.get_node(marker.id).access_count > previous_access_count
    assert opened
    for connection in opened:
        with pytest.raises(sqlite3.ProgrammingError, match='closed'):
            connection.execute('SELECT 1')



def _remove_profile_revision(store, missing):
    if missing == 'table':
        names = [row[0] for row in store.conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'trigger' AND name LIKE 'warm_profile_%'")]
        for name in names:
            store.conn.execute(f'DROP TRIGGER "{name}"')
        store.conn.execute('DROP TABLE memory_graph_revisions')
    else:
        store.conn.execute("DELETE FROM memory_graph_revisions WHERE name = 'warm_profile'")
    store.conn.commit()


@pytest.mark.parametrize('missing', ['table', 'counter'])
@pytest.mark.parametrize('cached', [False, True])
def test_unversioned_graph_profile_remains_available_and_fresh(reply_context, missing, cached):
    store, reply, _ = reply_context
    node = store.create_node('Marker', 'Synthetic fact', data='Cache-visible fact: Alpha', parent_id='user')
    if cached:
        assert reply() == 'Supplied memory: Alpha'
    _remove_profile_revision(store, missing)
    before = tuple(store.conn.iterdump())
    assert reply() == 'Supplied memory: Alpha'
    assert tuple(store.conn.iterdump()) == before, 'Profile reads must not bootstrap or edit saved data'
    store.conn.execute('UPDATE memory_nodes SET data = ? WHERE id = ?', ('Cache-visible fact: Beta', node.id))
    store.conn.commit()
    assert reply() == 'Supplied memory: Beta'


@pytest.mark.parametrize('missing', ['table', 'counter'])
def test_restored_revision_cannot_revive_an_old_profile(reply_context, missing):
    store, reply, _ = reply_context
    node = store.create_node('Marker', 'Synthetic fact', data='Cache-visible fact: Alpha', parent_id='user')
    assert reply() == 'Supplied memory: Alpha'
    revision = store.get_warm_profile_revision()
    _remove_profile_revision(store, missing)
    store.conn.execute('UPDATE memory_nodes SET data = ? WHERE id = ?', ('Cache-visible fact: Beta', node.id))
    store.conn.commit()
    assert reply() == 'Supplied memory: Beta'
    if missing == 'table':
        store.conn.execute('CREATE TABLE memory_graph_revisions (name TEXT PRIMARY KEY, revision INTEGER NOT NULL)')
    store.conn.execute('INSERT INTO memory_graph_revisions(name, revision) VALUES(?, ?)', ('warm_profile', revision))
    store.conn.commit()
    assert reply() == 'Supplied memory: Beta'



def test_unavailable_revision_reader_does_not_reuse_a_cached_profile(reply_context, monkeypatch):
    store, reply, _ = reply_context
    store.create_node('Marker', 'Synthetic fact', data='Cache-visible fact: Alpha', parent_id='user')
    assert reply() == 'Supplied memory: Alpha'
    def unavailable(_store):
        raise sqlite3.OperationalError('Synthetic revision reader unavailable')
    monkeypatch.setattr(GraphMemoryStore, 'get_warm_profile_revision', unavailable)
    assert reply() == 'Supplied memory: Absent'
