"""Committed graph edits from other connections reach the next reply context."""
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
        yield store, lambda: engine.run_reply_engine(db, mock_config, None, 'Check the stored fact.', dialogue_memory), mock_config
    finally:
        store.close()


@pytest.mark.parametrize('branch', ['user', 'directives'])
@pytest.mark.parametrize('change', ['update', 'delete', 'insert', 'move_out', 'move_in'])
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
