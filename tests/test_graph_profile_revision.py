"""Profile revisions survive reopen and reflect nested User/Directives writes."""
import sqlite3

import pytest

from jarvis.memory.graph import GraphMemoryStore

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('branch', ['user', 'directives'])
@pytest.mark.parametrize('depth', [1, 4])
def test_nested_changes_advance_persisted_revision_without_touch_invalidation(tmp_path, branch, depth):
    path = str(tmp_path / 'graph.db')
    store = GraphMemoryStore(path)
    try:
        parent = branch
        for index in range(depth):
            node = store.create_node(f'Level {index}', 'Synthetic category', parent_id=parent)
            parent = node.id
        before = store.get_warm_profile_revision()
        store.conn.execute('UPDATE memory_nodes SET data = ? WHERE id = ?', ('A durable fact', node.id))
        store.conn.commit()
        changed = store.get_warm_profile_revision()
        assert changed > before
        store.touch_node(node.id)
        assert store.get_warm_profile_revision() == changed
    finally:
        store.close()
    reopened = GraphMemoryStore(path)
    try:
        assert reopened.get_warm_profile_revision() == changed
        assert reopened.get_node(node.id).data == 'A durable fact'
    finally:
        reopened.close()


def test_existing_graph_data_survives_revision_schema_initialisation(tmp_path):
    path = str(tmp_path / 'graph.db')
    with sqlite3.connect(path) as connection:
        connection.executescript('''CREATE TABLE memory_nodes (
            id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT NOT NULL,
            data TEXT NOT NULL DEFAULT '', parent_id TEXT REFERENCES memory_nodes(id) ON DELETE SET NULL,
            access_count INTEGER NOT NULL DEFAULT 0, last_accessed TEXT NOT NULL,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            data_token_count INTEGER NOT NULL DEFAULT 0
        );''')
        for node_id, parent, data in [('root', None, ''), ('user', 'root', 'Keep this stored fact')]:
            connection.execute('''INSERT INTO memory_nodes
                (id, name, description, data, parent_id, last_accessed, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                (node_id, node_id, 'Existing node', data, parent,
                 '2030-01-01T00:00:00+00:00', '2030-01-01T00:00:00+00:00', '2030-01-01T00:00:00+00:00'))
    store = GraphMemoryStore(path)
    try:
        assert store.get_node('user').data == 'Keep this stored fact'
        before = store.get_warm_profile_revision()
        store.append_to_node('user', 'New fact')
        assert store.get_warm_profile_revision() > before
        assert store.get_node('user').data == 'Keep this stored fact\nNew fact'
    finally:
        store.close()
