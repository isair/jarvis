"""Failed graph bootstrap releases its SQLite connection and writer lock."""
import sqlite3

import pytest

from jarvis.memory import graph

pytestmark = pytest.mark.unit


@pytest.fixture
def captured_connections(monkeypatch):
    original = sqlite3.connect
    opened = []

    def connect(*args, **kwargs):
        connection = original(*args, **kwargs)
        opened.append(connection)
        return connection

    monkeypatch.setattr(graph.sqlite3, 'connect', connect)
    try:
        yield original, opened
    finally:
        for connection in opened:
            connection.close()


def test_schema_failure_closes_connection_and_preserves_existing_rows(tmp_path, captured_connections):
    connect, opened = captured_connections
    path = str(tmp_path / 'malformed.db')
    with connect(path) as seed:
        seed.execute('CREATE TABLE memory_nodes (id TEXT PRIMARY KEY)')
        seed.execute('INSERT INTO memory_nodes VALUES (?)', ('Existing synthetic row',))
    seed.close()
    with pytest.raises(sqlite3.OperationalError, match='no such column'):
        graph.GraphMemoryStore(path)
    failed = opened[-1]
    with pytest.raises(sqlite3.ProgrammingError, match='closed'):
        failed.execute('SELECT 1')
    with connect(path) as check:
        assert check.execute('SELECT id FROM memory_nodes').fetchall() == [('Existing synthetic row',)]
    check.close()


def test_seed_failure_closes_connection_releases_lock_and_allows_recovery(tmp_path, captured_connections):
    connect, opened = captured_connections
    path = str(tmp_path / 'graph.db')
    seed = graph.GraphMemoryStore(path)
    retained = seed.create_node('Retained', 'Synthetic fact', data='Keep this fact', parent_id=graph.BRANCH_WORLD)
    branch_id = graph.FIXED_BRANCHES[0][0]
    seed.conn.execute('DELETE FROM memory_nodes WHERE id = ?', (branch_id,))
    seed.conn.commit()
    seed.conn.executescript("""CREATE TRIGGER reject_bootstrap BEFORE INSERT ON memory_nodes
        BEGIN SELECT RAISE(ABORT, 'synthetic seed rejected'); END;""")
    seed.close()
    with pytest.raises(sqlite3.IntegrityError, match='synthetic seed rejected'):
        graph.GraphMemoryStore(path)
    failed = opened[-1]
    with pytest.raises(sqlite3.ProgrammingError, match='closed'):
        failed.execute('SELECT 1')
    with connect(path, timeout=0.1) as repair:
        repair.execute('DROP TRIGGER reject_bootstrap')
    repair.close()
    recovered = graph.GraphMemoryStore(path)
    try:
        assert recovered.get_node(branch_id) is not None
        assert recovered.get_node(retained.id).data == retained.data
    finally:
        recovered.close()
