"""Failed diary initialisation releases its SQLite transaction and handle."""

from contextlib import closing
import sqlite3
import pytest
from jarvis.memory.db import Database

@pytest.mark.unit
@pytest.mark.parametrize("stage", ["vector", "schema"])
@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_failed_initialisation_closes_diary_connection(tmp_path, monkeypatch, stage, failure):
    path = str(tmp_path / 'diary.db')
    with closing(Database(path)) as original:
        ident = original.upsert_conversation_summary('2026-01-01', 'A retained diary fact')
    connections = []
    connect = sqlite3.connect
    def track_connect(*args, **kwargs):
        conn = connect(*args, **kwargs)
        connections.append(conn)
        return conn
    def unavailable_index(*args, **kwargs):
        raise failure('Synthetic diary initialisation failure')
    monkeypatch.setattr(sqlite3, 'connect', track_connect)
    if stage == 'vector':
        monkeypatch.setattr('jarvis.utils.vector_store.get_best_vector_store', unavailable_index)
    else:
        def failed_schema(self):
            self.conn.execute('BEGIN IMMEDIATE')
            self.conn.execute('UPDATE conversation_summaries SET summary=? WHERE id=?', ('Uncommitted edit', ident))
            raise failure('Synthetic diary initialisation failure')
        monkeypatch.setattr(Database, '_init_schema', failed_schema)
    with pytest.raises(failure, match='Synthetic diary initialisation'):
        Database(path)
    try:
        with pytest.raises(sqlite3.ProgrammingError, match='closed'):
            connections[0].execute('SELECT 1')
        with closing(connect(path, timeout=0)) as observer:
            observer.execute('BEGIN IMMEDIATE')
            assert observer.execute('SELECT summary FROM conversation_summaries WHERE id=?', (ident,)).fetchone()[0] == 'A retained diary fact'
            observer.rollback()
    finally:
        for conn in connections:
            conn.close()
