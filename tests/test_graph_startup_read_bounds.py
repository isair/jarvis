"""Startup recognises saved root text without eagerly loading whole notes."""
import gc
import sqlite3
import tracemalloc

import pytest

from jarvis.memory.graph import GraphMemoryStore

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('encoding', ['UTF-8', 'UTF-16le', 'UTF-16be'])
@pytest.mark.parametrize('text, expected', [
    ('\u2003\t\n ' * 3000, False),
    ('\u2003\t\n ' * 3000 + '保管', True),
    (' \x00 ', True),
    (None, False),
])
def test_root_inspection_handles_unicode_and_nulls_without_mutation(tmp_path, encoding, text, expected):
    path = tmp_path / 'graph.db'
    with sqlite3.connect(path) as connection:
        connection.execute(f"PRAGMA encoding='{encoding}'")
        connection.execute('CREATE TABLE encoding_marker (value TEXT)')
    store = GraphMemoryStore(str(path))
    try:
        if text is None:
            with pytest.raises(sqlite3.IntegrityError):
                store.conn.execute("UPDATE memory_nodes SET data=? WHERE id='root'", (text,))
            store.conn.rollback()
        else:
            store.conn.execute("UPDATE memory_nodes SET data=? WHERE id='root'", (text,))
            store.conn.commit()
        before = list(store.conn.iterdump())
        assert store.has_unclassified_root_content() is expected
        assert list(store.conn.iterdump()) == before
    finally:
        store.close()


@pytest.mark.parametrize('suffix,expected', [('', False), ('A saved fact', True)])
def test_root_inspection_has_bounded_python_allocations(tmp_path, suffix, expected):
    store = GraphMemoryStore(str(tmp_path / 'graph.db'))
    text = ' ' * (8 * 1024 * 1024) + suffix
    try:
        store.update_node('root', data=text)
        gc.collect()
        tracemalloc.start()
        try:
            assert store.has_unclassified_root_content() is expected
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert peak < len(text) // 8
    finally:
        store.close()


def test_root_inspection_does_not_commit_the_callers_pending_changes(tmp_path):
    path = tmp_path / 'graph.db'
    store = GraphMemoryStore(str(path))
    try:
        store.conn.execute("UPDATE memory_nodes SET data='Pending root fact' WHERE id='root'")
        assert store.has_unclassified_root_content()
        with sqlite3.connect(path) as observer:
            assert observer.execute("SELECT data FROM memory_nodes WHERE id='root'").fetchone()[0] == ''
        store.conn.rollback()
        assert not store.has_unclassified_root_content()
    finally:
        store.close()
