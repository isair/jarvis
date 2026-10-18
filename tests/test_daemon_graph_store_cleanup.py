"""Startup graph inspection releases its graph connection without blocking startup."""
import sqlite3
from contextlib import closing
from types import SimpleNamespace

import pytest

from jarvis import daemon
from jarvis.memory import graph
from jarvis.memory.db import Database

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('outcome', ['unchanged', 'root-child', 'root-data', 'read-error', 'write-error'])
def test_startup_releases_graph_store_after_inspection(tmp_path, mock_config, monkeypatch, capsys, outcome):
    path = str(tmp_path / 'synthetic.db')
    mock_config.db_path = path
    mock_config.location_enabled = False
    mock_config.use_stdin = True
    mock_config.dictation_enabled = False
    with closing(Database(path, sqlite_vss_path=None)) as diary:
        diary.upsert_conversation_summary('2026-01-01', 'Synthetic retained diary.', 'synthetic', 'jarvis')
    original_store = graph.GraphMemoryStore
    with closing(original_store(path)) as seed:
        retained = seed.create_node('Retained', 'Synthetic world fact', data='Keep this fact', parent_id='world')
        if outcome == 'root-child':
            seed.create_node('Manual note', 'User-created note', data='Keep this note too', parent_id='root')
        elif outcome == 'root-data':
            seed.update_node('root', data='A manually saved root fact')
        saved_nodes = [tuple(row) for row in seed.conn.execute('SELECT * FROM memory_nodes ORDER BY id')]

    if outcome.endswith('error'):
        def unavailable(store):
            if outcome == 'write-error':
                store.conn.execute('BEGIN IMMEDIATE')
                store.conn.execute('UPDATE memory_nodes SET data = ? WHERE id = ?', ('Uncommitted change', retained.id))
            raise RuntimeError('Synthetic inspection failure')
        monkeypatch.setattr(original_store, 'has_unclassified_root_content', unavailable)

    opened = []
    def open_graph(*args, **kwargs):
        store = original_store(*args, **kwargs)
        opened.append(store)
        return store

    monkeypatch.setattr(graph, 'GraphMemoryStore', open_graph)
    monkeypatch.setattr(daemon, 'load_settings', lambda: mock_config)
    monkeypatch.setattr(daemon, '_install_signal_handlers', lambda: None)
    monkeypatch.setattr(daemon, 'VoiceListener', lambda *args, **kwargs: SimpleNamespace(
        start=lambda: None, stop=lambda: None, join=lambda **kwargs: None,
    ))
    monkeypatch.setattr(daemon, 'create_tts_engine', lambda **kwargs: SimpleNamespace(enabled=False, stop=lambda: None))
    monkeypatch.setattr('jarvis.tools.external.mcp_runtime.shutdown_runtime', lambda: None)
    try:
        daemon._run_daemon(smoke_test=True)
        assert 'SMOKE_TEST_INIT_OK' in capsys.readouterr().out
        assert opened
        for store in opened:
            with pytest.raises(sqlite3.ProgrammingError, match='closed'):
                store.conn.execute('SELECT 1')
        with closing(sqlite3.connect(path, timeout=0.1)) as check:
            check.execute('BEGIN IMMEDIATE')
            assert check.execute('SELECT summary FROM conversation_summaries').fetchall() == [('Synthetic retained diary.',)]
            assert check.execute('SELECT * FROM memory_nodes ORDER BY id').fetchall() == saved_nodes
            check.rollback()
    finally:
        for store in opened:
            store.close()
