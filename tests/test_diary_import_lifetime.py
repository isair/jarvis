"""Diary import releases its owned connection on every stream exit."""
from contextlib import closing
import json
import sqlite3
from types import SimpleNamespace

import pytest

from desktop_app import memory_viewer
from jarvis import config
from jarvis.memory import db as database, graph_ops

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures('no_network')]


@pytest.mark.parametrize('exit_path', [
    'empty', 'complete', 'read-error', 'graph-error',
    'cancel-before-graph', 'cancel-midway',
])
def test_import_releases_diary_connection(tmp_path, mock_config, monkeypatch, exit_path):
    path = str(tmp_path / 'synthetic.db')
    original_database = database.Database
    with closing(original_database(path, sqlite_vss_path=None)) as seed:
        if exit_path != 'empty':
            seed.upsert_conversation_summary('2026-01-01', 'Synthetic user fact', None, 'jarvis')
            seed.upsert_conversation_summary('2026-01-02', 'Another synthetic fact', None, 'jarvis')

    opened = []

    def open_diary(*args, **kwargs):
        diary = original_database(*args, **kwargs)
        opened.append(diary)
        if exit_path == 'read-error':
            def unavailable():
                raise RuntimeError('Synthetic diary read failure')
            monkeypatch.setattr(diary, 'get_all_conversation_summaries', unavailable)
        return diary

    def graph_store():
        if exit_path == 'graph-error':
            raise RuntimeError('Synthetic graph open failure')
        return object()

    monkeypatch.setattr(database, 'Database', open_diary)
    monkeypatch.setattr(config, 'load_settings', lambda: mock_config)
    monkeypatch.setattr(memory_viewer, '_get_db_path', lambda: path)
    monkeypatch.setattr(memory_viewer, 'get_graph_store', graph_store)
    monkeypatch.setattr(graph_ops, 'update_graph_from_dialogue',
                        lambda **kwargs: SimpleNamespace(stored=[]))
    monkeypatch.setitem(memory_viewer.app.config, 'TESTING', True)

    try:
        with memory_viewer.app.test_client().post('/api/graph/import-diary') as response:
            assert response.status_code == 200
            if exit_path.startswith('cancel'):
                stream = iter(response.response)
                assert json.loads(next(stream))['type'] == 'start'
                if exit_path == 'cancel-midway':
                    assert json.loads(next(stream))['type'] == 'progress'
            else:
                events = [json.loads(line) for line in response.data.splitlines()]
                terminal = 'error' if exit_path.endswith('error') else 'complete'
                assert events[-1]['type'] == terminal
        assert opened
        for diary in opened:
            with pytest.raises(sqlite3.ProgrammingError, match='closed'):
                diary.conn.execute('SELECT 1')
    finally:
        for diary in opened:
            diary.close()
