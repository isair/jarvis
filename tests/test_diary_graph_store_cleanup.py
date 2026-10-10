"""Every diary graph pass releases its temporary database connection."""
import sqlite3
from contextlib import closing
from datetime import datetime, timezone

import pytest

from jarvis.memory import conversation, graph, graph_ops
from jarvis.memory.db import Database

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('outcome', ['stored', 'update-error', 'no-summary', 'summary-read-error'])
def test_diary_save_releases_graph_store_for_every_graph_outcome(tmp_path, mock_config, dialogue_memory, monkeypatch, outcome):
    original_store = graph.GraphMemoryStore
    opened = []

    def open_graph(path):
        store = original_store(path)
        opened.append(store)
        return store

    monkeypatch.setattr(graph, 'GraphMemoryStore', open_graph)
    summary = 'The user has a cat named Juniper.'
    monkeypatch.setattr(conversation, 'generate_conversation_summary', lambda *args, **kwargs: (summary, 'pets'))
    dialogue_memory.add_message('user', 'My cat is called Juniper.')
    dialogue_memory.add_message('assistant', 'You have a cat called Juniper.')
    with closing(Database(str(tmp_path / 'synthetic.db'), sqlite_vss_path=None)) as db:
        read_summary = db.get_conversation_summary
        if outcome == 'no-summary':
            monkeypatch.setattr(db, 'get_conversation_summary', lambda *args: None)
        elif outcome == 'summary-read-error':
            def unavailable_after_save(*args):
                row = read_summary(*args)
                if row is not None:
                    raise RuntimeError('Synthetic graph read failure')
                return row
            monkeypatch.setattr(db, 'get_conversation_summary', unavailable_after_save)

        def update_graph(*, store, **kwargs):
            if outcome == 'update-error':
                raise RuntimeError('Synthetic graph update failure')
            node = store.create_node('Pet', 'Synthetic pet fact', data=summary, parent_id='user')
            return graph_ops.GraphUpdateResult(stored=[(summary, node.name)], skipped=0)

        monkeypatch.setattr(graph_ops, 'update_graph_from_dialogue', update_graph)
        try:
            summary_id = conversation.update_diary_from_dialogue_memory(db, dialogue_memory, mock_config, force=True)
            row = read_summary(datetime.now(timezone.utc).date().isoformat(), 'jarvis')
            assert summary_id is not None
            assert row['summary'] == summary
            assert dialogue_memory.get_pending_chunks() == []
            assert opened
            for store in opened:
                with pytest.raises(sqlite3.ProgrammingError, match='closed'):
                    store.conn.execute('SELECT 1')
            with closing(original_store(db.db_path)) as reader:
                saved_facts = {node.data for node in reader.get_children('user')}
                assert (summary in saved_facts) == (outcome == 'stored')
        finally:
            for store in opened:
                store.close()
