"""Opening an initialised graph does not require an unnecessary writer lock."""
import sqlite3
from contextlib import closing

import pytest

from jarvis.memory import graph

pytestmark = pytest.mark.unit


def test_initialised_graph_opens_and_reads_while_an_independent_writer_is_active(tmp_path, monkeypatch):
    path = str(tmp_path / 'graph.db')
    with closing(graph.GraphMemoryStore(path)) as seed:
        seed.conn.execute('PRAGMA journal_mode=WAL')
        fact = seed.create_node('Cafe Aurora', 'Opening time', data='Opens at 06:35', parent_id='world')
        revision = seed.get_warm_profile_revision()
    connect = sqlite3.connect
    def fail_fast_connect(*args, **kwargs):
        kwargs.setdefault('timeout', 0.01)
        return connect(*args, **kwargs)
    monkeypatch.setattr(graph.sqlite3, 'connect', fail_fast_connect)
    with closing(connect(path)) as writer:
        writer.execute('BEGIN IMMEDIATE')
        writer.execute('UPDATE memory_nodes SET data = ? WHERE id = ?', ('Pending independent change', fact.id))
        with closing(graph.GraphMemoryStore(path)) as reader:
            assert reader.get_node(fact.id).data == fact.data
            assert reader.get_warm_profile_revision() == revision
            writer.rollback()
            reader.append_to_node(fact.id, 'Closed on Sunday')
            assert reader.get_node(fact.id).data == fact.data + '\nClosed on Sunday'


@pytest.mark.parametrize('missing', ['revision', 'branch'])
def test_bootstrap_repairs_missing_seed_data_without_rewriting_existing_facts(tmp_path, missing):
    path = str(tmp_path / 'graph.db')
    with closing(graph.GraphMemoryStore(path)) as seed:
        fact = seed.create_node('Retained', 'Synthetic fact', data='Keep this fact', parent_id='world')
        seed.touch_node(fact.id)
        fact = seed.get_node(fact.id)
        if missing == 'revision':
            seed.conn.execute("DELETE FROM memory_graph_revisions WHERE name = 'warm_profile'")
        else:
            seed.conn.execute('DELETE FROM memory_nodes WHERE id = ?', (graph.BRANCH_DIRECTIVES,))
        seed.conn.commit()
    with closing(graph.GraphMemoryStore(path)) as repaired:
        assert repaired.get_node(fact.id) == fact
        assert repaired.get_node(graph.BRANCH_DIRECTIVES).parent_id == 'root'
        revision = repaired.get_warm_profile_revision()
    with closing(graph.GraphMemoryStore(path)) as reopened:
        assert reopened.get_node(fact.id) == fact
        assert reopened.get_warm_profile_revision() == revision
