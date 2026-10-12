"""Optional graph access tracking does not prevent reading saved facts."""
from contextlib import closing
from concurrent.futures import ThreadPoolExecutor
import sqlite3

import pytest

from jarvis.memory.graph import GraphMemoryStore

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('journal_mode', ['DELETE', 'WAL'])
@pytest.mark.parametrize('tracking', ['available', 'busy', 'read_only', 'rejected'])
@pytest.mark.parametrize('surface', ['search', 'viewer'])
def test_saved_fact_remains_readable_when_tracking_is_unavailable(tmp_path, monkeypatch, tracking, surface, journal_mode):
    from desktop_app import memory_viewer

    path = str(tmp_path / 'synthetic.db')
    with closing(GraphMemoryStore(path)) as owner:
        owner.conn.execute(f'PRAGMA journal_mode={journal_mode}')
        fact = owner.create_node('Cafe Aurora', 'Opening time', data='Opens at 06:35', parent_id='world')
        saved = owner.get_node(fact.id).to_dict()
        if tracking == 'rejected':
            owner.conn.execute("""CREATE TRIGGER reject_access BEFORE UPDATE ON memory_nodes
                WHEN NEW.access_count != OLD.access_count
                BEGIN SELECT RAISE(ABORT, 'Synthetic access failure'); END""")
            owner.conn.commit()
        with closing(GraphMemoryStore(path, read_only=tracking == 'read_only')) as reader:
            configured_wait = 70
            reader.conn.execute(f'PRAGMA busy_timeout = {configured_wait}')
            if tracking == 'busy':
                owner.conn.execute('BEGIN IMMEDIATE')
            try:
                if surface == 'search':
                    matches = reader.search_nodes('Cafe Aurora')
                    assert [(node.id, node.data) for node in matches] == [(fact.id, saved['data'])]
                else:
                    monkeypatch.setattr(memory_viewer, '_graph_store', reader)
                    response = memory_viewer.app.test_client().get(f'/api/graph/node/{fact.id}')
                    assert response.status_code == 200
                    assert response.get_json()['node']['data'] == saved['data']
                assert reader.conn.execute('PRAGMA busy_timeout').fetchone()[0] == configured_wait
                recorded = reader.get_node(fact.id).to_dict()
                if tracking == 'available':
                    assert recorded['access_count'] == saved['access_count'] + 1
                else:
                    assert recorded == saved
            finally:
                owner.conn.rollback()


def test_graph_read_failure_remains_an_error(tmp_path):
    with closing(GraphMemoryStore(str(tmp_path / 'synthetic.db'))) as store:
        store.conn.execute('DROP TABLE memory_nodes')
        store.conn.commit()
        with pytest.raises(sqlite3.OperationalError):
            store.search_nodes('Cafe Aurora')


def test_access_tracking_does_not_wait_for_another_writer(tmp_path):
    path = str(tmp_path / 'synthetic.db')
    with closing(GraphMemoryStore(path)) as owner, closing(GraphMemoryStore(path)) as reader:
        fact = owner.create_node('Marker', 'Synthetic fact', data='Keep facts', parent_id='world')
        configured_wait = reader.conn.execute('PRAGMA busy_timeout').fetchone()[0]
        owner.conn.execute('BEGIN IMMEDIATE')
        with ThreadPoolExecutor(max_workers=1) as worker:
            attempt = worker.submit(reader.touch_node, fact.id)
            try:
                attempt.result(timeout=configured_wait / 1000 / 10)
            finally:
                owner.conn.rollback()
        assert reader.get_node(fact.id).access_count == fact.access_count
        assert reader.conn.execute('PRAGMA busy_timeout').fetchone()[0] == configured_wait


@pytest.mark.parametrize('outcome', ['commit', 'rollback'])
def test_tracking_preserves_the_callers_pending_content_transaction(tmp_path, outcome):
    path = str(tmp_path / 'synthetic.db')
    with closing(GraphMemoryStore(path)) as owner, closing(GraphMemoryStore(path)) as peer:
        fact = owner.create_node('Marker', 'Synthetic fact', data='Committed fact', parent_id='world')
        owner.conn.execute('BEGIN IMMEDIATE')
        owner.conn.execute('UPDATE memory_nodes SET data=? WHERE id=?', ('Pending fact', fact.id))
        owner.touch_node(fact.id)
        assert peer.get_node(fact.id).data == fact.data
        assert owner.get_node(fact.id).data == 'Pending fact'
        getattr(owner.conn, outcome)()
        expected = 'Pending fact' if outcome == 'commit' else fact.data
        assert peer.get_node(fact.id).data == expected
        assert peer.get_node(fact.id).access_count == fact.access_count


def test_tracking_preserves_the_callers_read_snapshot(tmp_path):
    path = str(tmp_path / 'synthetic.db')
    with closing(GraphMemoryStore(path)) as snapshot, closing(GraphMemoryStore(path)) as peer:
        fact = snapshot.create_node('Marker', 'Synthetic fact', data='Original fact', parent_id='world')
        snapshot.conn.execute('PRAGMA journal_mode=WAL')
        snapshot.conn.execute('BEGIN')
        assert snapshot.get_node(fact.id).data == fact.data
        peer.update_node(fact.id, data='Later committed fact')
        snapshot.touch_node(fact.id)
        assert snapshot.get_node(fact.id).data == fact.data
        snapshot.conn.rollback()
        assert snapshot.get_node(fact.id).data == 'Later committed fact'
