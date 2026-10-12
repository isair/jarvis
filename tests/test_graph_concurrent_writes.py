"""Concurrent graph writes retain every independently supplied fact."""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from jarvis.memory.graph import (
    GraphMemoryStore,
    register_graph_mutation_listener,
    unregister_graph_mutation_listener,
)

pytestmark = pytest.mark.unit


@pytest.fixture(params=[1, 2], ids=['shared-store', 'independent-stores'])
def stores(tmp_path, request):
    graphs = [GraphMemoryStore(str(tmp_path / 'graph.db')) for _ in range(request.param)]
    try:
        yield graphs
    finally:
        for graph in graphs:
            graph.close()


def test_concurrent_appends_retain_every_fact(stores):
    store = stores[0]
    node = store.create_node('User facts', 'Concurrent facts', parent_id='user')
    workers, rounds = 8, 12
    barrier = Barrier(workers)

    def append(worker):
        for index in range(rounds):
            barrier.wait(timeout=5)
            stores[worker % len(stores)].append_to_node(node.id, f'fact-{worker}-{index}')

    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(append, range(workers)))
    result = store.get_node(node.id)
    expected = {f'fact-{worker}-{index}' for worker in range(workers) for index in range(rounds)}
    assert set(result.data.splitlines()) == expected
    assert len(result.data.splitlines()) == len(expected)


def test_concurrent_updates_preserve_independent_fields(stores):
    store = stores[0]
    node = store.create_node('Initial name', 'Initial description', parent_id='user')
    barrier = Barrier(2)

    def update(field):
        snapshots = []
        for index in range(12):
            barrier.wait(timeout=5)
            writer = stores[0 if field == 'name' else -1]
            writer.update_node(node.id, **{field: f'{field}-{index}'})
            barrier.wait(timeout=5)
            result = store.get_node(node.id)
            snapshots.append((index, result.name, result.description))
            barrier.wait(timeout=5)
        return snapshots

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(update, ('name', 'description')))
    for snapshots in results:
        for index, name, description in snapshots:
            assert (name, description) == (f'name-{index}', f'description-{index}')


@pytest.mark.parametrize('operation', ['update', 'append'])
def test_mutation_listener_can_read_committed_data_from_another_thread(stores, operation):
    store = stores[0]
    node = store.create_node('User facts', 'Listener visibility', parent_id='user')
    observed = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        def listener(**event):
            if event['node_id'] == node.id:
                persisted = executor.submit(stores[-1].get_node, node.id).result(timeout=2)
                observed.append(persisted.data)

        register_graph_mutation_listener(listener)
        try:
            if operation == 'update':
                store.update_node(node.id, data='Committed fact')
            else:
                store.append_to_node(node.id, 'Committed fact')
            assert observed == ['Committed fact']
        finally:
            unregister_graph_mutation_listener(listener)


@pytest.mark.parametrize('failure_stage', ['statement', 'commit'])
@pytest.mark.parametrize('operation', ['update', 'append', 'delete', 'touch'])
def test_failed_write_does_not_block_another_writer(stores, operation, failure_stage):
    import sqlite3

    first, other = stores[0], stores[-1]
    blocked = first.create_node('Reject this write', 'Synthetic failure', data='Keep facts', parent_id='user')
    writable = first.create_node('Other node', 'Writable', parent_id='user')
    saved = first.get_node(blocked.id).to_dict()
    event = 'DELETE' if operation == 'delete' else 'UPDATE'
    if failure_stage == 'commit':
        failure = """INSERT INTO memory_nodes
            (id, name, description, parent_id, last_accessed, created_at, updated_at)
            VALUES ('synthetic-orphan', 'Rejected child', 'Commit failure', 'missing-parent',
                    OLD.last_accessed, OLD.created_at, OLD.updated_at);"""
    else:
        failure = "SELECT RAISE(ABORT, 'Synthetic write failure');"
    first.conn.execute(f"""CREATE TRIGGER reject_one_node BEFORE {event} ON memory_nodes
        WHEN OLD.id = '{blocked.id}' BEGIN {failure} END""")
    first.conn.commit()
    if failure_stage == 'commit':
        first.conn.execute('PRAGMA defer_foreign_keys = ON')
    other.conn.execute('PRAGMA busy_timeout = 100')
    with pytest.raises(sqlite3.IntegrityError):
        if operation == 'update':
            first.update_node(blocked.id, data='Rejected facts')
        elif operation == 'append':
            first.append_to_node(blocked.id, 'Rejected facts')
        elif operation == 'delete':
            first.delete_node(blocked.id)
        else:
            first.touch_node(blocked.id)
    assert other.update_node(writable.id, data='Committed fact') is not None
    assert first.get_node(blocked.id).to_dict() == saved
    assert first.get_node('synthetic-orphan') is None
    assert first.get_node(writable.id).data == 'Committed fact'
