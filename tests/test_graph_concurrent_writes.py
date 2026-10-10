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


@pytest.fixture
def store(tmp_path):
    graph = GraphMemoryStore(str(tmp_path / 'graph.db'))
    try:
        yield graph
    finally:
        graph.close()


def test_concurrent_appends_retain_every_fact(store):
    node = store.create_node('User facts', 'Concurrent facts', parent_id='user')
    workers, rounds = 8, 12
    barrier = Barrier(workers)

    def append(worker):
        for index in range(rounds):
            barrier.wait(timeout=5)
            store.append_to_node(node.id, f'fact-{worker}-{index}')

    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(append, range(workers)))
    result = store.get_node(node.id)
    expected = {f'fact-{worker}-{index}' for worker in range(workers) for index in range(rounds)}
    assert set(result.data.splitlines()) == expected
    assert len(result.data.splitlines()) == len(expected)


def test_concurrent_updates_preserve_independent_fields(store):
    node = store.create_node('Initial name', 'Initial description', parent_id='user')
    barrier = Barrier(2)

    def update(field):
        snapshots = []
        for index in range(12):
            barrier.wait(timeout=5)
            store.update_node(node.id, **{field: f'{field}-{index}'})
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
def test_mutation_listener_can_read_committed_data_from_another_thread(store, operation):
    node = store.create_node('User facts', 'Listener visibility', parent_id='user')
    observed = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        def listener(**event):
            if event['node_id'] == node.id:
                persisted = executor.submit(store.get_node, node.id).result(timeout=2)
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
