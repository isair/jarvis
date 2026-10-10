"""Graph splits commit their children and parent together from current input."""
import json

import pytest

from jarvis.memory import graph_ops
from jarvis.memory.graph import GraphMemoryStore, SPLIT_THRESHOLD

pytestmark = pytest.mark.unit


@pytest.fixture(params=[False, True], ids=['same-store', 'independent-store'])
def stores(tmp_path, request):
    store = GraphMemoryStore(str(tmp_path / 'graph.db'))
    other = GraphMemoryStore(store.db_path) if request.param else store
    try:
        yield store, other
    finally:
        if other is not store:
            other.close()
        store.close()


def large_node(store):
    return store.create_node('Synthetic topic', 'Original description',
        data='Original synthetic fact.\n' * (SPLIT_THRESHOLD + 1), parent_id='user')


def proposed_split():
    return json.dumps({'categories': [
        {'name': 'First', 'description': 'First description', 'facts': ['First fact.']},
        {'name': 'Second', 'description': 'Second description', 'facts': ['Second fact.']},
    ], 'summary': 'Synthetic summary'})


@pytest.mark.parametrize('change', ['append', 'name', 'description', 'delete'])
def test_stale_split_preserves_concurrent_changes_without_creating_children(stores, monkeypatch, change):
    store, other = stores
    node = large_node(store)

    def organise(**kwargs):
        if change == 'append':
            other.append_to_node(node.id, 'Concurrent fact.')
        elif change == 'delete':
            other.delete_node(node.id)
        else:
            other.update_node(node.id, **{change: 'Concurrent edit'})
        return proposed_split()

    monkeypatch.setattr(graph_ops, 'call_llm_direct', organise)
    assert graph_ops.auto_split_node(store, node.id, None, 'synthetic') is False
    assert store.get_children(node.id) == []
    current = store.get_node(node.id)
    if change == 'delete':
        assert current is None
    else:
        assert current.data == (node.data + '\nConcurrent fact.' if change == 'append' else node.data)
        if change in ('name', 'description'):
            assert getattr(current, change) == 'Concurrent edit'


def test_failed_child_write_rolls_back_entire_split_and_releases_writer(stores, monkeypatch):
    store, other = stores
    node = large_node(store)
    existing = store.create_node('Existing child', 'Keep', data='Retained fact.', parent_id=node.id)
    store.conn.executescript("""CREATE TRIGGER reject_second_child BEFORE INSERT ON memory_nodes
        WHEN NEW.name = 'Second' BEGIN SELECT RAISE(ABORT, 'synthetic failure'); END;""")
    monkeypatch.setattr(graph_ops, 'call_llm_direct', lambda **kwargs: proposed_split())
    assert graph_ops.auto_split_node(store, node.id, None, 'synthetic') is False
    assert store.get_node(node.id).data == node.data
    assert store.get_node(node.id).description == node.description
    assert [child.id for child in store.get_children(node.id)] == [existing.id]
    assert other.append_to_node(existing.id, 'Writer is available.').data.endswith('Writer is available.')


@pytest.mark.parametrize('change', ['touch', 'unrelated'])
def test_split_preserves_edits_outside_model_inputs(stores, monkeypatch, change):
    store, other = stores
    node = large_node(store)
    unrelated = store.create_node('Other', 'Keep', data='Other fact.', parent_id='world')

    def organise(**kwargs):
        if change == 'touch':
            other.touch_node(node.id)
        else:
            other.append_to_node(unrelated.id, 'Another fact.')
        return proposed_split()

    monkeypatch.setattr(graph_ops, 'call_llm_direct', organise)
    assert graph_ops.auto_split_node(store, node.id, None, 'synthetic') is True
    parent = store.get_node(node.id)
    assert parent.data == ''
    assert parent.description == 'Synthetic summary'
    assert {child.data for child in store.get_children(node.id)} == {'First fact.', 'Second fact.'}
    if change == 'touch':
        assert parent.access_count == node.access_count + 1
    else:
        assert store.get_node(unrelated.id).data.endswith('Another fact.')


def test_mutation_readers_only_observe_committed_complete_splits(stores, monkeypatch):
    from contextlib import closing
    from jarvis.memory.graph import register_graph_mutation_listener, unregister_graph_mutation_listener

    store, _ = stores
    node = large_node(store)
    views = []

    def observe(**event):
        with closing(GraphMemoryStore(store.db_path, read_only=True)) as reader:
            views.append((reader.get_node(node.id).data,
                          frozenset(child.data for child in reader.get_children(node.id))))

    monkeypatch.setattr(graph_ops, 'call_llm_direct', lambda **kwargs: proposed_split())
    register_graph_mutation_listener(observe)
    try:
        assert graph_ops.auto_split_node(store, node.id, None, 'synthetic') is True
    finally:
        unregister_graph_mutation_listener(observe)
    assert views
    assert set(views) == {('', frozenset({'First fact.', 'Second fact.'}))}
