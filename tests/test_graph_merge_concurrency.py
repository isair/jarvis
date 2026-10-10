"""Graph rewrites retain edits committed while the model is working."""
import json

import pytest

from jarvis.memory.graph import GraphMemoryStore
from jarvis.memory import graph_ops

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


@pytest.mark.parametrize('change', ['append', 'correct', 'delete'])
@pytest.mark.parametrize('new_facts', [[], ['New fact.']], ids=['consolidate', 'merge'])
def test_outdated_rewrite_cannot_overwrite_or_claim_a_changed_node(stores, monkeypatch, change, new_facts):
    store, other = stores
    node = store.create_node('Marker', 'Synthetic fact', data='Original fact.', parent_id='user')

    def rewrite(**kwargs):
        if change == 'append':
            other.append_to_node(node.id, 'Concurrent fact.')
        elif change == 'correct':
            other.update_node(node.id, data='Corrected fact.')
        else:
            other.delete_node(node.id)
        return json.dumps({'facts': ['Original fact.', *new_facts]})

    monkeypatch.setattr(graph_ops, 'call_llm_direct', rewrite)
    result = graph_ops.merge_node_data(store, node.id, new_facts, cfg=None, chat_model='synthetic')
    assert not result.success
    assert result.incorporated_indices == []
    current = store.get_node(node.id)
    if change == 'delete':
        assert current is None
    else:
        assert current.data == ('Original fact.\nConcurrent fact.' if change == 'append' else 'Corrected fact.')


@pytest.mark.parametrize('change', ['name', 'touch', 'world'])
def test_edits_outside_rewrite_inputs_preserve_success(stores, monkeypatch, change):
    store, other = stores
    node = store.create_node('Marker', 'Synthetic fact', data='Original fact.', parent_id='user')
    world = store.create_node('External', 'World fact', data='World fact.', parent_id='world')

    def rewrite(**kwargs):
        if change == 'name':
            other.update_node(node.id, name='Edited name')
        elif change == 'touch':
            other.touch_node(node.id)
        else:
            other.append_to_node(world.id, 'Another world fact.')
        return json.dumps({'facts': ['Original fact.', 'New fact.']})

    monkeypatch.setattr(graph_ops, 'call_llm_direct', rewrite)
    result = graph_ops.merge_node_data(store, node.id, ['New fact.'], cfg=None, chat_model='synthetic')
    assert result.success
    assert result.incorporated_indices == [0]
    current = store.get_node(node.id)
    assert current.data == 'Original fact.\nNew fact.'
    if change == 'name':
        assert current.name == 'Edited name'


@pytest.mark.parametrize('change', ['append', 'correct', 'delete'])
def test_graph_update_fallback_retains_concurrent_edits_and_reports_only_stored_facts(stores, monkeypatch, change):
    store, other = stores
    node = store.create_node('Marker', 'Synthetic fact', data='Original fact.', parent_id='user')
    monkeypatch.setattr(graph_ops, 'extract_graph_memories', lambda **kwargs: [('user', 'New fact.')])
    monkeypatch.setattr(graph_ops, 'find_best_node', lambda **kwargs: node.id)

    def rewrite(**kwargs):
        if change == 'append':
            other.append_to_node(node.id, 'Concurrent fact.')
        elif change == 'correct':
            other.update_node(node.id, data='Corrected fact.')
        else:
            other.delete_node(node.id)
        return json.dumps({'facts': ['Original fact.', 'New fact.']})

    monkeypatch.setattr(graph_ops, 'call_llm_direct', rewrite)
    result = graph_ops.update_graph_from_dialogue(store, 'Synthetic summary', cfg=None, chat_model='synthetic')
    current = store.get_node(node.id)
    if change == 'delete':
        assert current is None
        assert result.stored == []
    else:
        prefix = 'Original fact.\nConcurrent fact.' if change == 'append' else 'Corrected fact.'
        assert current.data == prefix + '\nNew fact.'
        assert result.stored == [('New fact.', node.name)]


def test_conditional_update_rejects_a_stale_input_without_partial_field_writes(stores):
    store, other = stores
    node = store.create_node('Marker', 'Synthetic fact', data='Original fact.', parent_id='user')
    other.update_node(node.id, data='Corrected fact.')
    result = store.update_node(node.id, name='Unwanted name', data='Outdated rewrite.', expected_data=node.data)
    assert result is None
    current = store.get_node(node.id)
    assert current.name == node.name
    assert current.data == 'Corrected fact.'
    committed = store.update_node(node.id, data='Current rewrite.', expected_data=current.data)
    assert committed is not None
    assert committed.data == store.get_node(node.id).data == 'Current rewrite.'
