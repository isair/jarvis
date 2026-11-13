"""Exact branch facts stay unique even when the picker changes destinations."""
from unittest.mock import patch

import pytest

from jarvis.memory.graph import GraphMemoryStore, BRANCH_USER, BRANCH_WORLD
from jarvis.memory import graph_ops as ops

pytestmark = pytest.mark.unit


@pytest.fixture
def store(tmp_path):
    graph = GraphMemoryStore(str(tmp_path / 'branch-dedupe.db'))
    yield graph
    graph.close()


@pytest.mark.parametrize('original,repeated', [
    ('The user lives in London.', 'The user lives in London.'),
    ('İstanbul is large.', 'i̇stanbul  is large.'),
    ('The user lives on Straße.', 'THE USER LIVES ON STRASSE.'),
    ('ＡＢＣ is a project.', 'ABC is a project.'),
])
def test_repeated_branch_fact_does_not_land_in_sibling(store, original, repeated):
    existing = store.create_node(name='Established', description='Established facts',
                                 data=original, parent_id=BRANCH_USER)
    destination = store.create_node(name='Alternative', description='Alternative route', parent_id=BRANCH_USER)
    access_before = store.get_node(existing.id).access_count
    with patch.object(ops, 'extract_graph_memories', return_value=[(BRANCH_USER, repeated)]), \
            patch.object(ops, 'find_best_node', return_value=destination.id), \
            patch.object(ops, 'merge_node_data', return_value=ops.MergeResult(False)):
        result = ops.update_graph_from_dialogue(store, 'A repeated declaration.', None, 'fixture')
    assert result.stored == []
    assert result.skipped == 1
    assert store.get_node(destination.id).data == ''
    assert store.get_node(existing.id).data == original
    assert store.get_node(existing.id).access_count == access_before


def test_repeated_fact_in_one_flush_is_unique_across_sibling_routes(store):
    left = store.create_node(name='Left', description='Left route', parent_id=BRANCH_USER)
    right = store.create_node(name='Right', description='Right route', parent_id=BRANCH_USER)
    fact = 'The user lives in London.'
    with patch.object(ops, 'extract_graph_memories', return_value=[(BRANCH_USER, fact), (BRANCH_USER, fact)]), \
            patch.object(ops, 'find_best_node', side_effect=[left.id, right.id]), \
            patch.object(ops, 'merge_node_data', return_value=ops.MergeResult(False)):
        result = ops.update_graph_from_dialogue(store, 'A repeated declaration.', None, 'fixture')
    assert len(result.stored) == 1
    assert result.skipped == 1
    assert sum(node.data.splitlines().count(fact) for node in store.get_all_nodes()) == 1


def test_branch_boundaries_and_different_facts_remain_independent(store):
    original = 'The user lives in London.'
    store.create_node(name='World quotation', description='A quoted example', data=original, parent_id=BRANCH_WORLD)
    destination = store.create_node(name='Identity', description='User facts', parent_id=BRANCH_USER)
    facts = [original, 'The user formerly lived in London.', 'The user does not live in London.']
    with patch.object(ops, 'extract_graph_memories', return_value=[(BRANCH_USER, fact) for fact in facts]), \
            patch.object(ops, 'find_best_node', return_value=destination.id), \
            patch.object(ops, 'merge_node_data', return_value=ops.MergeResult(False)):
        result = ops.update_graph_from_dialogue(store, 'Current and former declarations.', None, 'fixture')
    assert result.skipped == 0
    assert len(result.stored) == len(facts)
    assert store.get_node(destination.id).data.splitlines() == facts


def test_failed_routing_does_not_consume_pending_fact(store):
    destination = store.create_node(name='Identity', description='User facts', parent_id=BRANCH_USER)
    fact = 'The user lives in London.'
    with patch.object(ops, 'extract_graph_memories', return_value=[(BRANCH_USER, fact), (BRANCH_USER, fact)]), \
            patch.object(ops, 'find_best_node', side_effect=[RuntimeError('route unavailable'), destination.id]), \
            patch.object(ops, 'merge_node_data', return_value=ops.MergeResult(False)):
        result = ops.update_graph_from_dialogue(store, 'A repeated declaration.', None, 'fixture')
    assert result.skipped == 0
    assert result.stored == [(fact, 'Identity')]
    assert store.get_node(destination.id).data == fact
