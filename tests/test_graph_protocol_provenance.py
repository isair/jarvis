"""Model protocol delimiters cannot appear as newly invented graph content."""
import json

import pytest

from jarvis.memory import graph_ops
from jarvis.memory.graph import GraphMemoryStore

pytestmark = pytest.mark.unit

MARKERS = ('<tool_call|>', '<|synthetic_turn_end|>', '<unused347>', '</tool_response>')


def classify_all_as_durable(candidates):
    def infer(**kwargs):
        content = kwargs['user_content']
        if content.startswith('Extract'):
            return json.dumps(candidates)
        payload = json.loads(content)
        branches = {item['fact']: item['branch'] for item in candidates}
        return json.dumps({str(item['id']): {'verdict': 'DURABLE', 'kind': branches[item['fact']]}
                           for item in payload['candidates']})
    return infer


@pytest.mark.parametrize('marker', MARKERS)
def test_extraction_rejects_invented_protocol_tokens_without_losing_siblings(monkeypatch, marker):
    supported = 'The user enjoys sailing.'
    invented = f'Reply briefly {marker}'
    candidates = [{'branch': 'DIRECTIVES', 'fact': invented}, {'branch': 'USER', 'fact': supported}]
    monkeypatch.setattr(graph_ops, 'call_llm_direct', classify_all_as_durable(candidates))
    assert graph_ops.extract_graph_memories(
        'The user enjoys sailing and prefers brief replies.', None, 'synthetic',
    ) == [('user', supported)]


@pytest.mark.parametrize('marker', MARKERS)
def test_extraction_preserves_protocol_syntax_established_by_the_source(monkeypatch, marker):
    fact = f'The Arcadia protocol uses the literal delimiter {marker}.'
    candidates = [{'branch': 'WORLD', 'fact': fact}]
    monkeypatch.setattr(graph_ops, 'call_llm_direct', classify_all_as_durable(candidates))
    assert graph_ops.extract_graph_memories(fact, None, 'synthetic') == [('world', fact)]


@pytest.fixture
def store(tmp_path):
    graph = GraphMemoryStore(str(tmp_path / 'graph.db'))
    yield graph
    graph.close()


@pytest.mark.parametrize('marker', MARKERS)
def test_merge_rejects_invented_protocol_tokens_without_changing_saved_content(store, monkeypatch, marker):
    current = 'The user enjoys sailing.'
    new = 'The user grows orchids.'
    node = store.create_node('Profile', 'Personal facts', data=current, parent_id='user')
    before = store.get_graph_data()
    monkeypatch.setattr(graph_ops, 'call_llm_direct', lambda **kwargs: json.dumps({'facts': [current, f'{new} {marker}']}))
    assert not graph_ops.merge_node_data(store, node.id, [new], None, 'synthetic').success
    assert store.get_graph_data() == before
    assert store.get_node(node.id).data == current


@pytest.mark.parametrize('marker', MARKERS)
def test_merge_preserves_source_established_protocol_literals(store, monkeypatch, marker):
    current = 'The user enjoys sailing.'
    new = f'The Arcadia protocol uses the literal delimiter {marker}.'
    node = store.create_node('Profile', 'Personal facts', data=current, parent_id='user')
    monkeypatch.setattr(graph_ops, 'call_llm_direct', lambda **kwargs: json.dumps({'facts': [current, new]}))
    result = graph_ops.merge_node_data(store, node.id, [new], None, 'synthetic')
    assert result.success and result.incorporated_indices == [0]
    assert store.get_node(node.id).data == '\n'.join([current, new])


@pytest.mark.parametrize('marker', MARKERS)
@pytest.mark.parametrize('field', ['fact', 'name', 'description', 'summary'])
def test_split_rejects_protocol_corruption_in_all_generated_fields(store, monkeypatch, marker, field):
    current = 'A recorded observation. ' * (graph_ops.SPLIT_THRESHOLD + 1)
    node = store.create_node('Profile', 'Personal facts', data=current, parent_id='user')
    before = store.get_graph_data()
    proposal = {'categories': [
        {'name': 'First group', 'description': 'First collection', 'facts': ['First recorded observation.']},
        {'name': 'Second group', 'description': 'Second collection', 'facts': ['Second recorded observation.']},
    ], 'summary': 'Recorded observations'}
    if field == 'fact':
        proposal['categories'][0]['facts'][0] += marker
    elif field == 'summary':
        proposal['summary'] += marker
    else:
        proposal['categories'][0][field] += marker
    monkeypatch.setattr(graph_ops, 'call_llm_direct', lambda **kwargs: json.dumps(proposal))
    assert graph_ops.auto_split_node(store, node.id, None, 'synthetic') is False
    assert store.get_graph_data() == before
    assert store.get_node(node.id).data == current


@pytest.mark.parametrize('marker', MARKERS)
def test_split_preserves_protocol_literals_from_its_source(store, monkeypatch, marker):
    literal = f'The Arcadia protocol uses the literal delimiter {marker}.'
    current = literal + '\n' + 'A recorded observation. ' * (graph_ops.SPLIT_THRESHOLD + 1)
    node = store.create_node('Profile', 'Personal facts', data=current, parent_id='user')
    proposal = {'categories': [
        {'name': 'Protocol', 'description': 'Protocol details', 'facts': [literal]},
        {'name': 'Observations', 'description': 'Recorded observations', 'facts': ['A recorded observation.']},
    ], 'summary': 'Protocol and observations'}
    monkeypatch.setattr(graph_ops, 'call_llm_direct', lambda **kwargs: json.dumps(proposal))
    assert graph_ops.auto_split_node(store, node.id, None, 'synthetic') is True
    assert any(child.data == literal for child in store.get_children(node.id))


@pytest.mark.parametrize('marker', MARKERS)
def test_rejected_merge_keeps_saved_facts_and_appends_only_the_valid_new_fact(store, monkeypatch, marker):
    current = 'The user enjoys sailing.'
    new = 'The user grows orchids.'
    store.update_node('user', data=current)
    responses = iter([
        json.dumps([{'branch':'USER','fact':new}]),
        json.dumps({'facts':[current, f'{new} {marker}']}),
    ])
    monkeypatch.setattr(graph_ops, 'call_llm_direct', lambda **kwargs: next(responses))
    monkeypatch.setattr(graph_ops, '_review_graph_facts', lambda facts, *args: facts)
    result = graph_ops.update_graph_from_dialogue(store, new, None, 'synthetic')
    assert store.get_node('user').data.splitlines() == [current, new]
    assert [fact for fact, _ in result.stored] == [new]


def test_ordinary_markup_is_not_a_model_protocol_token(monkeypatch):
    fact = 'The user uses <Ada> as a nickname.'
    candidates = [{'branch':'USER','fact':fact}]
    monkeypatch.setattr(graph_ops, 'call_llm_direct', classify_all_as_durable(candidates))
    assert graph_ops.extract_graph_memories('The user uses Ada as a nickname.', None, 'synthetic') == [('user', fact)]


def test_source_established_token_does_not_authorise_a_different_token(monkeypatch):
    source = 'The Arcadia protocol uses <|start|>.'
    fact = 'The Arcadia protocol uses <|start|> and <|end|>.'
    candidates = [{'branch': 'WORLD', 'fact': fact}]
    monkeypatch.setattr(graph_ops, 'call_llm_direct', classify_all_as_durable(candidates))
    assert graph_ops.extract_graph_memories(source, None, 'synthetic') == []
