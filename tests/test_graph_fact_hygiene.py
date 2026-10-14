"""Graph memory retains only candidates with a complete semantic verdict."""
import json
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import MagicMock, patch

import pytest

from jarvis.memory import graph_ops
from jarvis.memory.graph_ops import extract_graph_memories

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("no_network")]


def review_response(labels):
    """Encode separate source verdicts and semantic types for an owned backend."""
    kinds = {'USER', 'DIRECTIVES', 'WORLD', 'REQUEST_HISTORY', 'ASSISTANT_SELF'}
    return json.dumps({index: {
        'verdict': 'DURABLE' if label in kinds else label,
        'kind': label if label in kinds else 'OTHER',
    } for index, label in labels.items()})


@pytest.mark.parametrize('branch', ['USER', 'DIRECTIVES', 'WORLD'])
@pytest.mark.parametrize('verdict', ['DURABLE', 'UNSUPPORTED', 'TRANSIENT', 'INTERACTION', 'ADVICE'])
def test_matching_branch_also_requires_a_durable_source_verdict(mock_config, branch, verdict):
    candidate = (branch.lower(), 'Source candidate')
    response = json.dumps({'0': {'verdict': verdict, 'kind': branch}})
    with patch('jarvis.memory.graph_ops.call_llm_direct', return_value=response):
        assert graph_ops._review_graph_facts(
            [candidate], mock_config, 'local-model', mock_config.llm_chat_timeout_sec,
            False, 'Owned source summary',
        ) == ([candidate] if verdict == 'DURABLE' else [])


CANDIDATES = [
    {'branch': 'WORLD', 'fact': 'A weekly weather forecast'},
    {'branch': 'USER', 'fact': 'The user follows an 1800 kcal meal plan'},
    {'branch': 'USER', 'fact': 'A question about camera systems'},
    {'branch': 'DIRECTIVES', 'fact': 'Always reply briefly'},
]


def test_independent_classification_rejects_misrouted_facts_without_rewriting_them(mock_config):
    candidates = [
        {'branch': 'USER', 'fact': 'Keep replies concise'},
        {'branch': 'DIRECTIVES', 'fact': 'The user likes coffee'},
        {'branch': 'USER', 'fact': 'The user lives in London'},
        {'branch': 'DIRECTIVES', 'fact': 'Answer in British English'},
        {'branch': 'WORLD', 'fact': 'This assistant can keep a diary'},
        {'branch': 'USER', 'fact': 'The user requested concise replies'},
    ]

    def infer(**kwargs):
        if kwargs['user_content'].startswith('Extract'):
            return json.dumps(candidates)
        review = json.loads(kwargs['user_content'])
        assert review['candidates'] == [
            {'id': index, 'fact': candidate['fact']}
            for index, candidate in enumerate(candidates)
        ], 'The reviewer must classify meaning without the proposed destination'
        return review_response({'0': 'DIRECTIVES', '1': 'USER', '2': 'USER',
                           '3': 'DIRECTIVES', '4': 'ASSISTANT_SELF', '5': 'REQUEST_HISTORY'})

    with patch('jarvis.memory.graph_ops.call_llm_direct', side_effect=infer):
        assert extract_graph_memories('Source summary', mock_config, 'local-model') == [
            ('user', candidates[2]['fact']), ('directives', candidates[3]['fact']),
        ]


@pytest.mark.parametrize('source_day', ['2027-12-31', None])
def test_fact_retains_the_reference_source_used_to_extract_it(mock_config, source_day):
    summary = 'The user plans to visit a museum tomorrow.'
    source = (f'(Date: {source_day}) ' if source_day else '') + summary
    fact = 'The user plans to visit a museum.'

    def infer(**kwargs):
        content = kwargs['user_content']
        if content.startswith('Extract'):
            assert content.endswith(source)
            return json.dumps([{'branch': 'USER', 'fact': fact}])
        review = json.loads(content)
        return review_response({'0': 'USER' if review['summary'] == source else 'UNSUPPORTED'})

    with patch('jarvis.memory.graph_ops.call_llm_direct', side_effect=infer):
        assert extract_graph_memories(summary, mock_config, 'local-model', date_utc=source_day) == [
            ('user', fact),
        ]


@pytest.mark.parametrize('invalid_fact', [True, 42, {'text': 'Invented fact'}, ['Invented fact'], None, '', '   '])
def test_malformed_fact_cannot_become_personal_memory(mock_config, invalid_fact):
    supported = 'The user prefers concise replies'
    candidates = [{'branch': 'USER', 'fact': invalid_fact},
                  {'branch': 'USER', 'fact': supported}]

    def infer(**kwargs):
        if kwargs['user_content'].startswith('Extract'):
            return json.dumps(candidates)
        review = json.loads(kwargs['user_content'])
        return review_response({str(item['id']): 'USER' for item in review['candidates']})

    with patch('jarvis.memory.graph_ops.call_llm_direct', side_effect=infer):
        assert extract_graph_memories('The user prefers concise replies', mock_config, 'local-model') == [
            ('user', supported),
        ]


def test_transient_and_interaction_candidates_are_not_returned(mock_config):
    with patch('jarvis.memory.graph_ops.call_llm_direct', side_effect=[
        json.dumps(CANDIDATES), review_response({'0': 'TRANSIENT', '1': 'USER', '2': 'INTERACTION', '3': 'DIRECTIVES'}),
    ]):
        assert extract_graph_memories('Mixed summary', mock_config, 'local-model') == [
            ('user', CANDIDATES[1]['fact']), ('directives', CANDIDATES[3]['fact']),
        ]


def invalid_reviews():
    valid = json.loads(review_response({'0': 'TRANSIENT', '1': 'USER',
                                        '2': 'INTERACTION', '3': 'DIRECTIVES'}))
    encoded = json.dumps(valid)
    yield from (None, '', '[]', 'null', 'All entries are valid',
                '{"0":"DURABLE","1":"DURABLE","2":"DURABLE","3":"DURABLE"}',
                json.dumps({'1': valid['1']}),
                json.dumps({**valid, '4': valid['0']}),
                encoded[:-1] + ', "0": ' + json.dumps(valid['0']) + '}',
                encoded.replace('"verdict": "TRANSIENT"',
                                '"verdict": "TRANSIENT", "verdict": "DURABLE"', 1))
    for malformed in (False, {}, 'DURABLE',
                      {'verdict': 'UNKNOWN', 'kind': 'USER'},
                      {'verdict': 'DURABLE', 'kind': 'UNKNOWN'},
                      {'verdict': False, 'kind': 'USER'},
                      {'verdict': 'DURABLE', 'kind': {}},
                      {'verdict': 'DURABLE'},
                      {'verdict': 'DURABLE', 'kind': 'USER', 'fact': 'Invented text'}):
        yield json.dumps({**valid, '0': malformed})


@pytest.mark.parametrize('verdict', invalid_reviews())
def test_incomplete_or_invalid_review_does_not_store_unreviewed_facts(mock_config, verdict):
    with patch('jarvis.memory.graph_ops.call_llm_direct', side_effect=[json.dumps(CANDIDATES), verdict]):
        assert extract_graph_memories('Mixed summary', mock_config, 'local-model') == []


def test_review_failure_does_not_escape_or_store_candidates(mock_config):
    with patch('jarvis.memory.graph_ops.call_llm_direct', side_effect=[
        json.dumps(CANDIDATES), RuntimeError('Unavailable local model'),
    ]):
        assert extract_graph_memories('Mixed summary', mock_config, 'local-model') == []


def test_durable_facts_keep_their_original_branch_text_and_date(mock_config):
    candidates = [
        {'branch': 'DIRECTIVES', 'fact': '[2026-10-03] Réponds brièvement'},
        {'branch': 'WORLD', 'fact': '[2026-10-03] Trenches Boxing Club offers evening classes'},
    ]
    with patch('jarvis.memory.graph_ops.call_llm_direct', side_effect=[
        json.dumps(candidates), review_response({'0': 'DIRECTIVES', '1': 'WORLD'}),
    ]):
        assert extract_graph_memories('Summary', mock_config, 'local-model', date_utc='2026-10-03') == [
            ('directives', '[2026-10-03] Réponds brièvement'),
            ('world', '[2026-10-03] Trenches Boxing Club offers evening classes'),
        ]


def test_unordered_verdicts_preserve_durable_facts_in_original_order(mock_config):
    with patch('jarvis.memory.graph_ops.call_llm_direct', side_effect=[
        json.dumps(CANDIDATES), review_response({'3': 'DIRECTIVES', '1': 'USER', '0': 'TRANSIENT', '2': 'INTERACTION'}),
    ]):
        assert extract_graph_memories('Mixed summary', mock_config, 'local-model') == [
            ('user', CANDIDATES[1]['fact']), ('directives', CANDIDATES[3]['fact']),
        ]


@pytest.fixture
def clock(monkeypatch):
    clock = [0.0]
    module_clock = MagicMock(wraps=time)
    module_clock.monotonic.side_effect = lambda: clock[0]
    monkeypatch.setattr(graph_ops, 'time', module_clock)
    return clock


@pytest.mark.parametrize('elapsed_fraction,background_wait', [
    (0.0, False), (0.2, False), (0.99, False), (1.0, False), (1.2, False),
    pytest.param(0.2, True, id='concurrent-within-budget'),
    pytest.param(1.0, True, id='concurrent-exhausted-budget'),
])
def test_review_uses_only_the_remaining_extraction_budget(mock_config, clock, elapsed_fraction, background_wait):
    budget = mock_config.llm_chat_timeout_sec
    observed = []

    def background_clock():
        before = time.monotonic()
        Event().wait(0.001)
        return before, time.monotonic()

    def backend(**kwargs):
        if kwargs['user_content'].startswith('Extract'):
            if background_wait:
                with ThreadPoolExecutor(max_workers=1) as pool:
                    observed.append(pool.submit(background_clock).result(timeout=5))
            clock[0] += budget * elapsed_fraction
            return json.dumps(CANDIDATES)
        assert kwargs['timeout_sec'] == pytest.approx(budget - clock[0])
        return review_response({'0': 'TRANSIENT', '1': 'USER', '2': 'INTERACTION', '3': 'DIRECTIVES'})

    with patch('jarvis.memory.graph_ops.call_llm_direct', side_effect=backend):
        result = extract_graph_memories('Summary', mock_config, 'local-model', timeout_sec=budget)

    assert result == ([
        ('user', CANDIDATES[1]['fact']), ('directives', CANDIDATES[3]['fact']),
    ] if elapsed_fraction < 1.0 else [])
    if background_wait:
        assert observed
        assert all(after > before for before, after in observed), "The graph budget clock froze an unrelated worker"


def test_nonpositive_budget_does_not_start_extraction(mock_config):
    with patch('jarvis.memory.graph_ops.call_llm_direct', side_effect=AssertionError('Unexpected inference')):
        assert extract_graph_memories('Summary', mock_config, 'local-model', timeout_sec=0) == []


def test_source_relationship_review_preserves_only_supported_candidates(mock_config):
    summary = 'The user asked to translate "I live in Bristol". They are vegetarian.'
    candidates = [{'branch': 'USER', 'fact': 'The user lives in Bristol'},
                  {'branch': 'USER', 'fact': 'The user is vegetarian'}]
    def infer(**kwargs):
        if kwargs['user_content'].startswith('Extract'):
            return json.dumps(candidates)
        source = json.loads(kwargs['user_content'])['summary']
        return review_response({'0': 'UNSUPPORTED', '1': 'USER'}) if source == summary else ''
    with patch('jarvis.memory.graph_ops.call_llm_direct', side_effect=infer):
        assert extract_graph_memories(summary, mock_config, 'local-model') == [
            ('user', 'The user is vegetarian'),
        ]
