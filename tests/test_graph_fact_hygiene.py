"""Graph memory retains only candidates with a complete semantic verdict."""
import json
from unittest.mock import patch

import pytest

from jarvis.memory.graph_ops import extract_graph_memories

pytestmark = pytest.mark.unit


CANDIDATES = [
    {'branch': 'WORLD', 'fact': 'A weekly weather forecast'},
    {'branch': 'USER', 'fact': 'The user follows an 1800 kcal meal plan'},
    {'branch': 'USER', 'fact': 'A question about camera systems'},
    {'branch': 'DIRECTIVES', 'fact': 'Always reply briefly'},
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
        return json.dumps({str(item['id']): 'DURABLE' for item in review['candidates']})

    with patch('jarvis.memory.graph_ops.call_llm_direct', side_effect=infer):
        assert extract_graph_memories('The user prefers concise replies', mock_config, 'local-model') == [
            ('user', supported),
        ]


def test_transient_and_interaction_candidates_are_not_returned(mock_config):
    with patch('jarvis.memory.graph_ops.call_llm_direct', side_effect=[
        json.dumps(CANDIDATES), json.dumps({'0': 'TRANSIENT', '1': 'DURABLE', '2': 'INTERACTION', '3': 'DURABLE'}),
    ]):
        assert extract_graph_memories('Mixed summary', mock_config, 'local-model') == [
            ('user', CANDIDATES[1]['fact']), ('directives', CANDIDATES[3]['fact']),
        ]


@pytest.mark.parametrize('verdict', [
    None, '', '{"1": "DURABLE"}',
    '{"0": "DURABLE", "0": "DURABLE", "1": "DURABLE", "2": "DURABLE", "3": "DURABLE"}',
    '{"0": "DURABLE", "1": "DURABLE", "2": "UNKNOWN", "3": "DURABLE"}',
    '{"0": "DURABLE", "1": "DURABLE", "2": "DURABLE", "4": "DURABLE"}',
    '[]', 'null',
    '{"0": false, "1": "DURABLE", "2": "INTERACTION", "3": "DURABLE"}',
    '{"0": {}, "1": "DURABLE", "2": "INTERACTION", "3": "DURABLE"}',
    'All entries are valid',
])
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
        json.dumps(candidates), json.dumps({'0': 'DURABLE', '1': 'DURABLE'}),
    ]):
        assert extract_graph_memories('Summary', mock_config, 'local-model', date_utc='2026-10-03') == [
            ('directives', '[2026-10-03] Réponds brièvement'),
            ('world', '[2026-10-03] Trenches Boxing Club offers evening classes'),
        ]


def test_unordered_verdicts_preserve_durable_facts_in_original_order(mock_config):
    with patch('jarvis.memory.graph_ops.call_llm_direct', side_effect=[
        json.dumps(CANDIDATES), json.dumps({'3': 'DURABLE', '1': 'DURABLE', '0': 'TRANSIENT', '2': 'INTERACTION'}),
    ]):
        assert extract_graph_memories('Mixed summary', mock_config, 'local-model') == [
            ('user', CANDIDATES[1]['fact']), ('directives', CANDIDATES[3]['fact']),
        ]


def test_exhausted_extraction_budget_leaves_graph_empty(mock_config):
    import time
    def slow_extraction(**kwargs):
        time.sleep(0.02)
        return json.dumps(CANDIDATES)
    with patch('jarvis.memory.graph_ops.call_llm_direct', side_effect=slow_extraction):
        assert extract_graph_memories('Summary', mock_config, 'local-model', timeout_sec=0.01) == []


def test_review_shares_the_remaining_extraction_budget(mock_config):
    import time
    budget = 0.1
    started = time.monotonic()
    def backend(**kwargs):
        if kwargs['user_content'].startswith('Extract'):
            time.sleep(budget / 5)
            return json.dumps(CANDIDATES)
        assert kwargs['timeout_sec'] <= budget - (time.monotonic() - started) + 0.005
        return json.dumps({'0': 'TRANSIENT', '1': 'DURABLE', '2': 'INTERACTION', '3': 'DURABLE'})
    with patch('jarvis.memory.graph_ops.call_llm_direct', side_effect=backend):
        assert extract_graph_memories('Summary', mock_config, 'local-model', timeout_sec=budget) == [
            ('user', CANDIDATES[1]['fact']), ('directives', CANDIDATES[3]['fact']),
        ]


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
        return json.dumps({'0': 'UNSUPPORTED', '1': 'DURABLE'}) if source == summary else ''
    with patch('jarvis.memory.graph_ops.call_llm_direct', side_effect=infer):
        assert extract_graph_memories(summary, mock_config, 'local-model') == [
            ('user', 'The user is vegetarian'),
        ]
