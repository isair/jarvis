"""Planned calls supply required fields before they can be dispatched."""
import json

import pytest

from evals.helpers import voice_config
from jarvis.reply import planner

pytestmark = pytest.mark.unit


def schema(required):
    return [{'type': 'function', 'function': {
        'name': 'localLookup',
        'parameters': {'type': 'object', 'properties': {
            'item': {'type': 'string'}, 'scope': {'type': 'string'},
        }, 'required': required},
    }}]


@pytest.mark.parametrize('step, arguments, required', [
    ('localLookup', None, ['item']),
    ("localLookup item='manual'", None, ['item', 'scope']),
    ('resolve <item>', {}, ['item']),
    ('resolve <item>', {'unknown': 'manual'}, ['item']),
    ('resolve <item>', {'item': 'manual'}, ['item', 'scope']),
])
def test_incomplete_required_arguments_cannot_resolve(monkeypatch, step, arguments, required):
    raw = 'null' if arguments is None else json.dumps({'name': 'localLookup', 'arguments': arguments})
    monkeypatch.setattr(planner, 'call_llm_direct', lambda **kwargs: raw)
    assert planner.resolve_next_tool_call(voice_config(), step, [], schema(required)) is None


def test_incomplete_concrete_step_can_be_completed_by_the_resolver(monkeypatch):
    raw = json.dumps({'name': 'localLookup', 'arguments': {'item': 'manual'}})
    monkeypatch.setattr(planner, 'call_llm_direct', lambda **kwargs: raw)
    assert planner.resolve_next_tool_call(voice_config(), 'localLookup', [('catalogue', '{}', 'Item: manual')], schema(['item'])) == (
        'localLookup', {'item': 'manual'},
    )


@pytest.mark.parametrize('arguments', [[], ['manual'], '', 'manual', 0, 42, False, True])
def test_non_object_arguments_cannot_become_an_empty_call(monkeypatch, arguments):
    raw = json.dumps({'name': 'localLookup', 'arguments': arguments})
    monkeypatch.setattr(planner, 'call_llm_direct', lambda **kwargs: raw)
    assert planner.resolve_next_tool_call(voice_config(), 'resolve <item>', [], schema([])) is None


@pytest.mark.parametrize('step, required, expected', [
    ('localLookup', [], {}),
    ("localLookup item='manual'", ['item'], {'item': 'manual'}),
    ("localLookup item='manual' scope='local'", ['item', 'scope'], {'item': 'manual', 'scope': 'local'}),
])
def test_complete_concrete_arguments_remain_available_without_inference(monkeypatch, step, required, expected):
    monkeypatch.setattr(planner, 'call_llm_direct', lambda **kwargs: pytest.fail('🗺️ Concrete valid arguments need no model'))
    assert planner.resolve_next_tool_call(voice_config(), step, [], schema(required)) == ('localLookup', expected)


@pytest.mark.parametrize('arguments', [None, {}])
def test_optional_empty_argument_objects_remain_available(monkeypatch, arguments):
    raw = json.dumps({'name': 'localLookup', 'arguments': arguments})
    monkeypatch.setattr(planner, 'call_llm_direct', lambda **kwargs: raw)
    assert planner.resolve_next_tool_call(voice_config(), 'resolve <item>', [], schema([])) == ('localLookup', {})


@pytest.mark.parametrize('arguments, expected', [
    ({}, None),
    ({'item': 'manual', 'extra': 'local'}, ('localLookup', {'item': 'manual', 'extra': 'local'})),
])
def test_freeform_schema_keeps_values_but_still_requires_declared_fields(monkeypatch, arguments, expected):
    freeform = schema(['item'])
    del freeform[0]['function']['parameters']['properties']
    raw = json.dumps({'name': 'localLookup', 'arguments': arguments})
    monkeypatch.setattr(planner, 'call_llm_direct', lambda **kwargs: raw)
    assert planner.resolve_next_tool_call(voice_config(), 'resolve <item>', [], freeform) == expected


def test_incomplete_plan_returns_to_chat_before_any_invalid_dispatch(monkeypatch, mock_config, db, dialogue_memory):
    from jarvis.reply import engine
    from jarvis.tools.types import ToolExecutionResult

    mock_config.llm_chat_model = 'gemma4:e2b'
    mock_config.location_enabled = False
    mock_config.embedding_model = None
    mock_config.memory_digest_enabled = False
    mock_config.tool_result_digest_enabled = False
    monkeypatch.setattr(engine, 'select_tools', lambda *args, **kwargs: ['webSearch', 'stop'])
    monkeypatch.setattr(engine, 'plan_query', lambda **kwargs: ['webSearch', 'Reply to the user.'])
    monkeypatch.setattr(planner, 'call_llm_direct', lambda **kwargs: 'null')
    responses = iter([
        {'message': {'content': '', 'tool_calls': [{'id': 'query', 'function': {
            'name': 'webSearch', 'arguments': {'search_query': 'document index'},
        }}]}},
        {'message': {'content': 'The catalogue includes a document guide.'}},
    ])
    monkeypatch.setattr(engine, 'chat_with_messages', lambda **kwargs: next(responses))
    dispatched = []

    def run_tool(*, tool_name, tool_args, **kwargs):
        dispatched.append((tool_name, tool_args))
        complete = bool((tool_args or {}).get('search_query'))
        return ToolExecutionResult(success=complete, reply_text='Document guide.' if complete else 'Missing search query.')

    monkeypatch.setattr(engine, 'run_tool_with_retries', run_tool)
    monkeypatch.setattr('requests.post', lambda *args, **kwargs: pytest.fail('🗺️ Dispatch regression must not contact a model'))
    result = engine.run_reply_engine(
        db=db, cfg=mock_config, tts=None, text='Search for a document index, Jarvis.',
        dialogue_memory=dialogue_memory,
    )
    assert result == 'The catalogue includes a document guide.'
    assert dispatched and all(name == 'webSearch' and args.get('search_query') for name, args in dispatched)
