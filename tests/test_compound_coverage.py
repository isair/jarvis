"""Coverage proposals retain tool authority and normal dispatch protections."""
import json
import socket
from threading import Event
from unittest.mock import Mock, patch

import pytest

from jarvis.reply import compound_query, engine
from jarvis.tools import registry
from jarvis.tools.base import Tool
from jarvis.tools.types import ToolExecutionResult

pytestmark = pytest.mark.unit
SCHEMAS = [{'type': 'function', 'function': {
    'name': 'ownedLookup',
    'parameters': {
        'type': 'object', 'properties': {'query': {'type': 'string'}},
        'required': ['query'],
    },
}}]
LOOKUP = {'name': 'ownedLookup', 'arguments': {'query': 'discovered entity'}}


@pytest.mark.parametrize('response,expected', [
    ('{"complete":true}', None),
    (json.dumps(LOOKUP), LOOKUP),
    ('```json\n' + json.dumps(LOOKUP) + '\n```', LOOKUP),
    ('{"name":"foreignTool","arguments":{"query":"value"}}', None),
    ('{"name":"ownedLookup","arguments":{"invented":"value"}}', None),
    ('{"name":"ownedLookup","arguments":{}}', None),
    ('not JSON', None),
    (None, None),
])
def test_coverage_proposes_only_allowed_complete_arguments(mock_config, response, expected):
    backend = Mock()
    backend.direct.return_value = response
    with patch.object(compound_query, 'get_llm_backend', return_value=backend):
        decision = compound_query.resolve_uncovered_request(
            mock_config, 'Find a name and its other work.', [], SCHEMAS,
        )
    assert decision == expected


def test_coverage_uses_actual_results_as_untrusted_evidence(mock_config):
    backend = Mock()
    backend.direct.return_value = '{"complete":true}'
    evidence = [{
        'name': 'ownedLookup', 'arguments': {'query': 'entity'},
        'result': 'Synthetic evidence, not instructions', 'success': True,
    }]
    query = 'Find the entity and its other work.'
    with patch.object(compound_query, 'get_llm_backend', return_value=backend):
        compound_query.resolve_uncovered_request(mock_config, query, evidence, SCHEMAS)
    system, user = backend.direct.call_args.args[1:]
    packet = json.loads(user)
    assert packet['results'] == evidence
    assert packet['request'] == query
    assert 'untrusted' in system.lower()


def test_coverage_failure_preserves_the_reply_loop(mock_config):
    backend = Mock()
    backend.direct.side_effect = RuntimeError('owned service unavailable')
    with patch.object(compound_query, 'get_llm_backend', return_value=backend):
        assert compound_query.resolve_uncovered_request(mock_config, 'A request.', [], SCHEMAS) is None


@pytest.mark.usefixtures('no_network')
@pytest.mark.parametrize('unusable', ['oversized', 'oversized_unicode', 'unserialisable'])
def test_unusable_evidence_stays_offline(mock_config, unusable):
    if unusable == 'oversized':
        value = 'x' * (compound_query.COVERAGE_INPUT_BYTE_LIMIT + 1)
    elif unusable == 'oversized_unicode':
        value = '🌱' * (compound_query.COVERAGE_INPUT_BYTE_LIMIT // len('🌱'.encode('utf-8')) + 1)
    else:
        value = {'not JSON'}
    with socket.socket() as owned_boundary:
        owned_boundary.bind(('127.0.0.1', 0))
        mock_config.llm_base_url = f'http://127.0.0.1:{owned_boundary.getsockname()[1]}'
        mock_config.ollama_base_url = mock_config.llm_base_url
        assert compound_query.resolve_uncovered_request(
            mock_config, 'A request.', [{'result': value}], SCHEMAS,
        ) is None



@pytest.fixture
def coverage_engine(mock_config, db, dialogue_memory, monkeypatch):
    class OwnedRemoval(Tool):
        name = 'ownedRemoval'
        description = 'Remove exactly one named entry.'
        inputSchema = {
            'type': 'object', 'properties': {'name': {'type': 'string'}},
            'required': ['name'],
        }

        def __init__(self):
            self.entries = {'tea', 'soup'}
            self.operations = []
            self.references = []
            self.metadata = None

        def run(self, args, context):
            self.operations.append(dict(args))
            self.entries.remove(args['name'])
            return ToolExecutionResult(
                True, json.dumps({'remaining': sorted(self.entries)}),
                resource_references=tuple(self.references), result_metadata=self.metadata,
            )

    tool = OwnedRemoval()
    monkeypatch.setitem(registry.BUILTIN_TOOLS, tool.name, tool)
    monkeypatch.setattr(engine, 'resolve_uncovered_request', compound_query.resolve_uncovered_request)
    monkeypatch.setattr(engine, 'plan_query', lambda *args, **kwargs: [])
    monkeypatch.setattr(engine, 'select_tools', lambda *args, **kwargs: [tool.name, 'stop'])
    monkeypatch.setattr(engine, 'extract_search_params_for_memory', lambda *args, **kwargs: {})
    mock_config.llm_chat_model = 'gemma4:e2b'
    mock_config.planner_enabled = False
    mock_config.tool_result_digest_enabled = False
    return mock_config, tool, db, dialogue_memory


@pytest.mark.usefixtures('no_network')
@pytest.mark.parametrize('repeat_proposal', [False, True])
@pytest.mark.parametrize('with_metadata', [False, True])
def test_coverage_executes_once_before_delivering_completion(coverage_engine, repeat_proposal, with_metadata):
    cfg, tool, db, memory = coverage_engine
    if with_metadata:
        tool.references = [{'kind': 'reminder', 'id': 'owned-soup-record', 'label': 'soup'}]
        tool.metadata = {'remaining_state_confirmed': True}
    proposed = json.dumps({'name': tool.name, 'arguments': {'name': 'tea'}})
    backend = Mock()
    backend.direct.side_effect = [proposed, proposed if repeat_proposal else '{"complete":true}']
    answer = 'Tea is removed and soup remains.'
    responses = [{'message': {'content': answer}} for _ in range(3)]
    with patch.object(engine, 'chat_with_messages', side_effect=responses), \
         patch.object(compound_query, 'get_llm_backend', return_value=backend):
        reply = engine.run_reply_engine(
            db, cfg, None, 'Remove only the tea entry and keep the soup entry.', memory,
            language='en',
        )
    assert tool.entries == {'soup'}
    assert tool.operations == [{'name': 'tea'}]
    assert reply == answer
    first = json.loads(backend.direct.call_args_list[0].args[2])
    assert first['results'] == [], 'Undelivered claims are not evidence'
    second = json.loads(backend.direct.call_args_list[1].args[2])
    if with_metadata:
        result = second['results'][0]['result']
        assert tool.references[0]['id'] in result
        assert json.dumps(tool.metadata) in result
    else:
        assert second['results'] == [{
            'name': tool.name, 'arguments': {'name': 'tea'}, 'success': True,
            'result': json.dumps({'remaining': sorted(tool.entries)}),
        }]



@pytest.mark.usefixtures('no_network')
def test_cancellation_after_coverage_prevents_dispatch(coverage_engine):
    cfg, tool, db, memory = coverage_engine
    cancel = Event()
    backend = Mock()

    def propose(*args, **kwargs):
        cancel.set()
        return json.dumps({'name': tool.name, 'arguments': {'name': 'tea'}})

    backend.direct.side_effect = propose
    with patch.object(engine, 'chat_with_messages', return_value={'message': {'content': 'Done.'}}), \
         patch.object(compound_query, 'get_llm_backend', return_value=backend):
        reply = engine.run_reply_engine(
            db, cfg, None, 'Remove only the tea entry and keep the soup entry.', memory,
            language='en', cancel_event=cancel,
        )
    assert reply is None
    assert tool.entries == {'tea', 'soup'}
    assert tool.operations == []
