"""Declared record actions require actual, namespaced identity before mutation."""
from copy import deepcopy
import json

import pytest

from evals.test_text_tool_schema import OwnedEntryManager, observed_entry_names
from jarvis.reply.record_binding import RECORD_CONTRACT, prepare_record_request, resolve_record_plan

pytestmark = pytest.mark.unit


def declared_tool():
    tool = OwnedEntryManager()
    schema = deepcopy(tool.inputSchema)
    return tool, [{'type': 'function', 'function': {'name': tool.name, 'description': tool.description, 'parameters': schema}}]


def intent(*, kind='C', literal=None, marker=None, topic='tea', action=2):
    return {'english': 'Delete the requested entry.', 'kind': kind, 'literal_target': literal,
            'naming_marker': marker, 'description': topic, 'action_index': action}


def resolve(query, tools, decision, current=(), prior=()):
    request = prepare_record_request(query, tools, list(current), list(prior))
    assert request is not None
    return resolve_record_plan(request, {'plan': [decision]})


def test_unknown_description_discovers_before_exactly_one_mutation():
    tool, tools = declared_tool()
    query = 'Delete only the tea reminder and keep the soup reminder.'
    first = resolve(query, tools, intent())
    assert first.call == {'name': tool.name, 'arguments': {'operation': 'enumerate_entries', 'scope': 'whole_collection'}}
    observed = tool.run(first.call['arguments'], None)
    rows = [{'name': tool.name, 'arguments': first.call['arguments'], 'result': observed.reply_text, 'success': observed.success}]
    second = resolve(query, tools, intent(), rows)
    assert second.call == {'name': tool.name, 'arguments': {'operation': 'discard_entries', 'scope': 'matching_labels', 'labels': ['tea']}}
    effect = tool.run(second.call['arguments'], None)
    rows.append({'name': tool.name, 'arguments': second.call['arguments'], 'result': effect.reply_text, 'success': effect.success})
    assert resolve(query, tools, intent(), rows).status == 'complete'
    assert len(tool.calls) == 2
    assert tool.entries == {'soup'}


@pytest.mark.parametrize('entries', [{'tea', 'soup'}, {'tea reminder', 'soup reminder'}])
def test_prior_observation_binds_complete_actual_name_without_lookup(entries):
    tool, tools = declared_tool()
    tool.entries = set(entries)
    prior = [observed_entry_names(tool)]
    tool.calls.clear()
    proposed = resolve('Delete only the tea reminder and keep the soup reminder.', tools, intent(), prior=prior)
    assert tool.run(proposed.call['arguments'], None).success
    assert len(tool.calls) == 1
    assert tool.entries == {name for name in entries if name.startswith('soup')}


@pytest.mark.parametrize('query,marker', [
    ('Delete only the entry named tea reminder and keep soup.', 'named'),
    ('Delete only the entry named "tea reminder" and keep soup.', None),
])
def test_literal_full_name_needs_no_namespace_observation(query, marker):
    tool, tools = declared_tool()
    tool.entries = {'tea reminder', 'soup'}
    proposed = resolve(query, tools, intent(kind='B', literal='tea reminder', marker=marker))
    assert tool.run(proposed.call['arguments'], None).success
    assert tool.entries == {'soup'}
    assert len(tool.calls) == 1


@pytest.mark.parametrize('entries', [{'tea', 'soup'}, {'reminder', 'soup'}, {'tea', 'tea reminder', 'soup'}])
def test_complete_missing_or_ambiguous_namespace_never_mutates(entries):
    tool, tools = declared_tool()
    tool.entries = set(entries)
    args = {'operation': 'enumerate_entries', 'scope': 'whole_collection'}
    observed = tool.run(args, None)
    topic = 'tea' if 'tea reminder' in entries else 'cocoa'
    query = f'Delete only the {topic} reminder and keep soup.'
    proposed = resolve(query, tools, intent(kind='B', literal=f'{topic} reminder', topic=topic),
        [{'name': tool.name, 'arguments': args, 'success': True, 'result': observed.reply_text}])
    assert proposed.status == 'blocked'
    assert proposed.call is None
    assert tool.entries == entries
    assert len(tool.calls) == 1


def test_fresh_complete_snapshot_supersedes_old_names_and_other_namespaces():
    tool, tools = declared_tool()
    prior = [observed_entry_names(tool), {'name': 'otherStore', 'result': '{"remaining_entries":["cocoa"]}'}]
    tool.entries = {'soup'}
    args = {'operation': 'enumerate_entries', 'scope': 'whole_collection'}
    observed = tool.run(args, None)
    rows = [{'name': tool.name, 'arguments': args, 'success': True, 'result': observed.reply_text}]
    assert resolve('Delete the tea reminder.', tools, intent(), rows, prior).status == 'blocked'
    assert resolve('Delete the cocoa reminder.', tools, intent(topic='cocoa'), rows, prior).status == 'blocked'
    assert tool.entries == {'soup'}


def test_previous_successful_action_does_not_execute_current_request():
    tool, tools = declared_tool()
    prior = [observed_entry_names(tool)]
    previous = tool.run({'operation': 'discard_entries', 'scope': 'matching_labels', 'labels': ['tea']}, None)
    prior.append({'name': tool.name, 'result': previous.reply_text})
    tool.entries.add('tea')
    tool.calls.clear()
    proposed = resolve('Delete the entry named tea.', tools, intent(kind='B', literal='tea', marker='named'), prior=prior)
    assert tool.run(proposed.call['arguments'], None).success
    assert tool.entries == {'soup'}
    assert len(tool.calls) == 1


@pytest.mark.parametrize('damage', ['foreign_discovery', 'unknown_selector', 'invalid_arguments', 'invalid_pointer'])
def test_malformed_declarations_do_not_authorise_calls(damage):
    _, tools = declared_tool()
    contract = tools[0]['function']['parameters'][RECORD_CONTRACT]
    if damage == 'foreign_discovery': contract['discovery'] = 99
    if damage == 'unknown_selector': contract['actions'][2]['selector'] = 'absent'
    if damage == 'invalid_arguments': contract['actions'][2]['arguments']['operation'] = 'invented_operation'
    if damage == 'invalid_pointer': contract['collection']['path'] = '/bad~2pointer'
    assert prepare_record_request('Delete my tea reminder.', tools, [], []) is None


def test_aggregate_required_input_limit_cannot_be_bypassed_by_narrow_prompt():
    _, tools = declared_tool()
    rows = [{'name': 'ownedEntryManager', 'success': True, 'result': 'x' * 6001}]
    assert prepare_record_request('Delete my tea reminder.', tools, rows, []) is None


def test_cached_plan_cannot_cross_request_boundary():
    from jarvis.reply.record_binding import decode_record_plan
    _, tools = declared_tool()
    old = prepare_record_request('Delete the entry named tea.', tools, [], [])
    plan = decode_record_plan(old, {'plan': [intent(kind='B', literal='tea', marker='named')]})
    current = prepare_record_request('Keep tea, list everything.', tools, [], [])
    assert resolve_record_plan(current, plan).status == 'blocked'


@pytest.mark.parametrize('planner_enabled', [False, True])
@pytest.mark.parametrize('model', ['gemma4:e2b', 'gpt-oss:20b'])
def test_engine_reuses_request_interpretation_after_discovery(
        mock_config, db, dialogue_memory, monkeypatch, planner_enabled, model):
    from unittest.mock import patch
    from jarvis.reply import engine
    from jarvis.tools import registry
    tool, _ = declared_tool()
    monkeypatch.setitem(registry.BUILTIN_TOOLS, tool.name, tool)
    mock_config.llm_chat_model = model
    mock_config.planner_enabled = planner_enabled
    mock_config.agentic_max_turns = 5
    model_inputs = []

    def chat(*args, **kwargs):
        model_inputs.append(kwargs)
        if len(model_inputs) == 1:
            return {'message': {'content': json.dumps({'plan': [intent()]})}}
        return {'message': {'content': 'The tea entry was removed, soup is preserved.'}}

    with patch.object(engine, 'select_tools', return_value=[tool.name, 'stop']), \
         patch.object(engine, 'chat_with_messages', side_effect=chat), \
         patch.object(engine, 'plan_query', side_effect=AssertionError('No separate record planner inference')), \
         patch.object(engine, 'run_tool_with_retries', side_effect=lambda **kw: tool.run(kw['tool_args'], None)), \
         patch.object(engine, 'extract_search_params_for_memory', return_value={}):
        reply = engine.run_reply_engine(db, mock_config, None,
            'Delete only the tea reminder and keep the soup reminder.', dialogue_memory)
    assert reply == 'The tea entry was removed, soup is preserved.'
    assert tool.entries == {'soup'}
    assert tool.calls == [
        {'operation': 'enumerate_entries', 'scope': 'whole_collection'},
        {'operation': 'discard_entries', 'scope': 'matching_labels', 'labels': ['tea']},
    ]
    assert len(model_inputs) == 2, 'Interpret once, compile discovery and mutation, then render the result'


@pytest.mark.parametrize('flags', [{'success': False}, {'content_truncated': True}, {'tool_feedback': True}])
def test_unusable_discovery_never_authorises_mutation_or_repeated_lookup(flags):
    tool, tools = declared_tool()
    row = {'name': tool.name, 'arguments': {'operation': 'enumerate_entries', 'scope': 'whole_collection'},
           'success': True, 'result': '{"remaining_entries":["tea"]}', **flags}
    resolution = resolve('Delete my tea reminder.', tools, intent(), [row])
    assert resolution.status == 'blocked'
    assert tool.entries == {'tea', 'soup'}
    assert not tool.calls


def test_discovery_cannot_complete_a_later_requested_read():
    tool, tools = declared_tool()
    query = 'Delete the tea reminder, then list all entries.'
    decisions = {'plan': [intent(), intent(kind='A', topic=None, action=0)]}
    rows = []
    for expected in ['enumerate_entries', 'discard_entries', 'enumerate_entries']:
        request = prepare_record_request(query, tools, rows, [])
        proposal = resolve_record_plan(request, decisions)
        assert proposal.call['arguments']['operation'] == expected
        effect = tool.run(proposal.call['arguments'], None)
        rows.append({'name': tool.name, 'arguments': proposal.call['arguments'], 'success': True, 'result': effect.reply_text})
    assert resolve_record_plan(prepare_record_request(query, tools, rows, []), decisions).status == 'complete'
    assert tool.entries == {'soup'}


def test_failed_discovery_does_not_use_stale_prior_names():
    tool, tools = declared_tool()
    prior = [observed_entry_names(tool)]
    row = {'name': tool.name, 'arguments': {'operation': 'enumerate_entries', 'scope': 'whole_collection'},
           'success': False, 'result': 'Read failed'}
    assert resolve('Delete my tea reminder.', tools, intent(), [row], prior).status == 'blocked'


def test_latest_complete_mutation_snapshot_replaces_earlier_collection():
    tool, tools = declared_tool()
    args = {'operation': 'enumerate_entries', 'scope': 'whole_collection'}
    before = tool.run(args, None)
    removal = {'operation': 'discard_entries', 'scope': 'matching_labels', 'labels': ['tea']}
    after = tool.run(removal, None)
    rows = [{'name': tool.name, 'arguments': args, 'success': True, 'result': before.reply_text},
            {'name': tool.name, 'arguments': removal, 'success': True, 'result': after.reply_text}]
    # A second requested removal cannot reuse an identity removed by the first.
    decisions = {'plan': [intent(), intent()]}
    resolution = resolve_record_plan(prepare_record_request('Delete tea reminder twice.', tools, rows, []), decisions)
    assert resolution.status == 'blocked'
    assert tool.entries == {'soup'}


def test_observed_numeric_identifier_completes_effect_using_its_display_name():
    _, tools = declared_tool()
    schema = tools[0]['function']['parameters']
    schema['properties']['labels']['items'] = {'type': 'integer'}
    contract = schema[RECORD_CONTRACT]
    contract['collection']['display'] = '/title'
    contract['actions'][2]['identity'] = '/key'
    lookup = {'operation': 'enumerate_entries', 'scope': 'whole_collection'}
    rows = [{'name': tools[0]['function']['name'], 'arguments': lookup, 'success': True,
             'result': '{"remaining_entries":[{"key":47,"title":"tea reminder"}]}'}]
    query = 'Delete the tea reminder.'
    call = resolve(query, tools, intent(), rows).call
    assert call['arguments']['labels'] == [47]
    rows.append({'name': call['name'], 'arguments': call['arguments'], 'success': True,
                 'result': '{"remaining_entries":[]}'})
    assert resolve(query, tools, intent(), rows).status == 'complete'


def test_mcp_discovery_keeps_the_canonical_record_contract(monkeypatch):
    from jarvis.tools import registry
    tool, _ = declared_tool()
    original = deepcopy(tool.inputSchema)

    class OwnedMCPClient:
        def __init__(self, config):
            pass

        def list_tools(self, server):
            return [{'name': tool.name, 'description': tool.description, 'inputSchema': deepcopy(original)}]

    monkeypatch.setattr(registry, 'MCPClient', OwnedMCPClient)
    discovered, errors = registry.discover_mcp_tools({'owned': {}})
    name = 'owned__' + tool.name
    assert not errors
    schemas = registry.generate_tools_json_schema([name], discovered)
    assert schemas[0]['function']['parameters'] == original
    assert original == tool.inputSchema
    proposed = resolve('Delete the tea reminder.', schemas, intent())
    assert proposed.call == {'name': name, 'arguments': {'operation': 'enumerate_entries', 'scope': 'whole_collection'}}
    observed = tool.run(proposed.call['arguments'], None)
    row = {'name': name, 'arguments': proposed.call['arguments'], 'success': True, 'result': observed.reply_text}
    mutation = resolve('Delete the tea reminder.', schemas, intent(), [row])
    assert tool.run(mutation.call['arguments'], None).success
    assert tool.entries == {'soup'}
    assert RECORD_CONTRACT in registry.generate_tools_description([name], discovered)


@pytest.mark.parametrize('mode', ['cancel_after_read', 'raw_mutation', 'stop', 'discover_other_tool'])
def test_engine_record_controls_preserve_authority(mock_config, db, dialogue_memory, monkeypatch, mode):
    from threading import Event
    from unittest.mock import patch
    from jarvis.reply import engine
    from jarvis.tools import registry
    from jarvis.tools.builtin.stop import STOP_SIGNAL
    from jarvis.tools.types import ToolExecutionResult
    tool, _ = declared_tool()
    monkeypatch.setitem(registry.BUILTIN_TOOLS, tool.name, tool)
    mock_config.llm_chat_model = 'gemma4:e2b'
    mock_config.planner_enabled = False
    mock_config.agentic_max_turns = 4
    cancel = Event()
    executed = []
    proposals = {
        'cancel_after_read': {'plan': [intent()]},
        'raw_mutation': {'tool_calls': [{'function': {'name': tool.name, 'arguments':
            {'operation': 'discard_entries', 'scope': 'whole_collection'}}}]},
        'stop': {'tool_calls': [{'function': {'name': 'stop', 'arguments': {}}}]},
        'discover_other_tool': {'tool_calls': [{'function': {'name': 'toolSearchTool', 'arguments': {'query': 'local time'}}}]},
    }
    responses = iter([{'message': {'content': json.dumps(proposals[mode])}},
                      {'message': {'content': 'No further operation was performed.'}}])

    def run(**kwargs):
        name, args = kwargs['tool_name'], kwargs['tool_args']
        executed.append(name)
        if name == tool.name:
            result = tool.run(args, None)
            if mode == 'cancel_after_read':
                cancel.set()
            return result
        return ToolExecutionResult(True, STOP_SIGNAL if name == 'stop' else 'No additional tools found.')

    with patch.object(engine, 'select_tools', return_value=[tool.name, 'stop']), \
         patch.object(engine, 'chat_with_messages', side_effect=lambda **kw: next(responses)), \
         patch.object(engine, 'run_tool_with_retries', side_effect=run), \
         patch.object(engine, 'extract_search_params_for_memory', return_value={}):
        reply = engine.run_reply_engine(db, mock_config, None,
            'Delete only the tea reminder and keep the soup reminder.', dialogue_memory, cancel_event=cancel)
    assert tool.entries == {'tea', 'soup'}
    if mode == 'cancel_after_read':
        assert executed == [tool.name]
        assert reply is None
    elif mode == 'raw_mutation':
        assert executed == []
    else:
        assert executed == ['stop' if mode == 'stop' else 'toolSearchTool']


def test_latest_prior_snapshot_does_not_resurrect_removed_names():
    tool, tools = declared_tool()
    prior = [{'name': tool.name, 'result': '{"remaining_entries":["tea","soup"]}'},
             {'name': tool.name, 'result': '{"remaining_entries":["soup"]}'}]
    call = resolve('Delete the tea reminder.', tools, intent(), prior=prior).call
    assert call['arguments'] == {'operation': 'enumerate_entries', 'scope': 'whole_collection'}


def test_feedback_cannot_complete_a_requested_effect():
    tool, tools = declared_tool()
    row = {'name': tool.name, 'arguments': {'operation': 'discard_entries', 'scope': 'matching_labels', 'labels': ['tea']},
           'success': True, 'tool_feedback': True, 'result': 'An earlier call succeeded.'}
    resolution = resolve('Delete the entry named tea.', tools, intent(kind='B', literal='tea', marker='named'), [row])
    assert resolution.status == 'call'
    assert tool.entries == {'tea', 'soup'}


def test_qualified_resource_reference_supplies_identity_without_a_collection_payload():
    tool, tools = declared_tool()
    contract = tools[0]['function']['parameters'][RECORD_CONTRACT]
    contract.update(reference_kind='entry', collection={'source': 'result_metadata', 'path': '', 'display': '/title', 'complete': True})
    contract['actions'][2]['identity'] = '/key'
    prior = [{'name': tool.name, 'resource_references': [{'kind': 'entry', 'key': 'tea', 'title': 'tea reminder'}]}]
    call = resolve('Delete the tea reminder.', tools, intent(), prior=prior).call
    assert call['arguments'] == {'operation': 'discard_entries', 'scope': 'matching_labels', 'labels': ['tea']}


def test_complete_snapshot_overrules_conflicting_resource_references():
    tool, tools = declared_tool()
    contract = tools[0]['function']['parameters'][RECORD_CONTRACT]
    contract['reference_kind'] = 'entry'
    contract['collection']['display'] = '/title'
    contract['actions'][2]['identity'] = '/key'
    row = {'name': tool.name, 'arguments': {'operation': 'enumerate_entries', 'scope': 'whole_collection'},
           'success': True, 'result': '{"remaining_entries":[]}',
           'resource_references': [{'kind': 'entry', 'key': 'tea', 'title': 'tea reminder'}]}
    assert resolve('Delete the tea reminder.', tools, intent(), [row]).status == 'blocked'


def test_selector_constraints_validate_observed_values_without_inventing_a_sample():
    _, tools = declared_tool()
    schema = tools[0]['function']['parameters']
    schema['properties']['labels']['items'] = {'$ref': '#/$defs/identifier'}
    schema['$defs'] = {'identifier': {'type': 'string', 'pattern': '^[a-z]{3}-[0-9]{4}$'}}
    contract = schema[RECORD_CONTRACT]
    contract['collection']['display'] = '/title'
    contract['actions'][2]['identity'] = '/key'
    row = {'name': tools[0]['function']['name'], 'success': True,
           'arguments': {'operation': 'enumerate_entries', 'scope': 'whole_collection'},
           'result': '{"remaining_entries":[{"key":"tea-0001","title":"tea reminder"}]}'}
    proposed = resolve('Delete the tea reminder.', tools, intent(), [row])
    assert proposed.call['arguments']['labels'] == ['tea-0001']


def test_literal_numeric_identifier_obeys_its_local_reference():
    _, tools = declared_tool()
    schema = tools[0]['function']['parameters']
    schema['properties']['labels']['items'] = {'$ref': '#/$defs/identifier'}
    schema['$defs'] = {'identifier': {'type': 'integer'}}
    call = resolve('Delete the record with ID 47.', tools,
                   intent(kind='B', literal='47', marker='ID', topic='47')).call
    assert call['arguments']['labels'] == [47]


@pytest.mark.parametrize('source', ['diary', 'graph', 'all'])
def test_declared_record_request_needs_no_query_memory_inference(
        mock_config, db, dialogue_memory, monkeypatch, source):
    from jarvis.reply import engine
    from jarvis.tools import registry
    tool, _ = declared_tool()
    monkeypatch.setitem(registry.BUILTIN_TOOLS, tool.name, tool)
    mock_config.llm_chat_model = 'gemma4:e2b'
    mock_config.planner_enabled = False
    mock_config.memory_enrichment_source = source
    db.upsert_conversation_summary('2020-01-01', 'A real owned saved fact.', '', source_app='fixture')
    responses = iter([{'message': {'content': json.dumps({'plan': [intent()]})}},
                      {'message': {'content': 'Tea is removed.'}}])

    class UnrelatedInference(BaseException):
        pass

    def forbidden(*args, **kwargs):
        raise UnrelatedInference('A record action must not infer unused memory keywords or digest')

    monkeypatch.setattr(engine, 'extract_search_params_for_memory', forbidden)
    monkeypatch.setattr(engine, 'digest_memory_for_query', forbidden)
    monkeypatch.setattr(engine, 'select_tools', lambda *args, **kwargs: [tool.name, 'stop'])
    monkeypatch.setattr(engine, 'chat_with_messages', lambda **kwargs: next(responses))
    monkeypatch.setattr(engine, 'run_tool_with_retries', lambda **kwargs: tool.run(kwargs['tool_args'], None))
    assert engine.run_reply_engine(db, mock_config, None, 'Delete the tea reminder.', dialogue_memory) == 'Tea is removed.'
    assert tool.entries == {'soup'}
    assert len(tool.calls) == 2


def test_engine_executes_a_fresh_requested_read_after_mutation(mock_config, db, dialogue_memory, monkeypatch):
    from jarvis.reply import engine
    from jarvis.tools import registry
    tool, _ = declared_tool()
    monkeypatch.setitem(registry.BUILTIN_TOOLS, tool.name, tool)
    mock_config.llm_chat_model = 'gemma4:e2b'
    mock_config.planner_enabled = False
    mock_config.agentic_max_turns = 5
    proposals = {'plan': [intent(), intent(kind='A', topic=None, action=0)]}
    responses = iter([{'message': {'content': json.dumps(proposals)}}, {'message': {'content': 'Only soup remains.'}}])
    monkeypatch.setattr(engine, 'select_tools', lambda *args, **kwargs: [tool.name, 'stop'])
    monkeypatch.setattr(engine, 'chat_with_messages', lambda **kwargs: next(responses))
    monkeypatch.setattr(engine, 'run_tool_with_retries', lambda **kwargs: tool.run(kwargs['tool_args'], None))
    monkeypatch.setattr(engine, 'digest_loop_for_max_turns', lambda **kwargs: 'The tool sequence is incomplete.')
    reply = engine.run_reply_engine(db, mock_config, None, 'Delete the tea reminder, then list the remaining entries.', dialogue_memory)
    assert reply == 'Only soup remains.'
    assert tool.entries == {'soup'}
    assert tool.calls == [{'operation': 'enumerate_entries', 'scope': 'whole_collection'},
                          {'operation': 'discard_entries', 'scope': 'matching_labels', 'labels': ['tea']},
                          {'operation': 'enumerate_entries', 'scope': 'whole_collection'}]


@pytest.mark.parametrize('overflow', ['query', 'catalogue'])
def test_initial_record_packet_overflow_cannot_admit_raw_mutation(mock_config, db, dialogue_memory, monkeypatch, overflow):
    from jarvis.reply import engine
    from jarvis.tools import registry
    tool, _ = declared_tool()
    monkeypatch.setitem(registry.BUILTIN_TOOLS, tool.name, tool)
    mock_config.llm_chat_model = 'gemma4:e2b'
    mock_config.planner_enabled = False
    query = 'Delete only the tea reminder.'
    if overflow == 'query':
        query += ' ' + 'x' * 6000
    else:
        tool.description += 'x' * 6000
    responses = iter([{'message': {'content': json.dumps({'tool_calls': [{'function': {'name': tool.name,
        'arguments': {'operation': 'discard_entries', 'scope': 'whole_collection'}}}]})}},
                      {'message': {'content': 'The request exceeds the supported input budget.'}}])
    monkeypatch.setattr(engine, 'select_tools', lambda *args, **kwargs: [tool.name, 'stop'])
    monkeypatch.setattr(engine, 'chat_with_messages', lambda **kwargs: next(responses))
    monkeypatch.setattr(engine, 'run_tool_with_retries', lambda **kwargs: tool.run(kwargs['tool_args'], None))
    engine.run_reply_engine(db, mock_config, None, query, dialogue_memory)
    assert tool.entries == {'tea', 'soup'}
    assert tool.calls == []


@pytest.mark.parametrize('model', ['gemma4:e2b', 'gpt-oss:20b'])
@pytest.mark.parametrize('source', ['result', 'result_metadata'])
def test_real_engine_carryover_keeps_known_record_identity(mock_config, db, dialogue_memory, monkeypatch, model, source):
    from jarvis.reply import engine
    from jarvis.tools import registry
    tool, _ = declared_tool()
    if source == 'result_metadata':
        from dataclasses import replace
        tool.inputSchema = deepcopy(tool.inputSchema)
        contract = tool.inputSchema[RECORD_CONTRACT]
        contract['collection'] = {'source': 'result_metadata', 'path': '/records', 'display': '/title', 'complete': True}
        contract['actions'][2]['identity'] = '/key'
        execute = tool.run

        def run(args, context):
            effect = execute(args, context)
            return replace(effect, reply_text='The current entries are available.', result_metadata={
                'records': [{'key': value, 'title': value} for value in sorted(tool.entries)],
                'unrelated_private_text': 'This is not identity data.',
            })

        tool.run = run
    monkeypatch.setitem(registry.BUILTIN_TOOLS, tool.name, tool)
    mock_config.llm_chat_model = model
    mock_config.planner_enabled = False
    responses = iter([{'message': {'content': json.dumps({'plan': [intent(kind='A', topic=None, action=0)]})}},
                      {'message': {'content': 'The current entries are listed.'}},
                      {'message': {'content': json.dumps({'plan': [intent()]})}},
                      {'message': {'content': 'Tea is removed.'}}])
    monkeypatch.setattr(engine, 'select_tools', lambda *args, **kwargs: [tool.name, 'stop'])
    monkeypatch.setattr(engine, 'chat_with_messages', lambda **kwargs: next(responses))
    monkeypatch.setattr(engine, 'run_tool_with_retries', lambda **kwargs: tool.run(kwargs['tool_args'], None))
    assert engine.run_reply_engine(db, mock_config, None, 'List all entries.', dialogue_memory)
    tool.calls.clear()
    assert engine.run_reply_engine(db, mock_config, None, 'Delete the tea reminder.', dialogue_memory) == 'Tea is removed.'
    assert tool.entries == {'soup'}
    assert tool.calls == [{'operation': 'discard_entries', 'scope': 'matching_labels', 'labels': ['tea']}]


def test_projected_identity_survives_prose_truncation_but_is_never_partially_retained():
    from jarvis.reply.record_binding import capture_record_observation
    from jarvis.reply.tool_identity import observed_tool_identities
    from jarvis.memory.conversation import DialogueMemory
    from jarvis.tools.types import ToolExecutionResult
    tool, tools = declared_tool()
    schema = tools[0]['function']['parameters']
    effect = ToolExecutionResult(True, '{"remaining_entries":["tea","soup"],"private_extra":"not an identifier"}')
    receipt = capture_record_observation(tool.name, schema, effect)
    assert receipt is not None
    assert 'private_extra' not in json.dumps(receipt)
    memory = DialogueMemory()
    memory.record_tool_turn([
        {'role': 'assistant', 'tool_calls': [{'id': 'owned-read', 'function': {'name': tool.name, 'arguments': {}}}]},
        {'role': 'tool', 'tool_name': tool.name, 'tool_call_id': 'owned-read', 'tool_failed': False,
         'content': 'Rendered result and guidance ' * 100, 'record_evidence': receipt},
    ])
    limit = len(json.dumps(receipt, ensure_ascii=False)) + 1
    messages = memory.get_recent_turns_with_tools(per_entry_chars=limit)
    prior = observed_tool_identities(messages)
    assert resolve('Delete the tea reminder.', tools, intent(), prior=prior).call['arguments']['labels'] == ['tea']
    assert observed_tool_identities(memory.get_recent_turns_with_tools(per_entry_chars=1)) == []
    again = memory.get_recent_turns_with_tools(per_entry_chars=limit)
    assert observed_tool_identities(again) == prior, 'A shorter read must not truncate stored identifiers'


def test_redaction_never_turns_a_private_identifier_into_an_admitted_name():
    from jarvis.reply.record_binding import capture_record_observation
    from jarvis.reply.tool_identity import observed_tool_identities
    from jarvis.memory.conversation import DialogueMemory
    from jarvis.tools.types import ToolExecutionResult
    tool, tools = declared_tool()
    secret = 'ab' * 32
    receipt = capture_record_observation(tool.name, tools[0]['function']['parameters'],
        ToolExecutionResult(True, json.dumps({'remaining_entries': [secret]})))
    memory = DialogueMemory()
    memory.record_tool_turn([
        {'role': 'assistant', 'tool_calls': [{'id': 'owned-secret-read', 'function': {'name': tool.name, 'arguments': {}}}]},
        {'role': 'tool', 'tool_name': tool.name, 'tool_call_id': 'owned-secret-read', 'tool_failed': False,
         'content': json.dumps({'remaining_entries': [secret]}), 'record_evidence': receipt},
    ])
    stored = memory.get_recent_turns_with_tools()
    assert secret not in json.dumps(stored)
    assert observed_tool_identities(stored) == [], 'Scrubbed text cannot supply a substitute identifier'
