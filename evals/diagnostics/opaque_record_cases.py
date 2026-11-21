"""Native text-protocol calls use the selected tool's declared argument contract."""
import json
from dataclasses import replace

import pytest

from evals.helpers import isolated_voice_settings, assert_not_fallback_reply
from jarvis.memory.conversation import DialogueMemory
from jarvis.memory.db import Database
from jarvis.reply import engine
from jarvis.tools import registry
from jarvis.tools.base import Tool
from jarvis.tools.types import ToolExecutionResult

pytestmark = pytest.mark.eval


class OwnedEntryManager(Tool):
    name = 'ownedEntryManager'
    description = 'Retrieve current local reminder entries or remove the requested entries.'
    inputSchema = {
        'type': 'object',
        'properties': {
            'operation': {
                'type': 'string',
                'enum': ['enumerate_entries', 'discard_entries'],
                'description': 'enumerate_entries lists entries; discard_entries removes entries.',
            },
            'scope': {
                'type': 'string',
                'enum': ['whole_collection', 'matching_labels'],
                'description': 'whole_collection selects all entries; matching_labels selects only supplied labels.',
            },
            'labels': {'type': 'array', 'items': {'type': 'string'},
                       'description': 'Entry labels selected when scope is matching_labels.'},
        },
        'required': ['operation', 'scope'],
        'additionalProperties': False,
    }

    def __init__(self):
        self.entries = {'tea', 'soup'}
        self.calls = []

    def run(self, args, context):
        self.calls.append(args)
        operation, scope = args.get('operation'), args.get('scope')
        properties = self.inputSchema['properties']
        if operation not in properties['operation']['enum'] or scope not in properties['scope']['enum']:
            return ToolExecutionResult(False, None, 'Arguments do not match the declared operation and scope.')
        if scope == 'matching_labels':
            labels = args.get('labels')
            if not isinstance(labels, list) or not all(isinstance(label, str) for label in labels):
                return ToolExecutionResult(False, None, 'Matching scope requires a list of labels.')
            selected = self.entries.intersection(labels)
        else:
            selected = self.entries.copy()
        if operation == 'discard_entries':
            self.entries.difference_update(selected)
        return ToolExecutionResult(True, json.dumps({'remaining_entries': sorted(self.entries)}))


def observed_entry_names(tool):
    """Obtain namespace evidence from the owned backend before the current request."""
    arguments = {'operation': 'enumerate_entries', 'scope': 'whole_collection'}
    result = tool.run(arguments, None)
    assert result.success
    return {'name': tool.name, 'tool_call_id': 'observed-entry-list',
            'result': result.reply_text, 'resource_references': list(result.resource_references)}


def remember_observed_entry_names(tool, dialogue):
    dialogue.add_message('user', 'List my current reminder entries.')
    observed = observed_entry_names(tool)
    dialogue.record_tool_turn([
        {'role': 'assistant', 'content': '', 'tool_calls': [{
            'id': observed['tool_call_id'], 'type': 'function', 'function': {
                'name': tool.name,
                'arguments': {'operation': 'enumerate_entries', 'scope': 'whole_collection'},
            },
        }]},
        {'role': 'tool', 'tool_name': tool.name, 'tool_call_id': observed['tool_call_id'],
         'tool_failed': False, 'content': observed['result']},
    ])
    dialogue.add_message('assistant', 'The current entries are listed in the tool result.')
    # Calls measure this request; the prior read belongs to conversation setup.
    tool.calls.clear()


@pytest.mark.parametrize('query,operation,scope,remaining', [
    ('List all my reminder entries.', 'enumerate_entries', 'whole_collection', {'tea', 'soup'}),
    ('Delete every reminder entry.', 'discard_entries', 'whole_collection', set()),
    ('Delete only the tea reminder and keep the soup reminder.', 'discard_entries', 'matching_labels', {'soup'}),
])
@pytest.mark.parametrize('spoken_output', [False, True])
def test_text_tool_call_preserves_declared_operation_and_scope(monkeypatch, query, operation, scope, remaining, record_property, spoken_output):
    tool = OwnedEntryManager()
    monkeypatch.setitem(registry.BUILTIN_TOOLS, tool.name, tool)
    monkeypatch.setattr(engine, 'select_tools', lambda *args, **kwargs: [tool.name, 'stop'])
    cfg = replace(isolated_voice_settings(), location_enabled=False, mcps={}, planner_enabled=False,
                  llm_chat_timeout_sec=60, planner_timeout_sec=15)
    db = Database(':memory:')
    try:
        dialogue = DialogueMemory()
        if scope == 'matching_labels':
            remember_observed_entry_names(tool, dialogue)
        reply = engine.run_reply_engine(db, cfg, None, query, dialogue, language='en', spoken_output=spoken_output)
        record_property('reply', reply)
        record_property('tool_calls', json.dumps(tool.calls))
        assert reply
        assert_not_fallback_reply(reply, 'text tool schema')
        assert len(tool.calls) == 1, 'Declared schema must support one correct call without argument repair or duplicate execution'
        assert tool.calls[0]['operation'] == operation
        assert tool.calls[0]['scope'] == scope
        assert tool.entries == remaining
    finally:
        db.close()


@pytest.mark.parametrize('spoken_output', [False, True])
@pytest.mark.parametrize('query', [
    'Delete only the entry named "tea reminder" and keep the entry named "soup reminder".',
    'Delete only the entry named tea reminder and keep the entry named soup reminder.',
])
def test_literal_multiword_labels_keep_their_complete_names(monkeypatch, record_property, spoken_output, query):
    tool = OwnedEntryManager()
    tool.entries = {'tea reminder', 'soup reminder'}
    monkeypatch.setitem(registry.BUILTIN_TOOLS, tool.name, tool)
    monkeypatch.setattr(engine, 'select_tools', lambda *args, **kwargs: [tool.name, 'stop'])
    cfg = replace(isolated_voice_settings(), location_enabled=False, mcps={}, planner_enabled=False,
                  llm_chat_timeout_sec=60, planner_timeout_sec=15)
    db = Database(':memory:')
    try:
        reply = engine.run_reply_engine(
            db, cfg, None,
            query,
            DialogueMemory(), language='en', spoken_output=spoken_output,
        )
        record_property('reply', reply)
        record_property('tool_calls', json.dumps(tool.calls))
        assert reply
        assert_not_fallback_reply(reply, 'literal multiword tool labels')
        assert len(tool.calls) == 1, 'Exact labels require one call without repair or duplicate execution'
        assert tool.calls[0] == {
            'operation': 'discard_entries', 'scope': 'matching_labels', 'labels': ['tea reminder'],
        }
        assert tool.entries == {'soup reminder'}
    finally:
        db.close()


@pytest.mark.parametrize('spoken_output', [False, True])
@pytest.mark.parametrize('entries,target,remaining', [
    ({'tea', 'soup'}, 'tea', {'soup'}),
    ({'tea reminder', 'soup reminder'}, 'tea reminder', {'soup reminder'}),
])
def test_unknown_record_names_are_observed_before_one_mutation(
        monkeypatch, spoken_output, record_property, entries, target, remaining):
    tool = OwnedEntryManager()
    tool.entries = set(entries)
    monkeypatch.setitem(registry.BUILTIN_TOOLS, tool.name, tool)
    monkeypatch.setattr(engine, 'select_tools', lambda *args, **kwargs: [tool.name, 'stop'])
    cfg = replace(isolated_voice_settings(), location_enabled=False, mcps={}, planner_enabled=False,
                  llm_chat_timeout_sec=60, planner_timeout_sec=15)
    db = Database(':memory:')
    try:
        reply = engine.run_reply_engine(
            db, cfg, None, 'Delete only the tea reminder and keep the soup reminder.',
            DialogueMemory(), language='en', spoken_output=spoken_output,
        )
        record_property('reply', reply)
        record_property('tool_calls', json.dumps(tool.calls))
        assert reply
        assert_not_fallback_reply(reply, 'unknown record identity')
        assert len(tool.calls) == 2, 'Unknown identity requires one lookup followed by one mutation'
        assert tool.calls[0]['operation'] == 'enumerate_entries'
        assert tool.calls[0]['scope'] == 'whole_collection'
        assert tool.calls[1]['operation'] == 'discard_entries'
        assert tool.calls[1]['scope'] == 'matching_labels'
        assert tool.calls[1]['labels'] == [target]
        assert tool.entries == remaining
    finally:
        db.close()


from evals.test_compound_result_guidance import OwnedFilmLookup
from jarvis.reply.compound_query import resolve_uncovered_request

@pytest.mark.parametrize('query,entries,operation,scope,remaining,known_namespace', [
    ('List all my reminder entries.', {'tea', 'soup'}, 'enumerate_entries', 'whole_collection', {'tea', 'soup'}, False),
    ('Delete every reminder entry.', {'tea', 'soup'}, 'discard_entries', 'whole_collection', set(), False),
    ('Delete only the tea reminder and keep the soup reminder.', {'tea', 'soup'}, 'discard_entries', 'matching_labels', {'soup'}, True),
    ('Delete only the entry named "tea reminder" and keep the entry named "soup reminder".', {'tea reminder', 'soup reminder'}, 'discard_entries', 'matching_labels', {'soup reminder'}, False),
    ('Delete only the entry named tea reminder and keep the entry named soup reminder.', {'tea reminder', 'soup reminder'}, 'discard_entries', 'matching_labels', {'soup reminder'}, False),
])
def test_recovery_selector_arguments_produce_the_requested_effect(
        monkeypatch, query, entries, operation, scope, remaining, known_namespace):
    tool = OwnedEntryManager()
    tool.entries = set(entries)
    monkeypatch.setitem(registry.BUILTIN_TOOLS, tool.name, tool)
    schemas = registry.generate_tools_json_schema([tool.name, 'stop', 'toolSearchTool'])
    prior = [observed_entry_names(tool)] if known_namespace else []
    tool.calls.clear()
    proposal = resolve_uncovered_request(isolated_voice_settings(), query, [], schemas, prior_results=prior)
    assert proposal is not None, 'An unresolved available action needs a concrete recovery proposal'
    assert proposal['name'] == tool.name
    assert proposal['arguments']['operation'] == operation
    assert proposal['arguments']['scope'] == scope
    result = tool.run(proposal['arguments'], None)
    assert result.success
    assert len(tool.calls) == 1
    assert tool.entries == remaining

@pytest.mark.parametrize('first_query', [
    'Who directed Possessor?',
    'director of Possessor and their filmography',
])
def test_recovery_query_uses_the_discovered_entity(monkeypatch, first_query):
    tool = OwnedFilmLookup()
    monkeypatch.setitem(registry.BUILTIN_TOOLS, tool.name, tool)
    first_args = {'query': first_query}
    first = tool.run(first_args, None)
    schemas = registry.generate_tools_json_schema([tool.name, 'stop', 'toolSearchTool'])
    proposal = resolve_uncovered_request(
        isolated_voice_settings(),
        'Who directed Possessor and what other films has that director made?',
        [{'name': tool.name, 'arguments': first_args, 'result': first.reply_text, 'success': first.success}],
        schemas, prior_results=[],
    )
    assert proposal is not None
    assert proposal['name'] == tool.name
    query = proposal['arguments']['query'].casefold()
    assert 'brandon' in query and 'cronenberg' in query
    result = tool.run(proposal['arguments'], None)
    assert result.success
    assert len(tool.calls) == 2
    assert 'Antiviral' in result.reply_text and 'Infinity Pool' in result.reply_text

@pytest.mark.parametrize('entries,target,remaining', [
    ({'tea', 'soup'}, 'tea', {'soup'}),
    ({'tea reminder', 'soup reminder'}, 'tea reminder', {'soup reminder'}),
])
def test_recovery_discovers_unknown_names_before_mutating(monkeypatch, entries, target, remaining):
    tool = OwnedEntryManager()
    tool.entries = set(entries)
    monkeypatch.setitem(registry.BUILTIN_TOOLS, tool.name, tool)
    schemas = registry.generate_tools_json_schema([tool.name, 'stop', 'toolSearchTool'])
    cfg = isolated_voice_settings()
    query = 'Delete only the tea reminder and keep the soup reminder.'
    first = resolve_uncovered_request(cfg, query, [], schemas, prior_results=[])
    assert first is not None and first['name'] == tool.name
    assert first['arguments']['operation'] == 'enumerate_entries'
    assert first['arguments']['scope'] == 'whole_collection'
    observed = tool.run(first['arguments'], None)
    assert observed.success
    second = resolve_uncovered_request(cfg, query, [{
        'name': tool.name, 'arguments': first['arguments'], 'success': True, 'result': observed.reply_text,
    }], schemas, prior_results=[])
    assert second is not None and second['name'] == tool.name
    assert second['arguments']['operation'] == 'discard_entries'
    assert second['arguments']['scope'] == 'matching_labels'
    assert second['arguments']['labels'] == [target]
    result = tool.run(second['arguments'], None)
    assert result.success
    assert len(tool.calls) == 2
    assert tool.entries == remaining

def test_previous_mutation_does_not_complete_a_new_request(monkeypatch):
    tool = OwnedEntryManager()
    monkeypatch.setitem(registry.BUILTIN_TOOLS, tool.name, tool)
    prior = [observed_entry_names(tool)]
    previous = tool.run({'operation': 'discard_entries', 'scope': 'matching_labels', 'labels': ['tea']}, None)
    assert previous.success
    prior.append({'name': tool.name, 'tool_call_id': 'previous-mutation',
                  'result': previous.reply_text, 'resource_references': []})
    tool.entries.add('tea')
    tool.calls.clear()
    schemas = registry.generate_tools_json_schema([tool.name, 'stop', 'toolSearchTool'])
    proposal = resolve_uncovered_request(
        isolated_voice_settings(), 'Delete only the entry named tea and keep the entry named soup.',
        [], schemas, prior_results=prior,
    )
    assert proposal is not None and proposal['name'] == tool.name
    assert proposal['arguments']['operation'] == 'discard_entries'
    assert proposal['arguments']['scope'] == 'matching_labels'
    assert proposal['arguments']['labels'] == ['tea']
    assert tool.run(proposal['arguments'], None).success
    assert len(tool.calls) == 1
    assert tool.entries == {'soup'}

def test_missing_identity_does_not_produce_a_mutation(monkeypatch):
    tool = OwnedEntryManager()
    monkeypatch.setitem(registry.BUILTIN_TOOLS, tool.name, tool)
    arguments = {'operation': 'enumerate_entries', 'scope': 'whole_collection'}
    observed = tool.run(arguments, None)
    schemas = registry.generate_tools_json_schema([tool.name, 'stop', 'toolSearchTool'])
    proposal = resolve_uncovered_request(
        isolated_voice_settings(), 'Delete only the cocoa reminder in this collection and keep the soup reminder.',
        [{'name': tool.name, 'arguments': arguments, 'success': True, 'result': observed.reply_text}],
        schemas, prior_results=[],
    )
    assert proposal is None, 'An observed namespace without the requested record requires clarification'
    assert len(tool.calls) == 1
    assert tool.entries == {'tea', 'soup'}

@pytest.mark.parametrize('entries,query',[
    ({'reminder','soup'},'Delete only the cocoa reminder in this collection and keep the soup reminder.'),
    ({'note','shopping'},'Remove only my travel note and keep my shopping note.'),
    ({'tea','soup'},'Delete only the cocoa reminder.'),
    ({'tea','tea reminder','soup'},'Delete only the tea reminder and keep the soup reminder.'),
    ({'tea green','tea black','soup'},'Delete only the tea reminder and keep the soup reminder.'),
])
def test_contract_snapshot_never_guesses_an_absent_or_ambiguous_target(monkeypatch,entries,query):
    tool=OwnedEntryManager();tool.entries=set(entries)
    monkeypatch.setitem(registry.BUILTIN_TOOLS,tool.name,tool)
    args={'operation':'enumerate_entries','scope':'whole_collection'}
    observed=tool.run(args,None)
    schemas=registry.generate_tools_json_schema([tool.name,'stop','toolSearchTool'])
    proposal=resolve_uncovered_request(isolated_voice_settings(),query,[{'name':tool.name,'arguments':args,'success':True,'result':observed.reply_text}],schemas,prior_results=[])
    assert proposal is None
    assert len(tool.calls)==1
    assert tool.entries==entries

def test_contract_fresh_complete_snapshot_replaces_stale_identity(monkeypatch):
    tool=OwnedEntryManager()
    monkeypatch.setitem(registry.BUILTIN_TOOLS,tool.name,tool)
    prior=[observed_entry_names(tool)]
    removed=tool.run({'operation':'discard_entries','scope':'matching_labels','labels':['tea']},None)
    assert removed.success
    tool.calls.clear()
    args={'operation':'enumerate_entries','scope':'whole_collection'};observed=tool.run(args,None)
    schemas=registry.generate_tools_json_schema([tool.name,'stop','toolSearchTool'])
    proposal=resolve_uncovered_request(isolated_voice_settings(),'Delete only the tea reminder and keep the soup reminder.',[{'name':tool.name,'arguments':args,'success':True,'result':observed.reply_text}],schemas,prior_results=prior)
    assert proposal is None
    assert len(tool.calls)==1
    assert tool.entries=={'soup'}

@pytest.mark.parametrize('query,target,keeper',[
    ('Entferne nur die Tee-Erinnerung und behalte Suppe.','Tee','Suppe'),
    ('Sadece çay hatırlatıcısını sil, çorba hatırlatıcısı kalsın.','çay','çorba'),
    ('Delete only the tea reminder and keep the soup reminder.','tea '+('kettle '*50),'soup'),
])
def test_contract_discovers_then_copies_the_actual_identifier(monkeypatch,query,target,keeper):
    tool=OwnedEntryManager();tool.entries={target,keeper}
    monkeypatch.setitem(registry.BUILTIN_TOOLS,tool.name,tool)
    cfg=isolated_voice_settings();schemas=registry.generate_tools_json_schema([tool.name,'stop','toolSearchTool'])
    first=resolve_uncovered_request(cfg,query,[],schemas,prior_results=[])
    assert first=={'name':tool.name,'arguments':{'operation':'enumerate_entries','scope':'whole_collection'}}
    observed=tool.run(first['arguments'],None);assert observed.success
    second=resolve_uncovered_request(cfg,query,[{'name':tool.name,'arguments':first['arguments'],'success':True,'result':observed.reply_text}],schemas,prior_results=[])
    assert second=={'name':tool.name,'arguments':{'operation':'discard_entries','scope':'matching_labels','labels':[target]}}
    assert tool.run(second['arguments'],None).success
    assert len(tool.calls)==2
    assert tool.entries=={keeper}
