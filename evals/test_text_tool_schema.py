"""Native text-protocol calls use the selected tool's declared argument contract."""
import json
from copy import deepcopy
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
        'x-jarvis-records': {
            'discovery': 0,
            'collection': {'source': 'result', 'path': '/remaining_entries', 'complete': True},
            'actions': [
                {'purpose': 'List all current saved entries',
                 'arguments': {'operation': 'enumerate_entries', 'scope': 'whole_collection'}, 'selector': None},
                {'purpose': 'Delete every saved entry',
                 'arguments': {'operation': 'discard_entries', 'scope': 'whole_collection'}, 'selector': None},
                {'purpose': 'Delete only selected existing entries and preserve every other entry',
                 'arguments': {'operation': 'discard_entries', 'scope': 'matching_labels'}, 'selector': 'labels'},
            ],
        },
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


@pytest.mark.parametrize('spoken_output', [False, True])
def test_declared_record_mutation_is_followed_by_the_requested_fresh_read(monkeypatch, record_property, spoken_output):
    tool = OwnedEntryManager()
    monkeypatch.setitem(registry.BUILTIN_TOOLS, tool.name, tool)
    monkeypatch.setattr(engine, 'select_tools', lambda *args, **kwargs: [tool.name, 'stop'])
    cfg = replace(isolated_voice_settings(), location_enabled=False, mcps={}, planner_enabled=False,
                  llm_chat_timeout_sec=60, planner_timeout_sec=15)
    db = Database(':memory:')
    try:
        reply = engine.run_reply_engine(db, cfg, None,
            'Delete only the tea reminder, then list the remaining entries.',
            DialogueMemory(), language='en', spoken_output=spoken_output)
        record_property('reply', reply)
        record_property('tool_calls', json.dumps(tool.calls))
        assert reply
        assert_not_fallback_reply(reply, 'declared record fresh read')
        assert tool.calls == [
            {'operation': 'enumerate_entries', 'scope': 'whole_collection'},
            {'operation': 'discard_entries', 'scope': 'matching_labels', 'labels': ['tea']},
            {'operation': 'enumerate_entries', 'scope': 'whole_collection'},
        ]
        assert tool.entries == {'soup'}
    finally:
        db.close()


@pytest.mark.parametrize('spoken_output', [False, True])
def test_actual_engine_observation_supplies_a_followup_identity(monkeypatch, record_property, spoken_output):
    tool = OwnedEntryManager()
    monkeypatch.setitem(registry.BUILTIN_TOOLS, tool.name, tool)
    monkeypatch.setattr(engine, 'select_tools', lambda *args, **kwargs: [tool.name, 'stop'])
    cfg = replace(isolated_voice_settings(), location_enabled=False, mcps={}, planner_enabled=False,
                  llm_chat_timeout_sec=60, planner_timeout_sec=15)
    db = Database(':memory:')
    try:
        dialogue = DialogueMemory()
        assert engine.run_reply_engine(db, cfg, None, 'List all my reminder entries.',
            dialogue, language='en', spoken_output=spoken_output)
        assert tool.calls == [{'operation': 'enumerate_entries', 'scope': 'whole_collection'}]
        tool.calls.clear()
        reply = engine.run_reply_engine(db, cfg, None, 'Delete only the tea reminder and keep the soup reminder.',
            dialogue, language='en', spoken_output=spoken_output)
        record_property('reply', reply)
        record_property('tool_calls', json.dumps(tool.calls))
        assert reply
        assert_not_fallback_reply(reply, 'actual prior record identity')
        assert tool.calls == [{'operation': 'discard_entries', 'scope': 'matching_labels', 'labels': ['tea']}]
        assert tool.entries == {'soup'}
    finally:
        db.close()


def test_declared_record_action_does_not_prepare_unrelated_saved_memory(monkeypatch, record_property):
    tool = OwnedEntryManager()
    monkeypatch.setitem(registry.BUILTIN_TOOLS, tool.name, tool)
    monkeypatch.setattr(engine, 'select_tools', lambda *args, **kwargs: [tool.name, 'stop'])
    cfg = replace(isolated_voice_settings(), location_enabled=False, mcps={}, planner_enabled=False,
                  memory_enrichment_source='all', llm_chat_timeout_sec=60, planner_timeout_sec=15)
    db = Database(':memory:')
    try:
        db.upsert_conversation_summary('2020-01-01', 'An unrelated stored fact.', '', source_app='fixture')
        reply = engine.run_reply_engine(db, cfg, None, 'Delete only the tea reminder and keep the soup reminder.',
            DialogueMemory(), language='en')
        record_property('reply', reply)
        record_property('tool_calls', json.dumps(tool.calls))
        assert reply
        assert_not_fallback_reply(reply, 'record action with saved memory')
        assert tool.calls == [
            {'operation': 'enumerate_entries', 'scope': 'whole_collection'},
            {'operation': 'discard_entries', 'scope': 'matching_labels', 'labels': ['tea']},
        ]
        assert tool.entries == {'soup'}
    finally:
        db.close()


class OwnedNumericRecordManager(OwnedEntryManager):
    name = 'ownedNumericRecordManager'
    inputSchema = deepcopy(OwnedEntryManager.inputSchema)
    inputSchema['properties']['labels']['items'] = {'$ref': '#/$defs/identifier'}
    inputSchema['$defs'] = {'identifier': {'type': 'integer'}}
    inputSchema['x-jarvis-records']['collection']['display'] = '/title'
    inputSchema['x-jarvis-records']['actions'][2]['identity'] = '/key'

    def __init__(self):
        self.entries = {47, 99}
        self.calls = []

    def run(self, args, context):
        self.calls.append(args)
        operation, scope = args.get('operation'), args.get('scope')
        if operation not in {'enumerate_entries', 'discard_entries'} or scope not in {'whole_collection', 'matching_labels'}:
            return ToolExecutionResult(False, None, 'Invalid operation or scope.')
        if scope == 'matching_labels':
            labels = args.get('labels')
            if not isinstance(labels, list) or not all(type(value) is int for value in labels):
                return ToolExecutionResult(False, None, 'Selectors require integer identifiers.')
            selected = self.entries.intersection(labels)
        else:
            selected = self.entries.copy()
        if operation == 'discard_entries':
            self.entries.difference_update(selected)
        return ToolExecutionResult(True, json.dumps({'remaining_entries': [
            {'key': value, 'title': str(value)} for value in sorted(self.entries)]}))


def test_literal_numeric_record_identifier_uses_the_declared_local_reference(monkeypatch, record_property):
    tool = OwnedNumericRecordManager()
    monkeypatch.setitem(registry.BUILTIN_TOOLS, tool.name, tool)
    monkeypatch.setattr(engine, 'select_tools', lambda *args, **kwargs: [tool.name, 'stop'])
    cfg = replace(isolated_voice_settings(), location_enabled=False, mcps={}, planner_enabled=False,
                  llm_chat_timeout_sec=60, planner_timeout_sec=15)
    db = Database(':memory:')
    try:
        reply = engine.run_reply_engine(db, cfg, None, 'Delete only the entry with ID 47 and keep ID 99.',
            DialogueMemory(), language='en')
        record_property('reply', reply)
        record_property('tool_calls', json.dumps(tool.calls))
        assert reply
        assert_not_fallback_reply(reply, 'literal numeric record identifier')
        assert tool.calls == [{'operation': 'discard_entries', 'scope': 'matching_labels', 'labels': [47]}]
        assert tool.entries == {99}
    finally:
        db.close()
