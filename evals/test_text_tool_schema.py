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


@pytest.mark.parametrize('query,operation,scope,remaining', [
    ('List all my reminder entries.', 'enumerate_entries', 'whole_collection', {'tea', 'soup'}),
    ('Delete every reminder entry.', 'discard_entries', 'whole_collection', set()),
    ('Delete only the tea reminder and keep the soup reminder.', 'discard_entries', 'matching_labels', {'soup'}),
])
def test_text_tool_call_preserves_declared_operation_and_scope(monkeypatch, query, operation, scope, remaining, record_property):
    tool = OwnedEntryManager()
    monkeypatch.setitem(registry.BUILTIN_TOOLS, tool.name, tool)
    monkeypatch.setattr(engine, 'select_tools', lambda *args, **kwargs: [tool.name, 'stop'])
    cfg = replace(isolated_voice_settings(), location_enabled=False, mcps={}, planner_enabled=False,
                  llm_chat_timeout_sec=60, planner_timeout_sec=15)
    db = Database(':memory:')
    try:
        reply = engine.run_reply_engine(db, cfg, None, query, DialogueMemory(), language='en')
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
