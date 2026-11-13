"""Actual result coverage governs continuation without a planner."""
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


class OwnedReminderRemoval(Tool):
    name = 'removeReminder'
    description = 'Delete one reminder by its exact name. The current names are tea and soup. Return every remaining reminder.'
    inputSchema = {'type': 'object', 'properties': {'name': {'type': 'string'}}, 'required': ['name']}

    def __init__(self):
        self.entries = {'tea', 'soup'}
        self.calls = []

    def run(self, args, context):
        self.calls.append(args)
        if args.get('name') not in self.entries:
            return ToolExecutionResult(False, None, 'No reminder has that exact name. Current names: ' + ', '.join(sorted(self.entries)))
        self.entries.remove(args['name'])
        return ToolExecutionResult(True, json.dumps({'remaining_reminders': sorted(self.entries)}))


class OwnedFilmLookup(Tool):
    name = 'filmLookup'
    description = 'Look up film information by query. A director lookup returns a name; a filmography lookup returns films.'
    inputSchema = {'type': 'object', 'properties': {'query': {'type': 'string'}}, 'required': ['query']}

    def __init__(self):
        self.calls = []

    def run(self, args, context):
        self.calls.append(args)
        if len(self.calls) == 1:
            return ToolExecutionResult(True, 'Possessor was directed by Brandon Cronenberg. No filmography is included in this result.')
        return ToolExecutionResult(True, 'Brandon Cronenberg directed Antiviral (2012) and Infinity Pool (2023).')


def _reply(monkeypatch, tool, query):
    monkeypatch.setitem(registry.BUILTIN_TOOLS, tool.name, tool)
    monkeypatch.setattr(engine, 'select_tools', lambda *args, **kwargs: [tool.name, 'stop'])
    cfg = replace(isolated_voice_settings(), location_enabled=False, mcps={}, planner_enabled=False,
                  tool_result_digest_enabled=False, llm_chat_timeout_sec=60)
    db = Database(':memory:')
    try:
        reply = engine.run_reply_engine(db, cfg, None, query, DialogueMemory(), language='en')
        assert reply
        assert_not_fallback_reply(reply, 'compound result coverage')
        return reply
    finally:
        db.close()


def test_preservation_constraint_needs_no_second_action(monkeypatch, record_property):
    tool = OwnedReminderRemoval()
    reply = _reply(monkeypatch, tool, 'Delete only the tea reminder and keep the soup reminder.')
    record_property('reply', reply)
    record_property('tool_calls', json.dumps(tool.calls))
    assert len(tool.calls) == 1, 'A confirmed remaining reminder does not require another action'
    assert tool.calls[0] == {'name': 'tea'}
    assert tool.entries == {'soup'}
    assert 'soup' in reply.lower()


def test_dependent_information_requires_missing_evidence(monkeypatch, record_property):
    tool = OwnedFilmLookup()
    reply = _reply(monkeypatch, tool, 'Who directed Possessor and what other films has that director made?')
    record_property('reply', reply)
    record_property('tool_calls', json.dumps(tool.calls))
    assert len(tool.calls) >= 2, 'The first result does not contain a filmography'
    assert any('brandon' in str(call).lower() or 'cronenberg' in str(call).lower() for call in tool.calls[1:])
    assert 'cronenberg' in reply.lower()
    assert 'antiviral' in reply.lower() or 'infinity pool' in reply.lower()
