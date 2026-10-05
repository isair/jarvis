"""Concrete parsing never dispatches silently truncated or ambiguous values."""
import json
import os

import pytest
from evals.helpers import voice_config
from jarvis.reply import planner
from jarvis.tools.base import ToolContext
from jarvis.tools.builtin.local_files import LocalFilesTool

pytestmark = pytest.mark.unit
SCHEMA = [{'type': 'function', 'function': {'name': 'localValue', 'parameters': {
    'type': 'object', 'properties': {'value': {'type': 'string'}}, 'required': ['value'],
}}}]


@pytest.mark.parametrize('step', [
    "localValue value='first' value='second'",
    "localValue please value='first'",
    "localValue value='first' and use something else",
    "localValue value='broken",
    "localValue value='It's a quiet morning.'",
    'localValue value="He said \\"hello\\"."',
])
def test_ambiguous_concrete_step_cannot_dispatch_a_partial_value(monkeypatch, step):
    monkeypatch.setattr(planner, 'call_llm_direct', lambda **kwargs: 'null')
    assert planner.resolve_next_tool_call(voice_config(), step, [], SCHEMA) is None


def test_apostrophe_content_reaches_a_private_file_without_truncation(monkeypatch, tmp_path):
    tool = LocalFilesTool()
    target = tmp_path / 'note.txt'
    content = "It's a quiet morning."
    raw = json.dumps({'name': tool.name, 'arguments': {'operation': 'write', 'path': str(target), 'content': content}})
    monkeypatch.setattr(planner, 'call_llm_direct', lambda **kwargs: raw)
    original_expand = os.path.expanduser
    monkeypatch.setattr(os.path, 'expanduser', lambda path: str(tmp_path) if path == '~' else original_expand(path))
    schema = [{'type': 'function', 'function': {'name': tool.name, 'parameters': tool.inputSchema}}]
    resolved = planner.resolve_next_tool_call(voice_config(), f"localFiles operation='write' path='{target}' content='{content}'", [], schema)
    assert resolved is not None
    result = tool.run(resolved[1], ToolContext(None, voice_config(), '', '', '', 0, lambda message: None))
    assert result.success
    assert target.read_text() == content


@pytest.mark.parametrize('step, expected', [
    ('localValue value="It\'s a quiet morning."', "It's a quiet morning."),
    ('localValue value=\'He said "hello".\'', 'He said "hello".'),
    ("localValue value='first'.", 'first'),
    ("localValue value='first'", 'first'),
    ("localValue value=''", ''),
    ('localValue value=St.Gallen', 'St.Gallen'),
])
def test_complete_literal_value_remains_available_without_inference(monkeypatch, step, expected):
    monkeypatch.setattr(planner, 'call_llm_direct', lambda **kwargs: pytest.fail('🗺️ Complete literals need no model'))
    assert planner.resolve_next_tool_call(voice_config(), step, [], SCHEMA) == ('localValue', {'value': expected})


@pytest.mark.parametrize('separator', [' ', ', '])
def test_complete_pairs_keep_whitespace_and_comma_separators(monkeypatch, separator):
    schema = [{'type': 'function', 'function': {'name': 'localValue', 'parameters': {
        'type': 'object', 'properties': {'value': {'type': 'string'}, 'scope': {'type': 'string'}},
        'required': ['value'],
    }}}]
    monkeypatch.setattr(planner, 'call_llm_direct', lambda **kwargs: pytest.fail('🗺️ Complete pairs need no model'))
    step = f"localValue value='first'{separator}scope='local'"
    assert planner.resolve_next_tool_call(voice_config(), step, [], schema) == ('localValue', {'value': 'first', 'scope': 'local'})
