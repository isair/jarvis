"""Concrete plan values retain the types declared by their tool schema."""
import pytest
from evals.helpers import voice_config
from jarvis.reply import planner
from jarvis.tools.builtin.local_files import LocalFilesTool
from jarvis.tools.base import ToolContext

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('field_type, literal, expected', [
    ('boolean', 'false', False), ('boolean', 'true', True),
    ('boolean', "'false'", False), ('integer', '7', 7),
    ('integer', "'7'", 7), ('number', '2.5', 2.5),
    ('number', '-2.5', -2.5), ('null', 'null', None),
    ('string', 'false', 'false'), ('string', '7', '7'),
    ('string', 'null', 'null'), (None, 'false', 'false'),
    ('number', '7', 7), ('integer', '-7', -7),
    (['integer', 'string'], '7', 7),
    (['integer', 'string'], "'note-84'", 'note-84'),
    (['integer', 'string'], "'007'", '007'),
    (['string', 'null'], 'null', None),
    (['integer', 'boolean'], 'true', True),
])
def test_concrete_value_matches_declared_type(monkeypatch, field_type, literal, expected):
    schema = [{'type': 'function', 'function': {
        'name': 'localValue', 'parameters': {'type': 'object',
            'properties': {'value': {'type': field_type} if field_type is not None else {}}, 'required': ['value']},
    }}]
    monkeypatch.setattr(planner, 'call_llm_direct', lambda **kwargs: pytest.fail('🗺️ Concrete primitive values need no model'))
    resolved = planner.resolve_next_tool_call(voice_config(), f'localValue value={literal}', [], schema)
    assert resolved is not None and resolved[1] == {'value': expected}
    assert type(resolved[1]['value']) is type(expected)


@pytest.mark.parametrize('recursive', [False, True])
def test_planned_recursion_respects_the_directory_boundary(monkeypatch, tmp_path, recursive):
    tool = LocalFilesTool()
    (tmp_path / 'visible.txt').write_text('top-level')
    nested = tmp_path / 'nested'
    nested.mkdir()
    (nested / 'hidden.txt').write_text('nested')
    original_expand = __import__('os').path.expanduser
    monkeypatch.setattr('os.path.expanduser', lambda path: str(tmp_path) if path == '~' else original_expand(path))
    monkeypatch.setattr(planner, 'call_llm_direct', lambda **kwargs: pytest.fail('📂 Fully specified listing needs no model'))
    schema = [{'type': 'function', 'function': {'name': tool.name, 'parameters': tool.inputSchema}}]
    resolved = planner.resolve_next_tool_call(voice_config(), f"localFiles operation='list' path='{tmp_path}' recursive={str(recursive).lower()}", [], schema)
    assert resolved is not None
    context = ToolContext(None, voice_config(), '', '', '', 0, lambda message: None)
    result = tool.run(resolved[1], context)
    assert result.success
    assert 'visible.txt' in result.reply_text
    assert ('hidden.txt' in result.reply_text) is recursive


@pytest.mark.parametrize('field_type, literal', [
    ('boolean', 'no'), ('boolean', '0'), ('integer', '1.5'),
    ('integer', 'true'), ('number', 'NaN'), ('number', 'Infinity'),
    ('number', '1e309'), ('null', 'false'),
    ('array', "'[1,2]'"), ('object', "'{key:value}'"),
    (['integer', 'null'], 'true'),
    (['integer', 'boolean'], '1.5'),
])
def test_unresolved_typed_values_return_to_the_model(monkeypatch, field_type, literal):
    schema = [{'type': 'function', 'function': {
        'name': 'localValue', 'parameters': {'type': 'object',
            'properties': {'value': {'type': field_type}}, 'required': ['value']},
    }}]
    monkeypatch.setattr(planner, 'call_llm_direct', lambda **kwargs: 'null')
    assert planner.resolve_next_tool_call(voice_config(), f'localValue value={literal}', [], schema) is None


def test_unresolved_typed_value_can_be_completed_by_the_model(monkeypatch):
    schema = [{'type': 'function', 'function': {
        'name': 'localValue', 'parameters': {'type': 'object',
            'properties': {'value': {'type': 'boolean'}}, 'required': ['value']},
    }}]
    monkeypatch.setattr(planner, 'call_llm_direct', lambda **kwargs: '{"name":"localValue","arguments":{"value":false}}')
    assert planner.resolve_next_tool_call(voice_config(), 'localValue value=no', [], schema) == ('localValue', {'value': False})
