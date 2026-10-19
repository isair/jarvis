"""Text tool help retains the same input contract as native function calling."""
import copy
import json

import pytest
from jarvis.tools import registry

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('source', ['builtin', 'mcp'])
@pytest.mark.parametrize('schema', [
    {
        'type': 'object',
        'properties': {
            'operation': {'type': 'string', 'enum': ['enumerate', 'discard'],
                          'description': 'Choose the operation to perform.'},
            'scope': {'type': 'string', 'enum': ['everything', 'named'],
                      'description': 'Everything includes all entries; named uses labels.'},
            'labels': {'type': 'array', 'items': {'type': 'string', 'minLength': 1},
                       'minItems': 1, 'uniqueItems': True},
        },
        'required': ['operation', 'scope'],
        'if': {'properties': {'scope': {'const': 'named'}}},
        'then': {'required': ['labels']},
        'additionalProperties': False,
    },
    {'type': 'object', 'properties': {'message': {
        'anyOf': [{'type': 'null'}, {'type': 'string', 'maxLength': 40}],
        'description': '文字列 "quoted"\nsecond line',
    }}, 'required': ['message']},
    {'type': 'object', 'properties': {}, 'maxProperties': 0},
])
def test_text_and_native_tool_contracts_preserve_complete_schema(monkeypatch, source, schema):
    original = copy.deepcopy(schema)
    tool = registry.ToolSpec(name='ownedTool', description='Owned local operation', inputSchema=schema)
    monkeypatch.setattr(registry, 'BUILTIN_TOOLS', {'ownedTool': tool} if source == 'builtin' else {})
    external = {'ownedTool': tool} if source == 'mcp' else None
    native = registry.generate_tools_json_schema(['ownedTool'], external)
    text = registry.generate_tools_description(['ownedTool'], external)
    rendered = [line.removeprefix('Input schema: ') for line in text.splitlines()
                if line.startswith('Input schema: ')]
    assert len(rendered) == 1, 'Every admitted tool must expose its complete input schema'
    assert json.loads(rendered[0]) == native[0]['function']['parameters'] == original
    assert schema == original


def test_text_tool_help_does_not_expose_unselected_tools(monkeypatch):
    selected = registry.ToolSpec(name='selected', description='Selected tool', inputSchema={'type': 'object'})
    other = registry.ToolSpec(name='unselected', description='Private unrelated description', inputSchema={'type': 'object'})
    monkeypatch.setattr(registry, 'BUILTIN_TOOLS', {'selected': selected, 'unselected': other})
    text = registry.generate_tools_description(['selected'])
    assert selected.description in text
    assert other.name not in text
    assert other.description not in text
