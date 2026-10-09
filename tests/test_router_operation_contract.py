"""Tool choices come from the structured selection, not operation prose."""
import json
from unittest.mock import Mock

import pytest

from jarvis.tools.registry import ToolSpec
from jarvis.tools.selection import select_tools, ToolSelectionStrategy

pytestmark = pytest.mark.unit


def route(response, catalogue=None):
    tools = catalogue or {
        'getWeather': ToolSpec('getWeather', 'Get weather conditions.', {}),
        'webSearch': ToolSpec('webSearch', 'Search the web.', {}),
        'stop': ToolSpec('stop', 'End the conversation.', {}),
    }
    return select_tools('weather', tools, {}, strategy=ToolSelectionStrategy.LLM,
                        llm_backend=Mock(direct=Mock(return_value=response)), llm_model='test')


def test_operation_prose_cannot_add_an_unselected_tool():
    selected = route(json.dumps({'requested_operation': 'webSearch rather than getWeather',
                                 'tools': ['webSearch']}))
    assert selected == ['webSearch', 'stop']


def test_empty_selection_returns_only_mandatory_tools():
    assert route(json.dumps({'requested_operation': 'Answer from context', 'tools': []})) == ['stop']


@pytest.mark.parametrize('payload', [
    {'requested_operation': 'Look up weather', 'tools': 'webSearch'},
    {'requested_operation': ['webSearch'], 'tools': ['webSearch']},
    {'tools': ['webSearch']},
    {'requested_operation': '', 'tools': ['webSearch']},
    {'requested_operation': 'Look up weather', 'tools': ['webSearch', 123]},
])
def test_invalid_routing_shape_uses_keyword_fallback(payload):
    assert route(json.dumps(payload)) == route(None)


def test_names_are_matched_literally_without_stripping_registered_punctuation():
    tools = {name: ToolSpec(name, 'Registered action.', {}) for name in ('_local.tool_', 'otherAction', 'stop')}
    response = json.dumps({'requested_operation': 'Use the registered action',
                           'tools': ['_local.tool_']})
    assert route(response, tools) == ['_local.tool_', 'stop']
