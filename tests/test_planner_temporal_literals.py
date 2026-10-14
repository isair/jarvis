"""Declared temporal arguments preserve ISO values and resolve relative labels."""
import json

import pytest

from evals.helpers import voice_config
from jarvis.reply import planner

pytestmark = pytest.mark.unit


def _catalogue(format_name, field=None):
    parameters = {'type': 'object', 'properties': {
        'value': field or {'type': 'string', 'format': format_name},
    }, 'required': ['value']}
    return [{'type': 'function', 'function': {'name': 'retrieve', 'parameters': parameters}}]


@pytest.mark.parametrize('format_name, resolved', [
    ('date', '2029-02-06'), ('time', '16:30:00+00:00'),
    ('date-time', '2029-02-06T16:30:00+00:00'),
])
@pytest.mark.parametrize('label', ['today', 'bugün', 'aujourd’hui'])
def test_relative_temporal_literal_uses_resolver(monkeypatch, format_name, resolved, label):
    monkeypatch.setattr(planner, 'call_llm_direct', lambda **_: json.dumps({
        'name': 'retrieve', 'arguments': {'value': resolved},
    }))
    step = f"retrieve value='{label}'"
    assert planner.resolve_next_tool_call(voice_config(), step, [], _catalogue(format_name),
                                          original_query=step) == ('retrieve', {'value': resolved})


@pytest.mark.parametrize('format_name, literal', [
    ('date', '2029-02-06'), ('time', '16:30:00+00:00'),
    ('date-time', '2029-02-06T16:30:00+00:00'),
])
def test_valid_iso_literal_keeps_no_inference_path(monkeypatch, format_name, literal):
    def forbidden(**_):
        raise AssertionError('Valid temporal literal must not require inference')
    monkeypatch.setattr(planner, 'call_llm_direct', forbidden)
    step = f"retrieve value='{literal}'"
    assert planner.resolve_next_tool_call(voice_config(), step, [], _catalogue(format_name),
                                          original_query=step) == ('retrieve', {'value': literal})


@pytest.mark.parametrize('label', ['today', 'bugün', 'aujourd’hui'])
def test_ordinary_string_keeps_literal_value(monkeypatch, label):
    monkeypatch.setattr(planner, 'call_llm_direct', lambda **_: pytest.fail('Ordinary literal requires no inference'))
    step = f"retrieve value='{label}'"
    schema = _catalogue(None, {'type': 'string'})
    assert planner.resolve_next_tool_call(voice_config(), step, [], schema, original_query=step) == (
        'retrieve', {'value': label},
    )


def test_nullable_temporal_format_resolves_relative_literal(monkeypatch):
    field = {'anyOf': [{'type': 'string', 'format': 'date-time'}, {'type': 'null'}]}
    expected = '2029-02-06T16:30:00+00:00'
    monkeypatch.setattr(planner, 'call_llm_direct', lambda **_: json.dumps({
        'name': 'retrieve', 'arguments': {'value': expected},
    }))
    step = "retrieve value='today'"
    assert planner.resolve_next_tool_call(voice_config(), step, [], _catalogue(None, field),
                                          original_query=step) == ('retrieve', {'value': expected})
