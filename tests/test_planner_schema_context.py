"""Resolver backends receive complete argument contracts."""
import json

import pytest

from evals.helpers import voice_config
from jarvis.reply import planner

pytestmark = pytest.mark.unit


def test_declared_enum_can_drive_a_complete_resolved_call(monkeypatch):
    choices = ['source-specific-value', 'another-value']
    parameters = {'type': 'object', 'properties': {
        'filter': {'type': 'object', 'properties': {
            'state': {'type': 'string', 'enum': choices},
        }, 'required': ['state']},
    }, 'required': ['filter']}
    schema = [{'type': 'function', 'function': {
        'name': 'lookup', 'parameters': parameters,
    }}]

    def backend(**kwargs):
        text = kwargs['user_content'].partition('ALLOWED TOOLS:\n')[2]
        catalogue, _ = json.JSONDecoder().raw_decode(text)
        tool = catalogue[0]
        permitted = tool['parameters']['properties']['filter']['properties']['state']['enum']
        return json.dumps({'name': tool['name'], 'arguments': {'filter': {'state': permitted[0]}}})

    monkeypatch.setattr(planner, 'call_llm_direct', backend)
    assert planner.resolve_next_tool_call(voice_config(), 'lookup selected records', [], schema) == (
        'lookup', {'filter': {'state': choices[0]}},
    )
