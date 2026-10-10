"""Relative arguments can be grounded in the current clock."""
import json
from datetime import datetime, timezone

import pytest

from evals.helpers import voice_config
from jarvis.reply import planner

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('nested', [False, True])
def test_clock_grounded_call_uses_supplied_instant(monkeypatch, nested):
    instant = datetime(2029, 2, 6, 16, 30, tzinfo=timezone.utc)

    class Clock:
        @staticmethod
        def now(tz):
            return instant.astimezone(tz)

    def backend(**kwargs):
        text = kwargs['user_content'].partition('CURRENT CLOCK:\n')[2]
        clock, _ = json.JSONDecoder().raw_decode(text)
        arguments = {'until': clock['utc']}
        return json.dumps({'name': 'retrieve', 'arguments': {'range': arguments} if nested else arguments})

    monkeypatch.setattr(planner, 'datetime', Clock, raising=False)
    monkeypatch.setattr(planner, 'call_llm_direct', backend)
    parameters = {'type': 'object', 'properties': {
        'until': {'type': 'string', 'format': 'date-time'},
    }, 'required': ['until']}
    if nested:
        parameters = {'type': 'object', 'properties': {'range': parameters}, 'required': ['range']}
    schema = [{'type': 'function', 'function': {'name': 'retrieve', 'parameters': parameters}}]
    expected = {'until': instant.isoformat()}
    assert planner.resolve_next_tool_call(voice_config(), 'retrieve records up to now', [], schema) == (
        'retrieve', {'range': expected} if nested else expected,
    )


@pytest.mark.parametrize('other_temporal_tool', [False, True])
def test_unrelated_step_keeps_its_argument_context(monkeypatch, other_temporal_tool):
    schema = [{'type': 'function', 'function': {
        'name': 'localValue', 'parameters': {'type': 'object', 'properties': {'value': {'type': 'string'}}},
    }}]
    if other_temporal_tool:
        schema.append({'type': 'function', 'function': {
            'name': 'dateValue', 'parameters': {'type': 'object', 'properties': {
                'date': {'type': 'string', 'format': 'date'},
            }},
        }})

    def backend(**kwargs):
        value = 'literal' if 'CURRENT CLOCK:' not in kwargs['user_content'] and kwargs['system_prompt'] == planner._STEP_RESOLVER_SYSTEM else 'altered'
        return json.dumps({'name': 'localValue', 'arguments': {'value': value}})

    monkeypatch.setattr(planner, 'call_llm_direct', backend)
    assert planner.resolve_next_tool_call(voice_config(), 'localValue preserve the selected value', [], schema) == (
        'localValue', {'value': 'literal'},
    )
