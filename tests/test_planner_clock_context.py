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
    assert planner.resolve_next_tool_call(voice_config(), 'retrieve records up to now', [], schema, original_query='retrieve records up to now') == (
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
    assert planner.resolve_next_tool_call(voice_config(), 'localValue preserve the selected value', [], schema, original_query='localValue preserve the selected value') == (
        'localValue', {'value': 'literal'},
    )


@pytest.mark.skipif(not hasattr(__import__('time'), 'tzset'), reason='Explicit POSIX timezone control is unavailable')
@pytest.mark.parametrize('timezone_name, timestamp', [
    ('Europe/London', '2029-03-25T12:00:00+00:00'),
    ('Europe/London', '2029-10-28T12:00:00+00:00'),
    ('Asia/Kuala_Lumpur', '2029-02-06T16:30:00+00:00'),
])
@pytest.mark.parametrize('offset', [-1, 0, 1])
@pytest.mark.parametrize('inclusive', [False, True])
def test_calendar_windows_preserve_local_days_across_offsets_and_dst(
    monkeypatch, timezone_name, timestamp, offset, inclusive,
):
    import time
    from datetime import timedelta
    from zoneinfo import ZoneInfo

    instant = datetime.fromisoformat(timestamp)
    local = instant.astimezone(ZoneInfo(timezone_name))
    start = local.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=offset)
    end = (start + timedelta(days=1)).astimezone(timezone.utc)
    if inclusive:
        end -= timedelta(microseconds=1)
    expected = {'lower': start.astimezone(timezone.utc).isoformat(), 'upper': end.isoformat()}

    class Clock:
        @staticmethod
        def now(tz):
            return instant.astimezone(tz)

    def backend(**kwargs):
        clock, _ = json.JSONDecoder().raw_decode(kwargs['user_content'].partition('CURRENT CLOCK:\n')[2])
        day = next(day for day in clock['calendar_days'] if day['day_offset'] == offset)
        assert day['date'] == start.date().isoformat()
        upper = 'end_inclusive_utc' if inclusive else 'end_exclusive_utc'
        return json.dumps({'name': 'retrieve', 'arguments': {'lower': day['start_utc'], 'upper': day[upper]}})

    monkeypatch.setattr(planner, 'datetime', Clock)
    monkeypatch.setattr(planner, 'call_llm_direct', backend)
    schema = [{'type': 'function', 'function': {'name': 'retrieve', 'parameters': {
        'type': 'object', 'properties': {key: {'type': 'string', 'format': 'date-time'} for key in expected},
        'required': list(expected),
    }}}]
    try:
        with monkeypatch.context() as scoped:
            scoped.setenv('TZ', timezone_name)
            time.tzset()
            result = planner.resolve_next_tool_call(voice_config(), 'retrieve the selected calendar interval', [], schema,
                                                     original_query='retrieve the selected calendar interval')
            assert result == ('retrieve', expected)
    finally:
        time.tzset()


@pytest.mark.parametrize('shape', ['direct', 'nullable', 'nested', 'array', 'definition'])
def test_calendar_references_expand_before_tool_dispatch(monkeypatch, shape):
    instant = datetime(2029, 2, 6, 16, 30, tzinfo=timezone.utc)
    reference = '@calendar:0:start_utc'
    date_field = {'type': 'string', 'format': 'date-time'}
    field = date_field
    if shape == 'nullable':
        field = {'anyOf': [date_field, {'type': 'null'}]}
    elif shape == 'nested':
        field = {'anyOf': [{'type': 'object', 'properties': {'start': date_field}}, {'type': 'null'}]}
    elif shape == 'array':
        field = {'type': 'array', 'items': date_field}
    elif shape == 'definition':
        field = {'$ref': '#/$defs/date-time'}
    parameters = {'type': 'object', 'properties': {'value': field, 'literal': {'type': 'string'}},
                  'required': ['value'], '$defs': {'date-time': date_field}}
    value = {'start': reference} if shape == 'nested' else [reference] if shape == 'array' else reference

    class Clock:
        @staticmethod
        def now(tz):
            return instant.astimezone(tz)

    monkeypatch.setattr(planner, 'datetime', Clock)
    monkeypatch.setattr(planner, 'call_llm_direct', lambda **_: json.dumps({
        'name': 'retrieve', 'arguments': {'value': value, 'literal': reference},
    }))
    schema = [{'type': 'function', 'function': {'name': 'retrieve', 'parameters': parameters}}]
    result = planner.resolve_next_tool_call(voice_config(), 'retrieve the current calendar range', [], schema,
                                             original_query='retrieve the current calendar range')
    assert result is not None
    output = result[1]['value']
    output = output['start'] if shape == 'nested' else output[0] if shape == 'array' else output
    local = datetime.fromisoformat(output).astimezone()
    assert local.date() == instant.astimezone().date()
    assert (local.hour, local.minute, local.second, local.microsecond) == (0, 0, 0, 0)
    assert result[1]['literal'] == reference


def test_unknown_calendar_reference_does_not_dispatch(monkeypatch):
    monkeypatch.setattr(planner, 'call_llm_direct', lambda **_: json.dumps({
        'name': 'retrieve', 'arguments': {'until': '@calendar:unavailable:end'},
    }))
    schema = [{'type': 'function', 'function': {'name': 'retrieve', 'parameters': {
        'type': 'object', 'properties': {'until': {'type': 'string', 'format': 'date-time'}},
        'required': ['until'],
    }}}]
    assert planner.resolve_next_tool_call(voice_config(), 'retrieve selected records', [], schema,
                                          original_query='retrieve selected records') is None
