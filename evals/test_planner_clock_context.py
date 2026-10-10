"""Relative date resolution follows the supplied clock in every language."""
from datetime import datetime, timedelta, timezone

import pytest

from conftest import requires_judge_llm
from helpers import voice_config
from jarvis.reply import planner
from jarvis.tools.builtin.nutrition.fetch_meals import FetchMealsTool, _normalise_time_range

pytestmark = [pytest.mark.eval, requires_judge_llm]


@pytest.mark.parametrize('step', [
    "fetchMeals time_range='today'",
    'fetchMeals bugün kaydedilen öğünleri getir',
])
def test_relative_calendar_day_uses_current_clock(monkeypatch, step):
    instant = datetime(2029, 2, 6, 16, 30, tzinfo=timezone.utc)

    class Clock:
        @staticmethod
        def now(tz):
            return instant.astimezone(tz)

    monkeypatch.setattr(planner, 'datetime', Clock, raising=False)
    tool = FetchMealsTool()
    schema = [{'type': 'function', 'function': {
        'name': tool.name, 'description': tool.description, 'parameters': tool.inputSchema,
    }}]
    result = planner.resolve_next_tool_call(voice_config(), step, [], schema, timeout_sec=60)
    assert result is not None and result[0] == tool.name, result
    assert {'since_utc', 'until_utc'} <= result[1].keys(), result
    since, until = _normalise_time_range(result[1])
    local_start = instant.astimezone().replace(hour=0, minute=0, second=0, microsecond=0)
    expected_start = local_start.astimezone(timezone.utc)
    assert datetime.fromisoformat(since) == expected_start, result
    assert instant <= datetime.fromisoformat(until) < expected_start + timedelta(days=1), result
