"""Saved plans retain their absolute dates against the reply clock."""
from datetime import datetime, timedelta, timezone
import json
from zoneinfo import ZoneInfo
from unittest.mock import patch

import pytest

from evals.conftest import requires_judge_llm
from evals.helpers import voice_config
from jarvis.llm import get_llm_backend
from jarvis.memory.graph_ops import format_warm_profile_block
from jarvis.reply import engine
from jarvis.system_prompt import build_system_prompt
from jarvis.utils.time_context import format_time_context

pytestmark = [pytest.mark.eval, requires_judge_llm]


@pytest.mark.parametrize('instant,zone', [
    (datetime(2026, 10, 12, tzinfo=timezone.utc), 'Europe/London'),
    (datetime(2027, 12, 31, 20, tzinfo=timezone.utc), 'Asia/Kuala_Lumpur'),
    (datetime(2028, 2, 28, 20, tzinfo=timezone.utc), 'America/Los_Angeles'),
])
@pytest.mark.parametrize('sample', range(3))
@pytest.mark.parametrize('offset', [-1, 1])
def test_absolute_plan_matches_local_tomorrow(instant, zone, offset, sample):
    today = instant.astimezone(ZoneInfo(zone)).date()
    day = today + timedelta(days=offset)
    profile = format_warm_profile_block({
        'user': ('The user lives in London. '
                 f'The user has an outing planned for {day.isoformat()} '
                 'in South London to see otters.'),
        'directives': '',
    })
    cfg = voice_config()
    with patch.object(engine, 'get_location_context_with_timezone',
                      return_value=('Location: Unknown', zone)), patch.object(
                          engine, 'format_time_context',
                          side_effect=lambda tz: format_time_context(tz, now_utc=instant)):
        clock = engine._live_time_location_string(cfg)
    answer = get_llm_backend(cfg).direct(
        cfg.llm_chat_model,
        build_system_prompt()+'\n'+profile+f'\n[Context: {clock}]',
        'From your saved information, is an outing established for tomorrow? '
        'Return only a JSON object with "outing_tomorrow" (true or false) '
        'and "home" (city).',
        timeout_sec=60, temperature=0.0, max_tokens=1024,
    )
    assert answer, 'No reply'
    start = answer.find('{')
    assert start >= 0, answer
    result, _ = json.JSONDecoder().raw_decode(answer[start:])
    assert result == {'outing_tomorrow': offset == 1, 'home': 'London'}, answer
