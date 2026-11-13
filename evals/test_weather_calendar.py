"""Local chat models preserve calendar dates from real weather-tool results."""
from datetime import date
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from evals.conftest import requires_judge_llm
from evals.helpers import voice_config
from jarvis.llm import get_llm_backend
from jarvis.system_prompt import build_system_prompt
from jarvis.tools.builtin.weather import WeatherTool

pytestmark = [pytest.mark.eval, requires_judge_llm]


@pytest.mark.parametrize('dates,zone', [
    (['2026-10-12', '2026-10-13', '2026-10-16'], 'Europe/London'),
    (['2027-12-31', '2028-01-01', '2028-01-02'], 'Pacific/Auckland'),
    (['2028-02-28', '2028-02-29', '2028-03-01'], 'America/Los_Angeles'),
])
def test_reply_preserves_forecast_calendar(dates, zone):
    cfg = voice_config()
    response = Mock()
    response.json.return_value = {
        'timezone': zone,
        'current': {'time': f'{dates[0]}T00:15', 'temperature_2m': 14, 'weather_code': 3},
        'daily': {'time': dates, 'weather_code': [61, 0, 0],
                  'temperature_2m_min': [13, 12, 11], 'temperature_2m_max': [17, 18, 19]},
    }
    context = SimpleNamespace(cfg=cfg, user_print=lambda text: None)
    with patch('jarvis.tools.builtin.weather.get_location_info', return_value={
        'latitude': 51.5, 'longitude': -0.1, 'city': 'Example city',
    }), patch('jarvis.tools.builtin.weather.requests.get', return_value=response):
        weather = WeatherTool().run({}, context)
    assert weather.success
    answer = get_llm_backend(cfg).direct(
        cfg.llm_chat_model, build_system_prompt(),
        'From the weather result below, return only a JSON array, one object per daily '
        'forecast row using exactly these keys: date (YYYY-MM-DD), weekday (English name), '
        'minimum (number in Celsius), maximum (number in Celsius). Preserve the recorded forecast location calendar.\n\n' + weather.reply_text,
        timeout_sec=60, temperature=0.0, max_tokens=1024,
    )
    assert answer, 'No reply'
    start = answer.find('[')
    assert start >= 0, answer
    readings, _ = json.JSONDecoder().raw_decode(answer[start:])
    expected = [{'date': day, 'weekday': date.fromisoformat(day).strftime('%A'),
                 'minimum': low, 'maximum': high}
                for day, low, high in zip(dates, [13, 12, 11], [17, 18, 19])]
    assert readings == expected, answer
