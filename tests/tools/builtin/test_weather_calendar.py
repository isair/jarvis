"""Forecast calendar labels follow the service's local dates."""
from datetime import date, timedelta
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from jarvis.tools.builtin.weather import WeatherTool

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("no_network")]


def weather_payload(day='2026-10-11', timezone='Europe/London'):
    start = date.fromisoformat(day)
    return {
        'timezone': timezone,
        'current': {'time': f'{day}T23:45', 'temperature_2m': 14, 'weather_code': 3},
        'daily': {
            'time': [(start + timedelta(days=i)).isoformat() for i in range(7)],
            'weather_code': [61, 61, 0, 0, 0, 0, 0],
            'temperature_2m_min': [13] * 7,
            'temperature_2m_max': [17] * 7,
        },
    }


def render_weather(payload):
    """Exercise the real tool without location inference or HTTP transport."""
    response = Mock()
    response.json.return_value = payload
    context = SimpleNamespace(cfg=SimpleNamespace(location_enabled=True), user_print=Mock())
    with patch('jarvis.tools.builtin.weather.get_location_info', return_value={
        'latitude': 51.5, 'longitude': -0.1, 'city': 'London',
    }), patch('jarvis.tools.builtin.weather.requests.get', return_value=response):
        result = WeatherTool().run({}, context)
    assert result.success, result.reply_text
    return result.reply_text


@pytest.mark.parametrize(('day', 'timezone'), [
    ('2026-10-11', 'Europe/London'),
    ('2027-12-31', 'Pacific/Auckland'),
    ('2028-02-28', 'America/Los_Angeles'),
])
def test_forecast_labels_use_service_calendar(day, timezone):
    payload = weather_payload(day, timezone)
    result = render_weather(payload)
    assert f"Local observation time: {payload['current']['time']} ({timezone})" in result
    for recorded in payload['daily']['time']:
        weekday = date.fromisoformat(recorded).strftime('%A')
        assert f'{recorded} ({weekday}): 13–17°C' in result


def test_bad_calendar_row_does_not_invent_weekday_or_discard_readings():
    payload = weather_payload()
    payload['daily']['time'][1] = 'not-a-date'
    result = render_weather(payload)
    assert 'not-a-date: 13–17°C' in result
    assert '2026-10-13 (Tuesday): 13–17°C' in result
