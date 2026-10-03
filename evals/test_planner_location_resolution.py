"""Dependent weather calls preserve punctuation in discovered place names."""
import pytest

from evals.helpers import voice_config
from evals.tool_routing import requires_judge_llm
from jarvis.reply.planner import resolve_next_tool_call
from jarvis.tools.builtin.weather import WeatherTool

pytestmark = [pytest.mark.eval, requires_judge_llm]


@pytest.mark.parametrize('place', ['St.Gallen', 'St.Albans', 'St.Louis'])
def test_weather_resolver_preserves_discovered_place(place):
    schema = [{'type': 'function', 'function': {
        'name': 'getWeather',
        'description': 'Get the weather for the specified place.',
        'parameters': WeatherTool().inputSchema,
    }}]
    resolved = resolve_next_tool_call(
        voice_config(), "getWeather location='<discovered city>'",
        [('localLookup', '{}', f'The selected city is {place}.')],
        schema, timeout_sec=60.0,
    )
    assert resolved is not None, '🌤️ A discovered city must remain resolvable'
    name, arguments = resolved
    assert name == 'getWeather', f'🌤️ Weather tool was changed: {resolved}'
    assert arguments.get('location', '').casefold() == place.casefold(), f'📍 City was altered: {resolved}'
