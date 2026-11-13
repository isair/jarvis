"""Explicit destinations take precedence over the user's detected location."""
from unittest.mock import patch

import pytest

from evals.conftest import requires_judge_llm
from evals.helpers import voice_config, judge_pass_fail
from jarvis.reply import engine
from jarvis.tools.types import ToolExecutionResult

pytestmark = [pytest.mark.eval, requires_judge_llm]
_DEFAULT_CITY = 'London'
_DETECTED_LOCATION = 'London, UK'
_TEMPERATURE = 15
_READING_CRITERIA = (
    f'PASS requires reporting the actual recorded temperature of {_TEMPERATURE} '
    'degrees Celsius and clear conditions, in any language. Any named place '
    'must agree with the recorded location. Missing, invented or contradictory '
    'readings require FAIL.'
)


@pytest.mark.parametrize('query,expected,previous', [
    ('What is the weather in Ankara?', 'Ankara', None),
    ('Wie ist das Wetter in Nürnberg?', 'Nürnberg', None),
    ('Paris hava durumu nasıl?', 'Paris', None),
    ('What is the weather in Washington D.C.?', 'Washington D.C.', None),
    ('What is the weather in Cambridge, Massachusetts?', 'Cambridge Massachusetts', None),
    ('How is the weather here?', None, None),
    ('Yarın hava nasıl olacak?', None, None),
    ('How is the weather there?', 'Paris', 'I will visit Paris tomorrow.'),
    ('Orada hava nasıl olacak?', 'Ankara', 'Yarın Ankara şehrine gideceğim.'),
])
def test_requested_city_is_not_replaced_by_detected_home(eval_db, eval_dialogue_memory, query, expected, previous):
    if previous:
        eval_dialogue_memory.add_message('user', previous)
        eval_dialogue_memory.add_message('assistant', 'Okay.')
    cfg = voice_config()
    cfg.location_enabled = True
    locations = []
    responses = []
    live_chat = engine.chat_with_messages
    def capture_reply(**kwargs):
        response = live_chat(**kwargs)
        responses.append(response)
        return response
    def fixture_weather(db, cfg, tool_name, tool_args, **kwargs):
        if tool_name != 'getWeather':
            return ToolExecutionResult(success=False, reply_text='No additional tools available.')
        location = (tool_args or {}).get('location')
        locations.append(location)
        return ToolExecutionResult(success=True, reply_text=f'Weather for {location or _DETECTED_LOCATION}: {_TEMPERATURE} C, clear.')
    with patch.object(engine, 'chat_with_messages', side_effect=capture_reply), \
            patch.object(engine, 'run_tool_with_retries', side_effect=fixture_weather), \
            patch.object(engine, 'select_tools', return_value=['getWeather', 'stop']), \
            patch.object(engine, 'plan_query', return_value=[]), \
            patch.object(engine, 'get_location_context_with_timezone', return_value=(f'Location: {_DETECTED_LOCATION}', None)), \
            patch.object(engine, 'extract_search_params_for_memory', return_value={'keywords': []}):
        reply = engine.run_reply_engine(eval_db, cfg, None, query, eval_dialogue_memory)
    assert locations, f'No weather lookup executed: {reply}; model outputs: {responses!r}'
    fold = lambda text: ''.join(c for c in str(text or '').casefold() if c.isalnum())
    if expected:
        assert all(fold(expected) in fold(location) for location in locations), (query, locations, reply)
    else:
        allowed_defaults = {fold(_DEFAULT_CITY), fold(_DETECTED_LOCATION)}
        assert all(not location or fold(location) in allowed_defaults for location in locations), (query, locations, reply)
    verdict = judge_pass_fail(
        _READING_CRITERIA,
        f'Request: {query}\nRecorded location: {expected or _DETECTED_LOCATION}\n'
        f'Recorded temperature: {_TEMPERATURE} C, clear.\nRecorded answer: {reply}',
    )
    assert verdict == 'PASS', (query, reply, verdict)


@pytest.mark.parametrize('place,temperature,expected', [
    (_DETECTED_LOCATION, _TEMPERATURE, 'PASS'),
    ('Paris', _TEMPERATURE, 'FAIL'),
    (_DETECTED_LOCATION, _TEMPERATURE + 3, 'FAIL'),
])
def test_weather_verifier_rejects_wrong_place_or_reading(place, temperature, expected):
    verdict = judge_pass_fail(
        _READING_CRITERIA,
        f'Recorded location: {_DETECTED_LOCATION}\nRecorded temperature: {_TEMPERATURE} C, clear.\n'
        f'Recorded answer: {place} is {temperature} C and clear.',
    )
    assert verdict == expected
