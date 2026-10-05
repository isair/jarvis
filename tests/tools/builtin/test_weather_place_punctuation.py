"""Weather fallback preserves punctuation within short geographic names."""
from types import SimpleNamespace

import pytest

from jarvis.tools.builtin import weather

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ('raw_place', 'expected'),
    [
        ('St. Petersburg', 'St. Petersburg'),
        ('Washington D.C.', 'Washington D.C'),
        ("'St. John's'.", "St. John's"),
        ('Frankfurt a. M.', 'Frankfurt a. M'),
    ],
)
def test_fallback_keeps_internal_place_punctuation(monkeypatch, raw_place, expected):
    cfg = SimpleNamespace(fast_model='local-model', llm_chat_model='local-model',
                          llm_tools_timeout_sec=8.)
    backend = SimpleNamespace(direct=lambda *args, **kwargs: raw_place)
    monkeypatch.setattr(weather, 'get_llm_backend', lambda _cfg: backend)

    assert weather._extract_place_from_user_text('weather at the named place', cfg) == expected


@pytest.mark.parametrize('provider', ['ollama', 'openai_compatible'])
def test_place_answer_has_generation_room_after_reasoning(monkeypatch, provider):
    from unittest.mock import MagicMock

    cfg = SimpleNamespace(
        llm_provider=provider, llm_base_url='http://127.0.0.1:1/v1',
        ollama_base_url='http://127.0.0.1:1', llm_api_key='',
        fast_model='local-reasoning-model', llm_chat_model='local-reasoning-model',
        llm_tools_timeout_sec=7.3,
    )
    raw_place = 'Washington D.C.'
    reasoning = ' '.join(['Resolve the city requested by the user.'] * 70)
    required_tokens = len(reasoning.split()) + len(raw_place.split())
    def post(url, **kwargs):
        assert kwargs['timeout'] == cfg.llm_tools_timeout_sec
        payload = kwargs['json']
        cap = payload.get('max_tokens', payload.get('options', {}).get('num_predict', 0))
        content = raw_place if cap >= required_tokens else ''
        message = {'content': content, 'reasoning_content': reasoning}
        response = MagicMock()
        response.__enter__.return_value = response
        response.json.return_value = (
            {'message': message} if provider == 'ollama'
            else {'choices': [{'message': message}]}
        )
        return response
    monkeypatch.setattr('requests.post', post)

    assert weather._extract_place_from_user_text('forecast for Washington D.C.', cfg) == raw_place.rstrip('.')


def test_fallback_city_reaches_geocoder_and_weather_result(monkeypatch):
    raw_place = 'Washington D.C.'
    cfg = SimpleNamespace(fast_model='local-model', llm_chat_model='local-model',
                          llm_tools_timeout_sec=8.)
    context = SimpleNamespace(cfg=cfg, redacted_text='forecast for Washington D.C.',
                              user_print=lambda *args: None)
    city = {'name': 'Washington', 'country': 'United States',
            'latitude': 38.9, 'longitude': -77.0}
    current = {'temperature_2m': 12.5, 'weather_code': 2}
    backend = SimpleNamespace(direct=lambda *args, **kwargs: raw_place)
    monkeypatch.setattr(weather, 'get_llm_backend', lambda _cfg: backend)
    monkeypatch.setattr(weather.WeatherTool, '_get_user_location', lambda self, _ctx: None)
    def get(url, *, params, timeout):
        if 'geocoding-api' in url:
            data = {'results': [city]} if params['name'] == raw_place.rstrip('.') else {'results': []}
        else:
            coordinates_match = (params['latitude'], params['longitude']) == (city['latitude'], city['longitude'])
            data = {'current': current if coordinates_match else {}}
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: data)
    monkeypatch.setattr('requests.get', get)

    result = weather.WeatherTool().run({}, context)
    assert result.success, result.reply_text
    assert city['name'] in result.reply_text
    assert str(current['temperature_2m']) in result.reply_text
