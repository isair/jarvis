"""URL hygiene preserves literal places and addresses in planned calls."""
import json

import pytest

from evals.helpers import voice_config
from jarvis.reply import planner

pytestmark = pytest.mark.unit


def resolve(monkeypatch, path, key, value, metadata):
    schema = [{'type': 'function', 'function': {
        'name': 'localLookup',
        'parameters': {'type': 'object', 'properties': {key: metadata}},
    }}]
    if path == 'concrete':
        monkeypatch.setattr(planner, 'call_llm_direct', lambda **kwargs: pytest.fail('🗺️ Concrete values need no model'))
        step = f"localLookup {key}='{value}'"
    else:
        raw = json.dumps({'name': 'localLookup', 'arguments': {key: value}})
        monkeypatch.setattr(planner, 'call_llm_direct', lambda **kwargs: raw)
        step = f'localLookup {key}=<discovered value>'
    return planner.resolve_next_tool_call(voice_config(), step, [], schema)


@pytest.mark.parametrize('path', ['concrete', 'model'])
@pytest.mark.parametrize('key, value, metadata', [
    ('location', 'St.Gallen', {'type': 'string'}),
    ('location', 'St.Albans', {'type': 'string'}),
    ('address', 'person@example.com', {'type': 'string', 'format': 'email'}),
    ('address', 'device.local', {'type': 'string'}),
    ('location', '[station](https://example.test)', {'type': 'string'}),
    ('address', '  device.local  ', {'type': 'string'}),
])
def test_non_url_fields_preserve_literal_values(monkeypatch, path, key, value, metadata):
    assert resolve(monkeypatch, path, key, value, metadata) == ('localLookup', {key: value})


@pytest.mark.parametrize('path', ['concrete', 'model'])
@pytest.mark.parametrize('key, value, metadata, expected', [
    ('url', 'example.test', {'type': 'string'}, 'https://example.test'),
    ('href', '[guide](https://example.test/guide)', {'type': 'string'}, 'https://example.test/guide'),
    ('location', 'example.test', {'type': 'string', 'format': 'uri'}, 'https://example.test'),
    ('address', '[guide](https://example.test/guide)', {'type': 'string', 'format': 'uri-reference'}, 'https://example.test/guide'),
    ('destination', 'example.test', {'type': 'string', 'format': 'uri'}, 'https://example.test'),
    ('url', 'https://example.test/guide', {'type': 'string'}, 'https://example.test/guide'),
])
def test_url_fields_retain_normalisation(monkeypatch, path, key, value, metadata, expected):
    assert resolve(monkeypatch, path, key, value, metadata) == ('localLookup', {key: expected})
