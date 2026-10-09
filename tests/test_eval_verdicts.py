"""Recorded-answer checks share a strict structured verdict contract."""
from unittest.mock import MagicMock, patch

import pytest

from evals import helpers

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('provider', ['ollama', 'openai_compatible'])
@pytest.mark.parametrize('expected', ['PASS', 'FAIL'])
def test_verdict_uses_constrained_output_without_changing_the_actor(monkeypatch, provider, expected):
    monkeypatch.setenv('EVAL_JUDGE_PROVIDER', provider)
    monkeypatch.setenv('EVAL_VERIFIER_MODEL', 'fixture-verifier')
    actor = helpers.voice_config().llm_chat_model
    payloads = []

    def post(url, **kwargs):
        payloads.append(kwargs['json'])
        response = MagicMock()
        answer = '{"verdict": "' + expected + '"}'
        response.json.return_value = {'message': {'content': answer},
                                     'choices': [{'message': {'content': answer}}]}
        return response

    with patch('requests.post', side_effect=post):
        verdict = helpers.judge_pass_fail('Requires the actual reading.', 'Recorded reading: 12 C.')
    assert verdict == expected
    payload = payloads[0]
    schema = (payload['format'] if provider == 'ollama'
              else payload['response_format']['json_schema']['schema'])
    assert schema['type'] == 'object'
    assert schema['required'] == ['verdict']
    assert schema['properties']['verdict']['enum'] == ['PASS', 'FAIL']
    assert schema['additionalProperties'] is False
    assert payload['model'] == 'fixture-verifier'
    assert helpers.voice_config().llm_chat_model == actor


@pytest.mark.parametrize('output', [
    None, {}, [], 7, '', 'PASS', 'PASS with explanation', '{"verdict":"PASS"} extra',
    '```json\n{"verdict":"PASS"}\n```', 'null', '[]', '{}', 'true', '7',
    '[["verdict","PASS"]]',
    '{"verdict":true}', '{"verdict":null}', '{"verdict":"UNKNOWN"}',
    '{"verdict":"pass"}', '{"verdict":"PASS","explanation":"fine"}',
    '{"verdict":"FAIL","verdict":"PASS"}',
])
def test_malformed_or_unavailable_verdict_is_unknown(output):
    with patch.object(helpers, 'call_judge_llm', return_value=output):
        assert helpers.judge_pass_fail('Requires the actual reading.', 'Recorded answer.') is None


@pytest.mark.parametrize('verdict', ['PASS', 'FAIL'])
def test_valid_verdict_retains_its_meaning(verdict):
    with patch.object(helpers, 'call_judge_llm', return_value='\n{"verdict":"' + verdict + '"}\n'):
        assert helpers.judge_pass_fail('Requires the actual reading.', 'Recorded answer.') == verdict
