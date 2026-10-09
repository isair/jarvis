"""Response verification can use a separate model on the selected transport."""
from unittest.mock import MagicMock, patch

import pytest

from evals import helpers

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('provider', ['ollama', 'openai_compatible'])
@pytest.mark.parametrize('verifier_model', [None, '   ', 'fixture-verifier'])
def test_verifier_override_preserves_the_model_under_evaluation(monkeypatch, provider, verifier_model):
    if verifier_model is None:
        monkeypatch.delenv('EVAL_VERIFIER_MODEL', raising=False)
    else:
        monkeypatch.setenv('EVAL_VERIFIER_MODEL', verifier_model)
    monkeypatch.setenv('EVAL_JUDGE_PROVIDER', provider)
    actor_model = helpers.voice_config().llm_chat_model
    requests = []
    def post(url, **kwargs):
        requests.append(kwargs['json'])
        response = MagicMock()
        response.json.return_value = {'message': {'content': 'PASS'},
            'choices': [{'message': {'content': 'PASS'}}]}
        return response
    with patch('requests.post', side_effect=post):
        verdict = helpers.call_judge_llm('Judge only.', 'Recorded answer.')
    assert verdict == 'PASS'
    assert requests[0]['model'] == ((verifier_model or '').strip() or helpers.JUDGE_MODEL)
    assert helpers.voice_config().llm_chat_model == actor_model
