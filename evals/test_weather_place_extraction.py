"""Local-model place extraction preserves geographic punctuation."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from evals.conftest import requires_judge_llm
from evals.helpers import voice_config
from jarvis.tools.builtin import weather


@pytest.mark.eval
@requires_judge_llm
@pytest.mark.parametrize(
    ('utterance', 'expected'),
    [
        ('What is the forecast for Washington D.C.?', 'Washington DC'),
        ('How is the weather in St. Petersburg?', 'St Petersburg'),
        ('Wie ist das Wetter in Nürnberg?', 'Nürnberg'),
        ('How is the weather in San Francisco tomorrow?', 'San Francisco'),
        ('Yarın hava nasıl olacak?', None),
    ],
)
def test_place_reaches_fallback_with_its_geographic_name(utterance, expected):
    cfg = voice_config()
    backend = weather.get_llm_backend(cfg)
    answers = []
    def direct(*args, **kwargs):
        answer = backend.direct(*args, **kwargs)
        answers.append(answer)
        return answer
    with patch.object(weather, 'get_llm_backend', return_value=SimpleNamespace(direct=direct)):
        place = weather._extract_place_from_user_text(utterance, cfg)
    assert answers and isinstance(answers[0], str) and answers[0].strip(), 'Empty inference is not successful extraction'
    if expected is None:
        assert place is None
    else:
        normalise = lambda text: ''.join(char for char in text.casefold() if char.isalnum())
        assert place and normalise(place) == normalise(expected), (utterance, place)
