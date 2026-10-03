"""Meal extraction and coaching can emit answers after local reasoning."""
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from jarvis.memory.db import Database
from jarvis.tools.builtin.nutrition import log_meal

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('provider', ['ollama', 'openai_compatible'])
@pytest.mark.parametrize('phase', ['extraction', 'coaching'])
def test_nutrition_answer_survives_reasoning_budget(monkeypatch, provider, phase):
    cfg = SimpleNamespace(
        llm_provider=provider, llm_base_url='http://127.0.0.1:1/v1',
        ollama_base_url='http://127.0.0.1:1', llm_chat_model='local-reasoning-model',
        llm_chat_timeout_sec=7.3, llm_thinking_enabled=False,
    )
    meal = {'description': 'eggs with toast', 'calories_kcal': 220,
            'protein_g': 14, 'carbs_g': 18, 'fat_g': 11, 'confidence': 0.8}
    coaching = 'Drink water and add vegetables to your next meal.'
    reasoning = ' '.join(['Consider the meal and typical portions.'] * 80)
    observed_timeouts = []
    def post(url, **kwargs):
        observed_timeouts.append(kwargs['timeout'])
        payload = kwargs['json']
        is_extraction = payload['messages'][0]['content'] == log_meal.NUTRITION_SYS
        answer = json.dumps(meal) if is_extraction else coaching
        cap = payload.get('max_tokens', payload.get('options', {}).get('num_predict', 0))
        required = len(reasoning.split()) + len(answer.split())
        response = MagicMock()
        response.__enter__.return_value = response
        message = {'content': answer if cap >= required else '', 'reasoning_content': reasoning}
        response.json.return_value = ({'message': message} if provider == 'ollama'
                                     else {'choices': [{'message': message}]})
        return response
    monkeypatch.setattr('requests.post', post)
    if phase == 'extraction':
        db = Database(':memory:', sqlite_vss_path=None)
        try:
            reply = log_meal.extract_and_log_meal(db, cfg, 'I ate eggs with toast', 'stdin')
            now = datetime.now(timezone.utc)
            rows = db.get_meals_between((now - timedelta(minutes=1)).isoformat(),
                                        (now + timedelta(minutes=1)).isoformat())
            assert rows and rows[0]['description'] == meal['description']
            assert rows[0]['calories_kcal'] == meal['calories_kcal']
            assert reply and meal['description'] in reply
        finally:
            db.close()
    else:
        assert log_meal.generate_followups_for_meal(cfg, meal['description'], 'approximate macros') == coaching
    assert observed_timeouts and all(timeout == cfg.llm_chat_timeout_sec for timeout in observed_timeouts)
