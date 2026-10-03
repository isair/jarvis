"""Local memory recall parameters preserve topic and requested date scope."""
from datetime import datetime, timedelta, timezone

import pytest

from evals.helpers import voice_config
from evals.tool_routing import requires_judge_llm
from jarvis.reply import enrichment

pytestmark = [pytest.mark.eval, requires_judge_llm]
ANCHOR = datetime(2026, 10, 3, 12, tzinfo=timezone.utc)
CASES = [
    ('What did we discuss about cooking?', ('cooking', 'food'), None, None),
    ('What did we discuss about the Python project?', ('python', 'project', 'programming'), None, None),
    ('What news might interest me?', ('interest', 'hobb', 'preference'), None, None),
    ('Recommend a restaurant I would enjoy.', ('restaurant', 'food', 'dining', 'cuisine'), None, None),
    ('What did I eat yesterday?', ('eat', 'food', 'meal', 'nutrition'), -1, None),
    ('Dün bisiklet hakkında ne konuştuk?', ('bisiklet', 'bike', 'bicycle', 'cycling'), -1, None),
    ('¿Qué comí ayer?', ('comida', 'comer', 'food', 'meal', 'eat', 'nutrition'), -1, None),
    ('What did we discuss about cooking today?', ('cooking', 'food'), 0, None),
    ('What did we discuss about cycling on 30 September 2026?', ('cycling', 'bicycle', 'bike'), -3, None),
    ('Recommend a restaurant I would enjoy.', ('restaurant', 'food', 'dining', 'cuisine'), None,
     'Current date/time: Saturday, 2026-10-03 12:00 UTC. Location: Hackney, London.'),
]


@pytest.mark.parametrize('query, topics, day_offset, context', CASES)
def test_recall_parameters_retain_requested_scope(monkeypatch, query, topics, day_offset, context):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return ANCHOR if tz else ANCHOR.replace(tzinfo=None)

    monkeypatch.setattr(enrichment, 'datetime', Clock)
    cfg = voice_config()
    result = enrichment.extract_search_params_for_memory(
        query, cfg, cfg.fast_model, timeout_sec=15.0, context_hint=context,
    )
    keywords = result.get('keywords')
    assert isinstance(keywords, list) and keywords and all(
        isinstance(word, str) and word.strip() for word in keywords
    ), f'🧠 Memory extraction returned no usable keywords: {result}'
    assert any(topic in word.casefold() for topic in topics for word in keywords), (
        f'🧠 Memory keywords lost the requested topic: {result}'
    )
    if day_offset is None:
        assert not result.get('from') and not result.get('to'), (
            f'📅 A timeless request received an unrequested date filter: {result}'
        )
    else:
        requested_date = (ANCHOR + timedelta(days=day_offset)).date()
        assert result.get('from') and result.get('to'), f'📅 Requested date range is missing: {result}'
        for key in ('from', 'to'):
            value = datetime.fromisoformat(result[key].replace('Z', '+00:00'))
            assert value.tzinfo is not None and value.astimezone(timezone.utc).date() == requested_date, (
                f'📅 {key} does not cover the requested day: {result}'
            )
    if context:
        questions = result.get('questions', [])
        assert not any('locat' in question.casefold() or 'where' in question.casefold()
                       for question in questions), f'📍 Location is already available: {result}'
