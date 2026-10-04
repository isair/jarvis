"""Personal defaults are verified against source meaning, not copied quotations."""
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit


def _run(db, monkeypatch, text, candidate, review):
    from jarvis.reply import personal_context
    db.upsert_conversation_summary(datetime.now(timezone.utc).date().isoformat(), text, 'personal', 'jarvis')
    responses = iter([json.dumps([candidate]), json.dumps(review)])
    monkeypatch.setattr(personal_context, 'get_llm_backend', lambda cfg:
                        SimpleNamespace(direct=lambda *a, **kw: next(responses)))
    cfg = SimpleNamespace(fast_model='fast', llm_chat_model='chat',
                          llm_tools_timeout_sec=8., memory_enrichment_source='diary')
    return personal_context.resolve_missing_context('location', db, cfg, 'weather tomorrow', [])


def test_supported_source_survives_paraphrased_model_quote(db, monkeypatch):
    candidate = dict(value='Ankara', source='diary:1', kind='home', evidence='The user lives in Ankara.')
    value = _run(db, monkeypatch, 'The user stated that they live in Ankara.',
                 candidate, [{'id': 0, 'supported': True}])
    assert value and value.value == 'Ankara'


def test_quoted_translation_does_not_establish_residence(db, monkeypatch):
    candidate = dict(value='Bristol', source='diary:1', kind='home', evidence='I live in Bristol')
    value = _run(db, monkeypatch, 'The user requested a translation of "I live in Bristol".',
                 candidate, [{'id': 0, 'supported': False}])
    assert value is None


@pytest.mark.parametrize('review', [[], [{'id': 0, 'supported': 'yes'}],
                                   [{'id': 1, 'supported': True}],
                                   [{'id': 0, 'supported': True}, {'id': 0, 'supported': True}]])
def test_incomplete_or_malformed_verification_requires_clarification(db, monkeypatch, review):
    candidate = dict(value='London', source='diary:1', kind='home', evidence='The user lives in London.')
    assert _run(db, monkeypatch, 'The user lives in London.', candidate, review) is None
