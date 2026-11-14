"""Silent reply language is independent of the saved local speech backend."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from evals.helpers import isolated_voice_settings, judge_pass_fail, assert_not_fallback_reply
from jarvis.memory.conversation import DialogueMemory
from jarvis.memory.db import Database
from jarvis.reply import engine

pytestmark = pytest.mark.eval


@pytest.mark.parametrize('backend', ['piper', 'chatterbox'])
@pytest.mark.parametrize('delivery', ['text', 'disabled', 'spoken'])
@pytest.mark.parametrize('query,language', [
    ('Sana neler sorabilirim? Lütfen tek cümleyle Türkçe yanıtla.', 'tr'),
    ('Wobei kannst du mir helfen? Antworte auf Deutsch mit einem kurzen Satz.', 'de'),
    ('¿En qué puedes ayudarme? Responde en español con una frase breve.', 'es'),
])
def test_reply_language_follows_actual_delivery(monkeypatch, backend, delivery, query, language, record_property):
    cfg = replace(isolated_voice_settings(), tts_engine=backend, tts_enabled=True,
                  planner_enabled=False, llm_chat_timeout_sec=60,
                  location_enabled=False, mcps={})
    monkeypatch.setattr(engine, 'select_tools', lambda *args, **kwargs: ['stop'])
    spoken = []
    speaker = None if delivery == 'text' else SimpleNamespace(enabled=delivery == 'spoken', speak=spoken.append)
    db = Database(':memory:')
    try:
        reply = engine.run_reply_engine(db, cfg, speaker, query, DialogueMemory(), quiet=True)
        assert reply, '💬 A language request needs a reply'
        assert_not_fallback_reply(reply, 'reply delivery language')
        record_property('reply', reply)
        record_property('query', query)
        assert spoken == ([reply] if delivery == 'spoken' else [])
        expected = 'en' if delivery == 'spoken' else language
        assert judge_pass_fail(
            f'The reply must be in language {expected}. Natural phrasing and proper names are valid. '
            'A reply predominantly in another language fails. Only judge the reply language.',
            f'Reply: {reply}',
        ) == 'PASS'
    finally:
        db.close()


@pytest.mark.parametrize('expected,reply', [
    ('en', 'I can help you find information.'),
    ('tr', 'Size bilgi bulmada yardımcı olabilirim.'),
    ('de', 'Ich kann Ihnen helfen, Informationen zu finden.'),
    ('es', 'Puedo ayudarte a buscar información.'),
])
@pytest.mark.parametrize('correct', [True, False])
def test_language_judge_accepts_matching_and_rejects_other_languages(expected, reply, correct):
    actual = expected if correct else {'en': 'tr', 'tr': 'de', 'de': 'es', 'es': 'en'}[expected]
    assert judge_pass_fail(
        f'The reply must be in language {actual}. Natural phrasing and proper names are valid. '
        'A reply predominantly in another language fails. Only judge the reply language.',
        f'Reply: {reply}',
    ) == ('PASS' if correct else 'FAIL')
