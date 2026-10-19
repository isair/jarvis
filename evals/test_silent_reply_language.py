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
        assert spoken == ([reply] if delivery == 'spoken' else [])
        expected = 'en' if delivery == 'spoken' else language
        assert judge_pass_fail(
            f'The reply must be in language {expected}. Natural phrasing and proper names are valid. '
            'A reply predominantly in another language fails. Only judge the reply language.',
            f'Request: {query}\nReply: {reply}',
        ) == 'PASS'
    finally:
        db.close()
