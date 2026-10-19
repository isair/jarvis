"""Speech constraints apply to spoken delivery rather than silent replies."""
from types import SimpleNamespace

import pytest
from jarvis.reply import engine

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures('no_network')]


@pytest.mark.parametrize('backend', ['piper', 'chatterbox'])
@pytest.mark.parametrize('speaker_enabled', [None, False, True])
def test_silent_replies_do_not_inherit_the_speakers_language(monkeypatch, mock_config, db, dialogue_memory, backend, speaker_enabled):
    mock_config.tts_engine = backend
    mock_config.tts_enabled = True
    mock_config.planner_enabled = False
    mock_config.memory_enrichment_source = 'diary'
    mock_config.memory_digest_enabled = False
    spoken = []
    speaker = None if speaker_enabled is None else SimpleNamespace(enabled=speaker_enabled, speak=spoken.append)
    monkeypatch.setattr(engine, 'select_tools', lambda **kwargs: ['stop'])
    monkeypatch.setattr(engine, 'plan_query', lambda **kwargs: [])
    monkeypatch.setattr(engine, 'extract_search_params_for_memory', lambda **kwargs: {'keywords': []})
    def respond(*args, **kwargs):
        messages = kwargs['messages']
        english_only = any(engine.ENGLISH_SPEECH_REPLY_GUIDANCE in message.get('content', '')
                           for message in messages if message['role'] == 'system')
        return {'message': {'content': 'Hello.' if english_only else 'Merhaba.'}}
    monkeypatch.setattr(engine, 'chat_with_messages', respond)
    result = engine.run_reply_engine(db, mock_config, speaker, 'Türkçe selam ver.', dialogue_memory, quiet=True)
    assert result == ('Hello.' if speaker_enabled else 'Merhaba.')
    assert spoken == (['Hello.'] if speaker_enabled else [])


@pytest.mark.parametrize('speaker_enabled', [None, False, True])
def test_voice_dispatch_declares_only_actual_spoken_delivery(monkeypatch, mock_config, speaker_enabled):
    from threading import Lock
    from jarvis import daemon
    from jarvis.listening import listener as module
    monkeypatch.setattr(module, 'create_intent_judge', lambda cfg: None)
    monkeypatch.setattr(daemon, '_chat_query_lock', Lock())
    monkeypatch.setattr(daemon, '_global_stop_requested', False)
    generated, spoken = [], []
    speaker = None if speaker_enabled is None else SimpleNamespace(
        enabled=speaker_enabled, speak=lambda text, **kwargs: spoken.append(text),
    )
    voice = module.VoiceListener(None, mock_config, speaker, None)
    voice._start_thinking_tune = lambda: None
    voice._stop_thinking_tune = lambda: None
    def reply(*args, spoken_output=False, **kwargs):
        text = 'Hello.' if spoken_output else 'Merhaba.'
        generated.append(text)
        return text
    monkeypatch.setattr(engine, 'run_reply_engine', reply)
    try:
        voice._dispatch_query('Türkçe selam ver.', language='tr', generation=voice._capture_generation)
        assert generated == (['Hello.'] if speaker_enabled else ['Merhaba.'])
        assert spoken == (['Hello.'] if speaker_enabled else [])
    finally:
        voice.stop()
