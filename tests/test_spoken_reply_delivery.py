"""Spoken rendering preserves the request, tools and shared conversation."""
from threading import Event
from types import SimpleNamespace

import pytest

from jarvis.reply import engine

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures('no_network')]


@pytest.mark.parametrize('delivery', ['silent', 'disabled', 'engine', 'listener'])
def test_english_rendering_is_shared_by_spoken_callers_only(
    monkeypatch, mock_config, db, dialogue_memory, delivery,
):
    mock_config.tts_engine = 'piper'
    mock_config.planner_enabled = False
    mock_config.memory_enrichment_source = 'diary'
    mock_config.memory_digest_enabled = False
    query = 'Türkçe selam ver.'
    draft, english = 'Merhaba, size nasıl yardımcı olabilirim?', 'Hello, how can I help you?'
    spoken, main_requests, rendering_requests = [], [], []
    speaker = None if delivery in ('silent', 'listener') else SimpleNamespace(
        enabled=delivery == 'engine', speak=spoken.append,
    )
    monkeypatch.setattr(engine, 'select_tools', lambda **kwargs: ['stop'])
    monkeypatch.setattr(engine, 'extract_search_params_for_memory', lambda **kwargs: {'keywords': []})

    def chat(**kwargs):
        main_requests.extend(row['content'] for row in kwargs['messages'] if row['role'] == 'user')
        return {'message': {'content': draft}}

    def direct(model, system_prompt, user_content, **kwargs):
        rendering_requests.append(user_content)
        assert query not in user_content
        assert draft in user_content
        return english

    monkeypatch.setattr(engine, 'chat_with_messages', chat)
    monkeypatch.setattr(engine, 'get_llm_backend', lambda cfg: SimpleNamespace(direct=direct))
    # The renderer uses the same provider boundary as the reply engine.
    from jarvis.reply import delivery as module
    monkeypatch.setattr(module, 'get_llm_backend', lambda cfg: SimpleNamespace(direct=direct))
    monkeypatch.setattr(engine, 'render_english_speech', module.render_english_speech)
    result = engine.run_reply_engine(
        db, mock_config, speaker, query, dialogue_memory, quiet=True,
        spoken_output=delivery == 'listener',
    )
    expected = english if delivery in ('engine', 'listener') else draft
    assert result == expected
    assert spoken == ([english] if delivery == 'engine' else [])
    assert main_requests == [query]
    assert bool(rendering_requests) == (delivery in ('engine', 'listener'))
    saved = dialogue_memory.get_recent_messages()
    assert any(row['role'] == 'user' and row['content'] == query for row in saved)
    assert any(row['role'] == 'assistant' and row['content'] == expected for row in saved)


@pytest.mark.parametrize('failure', [None, '', '   ', RuntimeError('unavailable')])
def test_unavailable_rendering_preserves_the_original_result(monkeypatch, mock_config, failure):
    from jarvis.reply import delivery as module

    def direct(*args, **kwargs):
        if isinstance(failure, Exception):
            raise failure
        return failure

    monkeypatch.setattr(module, 'get_llm_backend', lambda cfg: SimpleNamespace(direct=direct))
    text = 'El informe está en /tmp/report.txt: 42 elementos.'
    assert module.render_english_speech(mock_config, text) == text


def test_oversized_rendering_retains_the_complete_original(monkeypatch, mock_config):
    from jarvis.reply import delivery as module
    requests = []
    monkeypatch.setattr(module, 'get_llm_backend', lambda cfg: requests.append(cfg))
    text = '引用資料 "quoted"\n' * 1000
    assert module.render_english_speech(mock_config, text) == text
    assert requests == []


def test_cancellation_during_rendering_suppresses_delivery(monkeypatch, mock_config, db, dialogue_memory):
    from jarvis.reply import delivery as module
    cancelled, spoken = Event(), []
    mock_config.planner_enabled = False
    mock_config.memory_enrichment_source = 'diary'
    mock_config.memory_digest_enabled = False
    monkeypatch.setattr(engine, 'select_tools', lambda **kwargs: ['stop'])
    monkeypatch.setattr(engine, 'extract_search_params_for_memory', lambda **kwargs: {'keywords': []})
    monkeypatch.setattr(engine, 'chat_with_messages', lambda **kwargs: {'message': {'content': 'Merhaba.'}})

    def direct(*args, **kwargs):
        cancelled.set()
        return 'Hello.'

    monkeypatch.setattr(module, 'get_llm_backend', lambda cfg: SimpleNamespace(direct=direct))
    monkeypatch.setattr(engine, 'render_english_speech', module.render_english_speech)
    speaker = SimpleNamespace(enabled=True, speak=spoken.append)
    assert engine.run_reply_engine(db, mock_config, speaker, 'Selam ver.', dialogue_memory,
                                   quiet=True, cancel_event=cancelled) is None
    assert spoken == []
    assert dialogue_memory.get_recent_messages() == []


@pytest.mark.parametrize('quoted', ['"Guten Morgen"', '“Guten Morgen”', '«Guten Morgen»'])
def test_quoted_literals_are_restored_verbatim(monkeypatch, mock_config, quoted):
    from jarvis.reply import delivery as module
    import json

    def direct(model, prompt, content, **kwargs):
        masked = json.loads(content)['text_to_translate']
        assert quoted not in masked
        return masked.replace('El título es', 'The title is')

    monkeypatch.setattr(module, 'get_llm_backend', lambda cfg: SimpleNamespace(direct=direct))
    assert module.render_english_speech(mock_config, f'El título es {quoted}.') == f'The title is {quoted}.'


@pytest.mark.parametrize('rendered', ['The title is missing.', 'JARVIS_LITERAL_0 JARVIS_LITERAL_0'])
def test_broken_literal_roundtrip_preserves_the_original(monkeypatch, mock_config, rendered):
    from jarvis.reply import delivery as module
    monkeypatch.setattr(module, 'get_llm_backend', lambda cfg: SimpleNamespace(direct=lambda *args, **kwargs: rendered))
    original = 'El título es "Guten Morgen".'
    assert module.render_english_speech(mock_config, original) == original


def test_many_quoted_literals_remain_distinct(monkeypatch, mock_config):
    from jarvis.reply import delivery as module
    import json
    monkeypatch.setattr(module, 'get_llm_backend', lambda cfg: SimpleNamespace(
        direct=lambda model, prompt, content, **kwargs: json.loads(content)['text_to_translate'].replace('El título es', 'The title is'),
    ))
    literals = ', '.join(f'"Title {i}"' for i in range(12))
    assert module.render_english_speech(mock_config, f'El título es {literals}.') == f'The title is {literals}.'


def test_canned_malformed_guard_reply_is_not_rewritten(monkeypatch, mock_config, db):
    from jarvis.memory.conversation import DialogueMemory
    mock_config.planner_enabled = False
    mock_config.memory_enrichment_source = 'diary'
    mock_config.memory_digest_enabled = False
    monkeypatch.setattr(engine, 'select_tools', lambda **kwargs: ['stop'])
    monkeypatch.setattr(engine, 'extract_search_params_for_memory', lambda **kwargs: {'keywords': []})
    monkeypatch.setattr(engine, 'chat_with_messages', lambda **kwargs: {'message': {'content': '{"unfinished":'}})
    monkeypatch.setattr(engine, 'render_english_speech', lambda *args: pytest.fail('Canned guard replies must remain recognisable'))
    silent = engine.run_reply_engine(db, mock_config, None, 'Hello', DialogueMemory(), quiet=True)
    spoken = []
    speaker = SimpleNamespace(enabled=True, speak=spoken.append)
    result = engine.run_reply_engine(db, mock_config, speaker, 'Hello', DialogueMemory(), quiet=True)
    assert result == silent
    assert spoken == [silent]
