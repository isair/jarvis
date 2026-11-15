"""Recorded countdown facts ground final delivery independently of a flawed draft."""
import json
import time
from types import SimpleNamespace

import pytest

from jarvis.reply import delivery, engine, timer_delivery
from jarvis.tools.types import ToolExecutionResult

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures('no_network')]


@pytest.mark.parametrize('spoken', [False, True])
def test_recorded_creation_corrects_false_expiry_before_publication(monkeypatch, mock_config, db, dialogue_memory, spoken):
    monkeypatch.setattr(engine, 'render_written_reply', delivery.render_written_reply)
    monkeypatch.setattr(engine, 'render_english_speech', delivery.render_english_speech)
    monkeypatch.setattr(engine, 'timer_reply_source', timer_delivery.timer_reply_source)
    mock_config.planner_enabled = False
    monkeypatch.setattr(engine, 'select_tools', lambda **kw: ['timer', 'stop'])
    monkeypatch.setattr(engine, 'extract_search_params_for_memory', lambda **kw: {})
    args = {'action': 'set', 'minutes': 5, 'label': 'Tea', 'expiry_announcement': 'Your tea timer is up'}
    responses = iter([{'message': {'tool_calls': [{'id': 'owned', 'function': {'name': 'timer', 'arguments': args}}]}}, {'message': {'content': 'Your tea timer is up.'}}])
    monkeypatch.setattr(engine, 'chat_with_messages', lambda **kw: next(responses))
    committed = {'id': 'owned-countdown', 'label': 'Tea', 'duration_sec': 300, 'state': 'running'}
    effects = []
    def execute(**kw):
        effects.append(kw['tool_args'])
        return ToolExecutionResult(True, 'Countdown state: running\nElapsed: false', result_metadata={'created_timer': committed})
    monkeypatch.setattr(engine, 'run_tool_with_retries', execute)
    def render(model, system, content, **kw):
        packet = json.loads(content)
        if 'current_request' in packet:
            return 'en'
        assert set(packet) == {'text_to_translate'}
        assert 'Your tea timer is up.' not in packet['text_to_translate']
        assert committed['id'] not in content
        return packet['text_to_translate']
    def extract(model, messages, **kw):
        packet = json.loads(messages[1]['content'])
        assert packet == {'current_request': 'Start a five-minute tea timer.'}
        assert committed['id'] not in messages[1]['content']
        return {'message': {'content': '{"other_requests":[]}'}}
    backend = SimpleNamespace(direct=render, chat=extract)
    monkeypatch.setattr(delivery, 'get_llm_backend', lambda cfg: backend)
    monkeypatch.setattr(timer_delivery, 'get_llm_backend', lambda cfg: backend)
    result = engine.run_reply_engine(db, mock_config, None, 'Start a five-minute tea timer.', dialogue_memory, spoken_output=spoken)
    assert 'Started 1 timer' in result and '5 minutes' in result and 'Tea' in result
    assert 'running' in result and 'is up' not in result
    assert len(effects) == 1
    assert any(row['role'] == 'assistant' and row['content'] == result for row in dialogue_memory.get_recent_messages())


@pytest.mark.parametrize('spoken', [False, True])
def test_unavailable_rendering_keeps_a_truthful_creation_acknowledgement(monkeypatch, mock_config, spoken):
    backend = SimpleNamespace(direct=lambda *a, **kw: None, chat=lambda *a, **kw: None)
    monkeypatch.setattr(delivery, 'get_llm_backend', lambda cfg: backend)
    monkeypatch.setattr(timer_delivery, 'get_llm_backend', lambda cfg: backend)
    facts = {'created_count': 1, 'created_timers': [{'id': 'private-id', 'label': 'Tea', 'duration_sec': 300, 'state': 'running'}]}
    draft = 'Your tea timer is up.'
    deadline = time.monotonic() + mock_config.llm_chat_timeout_sec
    source = timer_delivery.timer_reply_source(mock_config, 'Start a tea timer.', draft, facts['created_timers'], deadline)
    result = (delivery.render_english_speech(mock_config, source, protect_numbers=True, deadline=deadline) if spoken else
              delivery.render_written_reply(mock_config, 'Start a tea timer.', source, protect_numbers=True, deadline=deadline))
    assert 'Started 1 timer' in result
    assert '5 minutes' in result and 'Tea' in result
    assert draft not in result and 'private-id' not in result


def test_creation_records_do_not_override_a_later_cancellation(monkeypatch, mock_config, db, dialogue_memory):
    monkeypatch.setattr(engine, 'render_written_reply', delivery.render_written_reply)
    monkeypatch.setattr(engine, 'timer_reply_source', timer_delivery.timer_reply_source)
    mock_config.planner_enabled = False
    mock_config.llm_chat_model = mock_config.ollama_chat_model = 'gpt-oss:20b'
    monkeypatch.setattr(engine, 'select_tools', lambda **kw: ['timer', 'stop'])
    monkeypatch.setattr(engine, 'extract_search_params_for_memory', lambda **kw: {})
    actions = [{'action': 'set', 'minutes': 5, 'expiry_announcement': 'Timer is up'}, {'action': 'cancel', 'all': True}]
    calls = [{'id': str(i), 'function': {'name': 'timer', 'arguments': args}} for i, args in enumerate(actions)]
    draft = 'The timer was started and then cancelled.'
    responses = iter([{'message': {'tool_calls': calls}}, {'message': {'content': draft}}])
    monkeypatch.setattr(engine, 'chat_with_messages', lambda **kw: next(responses))
    effects = []
    def execute(**kw):
        effects.append(kw['tool_args'])
        metadata = {'created_timer': {'id': 'owned', 'duration_sec': 300, 'label': None, 'state': 'running'}} if kw['tool_args']['action'] == 'set' else None
        return ToolExecutionResult(True, 'Recorded action completed.', result_metadata=metadata)
    monkeypatch.setattr(engine, 'run_tool_with_retries', execute)
    def render(model, system, content, **kw):
        packet = json.loads(content)
        if 'current_request' in packet:
            return 'en'
        assert 'timer_facts' not in packet
        return packet['text_to_translate']
    backend = SimpleNamespace(direct=render, chat=lambda *a, **kw: {'message': {'content': '{"additional_text":""}'}})
    monkeypatch.setattr(delivery, 'get_llm_backend', lambda cfg: backend)
    monkeypatch.setattr(timer_delivery, 'get_llm_backend', lambda cfg: backend)
    assert engine.run_reply_engine(db, mock_config, None, 'Start then cancel a five minute timer.', dialogue_memory) == draft
    assert effects == actions


def test_timer_only_request_rejects_a_drafts_invented_extra_goal(monkeypatch, mock_config):
    responses = iter([{'other_requests': []}, {'additional_text': 'Your tea timer is up.'}])
    monkeypatch.setattr(timer_delivery, 'get_llm_backend', lambda cfg: SimpleNamespace(chat=lambda *a, **kw: {
        'message': {'content': json.dumps(next(responses))}}))
    records = [{'id': 'private', 'label': 'Tea', 'duration_sec': 300, 'state': 'running'}]
    result = timer_delivery.timer_reply_source(mock_config, 'Start a tea timer.', 'Your tea timer is up.', records,
        time.monotonic() + mock_config.llm_chat_timeout_sec)
    assert result == timer_delivery.timer_confirmation(records)


def test_mixed_request_copies_only_selected_original_answer_spans(monkeypatch, mock_config):
    responses = iter([{'other_requests': ['tell me the capital of France']}, {'verbatim_spans': ['Paris is the capital of France.']}])
    monkeypatch.setattr(timer_delivery, 'get_llm_backend', lambda cfg: SimpleNamespace(chat=lambda *a, **kw: {
        'message': {'content': json.dumps(next(responses))}}))
    records = [{'id': 'private', 'label': 'Tea', 'duration_sec': 300, 'state': 'running'}]
    result = timer_delivery.timer_reply_source(mock_config, 'Start a tea timer and tell me the capital of France.',
        'Your tea timer is up. Paris is the capital of France.', records, time.monotonic() + mock_config.llm_chat_timeout_sec)
    assert result == timer_delivery.timer_confirmation(records) + '\nParis is the capital of France.'


def test_mixed_answer_sharing_a_sentence_cannot_reintroduce_false_expiry(monkeypatch, mock_config):
    responses = iter([{'other_requests': ['tell me the capital of France']},
                      {'verbatim_spans': ['Paris is the capital of France.']}])
    monkeypatch.setattr(timer_delivery, 'get_llm_backend', lambda cfg: SimpleNamespace(chat=lambda *a, **kw: {
        'message': {'content': json.dumps(next(responses))}}))
    records = [{'id': 'private', 'label': 'Tea', 'duration_sec': 300, 'state': 'running'}]
    result = timer_delivery.timer_reply_source(mock_config, 'Start a tea timer and tell me the capital of France.',
        'Your tea timer is up; Paris is the capital of France.', records, time.monotonic() + mock_config.llm_chat_timeout_sec)
    assert result == timer_delivery.timer_confirmation(records) + '\nParis is the capital of France.'


def test_cancel_during_goal_extraction_prevents_another_inference(monkeypatch, mock_config):
    from threading import Event
    cancelled = Event()
    inference_after_cancellation = Event()
    def backend(model, messages, **kw):
        if cancelled.is_set():
            inference_after_cancellation.set()
            return None
        cancelled.set()
        return {'message': {'content': '{"other_requests":["tell me the capital of France"]}'}}
    monkeypatch.setattr(timer_delivery, 'get_llm_backend', lambda cfg: SimpleNamespace(chat=backend))
    records = [{'id': 'private', 'label': 'Tea', 'duration_sec': 300, 'state': 'running'}]
    result = timer_delivery.timer_reply_source(mock_config, 'Start a tea timer and tell me the capital of France.',
        'Paris is the capital of France.', records, time.monotonic() + mock_config.llm_chat_timeout_sec, cancelled)
    assert not inference_after_cancellation.is_set()
    assert result == timer_delivery.timer_confirmation(records)
