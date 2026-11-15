"""Recorded countdown facts ground final delivery independently of a flawed draft."""
import json
from types import SimpleNamespace

import pytest

from jarvis.reply import delivery, engine
from jarvis.tools.types import ToolExecutionResult

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures('no_network')]


@pytest.mark.parametrize('spoken', [False, True])
def test_recorded_creation_corrects_false_expiry_before_publication(monkeypatch, mock_config, db, dialogue_memory, spoken):
    monkeypatch.setattr(engine, 'render_written_reply', delivery.render_written_reply)
    monkeypatch.setattr(engine, 'render_english_speech', delivery.render_english_speech)
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
    corrected = 'A five-minute timer for Tea has been started.'
    def render(model, system, content, **kw):
        packet = json.loads(content)
        if 'current_request' in packet:
            return 'en'
        assert packet['timer_facts'] == {'created_count': 1, 'created_timers': [committed]}
        assert 'Your tea timer is up.' in packet['text_to_translate']
        return corrected
    monkeypatch.setattr(delivery, 'get_llm_backend', lambda cfg: SimpleNamespace(direct=render))
    result = engine.run_reply_engine(db, mock_config, None, 'Start a five-minute tea timer.', dialogue_memory, spoken_output=spoken)
    assert result == corrected
    assert len(effects) == 1
    assert any(row['role'] == 'assistant' and row['content'] == corrected for row in dialogue_memory.get_recent_messages())


@pytest.mark.parametrize('spoken', [False, True])
def test_unavailable_rendering_keeps_a_truthful_creation_acknowledgement(monkeypatch, mock_config, spoken):
    monkeypatch.setattr(delivery, 'get_llm_backend', lambda cfg: SimpleNamespace(direct=lambda *a, **kw: None))
    facts = {'created_count': 1, 'created_timers': [{'id': 'private-id', 'label': 'Tea', 'duration_sec': 300, 'state': 'running'}]}
    draft = 'Your tea timer is up.'
    result = (delivery.render_english_speech(mock_config, draft, timer_facts=facts) if spoken else
              delivery.render_written_reply(mock_config, 'Start a tea timer.', draft, timer_facts=facts))
    assert 'Started 1 timer' in result
    assert '300 seconds' in result and 'Tea' in result
    assert draft not in result and 'private-id' not in result


def test_creation_records_do_not_override_a_later_cancellation(monkeypatch, mock_config, db, dialogue_memory):
    monkeypatch.setattr(engine, 'render_written_reply', delivery.render_written_reply)
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
    monkeypatch.setattr(delivery, 'get_llm_backend', lambda cfg: SimpleNamespace(direct=render))
    assert engine.run_reply_engine(db, mock_config, None, 'Start then cancel a five minute timer.', dialogue_memory) == draft
    assert effects == actions
