"""Timer admission localises future speech without changing the countdown action."""
import json
from threading import Event
from types import SimpleNamespace

import pytest

from jarvis.reply import delivery, engine, timer_delivery
from jarvis.tools.types import ToolExecutionResult

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures('no_network')]


@pytest.mark.parametrize('planned', [False, True])
@pytest.mark.parametrize('cancel_during_translation', [False, True])
def test_expiry_speech_is_localised_before_one_action_or_cancellation(
        monkeypatch, mock_config, db, dialogue_memory, planned, cancel_during_translation):
    monkeypatch.setattr(engine, 'prepare_timer_arguments', timer_delivery.prepare_timer_arguments)
    mock_config.llm_chat_model = mock_config.ollama_chat_model = 'gemma4:e2b'
    mock_config.planner_enabled = planned
    cancelled = Event()
    query = 'Stelle einen Timer für fünf Minuten für Tee.'
    args = {'action': 'set', 'minutes': 5, 'label': 'Tee', 'expiry_announcement': 'Your tea timer is up'}
    translated = 'Dein Tee-Timer ist abgelaufen.'
    monkeypatch.setattr(engine, 'select_tools', lambda **kw: ['timer', 'stop'])
    monkeypatch.setattr(engine, 'extract_search_params_for_memory', lambda **kw: {})
    monkeypatch.setattr(engine, 'plan_query', lambda **kw: ["timer action='set' minutes=5", 'Reply to the user.'] if planned else [])
    monkeypatch.setattr(engine, '_resolve_plan_step', lambda **kw: ('timer', dict(args)))
    responses = iter(([{'message': {'tool_calls': [{'id': 'owned', 'function': {'name': 'timer', 'arguments': dict(args)}}]}}] if not planned else []) + [{'message': {'content': 'Der Timer läuft.'}}])
    monkeypatch.setattr(engine, 'chat_with_messages', lambda **kw: next(responses))
    def render(model, system, content, **kw):
        assert 'into de.' in system
        assert json.loads(content) == {'text_to_translate': args['expiry_announcement']}
        assert query not in content
        if cancel_during_translation:
            cancelled.set()
        return translated
    def bind(*a, **kw):
        return {'message': {'content': json.dumps({'requested_count': 1, 'label': 'Tee', 'hours': 0, 'minutes': 5, 'seconds': 0})}}
    backend = SimpleNamespace(direct=render, chat=bind)
    monkeypatch.setattr(delivery, 'get_llm_backend', lambda cfg: backend)
    monkeypatch.setattr(timer_delivery, 'get_llm_backend', lambda cfg: backend)
    effects = []
    def execute(**kw):
        effects.append(kw['tool_args'])
        return ToolExecutionResult(True, 'Countdown state: running\nElapsed: false', result_metadata={'created_timer': {
            'id': 'owned', 'label': 'Tee', 'duration_sec': 300, 'state': 'running',
        }})
    monkeypatch.setattr(engine, 'run_tool_with_retries', execute)
    result = engine.run_reply_engine(db, mock_config, None, query, dialogue_memory, language='de', cancel_event=cancelled)
    assert effects == ([] if cancel_during_translation else [{**args, 'expiry_announcement': translated}])
    assert result == (None if cancel_during_translation else 'Der Timer läuft.')
    assert args['expiry_announcement'] == 'Your tea timer is up'


def test_text_expiry_uses_request_prose_not_a_written_language_override(monkeypatch, mock_config):
    query = 'Stelle einen Timer für Tee. Antworte auf Englisch.'
    packets = []
    def render(model, system, content, **kw):
        packet = json.loads(content)
        packets.append(packet)
        if 'current_request' in packet:
            assert 'actual prose' in system
            return 'de'
        return 'Dein Tee-Timer ist abgelaufen.'
    monkeypatch.setattr(delivery, 'get_llm_backend', lambda cfg: SimpleNamespace(direct=render))
    assert delivery.render_timer_announcement(mock_config, query, 'Your tea timer is up') == 'Dein Tee-Timer ist abgelaufen.'
    assert packets == [{'current_request': query}, {'text_to_translate': 'Your tea timer is up'}]


@pytest.mark.parametrize('language', ['de', None])
def test_oversized_expiry_text_never_reaches_a_model(monkeypatch, mock_config, language):
    monkeypatch.setattr(delivery, 'get_llm_backend', lambda cfg: pytest.fail('Oversized expiry packet reached a model'))
    text = '\x01' * (delivery._MAX_RENDER_INPUT_BYTES // 6 + 1)
    assert delivery.render_timer_announcement(mock_config, 'Stelle einen Timer.', text, language=language) == text


@pytest.mark.parametrize('tool,args', [('timer', {'action': 'list'}), ('timer', {'action': 'cancel', 'label': 'tea'}), ('getWeather', {'location': 'London'})])
def test_non_creation_actions_do_not_prepare_expiry(monkeypatch, mock_config, db, dialogue_memory, tool, args):
    mock_config.planner_enabled = False
    monkeypatch.setattr(engine, 'prepare_timer_arguments', lambda *a, **kw: pytest.fail('Non-creation action prepared expiry speech'))
    monkeypatch.setattr(engine, 'select_tools', lambda **kw: [tool, 'stop'])
    monkeypatch.setattr(engine, 'extract_search_params_for_memory', lambda **kw: {})
    responses = iter([{'message': {'tool_calls': [{'id': 'owned', 'function': {'name': tool, 'arguments': args}}]}}, {'message': {'content': 'Recorded state.'}}])
    monkeypatch.setattr(engine, 'chat_with_messages', lambda **kw: next(responses))
    monkeypatch.setattr(engine, 'run_tool_with_retries', lambda **kw: ToolExecutionResult(True, 'Recorded state.'))
    assert engine.run_reply_engine(db, mock_config, None, 'Report the recorded state.', dialogue_memory) == 'Recorded state.'


def test_oversized_numeric_binding_keeps_original_arguments(monkeypatch, mock_config):
    arguments = {'action': 'set', 'minutes': 5, 'expiry_announcement': 'Timer is up'}
    binding = {'requested_count': 1, 'label': None, 'hours': 10 ** 310, 'minutes': 0, 'seconds': 0}
    monkeypatch.setattr(timer_delivery, 'get_llm_backend', lambda cfg: SimpleNamespace(chat=lambda *a, **kw: {
        'message': {'content': json.dumps(binding)}}))
    monkeypatch.setattr(timer_delivery, 'render_timer_announcement', lambda cfg, request, text, **kw: text)
    prepared = timer_delivery.prepare_timer_arguments(mock_config, 'Start a five-minute timer.', arguments)
    assert prepared.arguments == arguments
    assert prepared.requested_count is None
