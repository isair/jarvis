"""Request multiplicity permits intended repeats and bounds additional admissions."""
import pytest

from jarvis.reply import engine
from jarvis.reply.timer_delivery import TimerPreparation, TimerRequest
from jarvis.tools.types import ToolExecutionResult

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures('no_network')]


@pytest.mark.parametrize('requested', [0, 1, 2])
@pytest.mark.parametrize('second_label', ['tea', 'soup'])
def test_current_request_count_owns_sequential_creation_admission(
        monkeypatch, mock_config, db, dialogue_memory, requested, second_label):
    mock_config.planner_enabled = False
    mock_config.llm_chat_model = mock_config.ollama_chat_model = 'gpt-oss:20b'
    monkeypatch.setattr(engine, 'prepare_timer_arguments', lambda cfg, request, args, **kw: TimerPreparation(dict(args), TimerRequest(tuple({} for _ in range(requested)))))
    monkeypatch.setattr(engine, 'select_tools', lambda **kw: ['timer', 'stop'])
    monkeypatch.setattr(engine, 'extract_search_params_for_memory', lambda **kw: {})
    def call(label, identity):
        return {'message': {'tool_calls': [{'id': identity, 'function': {'name': 'timer', 'arguments': {
            'action': 'set', 'minutes': 5, 'label': label, 'expiry_announcement': 'Timer is up',
        }}}]}}
    expected_reply = 'Recorded countdowns started.' if requested else 'No additional countdown was admitted.'
    responses = iter([call('tea', 'one'), call(second_label, 'two'), {'message': {'content': expected_reply}}])
    monkeypatch.setattr(engine, 'chat_with_messages', lambda **kw: next(responses))
    effects = []
    def execute(**kw):
        effects.append(kw['tool_args'])
        return ToolExecutionResult(True, 'Countdown started.', result_metadata={'created_timer': {
            'id': str(len(effects)), 'label': kw['tool_args']['label'], 'duration_sec': 300, 'state': 'running',
        }})
    monkeypatch.setattr(engine, 'run_tool_with_retries', execute)
    assert engine.run_reply_engine(db, mock_config, None, f'Start {requested} countdowns.', dialogue_memory) == expected_reply
    assert len(effects) == requested


def test_premature_confirmation_cannot_replace_a_remaining_requested_creation(monkeypatch, mock_config, db, dialogue_memory):
    mock_config.planner_enabled = False
    monkeypatch.setattr(engine, 'prepare_timer_arguments', lambda cfg, request, args, **kw: TimerPreparation(dict(args), TimerRequest(({}, {}))))
    monkeypatch.setattr(engine, 'select_tools', lambda **kw: ['timer', 'stop'])
    monkeypatch.setattr(engine, 'extract_search_params_for_memory', lambda **kw: {})
    call = {'message': {'tool_calls': [{'function': {'name': 'timer', 'arguments': {
        'action': 'set', 'minutes': 5, 'label': 'tea', 'expiry_announcement': 'Timer is up',
    }}}]}}
    responses = iter([call, {'message': {'content': 'Both timers are set.'}}, call, {'message': {'content': 'Two timers started.'}}])
    monkeypatch.setattr(engine, 'chat_with_messages', lambda **kw: next(responses))
    effects = []
    def execute(**kw):
        effects.append('started')
        return ToolExecutionResult(True, 'Countdown started.', result_metadata={'created_timer': {
            'id': str(len(effects)), 'label': 'tea', 'duration_sec': 300, 'state': 'running',
        }})
    monkeypatch.setattr(engine, 'run_tool_with_retries', execute)
    assert engine.run_reply_engine(db, mock_config, None, 'Start two tea timers.', dialogue_memory) == 'Two timers started.'
    assert effects == ['started', 'started']
    assert not any(row['role'] == 'assistant' and row['content'] == 'Both timers are set.' for row in dialogue_memory.get_recent_messages())
