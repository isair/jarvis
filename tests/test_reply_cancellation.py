"""Cancellation prevents a completed model request from starting a new action."""
from threading import Event

import pytest

from jarvis.reply import engine
from jarvis.tools.types import ToolExecutionResult

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures('no_network')]


@pytest.mark.parametrize('model,planned,empty_plan,unsupported', [
    ('gpt-oss:20b', False, False, False),
    ('gemma4:e2b', False, False, False),
    ('gemma4:e2b', True, False, False),
    ('gemma4:e2b', True, True, False),
    ('gpt-oss:20b', False, False, True),
], ids=['native-tool', 'text-tool', 'planned-tool', 'unresolved-plan', 'protocol-fallback'])
def test_cancellation_during_inference_prevents_the_requested_tool(
    monkeypatch, mock_config, db, dialogue_memory, tmp_path, model, planned, empty_plan, unsupported,
):
    cancelled = Event()
    effect = tmp_path/'executed'
    mock_config.llm_chat_model = model
    mock_config.planner_enabled = planned
    mock_config.memory_digest_enabled = False
    mock_config.tool_result_digest_enabled = False
    mock_config.tts_engine = 'none'
    monkeypatch.setattr(engine, 'extract_search_params_for_memory', lambda *a, **kw: {})
    monkeypatch.setattr(engine, 'select_tools', lambda *a, **kw: ['getWeather'])
    monkeypatch.setattr(engine, 'plan_query', lambda *a, **kw: ['getWeather'] if planned else [])

    def resolve(*args, **kwargs):
        cancelled.set()
        return None if empty_plan else ('getWeather', {'location': 'London'})

    def chat(*args, **kwargs):
        if cancelled.is_set():
            effect.write_text('A new inference request started after cancellation')
        cancelled.set()
        if unsupported:
            raise engine.ToolsNotSupportedError('Native tools unavailable')
        return {'message': {'role': 'assistant', 'content':
            'tool_calls: [{"name":"getWeather","arguments":{"location":"London"}}]',
            'tool_calls': [{'id': 'weather', 'function': {
                'name': 'getWeather', 'arguments': {'location': 'London'},
            }}]}}

    def run_tool(*args, **kwargs):
        effect.write_text('The tool performed its action')
        return ToolExecutionResult(success=True, reply_text='A result')

    monkeypatch.setattr(engine, '_resolve_plan_step', resolve)
    monkeypatch.setattr(engine, 'chat_with_messages', chat)
    monkeypatch.setattr(engine, 'run_tool_with_retries', run_tool)
    result = engine.run_reply_engine(db, mock_config, None, 'Check the forecast.',
                                     dialogue_memory, cancel_event=cancelled)
    assert result is None
    assert not effect.exists()
    assert not dialogue_memory.has_recent_messages()
