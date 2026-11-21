"""Fallback guidance preserves request parts without inventing completion state."""
from copy import deepcopy
from unittest.mock import patch

import pytest

from jarvis.reply import engine
from jarvis.tools.types import ToolExecutionResult

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('results', [
    ['Tea removed; soup remains.'],
    ['Tea removal failed; soup remains.'],
    ['Tea removed.', 'Soup remains.'],
])
def test_result_feedback_does_not_assign_completion_by_call_count(
    mock_config, db, dialogue_memory, results,
):
    mock_config.llm_chat_model = 'gemma4:e2b'
    mock_config.tool_result_digest_enabled = False
    requests = []
    responses = iter([
        *[{'message': {'content': '', 'tool_calls': [{'function': {
            'name': 'getWeather', 'arguments': {'location': str(index)},
        }}]}} for index in range(len(results))],
        {'message': {'content': 'Request assessed against the results.'}},
    ])

    def chat(cfg, messages, **kwargs):
        requests.append(deepcopy(messages))
        return next(responses)

    with patch.object(engine, 'plan_query', return_value=[]), \
         patch.object(engine, 'select_tools', return_value=['getWeather', 'stop']), \
         patch.object(engine, 'extract_search_params_for_memory', return_value={}), \
         patch.object(engine, 'run_tool_with_retries', side_effect=[
             ToolExecutionResult('failed' not in result, result) for result in results
         ]), \
         patch.object(engine, 'chat_with_messages', side_effect=chat):
        engine.run_reply_engine(db, mock_config, None,
            'Delete only the tea reminder and keep the soup reminder.',
            dialogue_memory, language='en')

    for request in requests[1:]:
        result = next(message for message in reversed(request) if message.get('tool_name'))
        feedback = result['content']
        assert 'Delete only the tea reminder' in feedback
        assert 'keep the soup reminder.' in feedback
        assert 'You have answered' not in feedback
        assert 'Still unanswered:' not in feedback
        assert 'MUST emit another' not in feedback
    assert 'ONE tool call per sub-question' not in requests[0][0]['content']
