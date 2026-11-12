"""Unsuccessful tool outcomes expose bounded diagnostics in every dispatch path."""
import json
from unittest.mock import patch

import pytest

from jarvis.reply import engine
from jarvis.tools.types import ToolExecutionResult

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('path', ['planner', 'text', 'native'])
@pytest.mark.parametrize('reply_text,error_message,reason,success', [
    ('Unable to finish the request.', 'Permission denied.', 'Permission denied.', False),
    ('Permission denied.', None, 'Permission denied.', False),
    (None, 'Permission denied.', 'Permission denied.', False),
    ('Unable to finish.', 'Permission denied.\n' + 'Details ' * 80, 'Permission denied.', False),
    ('Read the report.', 'Stale diagnostic metadata.', None, True),
])
def test_diagnostics_follow_outcome_across_dispatch_paths(
    mock_config, db, dialogue_memory, capsys, path, reply_text, error_message, reason, success,
):
    mock_config.llm_chat_model = 'gpt-oss:20b' if path == 'native' else 'gemma4:e2b'
    call = {'function': {'name': 'localFiles', 'arguments': {'operation': 'read', 'path': 'report.txt'}}}
    first = {'message': {'content': 'tool_calls: ' + json.dumps([call])}} if path == 'text' else {
        'message': {'content': '', 'tool_calls': [call]},
    }
    final = {'message': {'content': 'Read access is unavailable.'}}
    received_results = []
    responses = iter([final] if path == 'planner' else [first, final])
    outcome = ToolExecutionResult(success=success, reply_text=reply_text, error_message=error_message)
    plan = ["localFiles operation='read' path='report.txt'", 'Reply to the user.']

    def chat(**kwargs):
        received_results.extend(
            message['content'] for message in kwargs['messages'] if message.get('tool_name')
        )
        return next(responses)

    with patch.object(engine, 'run_tool_with_retries', return_value=outcome), \
            patch.object(engine, 'select_tools', return_value=['localFiles', 'stop']), \
            patch.object(engine, 'plan_query', return_value=plan), \
            patch.object(engine, 'extract_search_params_for_memory', return_value={'keywords': []}), \
            patch.object(engine, 'chat_with_messages', side_effect=chat):
        if path != 'planner':
            with patch.object(engine, '_resolve_plan_step', return_value=None):
                reply = engine.run_reply_engine(db, mock_config, None, 'Read report.txt.', dialogue_memory)
        else:
            reply = engine.run_reply_engine(db, mock_config, None, 'Read report.txt.', dialogue_memory)

    lines = [line for line in capsys.readouterr().out.splitlines() if '❌ localFiles error:' in line]
    if success:
        assert not lines
    else:
        assert len(lines) == 1
        assert reason in lines[0]
        assert len(lines[0].split('error: ', 1)[1]) <= engine._TOOL_ERROR_PREVIEW_CHAR_LIMIT
    assert any((reply_text or error_message) in text for text in received_results)
    assert reply == 'Read access is unavailable.'
