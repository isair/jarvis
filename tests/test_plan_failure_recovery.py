"""An unsuccessful prerequisite cannot blindly advance a concrete plan."""
from unittest.mock import patch

import pytest

from jarvis.reply import engine
from jarvis.tools.types import ToolExecutionResult

pytestmark = pytest.mark.unit
PLAN = [
    "localFiles operation='read' path='report.txt'",
    "localFiles operation='delete' path='report.txt'",
    'Reply to the user.',
]
QUERY = 'Read report.txt, then only if you read it successfully delete it.'


@pytest.mark.parametrize('with_reply', [True, False])
def test_failed_read_does_not_execute_planned_delete(mock_config, db, dialogue_memory, with_reply):
    operations = []
    received_results = []
    def run_tool(db, cfg, tool_name, tool_args, **kwargs):
        operations.append(tool_args['operation'])
        if tool_args['operation'] == 'read':
            return ToolExecutionResult(success=False,
                reply_text='Read permission is unavailable.' if with_reply else None,
                error_message=None if with_reply else 'Read permission denied.')
        return ToolExecutionResult(success=True, reply_text='Deleted report.txt.')
    def chat(*args, **kwargs):
        received_results.extend(m['content'] for m in kwargs['messages'] if m.get('tool_name'))
        assert 'ACTION PLAN' not in kwargs['messages'][0]['content']
        return {'message': {'role': 'assistant', 'content': 'Please check read access. The file was not deleted.'}}
    with patch.object(engine, 'run_tool_with_retries', side_effect=run_tool), \
            patch.object(engine, 'select_tools', return_value=['localFiles', 'stop']), \
            patch.object(engine, 'plan_query', return_value=PLAN), \
            patch.object(engine, 'extract_search_params_for_memory', return_value={'keywords': []}), \
            patch.object(engine, 'chat_with_messages', side_effect=chat):
        reply = engine.run_reply_engine(db, mock_config, None, QUERY, dialogue_memory)
    assert operations == ['read'], 'A failed prerequisite must be inspected before a later action'
    assert 'not deleted' in reply
    assert all('all tool steps executed' not in text for text in received_results)


@pytest.mark.parametrize("previous_failure", [False, True])
def test_successful_prerequisite_keeps_direct_plan_execution(mock_config, db, dialogue_memory, previous_failure):
    operations = []
    if previous_failure:
        dialogue_memory.add_message('user', 'An earlier unrelated request')
        dialogue_memory.record_tool_turn([{'role': 'user', 'content': '[Tool error] earlier failure',
                                          'tool_name': 'localFiles', 'tool_failed': True}])
        dialogue_memory.add_message('assistant', 'That earlier task failed.')
    def run_tool(db, cfg, tool_name, tool_args, **kwargs):
        operations.append(tool_args['operation'])
        return ToolExecutionResult(success=True, reply_text='The file was read.' if tool_args['operation'] == 'read' else 'Deleted report.txt.')
    with patch.object(engine, 'run_tool_with_retries', side_effect=run_tool), \
            patch.object(engine, 'select_tools', return_value=['localFiles', 'stop']), \
            patch.object(engine, 'plan_query', return_value=PLAN), \
            patch.object(engine, 'extract_search_params_for_memory', return_value={'keywords': []}), \
            patch.object(engine, 'chat_with_messages', return_value={'message': {'content': 'The report was read and deleted.'}}):
        engine.run_reply_engine(db, mock_config, None, QUERY, dialogue_memory)
    assert operations == ['read', 'delete']


def test_model_can_recover_using_corrected_arguments(mock_config, db, dialogue_memory):
    import json
    operations = []
    def run_tool(db, cfg, tool_name, tool_args, **kwargs):
        operation, path = tool_args['operation'], tool_args['path']
        operations.append((operation, path))
        if operation == 'read' and path == 'report.txt':
            return ToolExecutionResult(success=False, reply_text='Use ./report.txt for this fixture.')
        return ToolExecutionResult(success=True, reply_text='Read the report.' if operation == 'read' else 'Deleted the report.')
    responses = iter([
        {'message': {'content': 'tool_calls: ' + json.dumps([{'function': {'name': 'localFiles', 'arguments': {'operation': 'read', 'path': './report.txt'}}}])}},
        {'message': {'content': 'The report contents were read.'}},
    ])
    with patch.object(engine, 'run_tool_with_retries', side_effect=run_tool), \
            patch.object(engine, 'select_tools', return_value=['localFiles', 'stop']), \
            patch.object(engine, 'plan_query', return_value=[PLAN[0], PLAN[-1]]), \
            patch.object(engine, 'extract_search_params_for_memory', return_value={'keywords': []}), \
            patch.object(engine, 'chat_with_messages', side_effect=lambda **kwargs: next(responses)):
        reply = engine.run_reply_engine(db, mock_config, None, 'Read report.txt and tell me its contents.', dialogue_memory)
    assert operations == [('read', 'report.txt'), ('read', './report.txt')]
    assert 'contents were read' in reply


@pytest.mark.parametrize('model', ['gemma4:e2b', 'gpt-oss:20b'])
@pytest.mark.parametrize('with_reply', [True, False])
def test_model_issued_failure_reassesses_tasks_in_both_protocols(mock_config, db, dialogue_memory, model, with_reply):
    mock_config.llm_chat_model = model
    operations = []
    responses = iter([
        {'message': {'content': '', 'tool_calls': [{'function': {'name': 'localFiles', 'arguments': {'operation': 'read', 'path': 'report.txt'}}}]}},
        {'message': {'content': 'Read access is unavailable, so the file remains intact.'}},
    ])
    def run_tool(db, cfg, tool_name, tool_args, **kwargs):
        operations.append(tool_args['operation'])
        return ToolExecutionResult(success=False, reply_text='Cannot read the file.' if with_reply else None,
                                   error_message=None if with_reply else 'Permission denied.')
    def chat(**kwargs):
        if operations:
            assert 'ACTION PLAN' not in kwargs['messages'][0]['content']
            result_messages = [m for m in kwargs['messages'] if m.get('tool_name')]
            assert all('all tool steps executed' not in m['content'] for m in result_messages)
        return next(responses)
    with patch.object(engine, 'run_tool_with_retries', side_effect=run_tool), \
            patch.object(engine, 'select_tools', return_value=['localFiles', 'stop']), \
            patch.object(engine, 'plan_query', return_value=PLAN), \
            patch.object(engine, '_resolve_plan_step', return_value=None), \
            patch.object(engine, 'extract_search_params_for_memory', return_value={'keywords': []}), \
            patch.object(engine, 'chat_with_messages', side_effect=chat):
        reply = engine.run_reply_engine(db, mock_config, None, QUERY, dialogue_memory)
    assert operations == ['read']
    assert 'intact' in reply
