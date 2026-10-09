"""Live replies use preserved arguments from recoverable tool-call envelopes."""
from unittest.mock import patch

import pytest

from evals.conftest import requires_judge_llm
from evals.helpers import voice_config, call_judge_llm
from jarvis.reply import engine
from jarvis.tools.types import ToolExecutionResult

pytestmark = [pytest.mark.eval, requires_judge_llm]


@pytest.mark.parametrize('query', [
    'Read report.txt and tell me its contents.',
    'report.txt dosyasını oku ve içeriğini söyle.',
])
def test_recoverable_read_preserves_arguments_without_empty_defaults(eval_db, eval_dialogue_memory, query):
    cfg = voice_config()
    cfg.location_enabled = False
    executed = []
    def fixture_tool(db, cfg, tool_name, tool_args, **kwargs):
        if tool_name != 'localFiles':
            return ToolExecutionResult(success=True, reply_text='No additional tools found.')
        executed.append(tool_args)
        if tool_name == 'localFiles' and tool_args.get('operation') == 'read' and tool_args.get('path') == 'report.txt':
            return ToolExecutionResult(success=True, reply_text='report.txt contents: Project review is on Friday at 10:00.')
        return ToolExecutionResult(success=False, reply_text='The requested report was not read.')
    live_chat = engine.chat_with_messages
    first = True
    def chat(**kwargs):
        nonlocal first
        if first:
            first = False
            return {'message': {'content': 'tool_calls: [{"function": {"name": "localFiles", "arguments": "{\\"operation\\": \\"read\\", \\"path\\": \\"report.txt\\"}}}]'}}
        return live_chat(**kwargs)
    with patch.object(engine, 'run_tool_with_retries', side_effect=fixture_tool), \
            patch.object(engine, 'chat_with_messages', side_effect=chat), \
            patch.object(engine, 'select_tools', return_value=['localFiles', 'stop']), \
            patch.object(engine, 'plan_query', return_value=[]), \
            patch.object(engine, 'extract_search_params_for_memory', return_value={'keywords': []}):
        reply = engine.run_reply_engine(eval_db, cfg, None, query, eval_dialogue_memory)
    assert executed and all(args.get('operation') == 'read' and args.get('path') == 'report.txt' for args in executed), (executed, reply)
    verdict = call_judge_llm('Output PASS or FAIL only. PASS requires reporting that the project review is on Friday at 10:00 using the actual report contents, in any language.',
                             f'Query: {query}\nActual contents: Project review is on Friday at 10:00.\nReply: {reply}')
    assert verdict and verdict.strip().upper() == 'PASS', (reply, verdict)
