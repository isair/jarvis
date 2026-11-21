"""A real MCP fault reaches the reply loop without an invented successful action."""
from unittest.mock import patch
import sys

if sys.version_info < (3, 11):
    from exceptiongroup import ExceptionGroup

import pytest
from evals.helpers import voice_config
from evals.tool_routing import requires_judge_llm
from jarvis.reply import engine
from jarvis.tools import registry
from jarvis.tools.types import ToolExecutionResult

pytestmark = [pytest.mark.eval, requires_judge_llm]


@pytest.mark.parametrize('query', ['Open https://example.com in the browser.',
                                  'Tarayıcıda https://example.com adresini aç.'])
def test_mcp_failure_produces_an_honest_action_reply(eval_db, eval_dialogue_memory, configure_mcp_tools, query):
    cfg = voice_config()
    cfg.location_enabled = False
    # Local CPU models need longer to process the full reply context.
    cfg.llm_chat_timeout_sec = cfg.planner_timeout_sec = 180
    catalogue = configure_mcp_tools(registry.ToolSpec(
        name='chrome-devtools__navigate_page', description='Navigate the browser to a supplied URL.',
        inputSchema={'type': 'object', 'properties': {'url': {'type': 'string'}}, 'required': ['url']},
    ))
    cfg.mcps = {'chrome-devtools': {'command': 'local-browser-fixture'}}
    attempts = []
    class Client:
        def __init__(self, config):
            pass
        def invoke_tool(self, **kwargs):
            attempts.append(kwargs)
            raise ExceptionGroup('outer transport', [ExceptionGroup('inner task group', [FileNotFoundError('Browser executable unavailable')])])
    with patch.object(registry, 'MCPClient', Client), \
         patch.object(engine, 'select_tools', return_value=[*catalogue, 'stop']):
        reply = engine.run_reply_engine(eval_db, cfg, None, query, eval_dialogue_memory)
    assert not attempts, 'A consequential MCP action must not reach the tool before confirmation'
    assert reply
    assert 'CONFIRM ' in reply
    assert 'CANCEL ' in reply


def test_confirmed_mcp_result_is_reported_as_natural_language(monkeypatch):
    cfg = voice_config()
    cfg.llm_chat_timeout_sec = 180
    result = ToolExecutionResult(
        success=True,
        reply_text='{"message":"Email sent to Priya"}',
        result_metadata={
            "confirmed_mcp_action": (
                "gmail/send_email with arguments "
                '{"message":"Hello","recipient":"Priya"}'
            )
        },
    )
    monkeypatch.setattr(
        engine,
        "handle_mcp_confirmation",
        lambda db, text, cancel_event=None: (True, result),
    )

    reply = engine.run_reply_engine(
        None, cfg, None, "CONFIRM 12345678", dialogue_memory=None, language="en",
    )

    assert reply
    assert "{" not in reply
    assert "sent" in reply.lower()
