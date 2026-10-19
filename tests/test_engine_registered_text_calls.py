"""Registered MCP names execute through simplified text calls after discovery."""

from unittest.mock import patch

import pytest

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("call_format", ["colon", "function"])
def test_discovered_mcp_name_executes_as_a_text_call(
    mock_config, db, dialogue_memory, monkeypatch, call_format
):
    from jarvis.reply import engine
    from jarvis.tools import registry
    from jarvis.tools.registry import ToolSpec
    from jarvis.tools.types import ToolExecutionResult

    registered_name = "local-browser__navigate_page"
    target = "https://example.org"
    mock_config.llm_chat_model = "gemma4:e2b"
    mock_config.planner_enabled = False
    mock_config.mcps = {"local-browser": {}}
    catalogue = {registered_name: ToolSpec(
        name=registered_name,
        description="Navigate the local browser to a URL.",
        inputSchema={"type": "object", "properties": {"url": {"type": "string"}},
                     "required": ["url"]},
    )}
    monkeypatch.setattr(registry, "get_cached_mcp_tools", lambda: catalogue)
    monkeypatch.setattr(registry, "is_mcp_cache_initialized", lambda: False)
    content = (
        f"{registered_name}: url: {target}"
        if call_format == "colon"
        else f'{registered_name}({{"url": "{target}"}})'
    )
    responses = iter([
        {"message": {"role": "assistant", "content": "", "tool_calls": [{
            "function": {"name": "toolSearchTool", "arguments": {"query": "open browser"}}
        }]}},
        {"message": {"role": "assistant", "content": content}},
        {"message": {"role": "assistant", "content": "The page is open."}},
    ])
    executed = []

    def run_tool(db, cfg, tool_name, tool_args, **kwargs):
        executed.append((tool_name, tool_args))
        result = (
            f"{registered_name}: Navigate the local browser."
            if tool_name == "toolSearchTool" else "The page is open."
        )
        return ToolExecutionResult(success=True, reply_text=result)

    with patch.object(engine, "select_tools", return_value=["webSearch", "stop"]), \
         patch.object(engine, "plan_query", return_value=[]), \
         patch.object(engine, "chat_with_messages", side_effect=lambda *a, **k: next(responses)), \
         patch.object(engine, "run_tool_with_retries", side_effect=run_tool), \
         patch.object(engine, "extract_search_params_for_memory", return_value={"keywords": []}):
        reply = engine.run_reply_engine(
            db=db, cfg=mock_config, tts=None, text="Open the requested page.",
            dialogue_memory=dialogue_memory,
        )

    assert executed == [
        ("toolSearchTool", {"query": "open browser"}),
        (registered_name, {"url": target}),
    ]
    assert reply == "The page is open."
