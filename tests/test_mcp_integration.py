"""
Integration tests for MCP tools in the reply engine.

These tests require more complex setup and may not run in basic CI environments.
They can be run locally or in development environments with git hooks.
"""

import pytest
from unittest.mock import Mock, patch


@pytest.mark.integration
def test_mcp_tools_integrated_with_reply_engine():
    """Test that MCP tools are properly integrated with the reply engine's tool discovery."""
    from jarvis.tools.registry import discover_mcp_tools, generate_tools_description, BUILTIN_TOOLS

    # Mock MCP client
    class FakeMCPClient:
        def __init__(self, config):
            pass

        def list_tools(self, server_name):
            if server_name == "test-server":
                return [
                    {"name": "tool1", "description": "Test tool 1"},
                    {"name": "tool2", "description": "Test tool 2"}
                ]
            return []

    with patch('jarvis.tools.registry.MCPClient', FakeMCPClient):
        # Test discovery
        mcps_config = {"test-server": {"command": "fake"}}
        mcp_tools, _errors = discover_mcp_tools(mcps_config)

        # Test tool registration (simulate what reply engine does)
        allowed_tools = list(BUILTIN_TOOLS.keys())
        for mcp_tool_name in mcp_tools.keys():
            if mcp_tool_name not in allowed_tools:
                allowed_tools.append(mcp_tool_name)

        # Test tool descriptions include MCP tools
        description = generate_tools_description(allowed_tools, mcp_tools)
        
        # Assertions
        assert "test-server__tool1" in allowed_tools
        assert "test-server__tool2" in allowed_tools
        assert "test-server__tool1" in description
        assert "test-server__tool2" in description


@pytest.mark.integration 
def test_mcp_tool_execution_in_context():
    """Test MCP tool execution with proper context and error handling."""
    from jarvis.tools.registry import run_tool_with_retries, ToolExecutionResult
    
    class MockDB:
        pass
    
    class MockConfig:
        def __init__(self):
            self.mcps = {
                "test-server": {
                    "command": "fake",
                    "read_only_tools": ["example_tool"],
                }
            }
            self.voice_debug = False
    
    # Mock successful execution
    class FakeMCPClient:
        def __init__(self, config):
            pass
            
        def invoke_tool(self, server_name, tool_name, arguments):
            return {"text": f"Executed {tool_name} on {server_name}", "isError": False}
    
    with patch('jarvis.tools.registry.MCPClient', FakeMCPClient):
        result = run_tool_with_retries(
            db=MockDB(),
            cfg=MockConfig(),
            tool_name="test-server__example_tool",
            tool_args={"param": "value"},
            system_prompt="test",
            original_prompt="test",
            redacted_text="test",
            max_retries=0
        )
    assert result.success is True
    assert "Executed example_tool on test-server" in result.reply_text


@pytest.mark.integration
def test_mcp_error_handling_in_context():
    """Test that MCP errors are properly handled in execution context."""
    from jarvis.tools.registry import run_tool_with_retries
    
    class MockDB:
        pass
    
    class MockConfig:
        def __init__(self):
            self.mcps = {
                "test-server": {
                    "command": "fake",
                    "read_only_tools": ["failing_tool"],
                }
            }
            self.voice_debug = False
    
    # Mock failing execution
    class FailingMCPClient:
        def __init__(self, config):
            pass
            
        def invoke_tool(self, server_name, tool_name, arguments):
            return {"text": "Tool failed", "isError": True}
    
    with patch('jarvis.tools.registry.MCPClient', FailingMCPClient):
        result = run_tool_with_retries(
            db=MockDB(),
            cfg=MockConfig(),
            tool_name="test-server__failing_tool",
            tool_args={},
            system_prompt="test",
            original_prompt="test", 
            redacted_text="test",
            max_retries=0
        )
    assert result.success is False
    assert result.error_message == "Tool failed"


@pytest.mark.integration
def test_mcp_exception_with_empty_message_still_yields_diagnosable_error():
    """A bare exception (e.g. ``concurrent.futures.TimeoutError()``) with an
    empty ``str(e)`` must not collapse the error message down to a bare
    trailing colon — callers need at least the exception type name to
    diagnose the failure from the log line alone.
    """
    from jarvis.tools.registry import run_tool_with_retries

    class MockDB:
        pass

    class MockConfig:
        def __init__(self):
            self.mcps = {
                "test-server": {
                    "command": "fake",
                    "read_only_tools": ["slow_tool"],
                }
            }
            self.voice_debug = False

    class TimingOutMCPClient:
        def __init__(self, config):
            pass

        def invoke_tool(self, server_name, tool_name, arguments):
            raise TimeoutError()  # empty str(e), as raised on a bare invoke timeout

    with patch('jarvis.tools.registry.MCPClient', TimingOutMCPClient):
        result = run_tool_with_retries(
            db=MockDB(),
            cfg=MockConfig(),
            tool_name="test-server__slow_tool",
            tool_args={},
            system_prompt="test",
            original_prompt="test",
            redacted_text="test",
            max_retries=0
        )
    assert result.success is False
    assert result.error_message.strip().endswith(":") is False
    assert "TimeoutError" in result.error_message


@pytest.mark.integration
def test_mcp_action_requires_explicit_confirmation_before_execution():
    """A discovered MCP tool without a local read-only allowlist must not run."""
    from jarvis.tools.registry import (
        run_tool_with_retries,
        clear_pending_mcp_actions,
    )

    class MockDB:
        pass

    class MockConfig:
        mcps = {"test-server": {"command": "fake"}}
        voice_debug = False

    class TrackingMCPClient:
        calls = 0

        def __init__(self, config):
            pass

        def invoke_tool(self, server_name, tool_name, arguments):
            TrackingMCPClient.calls += 1
            return {"text": "sent", "isError": False}

    clear_pending_mcp_actions()
    db = MockDB()
    with patch("jarvis.tools.registry.MCPClient", TrackingMCPClient):
        result = run_tool_with_retries(
            db=db,
            cfg=MockConfig(),
            tool_name="test-server__send_message",
            tool_args={"recipient": "Asha", "message": "Hello"},
            system_prompt="test",
            original_prompt="Send Hello to Asha",
            redacted_text="Send Hello to Asha",
            max_retries=0,
        )

    assert result.success is False
    assert result.confirmation_required
    assert "CONFIRM " in result.reply_text
    assert TrackingMCPClient.calls == 0
    clear_pending_mcp_actions()


@pytest.mark.integration
def test_mcp_confirmation_prompt_shows_all_action_arguments():
    from jarvis.tools.registry import (
        run_tool_with_retries,
        clear_pending_mcp_actions,
    )

    class MockDB:
        pass

    class MockConfig:
        mcps = {"test-server": {"command": "fake"}}
        voice_debug = False

    clear_pending_mcp_actions()
    result = run_tool_with_retries(
        db=MockDB(),
        cfg=MockConfig(),
        tool_name="test-server__send_message",
        tool_args={"message": "x" * 600, "recipient": "Asha"},
        system_prompt="test",
        original_prompt="Send a message",
        redacted_text="Send a message",
        max_retries=0,
    )

    assert result.confirmation_required
    assert "recipient" in result.reply_text
    assert "Asha" in result.reply_text
    assert ("x" * 600) in result.reply_text
    clear_pending_mcp_actions()


@pytest.mark.integration
def test_mcp_exact_confirmation_executes_pending_action_once():
    from jarvis.tools.registry import (
        run_tool_with_retries,
        handle_mcp_confirmation,
        clear_pending_mcp_actions,
    )

    class MockDB:
        pass

    class MockConfig:
        mcps = {"test-server": {"command": "fake"}}
        voice_debug = False

    class TrackingMCPClient:
        calls = []

        def __init__(self, config):
            pass

        def invoke_tool(self, server_name, tool_name, arguments):
            TrackingMCPClient.calls.append((server_name, tool_name, arguments))
            return {"text": "Message sent", "isError": False}

    clear_pending_mcp_actions()
    db = MockDB()
    with patch("jarvis.tools.registry.MCPClient", TrackingMCPClient):
        held = run_tool_with_retries(
            db=db,
            cfg=MockConfig(),
            tool_name="test-server__send_message",
            tool_args={"recipient": "Asha", "message": "Hello"},
            system_prompt="test",
            original_prompt="Send Hello to Asha",
            redacted_text="Send Hello to Asha",
            max_retries=0,
        )
        nonce = held.confirmation_required
        matched, executed = handle_mcp_confirmation(db, f"CONFIRM {nonce}")
        repeated = handle_mcp_confirmation(db, f"CONFIRM {nonce}")

    assert matched is True
    assert executed.success is True
    assert executed.reply_text == "Message sent"
    assert executed.result_metadata == {
        "confirmed_mcp_action": "test-server/send_message with arguments "
        '{"message": "Hello", "recipient": "Asha"}'
    }
    assert TrackingMCPClient.calls == [
        ("test-server", "send_message", {"recipient": "Asha", "message": "Hello"})
    ]
    assert repeated[1].success is False
    clear_pending_mcp_actions()


@pytest.mark.integration
def test_expired_mcp_confirmation_does_not_execute_action(monkeypatch):
    from jarvis.tools import registry

    class MockDB:
        pass

    class MockConfig:
        mcps = {"test-server": {"command": "fake"}}
        voice_debug = False

    class TrackingMCPClient:
        calls = 0

        def __init__(self, config):
            pass

        def invoke_tool(self, server_name, tool_name, arguments):
            TrackingMCPClient.calls += 1
            return {"text": "sent", "isError": False}

    registry.clear_pending_mcp_actions()
    db = MockDB()
    monkeypatch.setattr(registry.time, "monotonic", lambda: 100.0)
    with patch("jarvis.tools.registry.MCPClient", TrackingMCPClient):
        held = registry.run_tool_with_retries(
            db=db,
            cfg=MockConfig(),
            tool_name="test-server__send_message",
            tool_args={"recipient": "Asha", "message": "Hello"},
            system_prompt="test",
            original_prompt="Send Hello to Asha",
            redacted_text="Send Hello to Asha",
            max_retries=0,
        )
        monkeypatch.setattr(registry.time, "monotonic", lambda: 1000.0)
        matched, result = registry.handle_mcp_confirmation(
            db,
            f"CONFIRM {held.confirmation_required}",
        )

    assert matched is True
    assert result.success is False
    assert "expired" in result.reply_text
    assert TrackingMCPClient.calls == 0
    registry.clear_pending_mcp_actions()


@pytest.mark.integration
def test_unallowlisted_mcp_tool_requires_confirmation():
    from jarvis.tools.registry import (
        run_tool_with_retries,
        clear_pending_mcp_actions,
    )

    class MockDB:
        pass

    class MockConfig:
        mcps = {"test-server": {"command": "fake"}}
        voice_debug = False

    class TrackingMCPClient:
        calls = 0

        def __init__(self, config):
            pass

        def invoke_tool(self, server_name, tool_name, arguments):
            TrackingMCPClient.calls += 1
            return {"text": "results", "isError": False}

    clear_pending_mcp_actions()
    try:
        with patch("jarvis.tools.registry.MCPClient", TrackingMCPClient):
            result = run_tool_with_retries(
                db=MockDB(),
                cfg=MockConfig(),
                tool_name="test-server__search",
                tool_args={"query": "Asha"},
                system_prompt="test",
                original_prompt="Search for Asha",
                redacted_text="Search for Asha",
                max_retries=0,
            )
        assert result.success is False
        assert result.confirmation_required
        assert "CONFIRM " in result.reply_text
        assert TrackingMCPClient.calls == 0
    finally:
        clear_pending_mcp_actions()


@pytest.mark.integration
def test_mcp_locally_allowlisted_read_only_tool_runs_without_confirmation():
    from jarvis.tools.registry import (
        run_tool_with_retries,
        clear_pending_mcp_actions,
    )

    class MockDB:
        pass

    class MockConfig:
        mcps = {
            "test-server": {
                "command": "fake",
                "read_only_tools": ["search"],
            }
        }
        voice_debug = False

    class TrackingMCPClient:
        calls = 0

        def __init__(self, config):
            pass

        def invoke_tool(self, server_name, tool_name, arguments):
            TrackingMCPClient.calls += 1
            return {"text": "results", "isError": False}

    clear_pending_mcp_actions()
    try:
        with patch("jarvis.tools.registry.MCPClient", TrackingMCPClient):
            result = run_tool_with_retries(
                db=MockDB(),
                cfg=MockConfig(),
                tool_name="test-server__search",
                tool_args={"query": "Asha"},
                system_prompt="test",
                original_prompt="Search for Asha",
                redacted_text="Search for Asha",
                max_retries=0,
            )
        assert result.success is True
        assert result.confirmation_required is None
        assert TrackingMCPClient.calls == 1
    finally:
        clear_pending_mcp_actions()


@pytest.mark.integration
def test_reply_engine_intercepts_confirmation_before_planner(monkeypatch):
    from jarvis.reply import engine
    from jarvis.tools.types import ToolExecutionResult

    class MockDB:
        pass

    class MockConfig:
        pass

    monkeypatch.setattr(
        engine,
        "handle_mcp_confirmation",
        lambda db, text, cancel_event=None: (
            True,
            ToolExecutionResult(success=True, reply_text="Message sent"),
        ),
    )
    monkeypatch.setattr(
        engine,
        "chat_with_messages",
        lambda *args, **kwargs: pytest.fail("planner/model ran after MCP confirmation"),
    )

    result = engine.run_reply_engine(
        MockDB(),
        MockConfig(),
        None,
        "CONFIRM 12345678",
        dialogue_memory=None,
    )
    assert result == "Message sent"


@pytest.mark.integration
def test_confirmed_mcp_result_is_formatted_without_dispatching_more_tools(monkeypatch):
    from jarvis.reply import engine
    from jarvis.tools.types import ToolExecutionResult

    class MockDB:
        pass

    class MockConfig:
        wake_word = "Jarvis"
        llm_chat_timeout_sec = 2
        llm_thinking_enabled = False
        tts_engine = "piper"

    action_result = ToolExecutionResult(
        success=True,
        reply_text='{"message":"Email sent to Priya"}',
        result_metadata={
            "confirmed_mcp_action": (
                "gmail/send_email with arguments "
                '{"message":"Hello","recipient":"Priya"}'
            )
        },
    )
    captured_messages = []

    monkeypatch.setattr(
        engine,
        "handle_mcp_confirmation",
        lambda db, text, cancel_event=None: (True, action_result),
    )

    def format_reply(cfg, messages, *, timeout_sec, extra_options=None,
                     tools=None, thinking=False):
        captured_messages.extend(messages)
        assert tools is None
        return {"message": {"content": "The email was sent to Priya."}}

    monkeypatch.setattr(engine, "chat_with_messages", format_reply)
    monkeypatch.setattr(
        engine,
        "run_tool_with_retries",
        lambda **kwargs: pytest.fail("a confirmed action reply must not dispatch tools"),
    )

    result = engine.run_reply_engine(
        MockDB(), MockConfig(), None, "CONFIRM 12345678", dialogue_memory=None,
        language="en",
    )

    assert result == "The email was sent to Priya."
    assert "Email sent to Priya" in captured_messages[-1]["content"]
    assert "gmail/send_email" in captured_messages[-1]["content"]


@pytest.mark.integration
def test_confirmed_mcp_result_uses_status_fallback_when_formatting_fails(monkeypatch):
    from jarvis.reply import engine
    from jarvis.tools.types import ToolExecutionResult

    class MockDB:
        pass

    class MockConfig:
        wake_word = "Jarvis"
        llm_chat_timeout_sec = 2
        llm_thinking_enabled = False
        tts_engine = "piper"

    action_result = ToolExecutionResult(
        success=True,
        reply_text='{"message":"Email sent to Priya"}',
        result_metadata={"confirmed_mcp_action": "gmail/send_email"},
    )
    monkeypatch.setattr(
        engine,
        "handle_mcp_confirmation",
        lambda db, text, cancel_event=None: (True, action_result),
    )

    def failed_format(*args, **kwargs):
        raise TimeoutError("local model did not respond")

    monkeypatch.setattr(engine, "chat_with_messages", failed_format)
    result = engine.run_reply_engine(
        MockDB(), MockConfig(), None, "CONFIRM 12345678", dialogue_memory=None,
    )

    assert result == "The confirmed action completed."


@pytest.mark.integration
def test_confirmed_action_with_empty_result_still_returns_a_reply(monkeypatch):
    from jarvis.reply import engine
    from jarvis.tools.types import ToolExecutionResult

    class MockDB:
        pass

    class MockConfig:
        pass

    monkeypatch.setattr(
        engine,
        "handle_mcp_confirmation",
        lambda db, text, cancel_event=None: (
            True,
            ToolExecutionResult(success=True, reply_text=None),
        ),
    )
    result = engine.run_reply_engine(
        MockDB(),
        MockConfig(),
        None,
        "CONFIRM 12345678",
        dialogue_memory=None,
    )
    assert result
