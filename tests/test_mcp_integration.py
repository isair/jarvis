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
    from jarvis.tools.registry import run_tool_with_retries, ToolExecutionResult, ToolSpec, _mcp_tools_cache, _mcp_tools_cache_lock
    
    class MockDB:
        pass
    
    class MockConfig:
        def __init__(self):
            self.mcps = {"test-server": {"command": "fake"}}
            self.voice_debug = False
    
    # Mock successful execution
    class FakeMCPClient:
        def __init__(self, config):
            pass
            
        def invoke_tool(self, server_name, tool_name, arguments):
            return {"text": f"Executed {tool_name} on {server_name}", "isError": False}
    
    with _mcp_tools_cache_lock:
        _mcp_tools_cache["test-server__example_tool"] = ToolSpec(
            name="test-server__example_tool", description="Example", readOnlyHint=True,
        )
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
    with _mcp_tools_cache_lock:
        _mcp_tools_cache.pop("test-server__example_tool", None)
        
        assert result.success is True
        assert "Executed example_tool on test-server" in result.reply_text


@pytest.mark.integration
def test_mcp_error_handling_in_context():
    """Test that MCP errors are properly handled in execution context."""
    from jarvis.tools.registry import run_tool_with_retries, ToolSpec, _mcp_tools_cache, _mcp_tools_cache_lock
    
    class MockDB:
        pass
    
    class MockConfig:
        def __init__(self):
            self.mcps = {"test-server": {"command": "fake"}}
            self.voice_debug = False
    
    # Mock failing execution
    class FailingMCPClient:
        def __init__(self, config):
            pass
            
        def invoke_tool(self, server_name, tool_name, arguments):
            return {"text": "Tool failed", "isError": True}
    
    with _mcp_tools_cache_lock:
        _mcp_tools_cache["test-server__failing_tool"] = ToolSpec(
            name="test-server__failing_tool", description="Failing", readOnlyHint=True,
        )
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
    with _mcp_tools_cache_lock:
        _mcp_tools_cache.pop("test-server__failing_tool", None)
        
        assert result.success is False
        assert result.error_message == "Tool failed"


@pytest.mark.integration
def test_mcp_exception_with_empty_message_still_yields_diagnosable_error():
    """A bare exception (e.g. ``concurrent.futures.TimeoutError()``) with an
    empty ``str(e)`` must not collapse the error message down to a bare
    trailing colon — callers need at least the exception type name to
    diagnose the failure from the log line alone.
    """
    from jarvis.tools.registry import run_tool_with_retries, ToolSpec, _mcp_tools_cache, _mcp_tools_cache_lock

    class MockDB:
        pass

    class MockConfig:
        def __init__(self):
            self.mcps = {"test-server": {"command": "fake"}}
            self.voice_debug = False

    class TimingOutMCPClient:
        def __init__(self, config):
            pass

        def invoke_tool(self, server_name, tool_name, arguments):
            raise TimeoutError()  # empty str(e), as raised on a bare invoke timeout

    with _mcp_tools_cache_lock:
        _mcp_tools_cache["test-server__slow_tool"] = ToolSpec(
            name="test-server__slow_tool", description="Slow", readOnlyHint=True,
        )
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
    with _mcp_tools_cache_lock:
        _mcp_tools_cache.pop("test-server__slow_tool", None)

        assert result.success is False
        assert result.error_message.strip().endswith(":") is False
        assert "TimeoutError" in result.error_message


@pytest.mark.integration
def test_mcp_action_requires_explicit_confirmation_before_execution():
    """A discovered MCP tool that is not marked read-only must stop before side effects."""
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
def test_mcp_exact_confirmation_executes_pending_action_once():
    from jarvis.tools.registry import (
        run_tool_with_retries,
        handle_mcp_confirmation,
        clear_pending_mcp_actions,
        _mcp_tools_cache,
        _mcp_tools_cache_lock,
        ToolSpec,
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
    assert TrackingMCPClient.calls == [
        ("test-server", "send_message", {"recipient": "Asha", "message": "Hello"})
    ]
    assert repeated[1].success is False
    clear_pending_mcp_actions()


@pytest.mark.integration
def test_mcp_read_only_annotation_allows_read_without_confirmation():
    from jarvis.tools.registry import (
        run_tool_with_retries,
        clear_pending_mcp_actions,
        _mcp_tools_cache,
        _mcp_tools_cache_lock,
        ToolSpec,
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
    with _mcp_tools_cache_lock:
        _mcp_tools_cache["test-server__search"] = ToolSpec(
            name="test-server__search",
            description="Search",
            readOnlyHint=True,
        )
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
        with _mcp_tools_cache_lock:
            _mcp_tools_cache.pop("test-server__search", None)


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
