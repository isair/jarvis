"""
Tests for MCP tool discovery and integration.

This test suite ensures that:
1. MCP tools are properly discovered from configured servers
2. Tool naming follows the server__toolname convention
3. Tools are properly integrated into the reply engine
4. The new OpenAI-standard tool calling format works correctly
"""

import pytest
from jarvis.tools.registry import discover_mcp_tools, generate_tools_description, run_tool_with_retries, ToolExecutionResult


class DummyCfg:
    def __init__(self):
        self.mcps = {}
        self.voice_debug = False


class DummyDB:
    pass


@pytest.mark.unit
def test_discover_mcp_tools_empty_config():
    """Test that empty MCP config returns empty tools dict."""
    result = discover_mcp_tools({})
    assert result == {}


@pytest.mark.unit
def test_discover_mcp_tools_with_fake_server(monkeypatch):
    """Test discovery of tools from a fake MCP server."""
    # Mock the MCPClient
    class FakeClient:
        def __init__(self, config):
            self.config = config

        def list_tools(self, server_name):
            if server_name == "test-server":
                return [
                    {"name": "read", "description": "Read a file"},
                    {"name": "write", "description": "Write to a file"},
                    {"name": "list", "description": "List directory contents"},
                ]
            return []

    import jarvis.tools.registry as registry_mod
    monkeypatch.setattr(registry_mod, "MCPClient", FakeClient)

    mcps_config = {
        "test-server": {
            "command": "fake-cmd",
            "args": ["--test"]
        }
    }

    result = discover_mcp_tools(mcps_config)

    # Should create tools with server__toolname format
    expected_tools = {
        "test-server__read",
        "test-server__write",
        "test-server__list"
    }

    assert set(result.keys()) == expected_tools

    # Check tool spec properties
    read_tool = result["test-server__read"]
    assert read_tool.name == "test-server__read"
    assert "Read a file" in read_tool.description


@pytest.mark.unit
def test_discover_mcp_tools_handles_server_errors(monkeypatch):
    """Test that discovery continues even if one server fails."""
    class FakeClient:
        def __init__(self, config):
            self.config = config

        def list_tools(self, server_name):
            if server_name == "good-server":
                return [{"name": "tool1", "description": "Good tool"}]
            elif server_name == "bad-server":
                raise Exception("Server failed")
            return []

    import jarvis.tools.registry as registry_mod
    monkeypatch.setattr(registry_mod, "MCPClient", FakeClient)

    mcps_config = {
        "good-server": {"command": "good"},
        "bad-server": {"command": "bad"}
    }

    result = discover_mcp_tools(mcps_config)

    # Should still get tools from the good server
    assert "good-server__tool1" in result
    assert len(result) == 1


@pytest.mark.unit
def test_generate_tools_description_includes_mcp_tools():
    """Test that MCP tools are included in the tools description."""
    from jarvis.tools.registry import ToolSpec

    mcp_tools = {
            "server__read": ToolSpec(
                name="server__read",
                description="Read a file from the server",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path to read"
                        }
                    },
                    "required": ["path"]
                }
            )
        }

    allowed_tools = ["server__read", "screenshot"]
    description = generate_tools_description(allowed_tools, mcp_tools)

    assert "server__read" in description
    assert "Read a file from the server" in description
    assert "screenshot" in description  # Should still include builtin tools


@pytest.mark.unit
def test_mcp_tool_execution_new_format(monkeypatch):
    """Test execution of MCP tools using the new server__toolname format."""
    db = DummyDB()
    cfg = DummyCfg()
    cfg.mcps = {"test-server": {"command": "fake", "args": []}}

    class FakeClient:
        def __init__(self, config):
            self.config = config

        def invoke_tool(self, server_name, tool_name, arguments):
            assert server_name == "test-server"
            assert tool_name == "read"
            assert arguments == {"path": "/test/file.txt"}
            return {"text": "file contents", "isError": False}

    import jarvis.tools.registry as registry_mod
    monkeypatch.setattr(registry_mod, "MCPClient", FakeClient)

    result = run_tool_with_retries(
        db=db,
        cfg=cfg,
        tool_name="test-server__read",
        tool_args={"path": "/test/file.txt"},
        system_prompt="",
        original_prompt="",
        redacted_text="",
        max_retries=0
    )

    assert result.success is True
    assert result.reply_text == "file contents"
    assert result.error_message is None


@pytest.mark.unit
def test_mcp_tool_execution_error_handling(monkeypatch):
    """Test that MCP tool errors are properly handled."""
    db = DummyDB()
    cfg = DummyCfg()
    cfg.mcps = {"test-server": {"command": "fake", "args": []}}

    class FakeClient:
        def __init__(self, config):
            self.config = config

        def invoke_tool(self, server_name, tool_name, arguments):
            return {"text": "Permission denied", "isError": True}

    import jarvis.tools.registry as registry_mod
    monkeypatch.setattr(registry_mod, "MCPClient", FakeClient)

    result = run_tool_with_retries(
        db=db,
        cfg=cfg,
        tool_name="test-server__read",
        tool_args={"path": "/forbidden/file.txt"},
        system_prompt="",
        original_prompt="",
        redacted_text="",
        max_retries=0
    )

    assert result.success is False
    assert result.error_message == "Permission denied"


@pytest.mark.unit
def test_mcp_tool_invalid_server_name():
    """Test that invalid server names in tool names are handled."""
    db = DummyDB()
    cfg = DummyCfg()
    cfg.mcps = {"valid-server": {"command": "fake", "args": []}}

    result = run_tool_with_retries(
        db=db,
        cfg=cfg,
        tool_name="invalid-server__read",
        tool_args={"path": "/test/file.txt"},
        system_prompt="",
        original_prompt="",
        redacted_text="",
        max_retries=0
    )

    # Should fail gracefully since server not configured
    assert result.success is False
    assert result.error_message is not None
    assert "invalid-server" in result.error_message.lower()


@pytest.mark.unit
def test_mcp_tool_exception_handling(monkeypatch):
    """Test that exceptions during MCP tool execution are caught."""
    db = DummyDB()
    cfg = DummyCfg()
    cfg.mcps = {"test-server": {"command": "fake", "args": []}}

    class FakeClient:
        def __init__(self, config):
            self.config = config

        def invoke_tool(self, server_name, tool_name, arguments):
            raise Exception("Connection failed")

    import jarvis.tools.registry as registry_mod
    monkeypatch.setattr(registry_mod, "MCPClient", FakeClient)

    result = run_tool_with_retries(
        db=db,
        cfg=cfg,
        tool_name="test-server__read",
        tool_args={"path": "/test/file.txt"},
        system_prompt="",
        original_prompt="",
        redacted_text="",
        max_retries=0
    )

    assert result.success is False
    assert "Connection failed" in result.error_message
