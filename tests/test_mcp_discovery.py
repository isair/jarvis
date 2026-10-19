"""MCP discovery preserves external names and isolates server failures."""

import pytest
import jarvis.tools.registry as registry
from jarvis.tools.registry import discover_mcp_tools, generate_tools_description, generate_tools_json_schema, run_tool_with_retries, ToolExecutionResult


class DummyCfg:
    def __init__(self):
        self.mcps = {}
        self.voice_debug = False


class DummyDB:
    pass


@pytest.fixture
def discover_servers(monkeypatch):
    def discover(replies):
        class FakeClient:
            def __init__(self, config):
                self.config = config

            def list_tools(self, server_name):
                reply = replies[server_name]
                if isinstance(reply, BaseException):
                    raise reply
                return reply

        monkeypatch.setattr(registry, 'MCPClient', FakeClient)
        return discover_mcp_tools({name: {'command': 'fake'} for name in replies})
    return discover


@pytest.mark.unit
def test_discover_mcp_tools_empty_config():
    assert discover_mcp_tools({}) == ({}, {})


@pytest.mark.unit
@pytest.mark.parametrize('names', [
    ('tool1',),
    ('read', 'write', 'list'),
    ('tool_with_underscores', 'tool-with-dashes', 'simpletool'),
])
def test_discover_mcp_tools_with_fake_server(discover_servers, names):
    server = 'test-server'
    replies = [{'name': name, 'description': f'Description of {name}',
                'inputSchema': {'type': 'object', 'properties': {}}} for name in names]
    tools, errors = discover_servers({server: replies})
    assert errors == {}
    assert set(tools) == {f'{server}__{name}' for name in names}
    for reply in replies:
        canonical = f"{server}__{reply['name']}"
        tool = tools[canonical]
        assert tool.name == canonical
        assert tool.description == reply['description']
        assert tool.inputSchema == reply['inputSchema']


@pytest.mark.unit
def test_discover_mcp_tools_handles_server_errors(discover_servers):
    error = RuntimeError('Server failed')
    tools, errors = discover_servers({
        'good-server': [{'name': 'working_tool', 'description': 'Good tool'}],
        'bad-server': error,
        'empty-server': [],
    })
    assert set(tools) == {'good-server__working_tool'}
    assert errors == {'bad-server': str(error)}


@pytest.mark.unit
def test_generate_tools_description_includes_mcp_tools():
    tools = [
        registry.ToolSpec(name='server__read', description='Read a file from the server',
                          inputSchema={'type': 'object', 'properties': {
                              'path': {'type': 'string', 'description': 'File path to read'}},
                              'required': ['path']}),
        registry.ToolSpec(name='other-server__write', description='Write to the second server',
                          inputSchema={'type': 'object', 'properties': {}}),
    ]
    description = generate_tools_description([tool.name for tool in tools] + ['screenshot'],
                                            {tool.name: tool for tool in tools})
    for tool in tools:
        assert tool.name in description
        assert tool.description in description
    assert 'screenshot' in description


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


@pytest.mark.unit
def test_generate_tools_json_schema_returns_openai_format():
    """Test that generate_tools_json_schema returns OpenAI-compatible format for native tool calling."""
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
    tools_schema = generate_tools_json_schema(allowed_tools, mcp_tools)

    # Should return a list
    assert isinstance(tools_schema, list)
    assert len(tools_schema) >= 2  # At least screenshot and server__read

    # Each tool should have the OpenAI format
    for tool in tools_schema:
        assert "type" in tool
        assert tool["type"] == "function"
        assert "function" in tool
        assert "name" in tool["function"]
        assert "description" in tool["function"]
        assert "parameters" in tool["function"]

    # Check that MCP tool is included
    tool_names = [t["function"]["name"] for t in tools_schema]
    assert "server__read" in tool_names
    assert "screenshot" in tool_names

    # Check MCP tool has correct schema
    server_read_tool = next(t for t in tools_schema if t["function"]["name"] == "server__read")
    assert server_read_tool["function"]["description"] == "Read a file from the server"
    assert "properties" in server_read_tool["function"]["parameters"]
    assert "path" in server_read_tool["function"]["parameters"]["properties"]


@pytest.mark.unit
def test_generate_tools_json_schema_handles_empty_input():
    """Test that generate_tools_json_schema handles empty or missing inputs gracefully."""
    # With no MCP tools
    tools_schema = generate_tools_json_schema(["screenshot"], None)
    assert isinstance(tools_schema, list)
    assert len(tools_schema) >= 1

    # With empty MCP tools dict
    tools_schema = generate_tools_json_schema(["screenshot"], {})
    assert isinstance(tools_schema, list)
    assert len(tools_schema) >= 1
