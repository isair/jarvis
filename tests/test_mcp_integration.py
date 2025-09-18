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
    from jarvis.tools.registry import discover_mcp_tools, generate_tools_description
    from jarvis.profile.profiles import PROFILE_ALLOWED_TOOLS
    
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
        mcp_tools = discover_mcp_tools(mcps_config)
        
        # Test tool registration (simulate what reply engine does)
        allowed_tools = PROFILE_ALLOWED_TOOLS.get("developer", []).copy()
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
            self.mcps = {"test-server": {"command": "fake"}}
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
            self.mcps = {"test-server": {"command": "fake"}}
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
