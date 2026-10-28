"""
End-to-end tests for MCP tool integration.

These tests verify that the complete MCP integration pipeline works:
1. Configuration loading
2. MCP tool discovery
3. Tool registration and availability
4. Reply engine integration

Note: These tests are marked as @pytest.mark.e2e and may not run in basic CI environments.
They are intended for local development and git hook testing.
"""

import sys
import os
import json
import tempfile
from pathlib import Path
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.jarvis.tools.registry import discover_mcp_tools, generate_tools_description
from src.jarvis.config import load_settings


@pytest.mark.e2e
def test_configuration_loading():
    """Test that MCP configuration is properly loaded."""
    print("🔧 Testing MCP configuration loading...")

    try:
        cfg = load_settings()
        mcps = getattr(cfg, 'mcps', {})

        print(f"   Found {len(mcps)} configured MCP servers")

        # Assert that we have at least some MCP configuration
        assert isinstance(mcps, dict), "MCP configuration should be a dictionary"
        print("   ✅ Configuration loading successful")

    except Exception as e:
        print(f"   ❌ Failed to load configuration: {e}")
        assert False, f"Failed to load configuration: {e}"


@pytest.mark.e2e
def test_mcp_discovery_with_mock():
    """Discovery exposes every tool from a configured local server."""
    from unittest.mock import patch

    class FakeMCPClient:
        def __init__(self, config):
            self.config = config

        def list_tools(self, server_name):
            return [
                {"name": "read", "description": "Read a file"},
                {"name": "write", "description": "Write a file"},
                {"name": "list", "description": "List directory contents"},
            ]

    with patch('src.jarvis.tools.registry.MCPClient', FakeMCPClient):
        tools, errors = discover_mcp_tools({"test-server": {"command": "local-mcp"}})

    assert errors == {}
    assert set(tools) == {
        "test-server__read",
        "test-server__write",
        "test-server__list",
    }
    assert tools["test-server__read"].description == "Read a file"


@pytest.mark.e2e
def test_tool_registration_in_descriptions():
    """Descriptions include each allowed MCP tool and its purpose."""
    from src.jarvis.tools.registry import ToolSpec

    mcp_tools = {
        "server1__tool1": ToolSpec(
            name="server1__tool1",
            description="Test tool 1 from server1",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
        "server2__tool2": ToolSpec(
            name="server2__tool2",
            description="Test tool 2 from server2",
            inputSchema={"type": "object", "properties": {}, "required": []},
        ),
    }
    description = generate_tools_description(list(mcp_tools), mcp_tools)

    assert "server1__tool1" in description
    assert "Test tool 1 from server1" in description
    assert "server2__tool2" in description
    assert "Test tool 2 from server2" in description


@pytest.mark.e2e
def test_tool_name_format():
    """Test that tool names follow the server__toolname format."""
    print("🏷️  Testing tool naming convention...")

    from unittest.mock import patch

    class FakeMCPClient:
        def __init__(self, config):
            self.config = config

        def list_tools(self, server_name):
            return [
                {"name": "tool_with_underscores", "description": "Test tool"},
                {"name": "tool-with-dashes", "description": "Another test tool"},
                {"name": "simpletool", "description": "Simple tool"}
            ]

    try:
        with patch('src.jarvis.tools.registry.MCPClient', FakeMCPClient):
            mcps_config = {"my-server": {"command": "test"}}
            mcp_tools, _errors = discover_mcp_tools(mcps_config)

        expected_names = {
            "my-server__tool_with_underscores",
            "my-server__tool-with-dashes",
            "my-server__simpletool"
        }

        actual_names = set(mcp_tools.keys())

        assert actual_names == expected_names, f"Naming mismatch. Expected: {expected_names}, Got: {actual_names}"
        print("   ✅ Tool naming convention is correct")

    except Exception as e:
        print(f"   ❌ Naming test failed: {e}")
        assert False, f"Naming test failed: {e}"


@pytest.mark.e2e
def test_error_handling():
    """Test that MCP errors are handled gracefully."""
    print("⚠️  Testing error handling...")

    from unittest.mock import patch, Mock

    client = Mock()
    client.list_tools.side_effect = [
        [{"name": "working_tool", "description": "This works"}],
        ConnectionError("Server connection failed"),
        [],
    ]

    try:
        with patch('src.jarvis.tools.registry.MCPClient', return_value=client):
            mcps_config = {
                "good-server": {"command": "good"},
                "bad-server": {"command": "bad"},
                "empty-server": {"command": "empty"}
            }
            mcp_tools, mcp_errors = discover_mcp_tools(mcps_config)

        # Should only get tools from the good server
        assert set(mcp_tools) == {"good-server__working_tool"}
        assert mcp_errors == {"bad-server": "Server connection failed"}

    except Exception as e:
        print(f"   ❌ Error handling test failed: {e}")
        assert False, f"Error handling test failed: {e}"
