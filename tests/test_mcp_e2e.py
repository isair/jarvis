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
        for server_name, server_config in mcps.items():
            command = server_config.get('command', 'unknown')
            print(f"   - {server_name}: {command}")

        # Assert that we have at least some MCP configuration
        assert isinstance(mcps, dict), "MCP configuration should be a dictionary"
        print("   ✅ Configuration loading successful")

    except Exception as e:
        print(f"   ❌ Failed to load configuration: {e}")
        assert False, f"Failed to load configuration: {e}"


@pytest.mark.e2e
def test_mcp_discovery_with_mock():
    """Test MCP discovery with mocked servers."""
    print("🔍 Testing MCP tool discovery (mocked)...")

    # Create a fake MCP configuration
    fake_mcps = {
        "test-server": {
            "command": "echo",
            "args": ["test"]
        }
    }

    # Mock the MCPClient to avoid actual server connections
    from unittest.mock import patch, Mock

    class FakeMCPClient:
        def __init__(self, config):
            self.config = config

        def list_tools(self, server_name):
            return [
                {"name": "read", "description": "Read a file"},
                {"name": "write", "description": "Write a file"},
                {"name": "list", "description": "List directory contents"}
            ]

    try:
        with patch('src.jarvis.tools.registry.MCPClient', FakeMCPClient):
            mcp_tools = discover_mcp_tools(fake_mcps)

        expected_tools = {
            "test-server__read",
            "test-server__write",
            "test-server__list"
        }

        actual_tools = set(mcp_tools.keys())

        assert actual_tools == expected_tools, f"Tool mismatch. Expected: {expected_tools}, Got: {actual_tools}"
        print(f"   ✅ Successfully discovered {len(mcp_tools)} tools")
        for tool_name in mcp_tools:
            print(f"      - {tool_name}")

    except Exception as e:
        print(f"   ❌ Discovery failed: {e}")
        import traceback
        traceback.print_exc()
        assert False, f"Discovery failed: {e}"


@pytest.mark.e2e
def test_tool_registration_in_descriptions():
    """Test that discovered tools appear in tool descriptions."""
    print("📝 Testing tool description generation...")

    from src.jarvis.tools.registry import ToolSpec

    # Create mock MCP tools
    mock_mcp_tools = {
            "server1__tool1": ToolSpec(
                name="server1__tool1",
                description="Test tool 1 from server1",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            ),
            "server2__tool2": ToolSpec(
                name="server2__tool2",
                description="Test tool 2 from server2",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            )
        }

    try:
        allowed_tools = ["screenshot", "webSearch", "server1__tool1", "server2__tool2"]
        description = generate_tools_description(allowed_tools, mock_mcp_tools)

        # Check that MCP tools appear in description
        success = True
        for tool_name in mock_mcp_tools:
            if tool_name not in description:
                print(f"   ❌ Tool {tool_name} not found in description")
                success = False

        assert success, "Not all MCP tools appear in descriptions"
        print("   ✅ All MCP tools appear in descriptions")
        print(f"   📄 Description length: {len(description)} characters")

    except Exception as e:
        print(f"   ❌ Description generation failed: {e}")
        assert False, f"Description generation failed: {e}"


@pytest.mark.e2e
def test_tool_name_format():
    """Test that tool names follow the server__toolname format."""
    print("🏷️  Testing tool naming convention...")

    from unittest.mock import patch

    class FakeMCPClient:
        def __init__(self, config):
            self.config = config

        def list_tools(self, server_name):
            if server_name == "my-server":
                return [
                    {"name": "tool_with_underscores", "description": "Test tool"},
                    {"name": "tool-with-dashes", "description": "Another test tool"},
                    {"name": "simpletool", "description": "Simple tool"}
                ]
            return []

    try:
        with patch('src.jarvis.tools.registry.MCPClient', FakeMCPClient):
            mcps_config = {"my-server": {"command": "test"}}
            mcp_tools = discover_mcp_tools(mcps_config)

        expected_names = {
            "my-server__tool_with_underscores",
            "my-server__tool-with-dashes",
            "my-server__simpletool"
        }

        actual_names = set(mcp_tools.keys())

        assert actual_names == expected_names, f"Naming mismatch. Expected: {expected_names}, Got: {actual_names}"
        print("   ✅ Tool naming convention is correct")
        for name in actual_names:
            print(f"      - {name}")

    except Exception as e:
        print(f"   ❌ Naming test failed: {e}")
        assert False, f"Naming test failed: {e}"


@pytest.mark.e2e
def test_error_handling():
    """Test that MCP errors are handled gracefully."""
    print("⚠️  Testing error handling...")

    from unittest.mock import patch

    class FaultyMCPClient:
        def __init__(self, config):
            pass

        def list_tools(self, server_name):
            if server_name == "good-server":
                return [{"name": "working_tool", "description": "This works"}]
            elif server_name == "bad-server":
                raise Exception("Server connection failed")
            return []

    try:
        with patch('src.jarvis.tools.registry.MCPClient', FaultyMCPClient):
            mcps_config = {
                "good-server": {"command": "good"},
                "bad-server": {"command": "bad"},
                "empty-server": {"command": "empty"}
            }
            mcp_tools = discover_mcp_tools(mcps_config)

        # Should only get tools from the good server
        if len(mcp_tools) == 1 and "good-server__working_tool" in mcp_tools:
            print("   ✅ Error handling works correctly")
            print("      - Good server tools discovered")
            print("      - Bad server errors handled gracefully")
        else:
            print(f"   ❌ Expected 1 tool, got {len(mcp_tools)}: {list(mcp_tools.keys())}")
            assert False, f"Expected 1 tool, got {len(mcp_tools)}: {list(mcp_tools.keys())}"

    except Exception as e:
        print(f"   ❌ Error handling test failed: {e}")
        assert False, f"Error handling test failed: {e}"
