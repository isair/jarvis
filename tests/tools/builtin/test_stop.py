"""Tests for stop tool."""

import pytest
from unittest.mock import Mock

from src.jarvis.tools.builtin.stop import StopTool, STOP_SIGNAL
from src.jarvis.tools.base import ToolContext
from src.jarvis.tools.types import ToolExecutionResult


class TestStopTool:
    """Test stop tool functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tool = StopTool()
        self.context = Mock(spec=ToolContext)
        self.context.user_print = Mock()

    def test_tool_properties(self):
        """Test tool metadata properties."""
        assert self.tool.name == "stop"
        assert "end" in self.tool.description.lower()
        assert "conversation" in self.tool.description.lower()
        assert self.tool.inputSchema["type"] == "object"
        assert self.tool.inputSchema["required"] == []
        assert self.tool.inputSchema["properties"] == {}

    def test_run_returns_stop_signal(self):
        """Test that run returns the special stop signal."""
        result = self.tool.run({}, self.context)

        assert isinstance(result, ToolExecutionResult)
        assert result.success is True
        assert result.reply_text == STOP_SIGNAL
        assert result.error_message is None

    def test_run_with_none_args(self):
        """Test that run works with None args."""
        result = self.tool.run(None, self.context)

        assert isinstance(result, ToolExecutionResult)
        assert result.success is True
        assert result.reply_text == STOP_SIGNAL

    def test_stop_signal_is_unique(self):
        """Test that stop signal is a unique value unlikely to be confused with real content."""
        assert STOP_SIGNAL.startswith("__")
        assert STOP_SIGNAL.endswith("__")
        assert "JARVIS" in STOP_SIGNAL
        assert "STOP" in STOP_SIGNAL


class TestStopSignalIntegration:
    """Test stop signal integration with registry."""

    def test_stop_tool_in_registry(self):
        """Test that stop tool is registered in BUILTIN_TOOLS."""
        from src.jarvis.tools.registry import BUILTIN_TOOLS

        assert "stop" in BUILTIN_TOOLS
        assert isinstance(BUILTIN_TOOLS["stop"], StopTool)

    def test_stop_tool_in_all_profiles(self):
        """Test that stop tool is available in all profiles."""
        from src.jarvis.profile.profiles import PROFILE_ALLOWED_TOOLS

        for profile_name, allowed_tools in PROFILE_ALLOWED_TOOLS.items():
            assert "stop" in allowed_tools, f"stop tool missing from {profile_name} profile"
