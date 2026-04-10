"""Tests for the macOS AppleScript tool."""

from unittest.mock import Mock, patch

from src.jarvis.profile.profiles import PROFILE_ALLOWED_TOOLS
from src.jarvis.tools.base import ToolContext
from src.jarvis.tools.builtin.applescript import AppleScriptTool
from src.jarvis.tools.registry import BUILTIN_TOOLS
from src.jarvis.tools.types import ToolExecutionResult


class TestAppleScriptTool:
    """Test AppleScript tool metadata and dispatch."""

    def setup_method(self):
        self.tool = AppleScriptTool()
        self.context = Mock(spec=ToolContext)
        self.context.user_print = Mock()

    def test_tool_properties(self):
        assert self.tool.name == "appleScript"
        assert "macos" in self.tool.description.lower()
        assert self.tool.inputSchema["type"] == "object"
        assert self.tool.inputSchema["required"] == ["action"]

    @patch("src.jarvis.tools.builtin.applescript._get_battery", return_value="Battery is at 82%, sir.")
    def test_run_dispatches_simple_action(self, mock_get_battery):
        result = self.tool.run({"action": "get_battery"}, self.context)

        assert isinstance(result, ToolExecutionResult)
        assert result.success is True
        assert result.reply_text == "Battery is at 82%, sir."
        mock_get_battery.assert_called_once_with()

    @patch("src.jarvis.tools.builtin.applescript._set_volume", return_value="Volume set to 30, sir.")
    def test_run_accepts_nested_args(self, mock_set_volume):
        result = self.tool.run({"action": "set_volume", "args": {"level": 30}}, self.context)

        assert result.success is True
        assert result.reply_text == "Volume set to 30, sir."
        mock_set_volume.assert_called_once_with(30)

    @patch("src.jarvis.tools.builtin.applescript._open_app", return_value="Opened Safari, sir.")
    def test_run_accepts_top_level_args(self, mock_open_app):
        result = self.tool.run({"action": "open_app", "app_name": "Safari"}, self.context)

        assert result.success is True
        assert result.reply_text == "Opened Safari, sir."
        mock_open_app.assert_called_once_with("Safari")

    def test_run_requires_known_action(self):
        result = self.tool.run({"action": "launch_batmobile"}, self.context)

        assert result.success is False
        assert "Unknown appleScript action" in result.reply_text


class TestAppleScriptIntegration:
    """Test registry and profile integration for AppleScript."""

    def test_tool_registered(self):
        assert "appleScript" in BUILTIN_TOOLS
        assert isinstance(BUILTIN_TOOLS["appleScript"], AppleScriptTool)

    def test_tool_available_in_all_profiles(self):
        for allowed_tools in PROFILE_ALLOWED_TOOLS.values():
            assert "appleScript" in allowed_tools
