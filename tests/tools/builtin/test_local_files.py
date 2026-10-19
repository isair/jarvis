"""Tests for local files tool."""

import pytest
from unittest.mock import Mock, patch
import os

from jarvis.tools.builtin.local_files import LocalFilesTool
from jarvis.tools.base import ToolContext
from jarvis.tools.types import ToolExecutionResult


pytestmark = pytest.mark.unit


class TestLocalFilesTool:
    """Test local files tool functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tool = LocalFilesTool()
        self.context = Mock(spec=ToolContext)
        self.context.user_print = Mock()

    def test_tool_properties(self):
        """Test tool metadata properties."""
        assert self.tool.name == "localFiles"
        assert "file" in self.tool.description.lower()
        assert self.tool.inputSchema["type"] == "object"
        assert "operation" in self.tool.inputSchema["required"]
        assert "path" in self.tool.inputSchema["required"]

    def test_run_no_args(self):
        """Test local files with no arguments."""
        result = self.tool.run(None, self.context)

        assert isinstance(result, ToolExecutionResult)
        assert result.success is False
        assert "requires a JSON object" in result.reply_text

    def test_run_missing_operation(self):
        """Test local files with missing operation."""
        args = {"path": "test.txt"}
        result = self.tool.run(args, self.context)

        assert isinstance(result, ToolExecutionResult)
        assert result.success is False
        assert "requires 'operation'" in result.reply_text

    def test_run_missing_path(self):
        """Test local files with missing path."""
        args = {"operation": "read"}
        result = self.tool.run(args, self.context)

        assert isinstance(result, ToolExecutionResult)
        assert result.success is False
        assert "requires 'operation' and 'path'" in result.reply_text

    @pytest.mark.parametrize("payload", [b"Test content", "café 🦦".encode("utf-8"), b"bad\xffUTF-8", b""])
    def test_run_read_success(self, tmp_path, monkeypatch, payload):
        """Read owned UTF-8 content, replacing invalid input without changing valid text."""
        source = tmp_path / "test.txt"
        source.write_bytes(payload)
        expanduser = os.path.expanduser
        monkeypatch.setattr(os.path, "expanduser", lambda path: str(tmp_path) if path == "~" else expanduser(path))
        result = self.tool.run({"operation": "read", "path": str(source)}, self.context)

        assert result.success
        assert result.reply_text == payload.decode("utf-8", errors="replace")

    @patch('pathlib.Path.exists')
    def test_run_read_not_found(self, mock_exists):
        """Test file read when file doesn't exist."""
        mock_exists.return_value = False

        args = {"operation": "read", "path": "~/nonexistent.txt"}
        result = self.tool.run(args, self.context)

        assert isinstance(result, ToolExecutionResult)
        assert result.success is False
        assert "not found" in result.reply_text.lower()

    @patch('pathlib.Path.write_text')
    @patch('pathlib.Path.mkdir')
    def test_run_write_success(self, mock_mkdir, mock_write_text):
        """Test successful file write."""
        args = {"operation": "write", "path": "~/test.txt", "content": "Test content"}
        result = self.tool.run(args, self.context)

        assert isinstance(result, ToolExecutionResult)
        assert result.success is True
        assert "Wrote" in result.reply_text

    def test_run_write_no_content(self):
        """Test file write without content."""
        args = {"operation": "write", "path": "~/test.txt"}
        result = self.tool.run(args, self.context)

        assert isinstance(result, ToolExecutionResult)
        assert result.success is False
        assert "requires string 'content'" in result.reply_text

    def test_run_unsafe_path(self):
        """Test with path outside home directory."""
        args = {"operation": "read", "path": "/etc/passwd"}
        result = self.tool.run(args, self.context)

        assert isinstance(result, ToolExecutionResult)
        assert result.success is False
        assert "not allowed" in result.reply_text.lower()

    def test_run_unknown_operation(self):
        """Test with unknown operation."""
        args = {"operation": "invalid", "path": "~/test.txt"}
        result = self.tool.run(args, self.context)

        assert isinstance(result, ToolExecutionResult)
        assert result.success is False
        assert "Unknown localFiles operation" in result.reply_text
