"""Tests for local files tool."""

import pytest
from unittest.mock import Mock, patch, mock_open
import tempfile
import os
from pathlib import Path

from src.jarvis.tools.builtin.local_files import LocalFilesTool
from src.jarvis.tools.base import ToolContext
from src.jarvis.tools.types import ToolExecutionResult


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

    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.is_file')
    @patch('pathlib.Path.read_text')
    def test_run_read_success(self, mock_read_text, mock_is_file, mock_exists):
        """Test successful file read."""
        mock_exists.return_value = True
        mock_is_file.return_value = True
        mock_read_text.return_value = "Test content"

        args = {"operation": "read", "path": "~/test.txt"}
        result = self.tool.run(args, self.context)

        assert isinstance(result, ToolExecutionResult)
        assert result.success is True
        assert "Test content" in result.reply_text

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
