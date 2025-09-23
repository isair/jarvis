"""Tests for web search tool."""

import pytest
from unittest.mock import Mock, patch
import requests

from src.jarvis.tools.builtin.web_search import WebSearchTool
from src.jarvis.tools.base import ToolContext
from src.jarvis.tools.types import ToolExecutionResult


class TestWebSearchTool:
    """Test web search tool functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tool = WebSearchTool()
        self.context = Mock(spec=ToolContext)
        self.context.user_print = Mock()
        self.context.cfg = Mock()
        self.context.cfg.web_search_enabled = True
        self.context.cfg.voice_debug = False

    def test_tool_properties(self):
        """Test tool metadata properties."""
        assert self.tool.name == "webSearch"
        assert "search" in self.tool.description.lower()
        assert self.tool.inputSchema["type"] == "object"
        assert "search_query" in self.tool.inputSchema["required"]

    @patch('requests.get')
    def test_run_success(self, mock_get):
        """Test successful web search."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = '<html><body><div class="result">Test result</div></body></html>'
        mock_get.return_value = mock_response

        args = {"search_query": "test query"}
        result = self.tool.run(args, self.context)

        assert isinstance(result, ToolExecutionResult)
        assert result.success is True
        assert "test query" in result.reply_text
        self.context.user_print.assert_called()

    def test_run_disabled(self):
        """Test web search when disabled."""
        self.context.cfg.web_search_enabled = False

        args = {"search_query": "test query"}
        result = self.tool.run(args, self.context)

        assert isinstance(result, ToolExecutionResult)
        assert result.success is False
        assert "disabled" in result.reply_text.lower()

    def test_run_empty_query(self):
        """Test web search with empty query."""
        args = {"search_query": ""}
        result = self.tool.run(args, self.context)

        assert isinstance(result, ToolExecutionResult)
        assert result.success is False
        assert "provide a search query" in result.reply_text.lower()

    def test_run_no_args(self):
        """Test web search with no arguments."""
        result = self.tool.run(None, self.context)

        assert isinstance(result, ToolExecutionResult)
        assert result.success is False
        assert "provide a search query" in result.reply_text.lower()

    def test_run_web_search_disabled(self):
        """Test web search when disabled in configuration."""
        # Simulate web search being disabled
        self.context.cfg.web_search_enabled = False

        args = {"search_query": "test query"}
        result = self.tool.run(args, self.context)

        assert isinstance(result, ToolExecutionResult)
        assert result.success is False
        assert "disabled" in result.reply_text.lower()

    @patch('requests.get')
    def test_run_network_failure_graceful(self, mock_get):
        """Test web search with network failure - shows graceful handling."""
        # Simulate network failure - web search handles this gracefully
        mock_get.side_effect = requests.exceptions.ConnectionError("Network unreachable")

        args = {"search_query": "test query"}
        result = self.tool.run(args, self.context)

        assert isinstance(result, ToolExecutionResult)
        # Web search is designed to be graceful - even with network failures,
        # it returns success=True with helpful guidance for the user
        assert result.success is True
        assert "search" in result.reply_text.lower()
        assert "wasn't able to find" in result.reply_text.lower()
