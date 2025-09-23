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
    def test_run_success_with_instant_and_lite(self, mock_get):
        """Test successful web search with instant answer + lite HTML page parsing."""
        # First call: instant answer JSON
        instant = Mock()
        instant.status_code = 200
        instant.json.return_value = {"Abstract": "A quick fact", "AbstractURL": "https://example.com/fact"}
        instant.raise_for_status = Mock()
        # Second call: lite HTML page
        lite = Mock()
        lite.status_code = 200
        lite.content = (
            b'<html><body>'
            b'<a href="https://site1.test/">First site result about something</a>'
            b'<a href="https://site2.test/">Second site detailed result here</a>'
            b'</body></html>'
        )
        mock_get.side_effect = [instant, lite]

        args = {"search_query": "test query"}
        result = self.tool.run(args, self.context)

        assert isinstance(result, ToolExecutionResult)
        assert result.success is True
        assert "Quick Answer:" in result.reply_text
        # At least one parsed site result should appear
        assert ("First site result" in result.reply_text) or ("Second site" in result.reply_text)
        # Should include the query echo
        assert "test query" in result.reply_text
        # user_print called at least once for start + success/failure
        assert self.context.user_print.call_count >= 1
        # Ensure count interpolation happened (look for dynamic result line)
        printed = "\n".join(call.args[0] for call in self.context.user_print.call_args_list)
        assert "Found 2 results" in printed or "Found 1 results" in printed or "Found 3 results" in printed

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
        """Test web search with network failure - graceful fallback returns success with guidance."""
        # First request (instant) fails, second (lite) fails
        mock_get.side_effect = [requests.exceptions.ConnectionError("down"), requests.exceptions.ConnectionError("down")]  # both phases fail
        args = {"search_query": "test query"}
        result = self.tool.run(args, self.context)
        assert isinstance(result, ToolExecutionResult)
        assert result.success is True  # still returns guidance
        assert "wasn't able to find" in result.reply_text.lower()
