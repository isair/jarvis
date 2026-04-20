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

    @patch('src.jarvis.tools.builtin.web_search._fetch_page_content')
    @patch('requests.get')
    def test_fetch_cascades_through_results_when_first_fails(self, mock_get, mock_fetch):
        """If top result fetch fails, try result #2 — don't give up after one attempt.

        Field failure (2026-04-20) had the first fetch silently time out, producing
        a payload with no Content block and a reply that said 'here are some links'.
        The cascade gives us up to 3 attempts before falling back to the
        links-only envelope.
        """
        instant = Mock()
        instant.status_code = 200
        instant.json.return_value = {}  # no instant answer → fetch path runs
        instant.raise_for_status = Mock()
        lite = Mock()
        lite.status_code = 200
        lite.content = (
            b'<html><body>'
            b'<a href="https://site1.test/">First site result title</a>'
            b'<a href="https://site2.test/">Second site result title</a>'
            b'<a href="https://site3.test/">Third site result title</a>'
            b'</body></html>'
        )
        mock_get.side_effect = [instant, lite]
        # First fetch fails, second succeeds — asserts we didn't stop at #1.
        mock_fetch.side_effect = [None, "Page content about the topic."]

        result = self.tool.run({"search_query": "topic"}, self.context)

        assert result.success is True
        assert mock_fetch.call_count == 2, (
            f"Expected to retry after first fetch returned None; called {mock_fetch.call_count}x"
        )
        assert "Content from top result" in result.reply_text
        assert "Page content about the topic." in result.reply_text

    @patch('src.jarvis.tools.builtin.web_search._fetch_page_content')
    @patch('requests.get')
    def test_envelope_signals_when_all_fetches_fail(self, mock_get, mock_fetch):
        """When every fetch attempt returns None, envelope tells the model to admit it.

        Without this, the tool would emit "Use this information to reply" over a
        pure link list — which small models turn into "here are some links to
        Wikipedia" (the 2026-04-20 field failure). The new envelope instead tells
        the model to say it couldn't read the pages and offer retry, so the
        reply is honest instead of looking like a wrong answer.
        """
        instant = Mock()
        instant.status_code = 200
        instant.json.return_value = {}
        instant.raise_for_status = Mock()
        lite = Mock()
        lite.status_code = 200
        lite.content = (
            b'<html><body>'
            b'<a href="https://site1.test/">First site result title</a>'
            b'<a href="https://site2.test/">Second site result title</a>'
            b'<a href="https://site3.test/">Third site result title</a>'
            b'</body></html>'
        )
        mock_get.side_effect = [instant, lite]
        mock_fetch.side_effect = [None, None, None]

        result = self.tool.run({"search_query": "topic"}, self.context)

        assert result.success is True
        # Envelope must flag the fetch failure explicitly.
        assert "none of the top pages could be fetched" in result.reply_text.lower()
        # Must NOT tell the model to use the payload as an answer.
        assert "use this information to reply" not in result.reply_text.lower()
        # Must NOT advertise a Content block — there is none.
        assert "Content from top result" not in result.reply_text
        # Anti-confabulation guardrail must be in the envelope itself.
        assert "do not invent" in result.reply_text.lower() or "don't invent" in result.reply_text.lower()

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
