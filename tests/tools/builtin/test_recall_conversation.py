"""Tests for recall conversation tool."""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timezone

from src.jarvis.tools.builtin.recall_conversation import RecallConversationTool
from src.jarvis.tools.base import ToolContext
from src.jarvis.tools.types import ToolExecutionResult


class TestRecallConversationTool:
    """Test recall conversation tool functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tool = RecallConversationTool()
        self.context = Mock(spec=ToolContext)
        self.context.user_print = Mock()
        self.context.db = Mock()
        self.context.cfg = Mock()

    def test_tool_properties(self):
        """Test tool metadata properties."""
        assert self.tool.name == "recallConversation"
        assert "conversation" in self.tool.description.lower()
        assert self.tool.inputSchema["type"] == "object"
        assert "search_query" in self.tool.inputSchema["required"]

    @patch('src.jarvis.tools.builtin.recall_conversation.search_conversation_memory')
    def test_run_success_with_query(self, mock_search):
        """Test successful conversation recall with query."""
        mock_search.return_value = [
            {"content": "Previous conversation", "timestamp": "2023-01-01T10:00:00Z"}
        ]

        args = {"search_query": "test query"}
        result = self.tool.run(args, self.context)

        assert isinstance(result, ToolExecutionResult)
        assert result.success is True
        assert "Previous conversation" in result.reply_text
        self.context.user_print.assert_called()
        mock_search.assert_called_once()

    @patch('src.jarvis.tools.builtin.recall_conversation.search_conversation_memory')
    def test_run_success_no_results(self, mock_search):
        """Test conversation recall with no results."""
        mock_search.return_value = []

        args = {"search_query": "nonexistent query"}
        result = self.tool.run(args, self.context)

        assert isinstance(result, ToolExecutionResult)
        assert result.success is True
        assert "couldn't find" in result.reply_text.lower() or "no matching" in result.reply_text.lower()

    @patch('src.jarvis.tools.builtin.recall_conversation.search_conversation_memory')
    def test_run_with_time_range(self, mock_search):
        """Test conversation recall with time range."""
        mock_search.return_value = [
            {"content": "Time-filtered conversation", "timestamp": "2023-01-01T10:00:00Z"}
        ]

        args = {
            "search_query": "test",
            "from": "2023-01-01T00:00:00Z",
            "to": "2023-01-02T00:00:00Z"
        }
        result = self.tool.run(args, self.context)

        assert isinstance(result, ToolExecutionResult)
        assert result.success is True
        assert "Time-filtered conversation" in result.reply_text

    def test_run_no_args(self):
        """Test conversation recall with no arguments."""
        result = self.tool.run(None, self.context)

        assert isinstance(result, ToolExecutionResult)
        assert result.success is False
        assert "search query" in result.reply_text.lower() or "time range" in result.reply_text.lower()

    def test_run_empty_query(self):
        """Test conversation recall with empty query."""
        args = {"search_query": ""}
        result = self.tool.run(args, self.context)

        assert isinstance(result, ToolExecutionResult)
        assert result.success is False
        assert "search query" in result.reply_text.lower() or "time range" in result.reply_text.lower()

    @patch('src.jarvis.tools.builtin.recall_conversation.search_conversation_memory')
    def test_run_search_error(self, mock_search):
        """Test conversation recall with search error."""
        mock_search.side_effect = Exception("Database error")

        args = {"search_query": "test query"}
        result = self.tool.run(args, self.context)

        assert isinstance(result, ToolExecutionResult)
        assert result.success is False
        assert "trouble searching" in result.reply_text.lower()
