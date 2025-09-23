"""Tests for log meal tool."""

import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timezone

from src.jarvis.tools.builtin.nutrition.log_meal import LogMealTool
from src.jarvis.tools.base import ToolContext
from src.jarvis.tools.types import ToolExecutionResult


class TestLogMealTool:
    """Test log meal tool functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.tool = LogMealTool()
        self.context = Mock(spec=ToolContext)
        self.context.user_print = Mock()
        self.context.db = Mock()
        self.context.cfg = Mock()
        self.context.cfg.use_stdin = False
        self.context.redacted_text = "I ate a sandwich"
        self.context.max_retries = 1

    def test_tool_properties(self):
        """Test tool metadata properties."""
        assert self.tool.name == "logMeal"
        assert "meal" in self.tool.description.lower()
        assert self.tool.inputSchema["type"] == "object"
        assert "description" in self.tool.inputSchema["required"]
        assert "calories_kcal" in self.tool.inputSchema["required"]

    @patch('src.jarvis.tools.builtin.nutrition.log_meal.log_meal_from_args')
    @patch('src.jarvis.tools.builtin.nutrition.log_meal.generate_followups_for_meal')
    def test_run_with_complete_args(self, mock_followups, mock_log_meal):
        """Test successful meal logging with complete arguments."""
        mock_log_meal.return_value = 123
        mock_followups.return_value = "Drink water, eat vegetables"

        args = {
            "description": "Chicken sandwich",
            "calories_kcal": 400,
            "protein_g": 25,
            "carbs_g": 35,
            "fat_g": 15,
            "fiber_g": 3,
            "sugar_g": 5,
            "sodium_mg": 800,
            "potassium_mg": 300,
            "micros": {"iron_mg": 2},
            "confidence": 0.8
        }

        result = self.tool.run(args, self.context)

        assert isinstance(result, ToolExecutionResult)
        assert result.success is True
        assert "Logged meal #123" in result.reply_text
        assert "Chicken sandwich" in result.reply_text
        mock_log_meal.assert_called_once()
        mock_followups.assert_called_once()

    @patch('src.jarvis.tools.builtin.nutrition.log_meal.extract_and_log_meal')
    def test_run_with_extraction_fallback(self, mock_extract):
        """Test meal logging with text extraction fallback."""
        mock_extract.return_value = "Logged meal #456: sandwich - 300 kcal"

        # Incomplete args should trigger extraction
        args = {"description": "sandwich"}

        result = self.tool.run(args, self.context)

        assert isinstance(result, ToolExecutionResult)
        assert result.success is True
        assert "Logged meal #456" in result.reply_text
        mock_extract.assert_called_once()

    def test_run_failure(self):
        """Test meal logging failure."""
        # No args and no extraction success
        result = self.tool.run(None, self.context)

        assert isinstance(result, ToolExecutionResult)
        assert result.success is False
        assert result.reply_text == "Failed to log meal"
