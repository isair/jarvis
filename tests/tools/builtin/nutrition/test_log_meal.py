"""Tests for log meal tool."""

import pytest
from unittest.mock import Mock, patch

from src.jarvis.tools.builtin.nutrition.log_meal import LogMealTool
from src.jarvis.tools.base import ToolContext
from src.jarvis.tools.types import ToolExecutionResult
from src.jarvis.reply.planner import _parse_plan_step_concrete


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
        """Schema must expose a single 'meal' property so the planner's
        fast-path parser (key='value') can dispatch without an LLM resolver call."""
        assert self.tool.name == "logMeal"
        assert "meal" in self.tool.description.lower()
        schema = self.tool.inputSchema
        assert schema["type"] == "object"
        # Single 'meal' key — planner emits `logMeal meal='Big Mac'`
        assert "meal" in schema["properties"], (
            "'meal' must be a declared schema property so the fast-path parser accepts it"
        )
        # Numeric nutrition fields are implementation details resolved internally;
        # they must NOT appear in the public schema (they bloat the planner's
        # tool catalogue and cause the LLM resolver to attempt filling them in).
        assert "description" not in schema["properties"], (
            "'description' must not be a public schema key; use 'meal' instead"
        )
        assert "calories_kcal" not in schema.get("properties", {}), (
            "Nutrition fields must not appear in the public schema"
        )

    @patch('src.jarvis.tools.builtin.nutrition.log_meal.extract_and_log_meal')
    def test_run_with_meal_arg_passes_meal_text_to_extractor(self, mock_extract):
        """When the planner passes meal='Big Mac', the tool must pass that
        text to the extractor rather than the full redacted utterance."""
        mock_extract.return_value = "Logged meal #456: Big Mac - 550 kcal"

        result = self.tool.run({"meal": "Big Mac"}, self.context)

        assert result.success is True
        assert "Logged meal #456" in result.reply_text
        call_kwargs = mock_extract.call_args
        original_text = (
            call_kwargs.kwargs.get("original_text")
            or call_kwargs.args[2]
        )
        assert "Big Mac" in original_text, (
            "Extractor must use 'meal' arg as input text, not the full utterance"
        )

    @patch('src.jarvis.tools.builtin.nutrition.log_meal.extract_and_log_meal')
    def test_run_without_meal_arg_falls_back_to_redacted_text(self, mock_extract):
        """When no meal arg is provided, the extractor must use context.redacted_text."""
        mock_extract.return_value = "Logged meal #456: sandwich - 300 kcal"

        result = self.tool.run(None, self.context)

        assert isinstance(result, ToolExecutionResult)
        assert result.success is True
        assert "Logged meal #456" in result.reply_text
        call_kwargs = mock_extract.call_args
        original_text = (
            call_kwargs.kwargs.get("original_text")
            or call_kwargs.args[2]
        )
        assert original_text == self.context.redacted_text

    def test_run_failure(self):
        """When extraction returns nothing on all retries, return failure."""
        result = self.tool.run(None, self.context)

        assert isinstance(result, ToolExecutionResult)
        assert result.success is False
        assert result.reply_text == "Failed to log meal"


def test_planner_fast_path_accepts_meal_key():
    """The planner emits `logMeal meal='Big Mac'`. The fast-path parser must
    accept this and return ('logMeal', {'meal': 'Big Mac'}) without any LLM
    resolver call, so direct-exec works for small models."""
    tool = LogMealTool()
    allowed_names = ["logMeal"]
    allowed_props = {"logMeal": set(tool.inputSchema.get("properties", {}).keys())}

    result = _parse_plan_step_concrete(
        "logMeal meal='Big Mac'",
        allowed_names,
        allowed_props,
    )

    assert result is not None, (
        "Fast-path must accept 'logMeal meal=...' — 'meal' must be in the schema properties"
    )
    assert result[0] == "logMeal"
    assert result[1] == {"meal": "Big Mac"}
