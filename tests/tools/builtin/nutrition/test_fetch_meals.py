"""Tests for fetch meals tool."""

import pytest
from unittest.mock import Mock
from datetime import datetime, timezone, timedelta

from src.jarvis.tools.builtin.nutrition.fetch_meals import FetchMealsTool
from src.jarvis.tools.base import ToolContext
from src.jarvis.tools.types import ToolExecutionResult


class TestFetchMealsTool:
    """Test fetch meals tool functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tool = FetchMealsTool()
        self.context = Mock(spec=ToolContext)
        self.context.user_print = Mock()
        self.context.db = Mock()
    
    def test_tool_properties(self):
        """Test tool metadata properties."""
        assert self.tool.name == "fetchMeals"
        assert "meals" in self.tool.description.lower()
        assert self.tool.inputSchema["type"] == "object"
        assert self.tool.inputSchema["required"] == []
    
    def test_run_success(self):
        """Test successful meal fetching."""
        # Mock database response
        mock_meals = [
            {
                "description": "Breakfast",
                "calories_kcal": 300,
                "protein_g": 15,
                "carbs_g": 30,
                "fat_g": 10
            },
            {
                "description": "Lunch", 
                "calories_kcal": 500,
                "protein_g": 25,
                "carbs_g": 45,
                "fat_g": 20
            }
        ]
        self.context.db.get_meals_between.return_value = mock_meals
        
        args = {
            "since_utc": "2025-01-01T00:00:00Z",
            "until_utc": "2025-01-01T23:59:59Z"
        }
        
        result = self.tool.run(args, self.context)
        
        assert isinstance(result, ToolExecutionResult)
        assert result.success is True
        assert "Meals: 2" in result.reply_text
        assert "Total ~800 kcal" in result.reply_text
        assert "Breakfast" in result.reply_text
        assert "Lunch" in result.reply_text
    
    def test_run_no_args(self):
        """Test meal fetching with no time range (defaults to last 24h)."""
        self.context.db.get_meals_between.return_value = []
        
        result = self.tool.run(None, self.context)
        
        assert isinstance(result, ToolExecutionResult)
        assert result.success is True
        assert "Meals: 0" in result.reply_text
        # Should have called db with some time range
        self.context.db.get_meals_between.assert_called_once()
