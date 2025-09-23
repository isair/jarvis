"""Tests for delete meal tool."""

import pytest
from unittest.mock import Mock

from src.jarvis.tools.builtin.nutrition.delete_meal import DeleteMealTool
from src.jarvis.tools.base import ToolContext
from src.jarvis.tools.types import ToolExecutionResult


class TestDeleteMealTool:
    """Test delete meal tool functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tool = DeleteMealTool()
        self.context = Mock(spec=ToolContext)
        self.context.user_print = Mock()
        self.context.db = Mock()
    
    def test_tool_properties(self):
        """Test tool metadata properties."""
        assert self.tool.name == "deleteMeal"
        assert "delete" in self.tool.description.lower()
        assert self.tool.inputSchema["type"] == "object"
        assert "id" in self.tool.inputSchema["required"]
    
    def test_run_success(self):
        """Test successful meal deletion."""
        self.context.db.delete_meal.return_value = True
        
        args = {"id": 123}
        result = self.tool.run(args, self.context)
        
        assert isinstance(result, ToolExecutionResult)
        assert result.success is True
        assert "Meal deleted" in result.reply_text
        self.context.db.delete_meal.assert_called_once_with(123)
    
    def test_run_failure(self):
        """Test meal deletion failure."""
        self.context.db.delete_meal.return_value = False
        
        args = {"id": 999}
        result = self.tool.run(args, self.context)
        
        assert isinstance(result, ToolExecutionResult)
        assert result.success is False
        assert "couldn't delete" in result.reply_text.lower()
    
    def test_run_invalid_id(self):
        """Test deletion with invalid ID."""
        args = {"id": "not_a_number"}
        result = self.tool.run(args, self.context)
        
        assert isinstance(result, ToolExecutionResult)
        assert result.success is False
        # Should not call db.delete_meal with invalid ID
        self.context.db.delete_meal.assert_not_called()
