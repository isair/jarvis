"""Tests for screenshot tool."""

import pytest
from unittest.mock import Mock, patch

from src.jarvis.tools.builtin.screenshot import ScreenshotTool
from src.jarvis.tools.base import ToolContext
from src.jarvis.tools.types import ToolExecutionResult


class TestScreenshotTool:
    """Test screenshot tool functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.tool = ScreenshotTool()
        self.context = Mock(spec=ToolContext)
        self.context.user_print = Mock()
    
    def test_tool_properties(self):
        """Test tool metadata properties."""
        assert self.tool.name == "screenshot"
        assert "capture" in self.tool.description.lower()
        assert self.tool.inputSchema["type"] == "object"
        assert self.tool.inputSchema["required"] == []
    
    @patch('src.jarvis.tools.builtin.screenshot.capture_screenshot_and_ocr')
    def test_run_success(self, mock_ocr):
        """Test successful screenshot capture."""
        mock_ocr.return_value = "Sample OCR text"
        
        result = self.tool.run({}, self.context)
        
        assert isinstance(result, ToolExecutionResult)
        assert result.success is True
        assert result.reply_text == "Sample OCR text"
        self.context.user_print.assert_called()
        mock_ocr.assert_called_once_with(interactive=True)
    
    @patch('src.jarvis.tools.builtin.screenshot.capture_screenshot_and_ocr')
    def test_run_empty_ocr(self, mock_ocr):
        """Test screenshot with empty OCR result."""
        mock_ocr.return_value = None
        
        result = self.tool.run({}, self.context)
        
        assert isinstance(result, ToolExecutionResult)
        assert result.success is True
        assert result.reply_text == ""
