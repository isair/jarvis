"""Tests for screenshot tool."""

import pytest
from unittest.mock import Mock, patch
import sys

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

    @patch('shutil.which')
    @patch('subprocess.run')
    def test_run_success(self, mock_run, mock_which):
        """Test successful screenshot capture with inlined OCR logic."""
        # Lightweight stubs so dynamic imports succeed without heavy deps
        class _StubImgCtx:
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False
        class _StubImage:
            @staticmethod
            def open(*a, **k):
                return _StubImgCtx()

        sys.modules['pytesseract'] = type('StubTess', (), {
            'image_to_string': staticmethod(lambda *a, **k: 'Sample OCR text')
        })
        sys.modules['PIL'] = type('StubPIL', (), {'Image': _StubImage})
        sys.modules['PIL.Image'] = _StubImage

        # Indicate tools exist
        def which_side_effect(name):
            return f"/usr/bin/{name}" if name in ("screencapture", "tesseract") else None
        mock_which.side_effect = which_side_effect

        mock_proc = Mock()
        mock_proc.returncode = 0
        mock_run.return_value = mock_proc

        with patch('tempfile.mkdtemp', return_value='/tmp/jarvis_ocr_test'), \
             patch('os.path.exists', return_value=True), \
             patch('os.remove'), \
             patch('os.rmdir'):
            result = self.tool.run({}, self.context)

        assert isinstance(result, ToolExecutionResult)
        assert result.success is True
        assert result.reply_text == 'Sample OCR text'
        self.context.user_print.assert_called()

    @patch('shutil.which')
    @patch('subprocess.run')
    def test_run_empty_ocr(self, mock_run, mock_which):
        """Test screenshot with empty OCR result (tesseract missing)."""
        # screencapture present, tesseract missing
        def which_side_effect(name):
            if name == 'screencapture':
                return '/usr/bin/screencapture'
            return None
        mock_which.side_effect = which_side_effect
        mock_proc = Mock(); mock_proc.returncode = 0; mock_run.return_value = mock_proc
        with patch('tempfile.mkdtemp') as mock_tmp, \
             patch('os.path.exists') as mock_exists:
            mock_tmp.return_value = '/tmp/jarvis_ocr_test'
            mock_exists.return_value = True
            result = self.tool.run({}, self.context)
        assert isinstance(result, ToolExecutionResult)
        assert result.success is True
        assert result.reply_text == ''
