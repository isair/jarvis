"""Windows screenshot capture tests."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from src.jarvis.tools.builtin.screenshot import ScreenshotTool, capture_screen_ocr


@pytest.mark.unit
class TestScreenshotWindows:
    @patch("src.jarvis.tools.builtin.screenshot._ocr_png", return_value="Hello screen")
    @patch("src.jarvis.screen_capture.capture_display_png", return_value=True)
    @patch("src.jarvis.tools.builtin.screenshot.sys.platform", "win32")
    def test_capture_screen_ocr_windows(self, *_mocks):
        assert capture_screen_ocr() == "Hello screen"

    @patch(
        "src.jarvis.tools.builtin.screenshot.build_screen_context_for_query",
        return_value="UNTRUSTED SCREEN CAPTURE DATA\n```\nUI text\n```",
    )
    def test_run_returns_screen_block(self, _build):
        tool = ScreenshotTool()
        ctx = MagicMock()
        ctx.user_print = MagicMock()
        ctx.cfg = MagicMock()
        result = tool.run({}, ctx)
        assert result.success is True
        assert "UI text" in result.reply_text
