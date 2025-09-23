"""Screenshot tool implementation for OCR capture."""

from typing import Dict, Any, Optional
from ...debug import debug_log
from .ocr import capture_screenshot_and_ocr
from ..base import Tool, ToolContext
from ..types import ToolExecutionResult




class ScreenshotTool(Tool):
    """Tool for capturing screenshots and performing OCR."""

    @property
    def name(self) -> str:
        return "screenshot"

    @property
    def description(self) -> str:
        return "Capture a selected screen region and OCR the text. Use only if the OCR will materially help."

    @property
    def inputSchema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
            "required": []
        }

    def run(self, args: Optional[Dict[str, Any]], context: ToolContext) -> ToolExecutionResult:
        """Execute the screenshot tool."""
        context.user_print("📸 Capturing a screenshot for OCR…")
        debug_log("screenshot: capturing OCR...", "screenshot")
        ocr_text = capture_screenshot_and_ocr(interactive=True) or ""
        debug_log(f"screenshot: ocr_chars={len(ocr_text)}", "screenshot")
        context.user_print("✅ Screenshot processed.")
        # Return raw OCR text as tool result (no LLM processing here)
        return ToolExecutionResult(success=True, reply_text=ocr_text)
