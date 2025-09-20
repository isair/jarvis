"""Screenshot tool implementation for OCR capture."""

from typing import Optional
from ...debug import debug_log
from .ocr import capture_screenshot_and_ocr
from ..types import ToolExecutionResult


def execute_screenshot(_user_print: callable) -> ToolExecutionResult:
    """Capture a screenshot and perform OCR.
    
    Args:
        _user_print: Function to print user-facing messages
        
    Returns:
        ToolExecutionResult with OCR text
    """
    _user_print("📸 Capturing a screenshot for OCR…")
    debug_log("screenshot: capturing OCR...", "screenshot")
    ocr_text = capture_screenshot_and_ocr(interactive=True) or ""
    debug_log(f"screenshot: ocr_chars={len(ocr_text)}", "screenshot")
    _user_print("✅ Screenshot processed.")
    # Return raw OCR text as tool result (no LLM processing here)
    return ToolExecutionResult(success=True, reply_text=ocr_text)