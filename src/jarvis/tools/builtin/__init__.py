"""Builtin tools module.

This module contains all the built-in tools available to the Jarvis system.
Each tool is implemented using the common Tool interface for consistency.
"""

# Import all tool classes
from .screenshot import ScreenshotTool
from .web_search import WebSearchTool
from .local_files import LocalFilesTool
from .fetch_web_page import FetchWebPageTool
from .recall_conversation import RecallConversationTool
from .nutrition.log_meal import LogMealTool
from .nutrition.fetch_meals import FetchMealsTool
from .nutrition.delete_meal import DeleteMealTool

# Import supporting functions that may still be used elsewhere
from .ocr import capture_screenshot_and_ocr

__all__ = [
    # Tool classes
    'ScreenshotTool',
    'WebSearchTool', 
    'LocalFilesTool',
    'FetchWebPageTool',
    'RecallConversationTool',
    'LogMealTool',
    'FetchMealsTool',
    'DeleteMealTool',
    
    # Supporting functions
    'capture_screenshot_and_ocr',
]
