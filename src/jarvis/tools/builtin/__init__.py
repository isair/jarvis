"""Builtin tools module.

This module contains all the built-in tools available to the Jarvis system.
Each tool is implemented in its own file for better organization and maintainability.
"""

# Import all tool functions for easy access
from .web_search import execute_web_search
from .local_files import execute_local_files  
from .fetch_web_page import execute_fetch_web_page
from .recall_conversation import execute_recall_conversation
from .meal_tools import execute_log_meal, execute_fetch_meals, execute_delete_meal
from .screenshot import execute_screenshot
from .ocr import capture_screenshot_and_ocr
from .nutrition import extract_and_log_meal, log_meal_from_args, summarize_meals, generate_followups_for_meal

__all__ = [
    # Tool execution functions
    'execute_web_search',
    'execute_local_files',
    'execute_fetch_web_page',
    'execute_recall_conversation',
    'execute_log_meal',
    'execute_fetch_meals',
    'execute_delete_meal',
    'execute_screenshot',
    
    # Supporting functions
    'capture_screenshot_and_ocr',
    'extract_and_log_meal',
    'log_meal_from_args',
    'summarize_meals',
    'generate_followups_for_meal',
]
