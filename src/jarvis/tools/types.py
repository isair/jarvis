"""Common types and result classes for tools."""

from dataclasses import dataclass
from typing import Optional, Any


@dataclass
class ToolExecutionResult:
    """Result object for tool execution."""
    success: bool
    reply_text: Optional[str]
    error_message: Optional[str] = None
    missing_context: Optional[str] = None
    resource_references: tuple[dict[str, Any], ...] = ()
