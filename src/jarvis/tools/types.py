"""Common types and result classes for tools."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class RiskLevel(Enum):
    """Classification of action risk.

    Stored on each tool (via ``Tool.assess_risk``) so risk information
    stays co-located with the tool that owns it rather than in a separate
    parallel mapping that can diverge when tools are added or removed.
    """

    SAFE = "safe"       # Read-only or clearly reversible
    MODERATE = "moderate"  # Writes that are easily undone
    HIGH = "high"       # Potentially destructive or hard-to-undo


@dataclass
class ToolExecutionResult:
    """Result object for tool execution."""
    success: bool
    reply_text: Optional[str]
    error_message: Optional[str] = None
