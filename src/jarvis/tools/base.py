"""Base tool interface for Jarvis tools.

This module defines the common interface that all tools must implement,
ensuring consistency with MCP tool format and enabling dictionary-based execution.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable
from .types import ToolExecutionResult


class ToolContext:
    """Context object containing all the resources a tool might need."""

    def __init__(
        self,
        db,
        cfg,
        system_prompt: str,
        original_prompt: str,
        redacted_text: str,
        max_retries: int,
        user_print: Callable[[str], None]
    ):
        self.db = db
        self.cfg = cfg
        self.system_prompt = system_prompt
        self.original_prompt = original_prompt
        self.redacted_text = redacted_text
        self.max_retries = max_retries
        self.user_print = user_print


class Tool(ABC):
    """Base class for all Jarvis tools.

    This interface matches the MCP tool format with name, description, and inputSchema
        properties, while providing a simple execution interface focused on tool logic.

        Implementation guideline:
        - Put all operational logic directly in the `run` method.
        - Keep helper functions module-level only when they provide clear reuse (e.g. nutrition
            extraction helpers used by multiple code paths / tests). Otherwise inline.
        - `run` receives validated args (per schema) and a `ToolContext` giving access to db, cfg,
            prompts, redacted_text, retry allowance, and a user_print callable.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """The canonical tool identifier (camelCase)."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what the tool does."""
        pass

    @property
    @abstractmethod
    def inputSchema(self) -> Dict[str, Any]:
        """JSON Schema for tool arguments (matches MCP format)."""
        pass

    @abstractmethod
    def run(self, args: Optional[Dict[str, Any]], context: ToolContext) -> ToolExecutionResult:
        """Execute the tool with the given arguments and context.

        This is the only method tools need to implement. All common concerns
        like user printing, database access, config, etc. are provided via context.

        Args:
            args: Dictionary containing tool arguments (validated against inputSchema)
            context: ToolContext with db, cfg, user_print, etc.

        Returns:
            ToolExecutionResult with execution results
        """
        pass

    def execute(
        self,
        db,
        cfg,
        tool_args: Optional[Dict[str, Any]],
        system_prompt: str,
        original_prompt: str,
        redacted_text: str,
        max_retries: int,
        user_print: Callable[[str], None]
    ) -> ToolExecutionResult:
        """Execute the tool (internal method used by registry).

        This method creates the context and calls the tool's run method.
        Tools should implement run(), not this method.
        """
        context = ToolContext(
            db=db,
            cfg=cfg,
            system_prompt=system_prompt,
            original_prompt=original_prompt,
            redacted_text=redacted_text,
            max_retries=max_retries,
            user_print=user_print
        )
        return self.run(tool_args, context)
