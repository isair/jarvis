"""
Exception hierarchy for the reply engine.

These exceptions represent distinct failure modes within the agent loop and
are intended to be caught by the engine's error-handling layer or propagated
to the daemon for graceful degradation.
"""

from __future__ import annotations


class AgentError(Exception):
    """Base class for all reply-engine errors."""


class ApprovalRequiredError(AgentError):
    """Raised when a planned action requires explicit user approval before
    proceeding. Retained for API compatibility; the engine no longer raises
    this in the default act-then-undo flow."""


class LoopExhaustedError(AgentError):
    """Raised when the agentic loop reaches its maximum iteration count
    without producing a final response."""


class ModelOutputError(AgentError):
    """Raised when the language model returns output that cannot be parsed
    or is otherwise structurally invalid."""


class PolicyDeniedError(AgentError):
    """Raised when a policy check prevents an action from being executed."""


class ToolExecutionError(AgentError):
    """Raised when a tool call fails after all retry attempts are exhausted."""

    def __init__(self, tool_name: str, message: str) -> None:
        super().__init__(f"{tool_name}: {message}")
        self.tool_name = tool_name


class ToolSchemaError(AgentError):
    """Raised when a tool call cannot be constructed due to a schema validation
    failure (e.g. missing required arguments)."""

    def __init__(self, tool_name: str, message: str) -> None:
        super().__init__(f"{tool_name}: {message}")
        self.tool_name = tool_name
