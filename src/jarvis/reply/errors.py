"""
Formal exception hierarchy for the agent loop.

These exceptions are raised by the reply engine when specific failure
conditions are encountered.  Callers can catch individual classes to
apply targeted recovery logic.
"""

from __future__ import annotations

from typing import Optional


class AgentError(RuntimeError):
    """Base class for all agent loop errors."""


class ModelOutputError(AgentError):
    """
    Raised when the LLM returns output that cannot be processed.

    This includes:
    - Empty responses where a response was expected.
    - Malformed JSON that cannot be recovered.
    - Hallucinated API spec instead of conversational text.
    """

    def __init__(self, message: str, raw_content: Optional[str] = None) -> None:
        super().__init__(message)
        self.raw_content = raw_content


class ToolSchemaError(AgentError):
    """
    Raised when the LLM requests a tool with arguments that do not
    conform to the tool's declared JSON Schema.
    """

    def __init__(self, tool_name: str, reason: str) -> None:
        super().__init__(f"Tool schema error for '{tool_name}': {reason}")
        self.tool_name = tool_name
        self.reason = reason


class PolicyDeniedError(AgentError):
    """
    Raised when the policy engine blocks a tool invocation.

    Note: :class:`jarvis.policy.models.PolicyDeniedError` is the primary
    policy exception.  This subclass is re-raised from the agent loop so
    callers only need to import from this module.
    """

    def __init__(self, tool_name: str, reason: str) -> None:
        super().__init__(f"Policy denied tool '{tool_name}': {reason}")
        self.tool_name = tool_name
        self.reason = reason


class ApprovalRequiredError(AgentError):
    """
    Raised when a tool requires explicit user approval before executing
    and no prior grant covers the request.
    """

    def __init__(self, tool_name: str, prompt: str) -> None:
        super().__init__(f"Approval required for '{tool_name}': {prompt}")
        self.tool_name = tool_name
        self.prompt = prompt


class ToolExecutionError(AgentError):
    """
    Raised when a tool invocation returns an unexpected error after
    all retries have been exhausted.
    """

    def __init__(self, tool_name: str, reason: str, retry_count: int = 0) -> None:
        super().__init__(
            f"Tool '{tool_name}' failed after {retry_count} retries: {reason}"
        )
        self.tool_name = tool_name
        self.reason = reason
        self.retry_count = retry_count


class LoopExhaustedError(AgentError):
    """
    Raised when the agentic loop reaches ``agentic_max_turns`` without
    producing a final response.
    """

    def __init__(self, max_turns: int) -> None:
        super().__init__(f"Agent loop exhausted after {max_turns} turns without a response.")
        self.max_turns = max_turns
