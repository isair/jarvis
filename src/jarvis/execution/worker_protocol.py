"""
Worker protocol — JSON-serialisable request/response types for
subprocess tool execution.

Both classes are simple dataclasses that serialise to/from dicts so
they can be sent over a subprocess stdin/stdout pipe without any
special IPC dependencies.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class WorkerRequest:
    """Command sent from the main process to the worker subprocess."""

    tool_name: str
    """Canonical tool name."""

    tool_args: Dict[str, Any] = field(default_factory=dict)
    """Arguments to pass to the tool."""

    request_id: str = ""
    """Correlation ID echoed back in the response."""

    timeout_sec: float = 30.0
    """Maximum execution time after which the worker should self-terminate."""

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_dict(cls, d: dict) -> "WorkerRequest":
        return cls(
            tool_name=d.get("tool_name", ""),
            tool_args=d.get("tool_args", {}),
            request_id=d.get("request_id", ""),
            timeout_sec=float(d.get("timeout_sec", 30.0)),
        )

    @classmethod
    def from_json(cls, s: str) -> "WorkerRequest":
        return cls.from_dict(json.loads(s))


@dataclass
class WorkerResponse:
    """Response sent from the worker subprocess to the main process."""

    request_id: str
    success: bool
    reply_text: str = ""
    error_message: Optional[str] = None
    exit_code: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_dict(cls, d: dict) -> "WorkerResponse":
        return cls(
            request_id=d.get("request_id", ""),
            success=bool(d.get("success", False)),
            reply_text=d.get("reply_text", ""),
            error_message=d.get("error_message"),
            exit_code=int(d.get("exit_code", 0)),
        )

    @classmethod
    def from_json(cls, s: str) -> "WorkerResponse":
        return cls.from_dict(json.loads(s))
