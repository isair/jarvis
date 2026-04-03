"""
Tool runner — manages in-process and out-of-process tool execution.

The runner is the single execution funnel that tools pass through after
policy evaluation.  It decides, based on tool class and policy decision,
whether to execute in-process or spawn an isolated worker process.

Runner modes
------------
* ``IN_PROCESS``  — default for INFORMATIONAL, READ_ONLY_OPERATIONAL, and WRITE_OPERATIONAL tools.
* ``SUBPROCESS``  — forced for DESTRUCTIVE tools and optionally for WRITE_OPERATIONAL when
  the ``use_subprocess_for_writes`` flag is set in configuration.

Execution contract
------------------
Every execution attempt must:
1. Have a valid :class:`~jarvis.policy.models.PolicyDecision` (``allowed=True``).
2. Produce an :class:`ExecutionResult` regardless of internal failures.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from ..debug import debug_log
from ..policy.models import PolicyDecision, ToolClass


class RunnerMode(Enum):
    """Execution isolation mode."""
    IN_PROCESS = "in_process"
    SUBPROCESS = "subprocess"


@dataclass
class ExecutionResult:
    """Unified result from either in-process or subprocess execution."""
    success: bool
    reply_text: str = ""
    error_message: Optional[str] = None
    runner_mode: RunnerMode = RunnerMode.IN_PROCESS
    duration_ms: float = 0.0
    retry_count: int = 0
    execution_id: str = field(default_factory=lambda: uuid.uuid4().hex)


class ToolRunner:
    """
    Executes tools in-process or out-of-process based on policy.

    Args:
        cfg: Settings object.
        policy_decision_fn: Optional callable ``(tool_name, args) → PolicyDecision``
            that re-evaluates policy immediately before execution (defence-in-depth).
    """

    # Tools that are always isolated in a subprocess regardless of config
    _ALWAYS_SUBPROCESS: frozenset = frozenset()  # Populated from ToolClass.DESTRUCTIVE at runtime

    def __init__(self, cfg, policy_decision_fn=None) -> None:
        self._cfg = cfg
        self._policy_fn = policy_decision_fn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        tool_name: str,
        tool_args: Optional[Dict[str, Any]],
        *,
        context,
        decision: Optional[PolicyDecision] = None,
        max_retries: int = 2,
    ) -> ExecutionResult:
        """
        Execute *tool_name* with *tool_args* under the runner's isolation policy.

        Args:
            tool_name: Canonical tool identifier.
            tool_args: Arguments for the tool.
            context: :class:`~jarvis.tools.base.ToolContext` for in-process execution.
            decision: Pre-computed :class:`~jarvis.policy.models.PolicyDecision`.
                      If ``None``, the runner will call the policy function if set.
            max_retries: Number of retry attempts on transient failures.

        Returns:
            :class:`ExecutionResult` describing the outcome.
        """
        start = time.monotonic()
        retries = 0

        mode = self._choose_mode(tool_name, decision)

        last_error: Optional[str] = None
        for attempt in range(max_retries + 1):
            retries = attempt
            try:
                if mode == RunnerMode.SUBPROCESS:
                    result = self._run_subprocess(tool_name, tool_args or {})
                else:
                    result = self._run_in_process(tool_name, tool_args, context)
                duration = (time.monotonic() - start) * 1000
                result.runner_mode = mode
                result.duration_ms = duration
                result.retry_count = attempt
                return result
            except Exception as exc:
                last_error = str(exc)
                debug_log(f"runner: attempt {attempt + 1} failed for {tool_name}: {exc}", "runner")
                if attempt < max_retries:
                    time.sleep(0.5 * (attempt + 1))

        duration = (time.monotonic() - start) * 1000
        return ExecutionResult(
            success=False,
            error_message=last_error or "Unknown error",
            runner_mode=mode,
            duration_ms=duration,
            retry_count=retries,
        )

    # ------------------------------------------------------------------
    # Mode selection
    # ------------------------------------------------------------------

    def _choose_mode(
        self, tool_name: str, decision: Optional[PolicyDecision]
    ) -> RunnerMode:
        """Determine execution mode for *tool_name*."""
        use_subprocess_for_writes = getattr(
            self._cfg, "use_subprocess_for_writes", False
        )

        if decision is not None:
            if decision.tool_class == ToolClass.DESTRUCTIVE:
                return RunnerMode.SUBPROCESS
            if use_subprocess_for_writes and decision.tool_class == ToolClass.WRITE_OPERATIONAL:
                return RunnerMode.SUBPROCESS

        return RunnerMode.IN_PROCESS

    # ------------------------------------------------------------------
    # In-process execution
    # ------------------------------------------------------------------

    def _run_in_process(
        self,
        tool_name: str,
        tool_args: Optional[Dict[str, Any]],
        context,
    ) -> ExecutionResult:
        """Execute the tool within the current process."""
        from ..tools.registry import BUILTIN_TOOLS, get_cached_mcp_tools, run_mcp_tool

        if "__" in tool_name:
            # MCP tool
            mcp_tools = get_cached_mcp_tools()
            if tool_name not in mcp_tools:
                return ExecutionResult(
                    success=False,
                    error_message=f"MCP tool not found: {tool_name}",
                )
            try:
                mcp_result = run_mcp_tool(tool_name, tool_args or {}, mcp_tools)
                return ExecutionResult(
                    success=mcp_result.success,
                    reply_text=mcp_result.reply_text or "",
                    error_message=mcp_result.error_message,
                )
            except Exception as exc:
                return ExecutionResult(success=False, error_message=str(exc))

        tool = BUILTIN_TOOLS.get(tool_name)
        if tool is None:
            return ExecutionResult(
                success=False, error_message=f"Unknown built-in tool: {tool_name}"
            )

        raw = tool.run(tool_args, context)
        return ExecutionResult(
            success=raw.success,
            reply_text=raw.reply_text or "",
            error_message=raw.error_message,
        )

    # ------------------------------------------------------------------
    # Subprocess execution
    # ------------------------------------------------------------------

    def _run_subprocess(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        timeout_sec: float = 30.0,
    ) -> ExecutionResult:
        """Spawn an isolated worker process and execute the tool there."""
        from .worker_protocol import WorkerRequest, WorkerResponse

        # Forward path-safety constraints so the worker can enforce them
        # without a full cfg object (see WorkerRequest.safety_config).
        safety_config = {
            "workspace_roots": list(getattr(self._cfg, "workspace_roots", None) or []),
            "blocked_roots": list(getattr(self._cfg, "blocked_roots", None) or []),
            "read_only_roots": list(getattr(self._cfg, "read_only_roots", None) or []),
            "local_files_mode": str(getattr(self._cfg, "local_files_mode", "workspace")),
        }
        req = WorkerRequest(
            tool_name=tool_name,
            tool_args=tool_args,
            request_id=uuid.uuid4().hex,
            timeout_sec=timeout_sec,
            safety_config=safety_config,
        )

        python_exe = sys.executable
        worker_module = "jarvis.execution.subprocess_worker"

        debug_log(f"runner: spawning subprocess for {tool_name}", "runner")

        try:
            proc = subprocess.Popen(
                [python_exe, "-m", worker_module],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            stdout, stderr = proc.communicate(
                input=req.to_json() + "\n",
                timeout=timeout_sec + 5,
            )
        except subprocess.TimeoutExpired:
            proc.kill()
            return ExecutionResult(
                success=False,
                error_message=f"Subprocess worker timed out after {timeout_sec}s",
            )
        except Exception as exc:
            return ExecutionResult(
                success=False,
                error_message=f"Failed to spawn subprocess worker: {exc}",
            )

        if stderr:
            debug_log(f"runner: worker stderr: {stderr[:200]}", "runner")

        if not stdout.strip():
            return ExecutionResult(
                success=False,
                error_message=f"Subprocess worker produced no output (exit={proc.returncode})",
            )

        try:
            resp = WorkerResponse.from_json(stdout.strip())
            return ExecutionResult(
                success=resp.success,
                reply_text=resp.reply_text,
                error_message=resp.error_message,
                runner_mode=RunnerMode.SUBPROCESS,
            )
        except Exception as exc:
            return ExecutionResult(
                success=False,
                error_message=f"Failed to parse worker response: {exc}",
            )
