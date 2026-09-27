"""Bounded execution controls for one reply and its tool-call batches."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from ..debug import debug_log
from ..tools.types import ToolExecutionResult


class ExecutionCancelled(RuntimeError):
    """The user or daemon stopped this reply."""


class ExecutionDeadlineExceeded(RuntimeError):
    """The per-query wall-clock budget expired."""


class ExecutionControl:
    """One monotonic deadline shared by preparation, tools and reply output."""

    def __init__(
        self,
        cancel_event: threading.Event,
        timeout_sec: float,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.cancel_event = cancel_event
        self._clock = clock
        self.deadline = clock() + max(0.001, float(timeout_sec))
        self.task_store = None
        self.task_id: Optional[str] = None

    def check(self) -> None:
        if self.cancel_event.is_set():
            raise ExecutionCancelled("reply cancelled")
        if self._clock() >= self.deadline:
            raise ExecutionDeadlineExceeded("reply deadline exceeded")

    def remaining(self, cap: Optional[float] = None) -> float:
        self.check()
        remaining = self.deadline - self._clock()
        return min(remaining, float(cap)) if cap is not None else remaining

    def call(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Bound a blocking provider wait by this query's cancel/deadline.

        The underlying request may finish later. Callers must not use this
        to imply an already issued external side effect was rolled back.
        """
        self.check()
        completed: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

        def worker() -> None:
            if self.cancel_event.is_set():
                return
            try:
                completed.put((True, fn(*args, **kwargs)))
            except BaseException as exc:
                completed.put((False, exc))

        threading.Thread(target=worker, daemon=True, name="jarvis-provider-call").start()
        while True:
            self.check()
            try:
                success, value = completed.get(timeout=min(0.05, self.remaining()))
            except queue.Empty:
                continue
            self.check()
            if success:
                return value
            raise value


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict
    call_id: str


def is_read_only_call(call: ToolCall, mcp_tools: dict) -> bool:
    """Conservatively identify calls safe to overlap with independent reads."""
    if call.name in {"getTime", "getWeather", "webSearch", "fetchWebPage", "readTaskResult"}:
        return True
    if call.name == "localFiles":
        return call.args.get("operation") in {"read", "list"}
    spec = mcp_tools.get(call.name)
    return bool(spec is not None and getattr(spec, "read_only", False))


def is_mutating_call(call: ToolCall, mcp_tools: dict) -> bool:
    """Distinguish external writes from serial-only control/read operations."""
    return call.name not in {"stop", "toolSearchTool"} and not is_read_only_call(call, mcp_tools)


def tool_affinity(call: ToolCall) -> Optional[str]:
    """Calls sharing one persistent MCP session cannot overlap."""
    return call.name.split("__", 1)[0] if "__" in call.name else None


def _invoke_one(
    call: ToolCall,
    run: Callable[[ToolCall], ToolExecutionResult],
    control: ExecutionControl,
) -> ToolExecutionResult:
    control.check()
    completed: queue.Queue[ToolExecutionResult] = queue.Queue(maxsize=1)

    def worker() -> None:
        try:
            control.check()
            result = run(call)
        except (ExecutionCancelled, ExecutionDeadlineExceeded):
            return
        except Exception as exc:
            detail = str(exc) or type(exc).__name__
            debug_log(f"tool {call.name} failed: {detail}", "planning")
            result = ToolExecutionResult(False, None, detail)
        completed.put(result)

    threading.Thread(target=worker, daemon=True, name="jarvis-tool-call").start()
    while True:
        control.check()
        try:
            result = completed.get(timeout=min(0.05, control.remaining()))
            control.check()
            return result
        except queue.Empty:
            continue


def _run_parallel_reads(
    group: list[ToolCall],
    run: Callable[[ToolCall], ToolExecutionResult],
    control: ExecutionControl,
) -> list[ToolExecutionResult]:
    completed: queue.Queue[tuple[int, ToolExecutionResult]] = queue.Queue()

    def worker(index: int, call: ToolCall) -> None:
        try:
            result = run(call)
        except Exception as exc:
            detail = str(exc) or type(exc).__name__
            debug_log(f"tool {call.name} failed: {detail}", "planning")
            result = ToolExecutionResult(False, None, detail)
        completed.put((index, result))

    for index, call in enumerate(group):
        control.check()
        threading.Thread(
            target=worker, args=(index, call), daemon=True,
            name=f"jarvis-read-{index}",
        ).start()

    ordered: list[Optional[ToolExecutionResult]] = [None] * len(group)
    for _ in group:
        control.check()
        while True:
            try:
                index, result = completed.get(timeout=min(0.05, control.remaining()))
                ordered[index] = result
                break
            except queue.Empty:
                control.check()
    control.check()
    return [result for result in ordered if result is not None]


def execute_tool_batch(
    calls: list[ToolCall],
    run: Callable[[ToolCall], ToolExecutionResult],
    is_read_only: Callable[[ToolCall], bool],
    affinity: Callable[[ToolCall], Optional[str]],
    control: ExecutionControl,
    *,
    max_parallel_reads: int,
) -> list[ToolExecutionResult]:
    """Execute every model call, preserving order and write dependencies.

    Only consecutive independent reads overlap. A write is a barrier. Reads
    sharing a backend affinity (one MCP server session) also run serially.
    Already issued synchronous work can finish after cancellation, but no
    following call is issued and no cancelled result is returned to the loop.
    """
    results: list[ToolExecutionResult] = []
    group: list[ToolCall] = []
    affinities: set[str] = set()
    limit = max(1, int(max_parallel_reads))

    def flush() -> None:
        if not group:
            return
        if len(group) == 1:
            results.append(_invoke_one(group[0], run, control))
        else:
            results.extend(_run_parallel_reads(group, run, control))
        group.clear()
        affinities.clear()

    for call in calls:
        control.check()
        if not is_read_only(call):
            flush()
            results.append(_invoke_one(call, run, control))
            continue
        key = affinity(call)
        if len(group) >= limit or (key is not None and key in affinities):
            flush()
        group.append(call)
        if key is not None:
            affinities.add(key)
    flush()
    return results
