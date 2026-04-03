"""
Task State – session-scoped execution tracker.

Maintains the active task state during multi-step workflow execution,
supporting resumption within a session and providing clear execution
visibility for the desktop console.

Design principles:
- Stays in memory for the lifetime of the session (no persistence by default)
- Thread-safe via a simple lock
- Lightweight: does not drive execution, only observes and records it
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from .debug import debug_log


class TaskStatus(Enum):
    """Overall lifecycle of a task."""
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    AWAITING_APPROVAL = "awaiting_approval"
    DONE = "done"
    FAILED = "failed"


class StepStatus(Enum):
    """Lifecycle of a single execution step."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class TaskStep:
    """A single step in an execution plan."""
    description: str
    tool_name: Optional[str] = None
    status: StepStatus = StepStatus.PENDING
    result_summary: Optional[str] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None

    def start(self) -> None:
        self.status = StepStatus.RUNNING
        self.started_at = time.time()

    def complete(self, result_summary: Optional[str] = None) -> None:
        self.status = StepStatus.SUCCEEDED
        self.result_summary = result_summary
        self.finished_at = time.time()

    def fail(self, reason: Optional[str] = None) -> None:
        self.status = StepStatus.FAILED
        self.result_summary = reason
        self.finished_at = time.time()

    def skip(self, reason: Optional[str] = None) -> None:
        self.status = StepStatus.SKIPPED
        self.result_summary = reason
        self.finished_at = time.time()


@dataclass
class TaskState:
    """
    Session-scoped state for the currently executing task.

    Tracks the active intent, execution steps, and overall status so
    that the desktop console can display real-time progress and the
    engine can detect resumption opportunities.
    """
    intent: str = ""
    status: TaskStatus = TaskStatus.IDLE
    steps: List[TaskStep] = field(default_factory=list)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: Optional[str] = None

    def begin(self, intent: str) -> None:
        """Start tracking a new task."""
        self.intent = intent
        self.status = TaskStatus.PLANNING
        self.steps = []
        self.started_at = time.time()
        self.finished_at = None
        self.error = None
        debug_log(f"task started: {intent[:80]}", "task")

    def set_executing(self) -> None:
        """Transition from planning to active execution."""
        self.status = TaskStatus.EXECUTING
        debug_log("task executing", "task")

    def set_awaiting_approval(self) -> None:
        """Pause execution pending user approval."""
        self.status = TaskStatus.AWAITING_APPROVAL
        debug_log("task awaiting approval", "task")

    def add_step(self, description: str, tool_name: Optional[str] = None) -> TaskStep:
        """Add and return a new pending step."""
        step = TaskStep(description=description, tool_name=tool_name)
        self.steps.append(step)
        debug_log(f"step added: {description[:60]}", "task")
        return step

    def complete(self) -> None:
        """Mark the task as successfully completed."""
        self.status = TaskStatus.DONE
        self.finished_at = time.time()
        debug_log("task done", "task")

    def fail(self, reason: Optional[str] = None) -> None:
        """Mark the task as failed."""
        self.status = TaskStatus.FAILED
        self.error = reason
        self.finished_at = time.time()
        debug_log(f"task failed: {reason}", "task")

    def reset(self) -> None:
        """Return to idle state (between conversations)."""
        self.intent = ""
        self.status = TaskStatus.IDLE
        self.steps = []
        self.started_at = None
        self.finished_at = None
        self.error = None
        debug_log("task reset to idle", "task")

    def can_resume(self) -> bool:
        """
        Returns True when a task was interrupted and has pending steps
        that could be continued in the same session.
        """
        if self.status not in (TaskStatus.EXECUTING, TaskStatus.AWAITING_APPROVAL):
            return False
        return any(s.status == StepStatus.PENDING for s in self.steps)

    @property
    def completed_steps(self) -> List[TaskStep]:
        return [s for s in self.steps if s.status == StepStatus.SUCCEEDED]

    @property
    def failed_steps(self) -> List[TaskStep]:
        return [s for s in self.steps if s.status == StepStatus.FAILED]

    def summary(self) -> str:
        """Human-readable summary suitable for debug output."""
        parts = [f"Task: {self.intent[:60]}", f"Status: {self.status.value}"]
        if self.steps:
            parts.append(f"Steps: {len(self.completed_steps)}/{len(self.steps)} completed")
        if self.error:
            parts.append(f"Error: {self.error[:60]}")
        return " | ".join(parts)


# ---------------------------------------------------------------------------
# Module-level singleton – one active task per daemon session
# ---------------------------------------------------------------------------

_active_task: TaskState = TaskState()
_task_lock = threading.Lock()


def get_active_task() -> TaskState:
    """Return the singleton active task state (thread-safe read)."""
    with _task_lock:
        return _active_task


def begin_task(intent: str) -> TaskState:
    """Begin a new task, resetting any previous state."""
    with _task_lock:
        _active_task.begin(intent)
        return _active_task


def reset_task() -> None:
    """Reset the active task to idle."""
    with _task_lock:
        _active_task.reset()
