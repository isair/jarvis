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
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from .debug import debug_log


class TaskStatus(Enum):
    """Overall lifecycle of a task."""
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    REVERSIBLE = "reversible"   # Completed; last action(s) can still be undone
    DONE = "done"
    FAILED = "failed"

    # Backwards-compat alias so existing callers don't break during migration
    AWAITING_APPROVAL = "reversible"


class StepStatus(Enum):
    """Lifecycle of a single execution step."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    REVERSIBLE = "reversible"   # Succeeded; registered in UndoRegistry
    REVERSED = "reversed"       # Was undone by the user
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
    # Unique identifier — used to link this step to an UndoEntry
    step_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    # Set to True when a matching UndoEntry has been pushed to the registry
    reversible: bool = False
    undo_entry_id: Optional[str] = None

    def start(self) -> None:
        self.status = StepStatus.RUNNING
        self.started_at = time.time()

    def complete(self, result_summary: Optional[str] = None) -> None:
        self.status = StepStatus.SUCCEEDED
        self.result_summary = result_summary
        self.finished_at = time.time()

    def mark_reversible(self, undo_entry_id: str) -> None:
        """
        Transition a completed step to REVERSIBLE status.

        Called by the engine after pushing an UndoEntry to the registry
        so that the task log reflects that this step can still be undone.
        """
        self.status = StepStatus.REVERSIBLE
        self.reversible = True
        self.undo_entry_id = undo_entry_id
        debug_log(f"step marked reversible: {self.description[:60]}", "task")

    def mark_reversed(self, result_summary: Optional[str] = None) -> None:
        """
        Transition a reversible step to REVERSED status.

        Called by the engine after a successful undo execution so the
        task history accurately reflects that this step was undone.
        """
        self.status = StepStatus.REVERSED
        self.result_summary = result_summary or "undone by user"
        debug_log(f"step reversed: {self.description[:60]}", "task")

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

    def set_reversible(self) -> None:
        """
        Mark the task as completed with at least one reversible action.

        Set this instead of ``complete()`` when the engine has pushed one
        or more ``UndoEntry`` objects to the ``UndoRegistry``.
        """
        self.status = TaskStatus.REVERSIBLE
        self.finished_at = time.time()
        debug_log("task done (reversible actions registered)", "task")

    # Backwards-compat aliases
    def set_awaiting_approval(self) -> None:
        """Deprecated — use set_reversible() instead."""
        self.set_reversible()

    def set_approved(self) -> None:
        """Deprecated — previously resumed approval-gated execution."""
        if self.status == TaskStatus.REVERSIBLE:
            self.status = TaskStatus.EXECUTING
            debug_log("task approved (compat) — resuming execution", "task")

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
        if self.status not in (TaskStatus.EXECUTING,):
            return False
        return any(s.status == StepStatus.PENDING for s in self.steps)

    def can_undo(self) -> bool:
        """
        Returns True when the task completed with at least one reversible
        step that has not yet been undone or expired.
        """
        return self.status == TaskStatus.REVERSIBLE and any(
            s.status == StepStatus.REVERSIBLE for s in self.steps
        )

    @property
    def completed_steps(self) -> List[TaskStep]:
        return [
            s for s in self.steps
            if s.status in (StepStatus.SUCCEEDED, StepStatus.REVERSIBLE)
        ]

    @property
    def reversible_steps(self) -> List[TaskStep]:
        """Steps that completed successfully and can still be undone."""
        return [s for s in self.steps if s.status == StepStatus.REVERSIBLE]

    @property
    def failed_steps(self) -> List[TaskStep]:
        return [s for s in self.steps if s.status == StepStatus.FAILED]

    def summary(self) -> str:
        """Human-readable summary suitable for debug output."""
        parts = [f"Task: {self.intent[:60]}", f"Status: {self.status.value}"]
        if self.steps:
            parts.append(f"Steps: {len(self.completed_steps)}/{len(self.steps)} completed")
        rev = len(self.reversible_steps)
        if rev:
            parts.append(f"Reversible: {rev}")
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
