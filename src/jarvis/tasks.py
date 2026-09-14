"""Local task queue with persisted one-off and recurring schedules."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
import uuid
from typing import Callable, Optional

from .debug import debug_log
from .utils.redact import redact


def run_reply_engine(*args, **kwargs):
    """Load the reply engine only when a task actually starts."""
    from .reply.engine import run_reply_engine as _run_reply_engine
    return _run_reply_engine(*args, **kwargs)


class TaskStatus(str, Enum):
    QUEUED = "queued"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    PENDING_APPROVAL = "pending_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    id: str
    prompt: str
    status: TaskStatus = TaskStatus.QUEUED
    result: Optional[str] = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    action_summary: Optional[str] = None
    action_reason: Optional[str] = None
    action_risk: Optional[str] = None
    next_run_at: Optional[float] = None
    recurrence: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "action_summary": self.action_summary,
            "action_reason": self.action_reason,
            "action_risk": self.action_risk,
            "next_run_at": self.next_run_at,
            "recurrence": self.recurrence,
        }


class TaskManager:
    """Execute prompts through the normal reply engine in a bounded queue."""

    def __init__(
        self,
        db,
        cfg,
        dialogue_memory,
        tts=None,
        event_callback: Optional[Callable[[dict], None]] = None,
        max_workers: int = 1,
        poll_interval: float = 1.0,
    ) -> None:
        self.db = db
        self.cfg = cfg
        self.dialogue_memory = dialogue_memory
        self.tts = tts
        self.event_callback = event_callback
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="jarvis-task"
        )
        self._tasks: dict[str, Task] = {}
        self._futures: dict[str, Future] = {}
        self._cancel_events: dict[str, threading.Event] = {}
        self._lock = threading.RLock()
        self._approval_conditions: dict[str, threading.Condition] = {}
        self._approval_decisions: dict[str, Optional[bool]] = {}
        self._poll_interval = poll_interval
        self._stop_event = threading.Event()
        self._restore()
        self._scheduler = threading.Thread(
            target=self._poll_schedules,
            name="jarvis-task-scheduler",
            daemon=True,
        )
        self._scheduler.start()

    def submit(
        self,
        prompt: str,
        run_at: Optional[float] = None,
        recurrence: Optional[str] = None,
    ) -> str:
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("Task prompt cannot be empty")
        self._validate_schedule(run_at, recurrence)
        scheduled = run_at is not None
        task = Task(
            id=uuid.uuid4().hex[:12],
            prompt=prompt,
            status=TaskStatus.SCHEDULED if scheduled else TaskStatus.QUEUED,
            next_run_at=float(run_at) if scheduled else None,
            recurrence=recurrence,
        )
        with self._lock:
            self._tasks[task.id] = task
            self._cancel_events[task.id] = threading.Event()
            self._persist(task)
            self._emit(task)
            if not scheduled:
                self._futures[task.id] = self._executor.submit(self._run, task.id)
        debug_log(f"task submitted: {task.id}", "tasks")
        return task.id

    def reschedule(
        self, task_id: str, run_at: float, recurrence: Optional[str] = None
    ) -> bool:
        self._validate_schedule(run_at, recurrence)
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.status is not TaskStatus.SCHEDULED:
                return False
            task.next_run_at = float(run_at)
            task.recurrence = recurrence
            self._persist(task)
            self._emit(task)
            debug_log(f"task rescheduled: {task.id}", "tasks")
            return True

    def get(self, task_id: str) -> Optional[Task]:
        with self._lock:
            return self._tasks.get(task_id)

    def list_tasks(self) -> list[Task]:
        with self._lock:
            return sorted(
                self._tasks.values(), key=lambda task: task.created_at, reverse=True
            )

    def cancel(self, task_id: str) -> bool:
        with self._lock:
            task = self._tasks.get(task_id)
            future = self._futures.get(task_id)
            if task is None or task.status in {
                TaskStatus.COMPLETED,
                TaskStatus.FAILED,
                TaskStatus.CANCELLED,
            }:
                return False
            if future is not None:
                future.cancel()
            self._cancel_events.setdefault(task.id, threading.Event()).set()
            task.status = TaskStatus.CANCELLED
            task.completed_at = time.time()
            condition = self._approval_conditions.get(task_id)
            if condition is not None:
                self._approval_decisions[task_id] = False
                self._cancel_events.setdefault(task_id, threading.Event()).set()
                condition.notify_all()
            self._persist(task)
            self._emit(task)
            debug_log(f"task cancelled: {task.id}", "tasks")
            return True

    def wait_for_idle(self, timeout: float = 30.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if all(
                    task.status
                    not in {
                        TaskStatus.QUEUED,
                        TaskStatus.RUNNING,
                        TaskStatus.PENDING_APPROVAL,
                    }
                    for task in self._tasks.values()
                ):
                    return
            time.sleep(0.01)
        raise TimeoutError("Timed out waiting for task queue")

    def shutdown(self, wait: bool = True) -> None:
        self._stop_event.set()
        self._scheduler.join(timeout=max(1.0, self._poll_interval * 2))
        with self._lock:
            for task_id, task in self._tasks.items():
                if task.status in {
                    TaskStatus.QUEUED,
                    TaskStatus.RUNNING,
                    TaskStatus.PENDING_APPROVAL,
                }:
                    self._cancel_events.setdefault(task_id, threading.Event()).set()
                    task.status = TaskStatus.CANCELLED
                    task.completed_at = time.time()
                    self._approval_decisions[task_id] = False
                    condition = self._approval_conditions.get(task_id)
                    if condition is not None:
                        condition.notify_all()
                    self._persist(task)
                    self._emit(task)
        self._executor.shutdown(wait=wait, cancel_futures=True)

    def approve(self, task_id: str) -> bool:
        return self._decide_approval(task_id, True)

    def reject(self, task_id: str) -> bool:
        return self._decide_approval(task_id, False)

    def _decide_approval(self, task_id: str, decision: bool) -> bool:
        with self._lock:
            task = self._tasks.get(task_id)
            condition = self._approval_conditions.get(task_id)
            if (
                task is None
                or task.status is not TaskStatus.PENDING_APPROVAL
                or condition is None
            ):
                return False
            self._approval_decisions[task_id] = decision
            condition.notify_all()
            debug_log(
                f"task local action {'approved' if decision else 'rejected'}: {task_id}",
                "tasks",
            )
            return True

    def _request_approval(self, task_id: str, request: dict) -> bool:
        with self._lock:
            task = self._tasks[task_id]
            if task.status is TaskStatus.CANCELLED:
                return False
            condition = threading.Condition(self._lock)
            self._approval_conditions[task_id] = condition
            self._approval_decisions[task_id] = None
            task.status = TaskStatus.PENDING_APPROVAL
            task.action_summary = str(request.get("summary", "Local action"))
            task.action_risk = str(
                request.get("risk", "Local action may affect the device.")
            )
            task.action_reason = str(request.get("reason", ""))
            self._persist(task)
            self._emit(task)
            debug_log(f"task awaiting local action approval: {task_id}", "tasks")
            while self._approval_decisions[task_id] is None:
                condition.wait()
            decision = bool(self._approval_decisions.pop(task_id))
            self._approval_conditions.pop(task_id, None)
            if task.status is not TaskStatus.CANCELLED:
                task.status = TaskStatus.RUNNING
                self._persist(task)
                self._emit(task)
            return decision

    def _run(self, task_id: str) -> None:
        with self._lock:
            task = self._tasks[task_id]
            if task.status is TaskStatus.CANCELLED:
                return
            task.status = TaskStatus.RUNNING
            task.started_at = time.time()
            cancel_event = self._cancel_events.setdefault(
                task.id, threading.Event()
            )
            self._persist(task)
            self._emit(task)
        try:
            result = run_reply_engine(
                self.db,
                self.cfg,
                self.tts,
                task.prompt,
                self.dialogue_memory,
                cancel_event=cancel_event,
                approval_callback=lambda request: self._request_approval(
                    task_id, request
                ),
            )
            with self._lock:
                if task.status is not TaskStatus.CANCELLED:
                    task.result = result or ""
                    task.completed_at = time.time()
                    if task.recurrence:
                        task.next_run_at = self._next_occurrence(
                            task.next_run_at or time.time(), task.recurrence
                        )
                        task.status = TaskStatus.SCHEDULED
                    else:
                        task.next_run_at = None
                        task.status = TaskStatus.COMPLETED
                    self._persist(task)
                    self._emit(task)
        except Exception as exc:
            with self._lock:
                if task.status is not TaskStatus.CANCELLED:
                    task.status = TaskStatus.FAILED
                    task.error = str(exc)
                    task.completed_at = time.time()
                    self._persist(task)
                    self._emit(task)
            debug_log(f"task failed: {task_id}: {exc}", "tasks")

    def _poll_schedules(self) -> None:
        while not self._stop_event.wait(self._poll_interval):
            due: list[str] = []
            now = time.time()
            with self._lock:
                for task in self._tasks.values():
                    if (
                        task.status is TaskStatus.SCHEDULED
                        and task.next_run_at is not None
                        and task.next_run_at <= now
                    ):
                        task.status = TaskStatus.QUEUED
                        self._persist(task)
                        self._emit(task)
                        due.append(task.id)
                for task_id in due:
                    self._futures[task_id] = self._executor.submit(
                        self._run, task_id
                    )
            for task_id in due:
                debug_log(f"scheduled task is due: {task_id}", "tasks")

    @staticmethod
    def _next_occurrence(previous: float, recurrence: str) -> float:
        interval = 86400.0 if recurrence == "daily" else 604800.0
        next_run = previous + interval
        while next_run <= time.time():
            next_run += interval
        return next_run

    @staticmethod
    def _validate_schedule(
        run_at: Optional[float], recurrence: Optional[str]
    ) -> None:
        if run_at is None and recurrence is not None:
            raise ValueError("Recurrence requires a scheduled run time")
        if recurrence not in {None, "daily", "weekly"}:
            raise ValueError("Recurrence must be daily or weekly")

    def _persist(self, task: Task) -> None:
        if not hasattr(self.db, "upsert_task_record"):
            return
        record = task.as_dict()
        record["prompt"] = redact(task.prompt)
        if record["result"]:
            record["result"] = redact(str(record["result"]))
        if record["error"]:
            record["error"] = redact(str(record["error"]))
        try:
            self.db.upsert_task_record(record)
        except Exception as exc:
            debug_log(f"task persistence failed: {exc}", "tasks")

    def _restore(self) -> None:
        if not hasattr(self.db, "get_task_records"):
            return
        try:
            rows = self.db.get_task_records()
        except Exception as exc:
            debug_log(f"task restore failed: {exc}", "tasks")
            return
        for row in rows:
            keys = set(row.keys()) if hasattr(row, "keys") else set()
            get = lambda key, default=None: row[key] if key in keys else default
            task = Task(
                id=row["id"],
                prompt=row["prompt"],
                status=TaskStatus(row["status"]),
                result=row["result"],
                error=row["error"],
                created_at=row["created_at"],
                started_at=row["started_at"],
                completed_at=row["completed_at"],
                next_run_at=get("next_run_at"),
                recurrence=get("recurrence"),
            )
            if task.status in {
                TaskStatus.QUEUED,
                TaskStatus.RUNNING,
                TaskStatus.PENDING_APPROVAL,
            }:
                task.status = TaskStatus.FAILED
                task.error = "Task interrupted by daemon restart"
                task.completed_at = time.time()
                self._persist(task)
            self._tasks[task.id] = task
            self._cancel_events[task.id] = threading.Event()
            self._emit(task)

    def _emit(self, task: Task) -> None:
        if self.event_callback is None:
            return
        try:
            self.event_callback(task.as_dict())
        except Exception as exc:
            debug_log(f"task event callback failed: {exc}", "tasks")
