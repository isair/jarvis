"""Local task queue for prompts submitted by interactive clients."""

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
    RUNNING = "running"
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
    ) -> None:
        self.db = db
        self.cfg = cfg
        self.dialogue_memory = dialogue_memory
        self.tts = tts
        self.event_callback = event_callback
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="jarvis-task")
        self._tasks: dict[str, Task] = {}
        self._futures: dict[str, Future] = {}
        self._cancel_events: dict[str, threading.Event] = {}
        self._lock = threading.RLock()
        self._restore()

    def submit(self, prompt: str) -> str:
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("Task prompt cannot be empty")
        task = Task(id=uuid.uuid4().hex[:12], prompt=prompt)
        with self._lock:
            self._tasks[task.id] = task
            self._cancel_events[task.id] = threading.Event()
            self._persist(task)
            self._emit(task)
            self._futures[task.id] = self._executor.submit(self._run, task.id)
        debug_log(f"task submitted: {task.id}", "tasks")
        return task.id

    def get(self, task_id: str) -> Optional[Task]:
        with self._lock:
            return self._tasks.get(task_id)

    def list_tasks(self) -> list[Task]:
        with self._lock:
            return sorted(self._tasks.values(), key=lambda task: task.created_at, reverse=True)

    def cancel(self, task_id: str) -> bool:
        with self._lock:
            task = self._tasks.get(task_id)
            future = self._futures.get(task_id)
            if task is None or task.status in {
                TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED,
            }:
                return False
            if future is not None:
                future.cancel()
            self._cancel_events.setdefault(task.id, threading.Event()).set()
            task.status = TaskStatus.CANCELLED
            task.completed_at = time.time()
            self._persist(task)
            self._emit(task)
            debug_log(f"task cancelled: {task.id}", "tasks")
            return True

    def wait_for_idle(self, timeout: float = 30.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if all(task.status not in {TaskStatus.QUEUED, TaskStatus.RUNNING} for task in self._tasks.values()):
                    return
            time.sleep(0.01)
        raise TimeoutError("Timed out waiting for task queue")

    def shutdown(self, wait: bool = True) -> None:
        with self._lock:
            for task_id, task in self._tasks.items():
                if task.status in {TaskStatus.QUEUED, TaskStatus.RUNNING}:
                    self._cancel_events.setdefault(task_id, threading.Event()).set()
                    debug_log(f"task stop requested during shutdown: {task_id}", "tasks")
        self._executor.shutdown(wait=wait, cancel_futures=True)

    def _run(self, task_id: str) -> None:
        with self._lock:
            task = self._tasks[task_id]
            if task.status is TaskStatus.CANCELLED:
                return
            task.status = TaskStatus.RUNNING
            task.started_at = time.time()
            cancel_event = self._cancel_events.setdefault(task.id, threading.Event())
            self._persist(task)
            self._emit(task)
        try:
            result = run_reply_engine(
                self.db, self.cfg, self.tts, task.prompt, self.dialogue_memory,
                cancel_event=cancel_event,
            )
            with self._lock:
                if task.status is not TaskStatus.CANCELLED:
                    task.status = TaskStatus.COMPLETED
                    task.result = result or ""
                    task.completed_at = time.time()
                    self._persist(task)
                    self._emit(task)
                else:
                    self._persist(task)
        except Exception as exc:
            with self._lock:
                if task.status is not TaskStatus.CANCELLED:
                    task.status = TaskStatus.FAILED
                    task.error = str(exc)
                    task.completed_at = time.time()
                    self._persist(task)
                    self._emit(task)
            debug_log(f"task failed: {task_id}: {exc}", "tasks")

    def _emit(self, task: Task) -> None:
        if self.event_callback is None:
            return
        try:
            self.event_callback(task.as_dict())
        except Exception as exc:
            debug_log(f"task event callback failed: {exc}", "tasks")

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
            task = Task(
                id=row["id"], prompt=row["prompt"],
                status=TaskStatus(row["status"]),
                result=row["result"], error=row["error"],
                created_at=row["created_at"], started_at=row["started_at"],
                completed_at=row["completed_at"],
            )
            if task.status in {TaskStatus.QUEUED, TaskStatus.RUNNING}:
                task.status = TaskStatus.FAILED
                task.error = "Task interrupted by daemon restart"
                task.completed_at = time.time()
                self._persist(task)
            self._tasks[task.id] = task
            self._cancel_events[task.id] = threading.Event()
            self._emit(task)
