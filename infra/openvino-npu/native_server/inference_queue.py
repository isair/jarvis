from __future__ import annotations

import itertools
import queue
import threading
import time
from collections.abc import Callable
from typing import Any


class InferenceQueue:
    """One active NPU workload with priority 0 (rerank), 1 (query), 2 (index)."""

    MAX_CONSECUTIVE_RERANK_JOBS = 8

    def __init__(self) -> None:
        self._items: queue.PriorityQueue[tuple[int, int, Callable[[], Any], queue.Queue[Any]]] = queue.PriorityQueue()
        self._sequence = itertools.count()
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._run, name="openvino-npu-worker", daemon=True)
        self._worker.start()
        self.wait_count = 0
        self._consecutive_rerank_jobs = 0

    def submit(self, priority: int, operation: Callable[[], Any]) -> Any:
        return self.submit_timed(priority, operation)[0]

    def submit_timed(self, priority: int, operation: Callable[[], Any]) -> tuple[Any, float, float]:
        result: queue.Queue[Any] = queue.Queue(maxsize=1)
        submitted = time.perf_counter()
        self._items.put((priority, next(self._sequence), operation, result))
        self.wait_count += 1
        value = result.get()
        if isinstance(value, BaseException):
            raise value
        operation_result, started = value
        return operation_result, started - submitted, time.perf_counter() - started

    def close(self) -> None:
        self._stop.set()
        self._items.put((99, next(self._sequence), lambda: None, queue.Queue(maxsize=1)))
        self._worker.join(timeout=5)

    def _run(self) -> None:
        while not self._stop.is_set():
            priority, _, operation, result = self._next_item()
            if self._stop.is_set():
                return
            try:
                started = time.perf_counter()
                if priority == 0:
                    # Fairness is based on NPU-slot admission, not success.
                    self._consecutive_rerank_jobs += 1
                elif priority == 1:
                    self._consecutive_rerank_jobs = 0
                result.put((operation(), started))
            except BaseException as error:
                result.put(error)

    def _next_item(self) -> tuple[int, int, Callable[[], Any], queue.Queue[Any]]:
        first = self._items.get()
        if self._consecutive_rerank_jobs < self.MAX_CONSECUTIVE_RERANK_JOBS or first[0] != 0:
            return first
        pending = [first]
        while True:
            try:
                pending.append(self._items.get_nowait())
            except queue.Empty:
                break
        embeddings = [item for item in pending if item[0] == 1]
        if embeddings:
            selected = min(embeddings, key=lambda item: item[1])
            pending.remove(selected)
            for item in pending:
                self._items.put(item)
            return selected
        for item in pending[1:]:
            self._items.put(item)
        return pending[0]
