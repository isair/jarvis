"""Private local task journal and complete tool-result storage."""

from __future__ import annotations

import json
import hashlib
import os
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..memory.conversation import scrub_secrets
from ..utils.redact import redact


_store_lock = threading.RLock()


@dataclass(frozen=True)
class TaskRecord:
    task_id: str
    completed_write_signatures: set[str]


class TaskStore:
    """Persist compact outcomes and full evidence beside the configured DB."""

    def __init__(self, db_path: str | Path) -> None:
        self.root = Path(db_path).expanduser().resolve().parent / ".jarvis_tasks"
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.results_dir = self.root / "results"
        self.results_dir.mkdir(mode=0o700, exist_ok=True)

    @staticmethod
    def _safe_id(value: str) -> str:
        return uuid.UUID(value).hex

    def _task_path(self, task_id: str) -> Path:
        return self.root / f"{self._safe_id(task_id)}.json"

    def _result_path(self, result_id: str) -> Path:
        return self.results_dir / f"{self._safe_id(result_id)}.txt"

    @staticmethod
    def signature_digest(signature: str) -> str:
        """Keep write idempotency keys without storing argument text."""
        return hashlib.sha256(signature.encode("utf-8")).hexdigest()

    @staticmethod
    def _atomic_text(path: Path, content: str) -> None:
        descriptor, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def _load(self, task_id: str) -> dict:
        return json.loads(self._task_path(task_id).read_text(encoding="utf-8"))

    def _save(self, task: dict) -> None:
        task["updated_at"] = time.time()
        self._atomic_text(
            self._task_path(task["task_id"]),
            json.dumps(task, ensure_ascii=False, separators=(",", ":")),
        )

    @staticmethod
    def _record(task: dict) -> TaskRecord:
        return TaskRecord(
            task_id=task["task_id"],
            completed_write_signatures={
                result["signature"] for result in task["results"]
                if result["success"] and result["mutating"]
            },
        )

    def begin(self, objective: str, steps: list[str]) -> TaskRecord:
        task_id = uuid.uuid4().hex
        task = {
            "task_id": task_id,
            "objective": redact(objective),
            "steps": [
                {"text": redact(step), "status": "pending"} for step in steps
            ],
            "results": [],
            "missing_info": [],
            "status": "active",
            "created_at": time.time(),
        }
        with _store_lock:
            self._save(task)
        return self._record(task)

    def resume(self, task_id: str) -> TaskRecord:
        with _store_lock:
            task = self._load(task_id)
            task["status"] = "active"
            self._save(task)
            return self._record(task)

    def record_result(
        self,
        task_id: str,
        *,
        step_index: Optional[int],
        tool_name: str,
        success: bool,
        full_text: str,
        signature: str,
        mutating: bool,
    ) -> str:
        result_id = uuid.uuid4().hex
        full_text = str(full_text)
        with _store_lock:
            task = self._load(task_id)
            self._atomic_text(self._result_path(result_id), full_text)
            preview = scrub_secrets(full_text[:240]).replace("\n", " ").strip()
            task["results"].append({
                "result_id": result_id,
                "step_index": step_index,
                "tool_name": tool_name,
                "success": bool(success),
                "preview": preview[:180],
                "signature": self.signature_digest(signature),
                "mutating": bool(mutating),
            })
            if step_index is not None and 0 <= step_index < len(task["steps"]):
                task["steps"][step_index]["status"] = "done" if success else "failed"
            self._save(task)
        return result_id

    def finish(self, task_id: str, *, status: str, missing_info: list[str]) -> None:
        with _store_lock:
            task = self._load(task_id)
            task["status"] = status
            task["missing_info"] = [redact(item) for item in missing_info]
            self._save(task)

    def latest_incomplete(self) -> Optional[dict]:
        with _store_lock:
            candidates: list[dict] = []
            for path in self.root.glob("*.json"):
                try:
                    task = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                if task.get("status") in {"active", "interrupted", "deadline", "partial"}:
                    candidates.append(task)
            if not candidates:
                return None
            return max(candidates, key=lambda task: task.get("updated_at", 0))

    def compact_context(self, task_id: str, *, max_chars: int = 3500) -> str:
        with _store_lock:
            task = self._load(task_id)
        lines = [
            f"Task {task['task_id']} ({task['status']}): {task['objective']}",
            "This is prior task context. Choose next actions explicitly; do not repeat completed writes.",
        ]
        lines.extend(
            f"{index + 1}. [{step['status']}] {step['text']}"
            for index, step in enumerate(task["steps"])
        )
        lines.extend(
            f"{result['tool_name']}: {'ok' if result['success'] else 'failed'} "
            f"(result ID {result['result_id']}): {result['preview']}"
            for result in task["results"][-10:]
        )
        if task["missing_info"]:
            lines.append("Missing information: " + "; ".join(task["missing_info"]))
        return "\n".join(lines)[:max_chars]

    def read_result(self, result_id: str, *, offset: int = 0, limit: int = 4000) -> dict:
        if offset < 0 or limit < 1:
            raise ValueError("offset and limit must be positive")
        limit = min(limit, 4000)
        content = self._result_path(result_id).read_text(encoding="utf-8")
        return {
            "result_id": self._safe_id(result_id),
            "text": content[offset:offset + limit],
            "offset": offset,
            "total_chars": len(content),
            "has_more": offset + limit < len(content),
        }
