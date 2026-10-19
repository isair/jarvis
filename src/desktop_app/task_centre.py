"""Interactive desktop control centre for local Jarvis tasks."""

from __future__ import annotations

import json
import uuid
from typing import Callable, Optional

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
)

from desktop_app.themes import JARVIS_THEME_STYLESHEET
from jarvis.tasks import DEFAULT_TASK_HISTORY, TASK_EVENT_PREFIX, TaskStatus
from jarvis.utils.redact import redact

SUBMISSION_TIMEOUT_MS = 10000


_STATUS_LABELS = {
    "queued": "🟡 Queued",
    "running": "🔵 Running",
    "completed": "✅ Completed",
    "failed": "❌ Failed",
    "cancelled": "⚪ Cancelled",
}


class TaskCentreWindow(QDialog):
    """Create, monitor, and cancel prompts executed by Jarvis."""

    def __init__(
        self,
        submit_callback: Callable[..., Optional[str]],
        cancel_callback: Callable[[str], object],
        parent=None,
        *, max_history: int = DEFAULT_TASK_HISTORY,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("🧭 Jarvis Task Centre")
        self.setMinimumSize(720, 520)
        self.setStyleSheet(JARVIS_THEME_STYLESHEET)
        self._submit_callback = submit_callback
        self._cancel_callback = cancel_callback
        if isinstance(max_history, bool) or not isinstance(max_history, int) or max_history < 1:
            raise ValueError("Task history limit must be a positive integer")
        self._max_history = max_history
        self._tasks: dict[str, dict] = {}
        self._items: dict[str, QListWidgetItem] = {}
        self._pending_request = None
        self._submission_timer = QTimer(self)
        self._submission_timer.setSingleShot(True)
        self._submission_timer.timeout.connect(self._submission_timed_out)

        layout = QVBoxLayout(self)
        title = QLabel("🧭 Task Centre")
        title.setObjectName("title")
        layout.addWidget(title)
        subtitle = QLabel("Queue prompts for Jarvis and follow their progress.")
        subtitle.setObjectName("subtitle")
        layout.addWidget(subtitle)

        self.task_list = QListWidget()
        self.task_list.currentItemChanged.connect(self._show_task_details)
        layout.addWidget(self.task_list, 1)

        self.details = QTextEdit()
        self.details.setReadOnly(True)
        self.details.setAcceptRichText(False)
        self.details.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        self.details.setPlainText("Select a task to view its result.")
        layout.addWidget(self.details, 1)

        input_row = QHBoxLayout()
        self.prompt = QTextEdit()
        self.prompt.setPlaceholderText("Describe what Jarvis should do…")
        self.prompt.setFixedHeight(78)
        input_row.addWidget(self.prompt, 1)
        self.submit_button = QPushButton("🚀 Run Task")
        self.submit_button.setObjectName("primary")
        self.submit_button.clicked.connect(self._submit)
        input_row.addWidget(self.submit_button)
        layout.addLayout(input_row)

        actions = QHBoxLayout()
        self.cancel_button = QPushButton("🛑 Cancel Selected")
        self.cancel_button.setObjectName("danger")
        self.cancel_button.clicked.connect(self._cancel_selected)
        actions.addWidget(self.cancel_button)
        actions.addStretch()
        layout.addLayout(actions)

    def add_or_update_task(self, event: dict) -> bool:
        if not isinstance(event, dict) or event.get("type") != "task":
            return False
        task_id, status = event.get("id"), event.get("status")
        if not isinstance(task_id, str) or not task_id or len(task_id) > 128:
            return False
        if not isinstance(status, str) or status not in _STATUS_LABELS:
            return False
        if not isinstance(event.get("prompt"), str) or any(
            event.get(key) is not None and not isinstance(event[key], str)
            for key in ("result", "error")
        ):
            return False
        task = {"id": task_id, "status": status, "prompt": redact(event["prompt"]),
                "result": redact(event["result"]) if event.get("result") else None,
                "error": redact(event["error"]) if event.get("error") else None}
        self._tasks.pop(task_id, None)
        self._tasks[task_id] = task
        item = self._items.get(task_id)
        text = f"{_STATUS_LABELS[status]}  {task['prompt'][:100]}"
        if item is None:
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, task_id)
            self._items[task_id] = item
            self.task_list.insertItem(0, item)
        else:
            item.setText(text)
        terminal = [identity for identity, saved in self._tasks.items()
                    if saved['status'] not in {TaskStatus.QUEUED, TaskStatus.RUNNING}]
        for identity in terminal[:max(0, len(terminal) - self._max_history)]:
            removed = self._items.pop(identity)
            self.task_list.takeItem(self.task_list.row(removed))
            del self._tasks[identity]
        if self.task_list.currentItem() is None and self.task_list.count():
            self.task_list.setCurrentRow(0)
        self._show_task_details(self.task_list.currentItem(), None)
        return True

    def process_log_line(self, line: str) -> bool:
        if not isinstance(line, str) or not line.startswith(TASK_EVENT_PREFIX):
            return False
        try:
            event = json.loads(line[len(TASK_EVENT_PREFIX):])
        except (TypeError, ValueError):
            return False
        if not isinstance(event, dict):
            return False
        if event.get('type') == 'submission':
            identity, accepted = event.get('request_id'), event.get('accepted')
            if not isinstance(identity, str) or not identity or not isinstance(accepted, bool):
                return False
            field = 'task_id' if accepted else 'error'
            if not isinstance(event.get(field), str) or not event[field]:
                return False
            self._finish_submission(identity, accepted=accepted, error=event.get('error', ''))
            return True
        return self.add_or_update_task(event)

    def _finish_submission(self, identity: str, *, accepted: bool, error: str = '') -> None:
        if identity != self._pending_request:
            return
        self._submission_timer.stop()
        self._pending_request = None
        self.prompt.setReadOnly(False)
        self.submit_button.setEnabled(True)
        if accepted:
            self.prompt.clear()
        else:
            self.details.setPlainText(f"⚠️ {redact(error)}")

    def _submission_timed_out(self) -> None:
        if self._pending_request is not None:
            self._finish_submission(
                self._pending_request, accepted=False,
                error="No confirmation from Jarvis. Check the task list before trying again.",
            )

    def _submit(self) -> None:
        if self._pending_request is not None:
            return
        prompt = self.prompt.toPlainText().strip()
        if not prompt:
            self.details.setPlainText("⚠️ Enter a task before running it.")
            return
        identity = uuid.uuid4().hex
        self._pending_request = identity
        self.prompt.setReadOnly(True)
        self.submit_button.setEnabled(False)
        self._submission_timer.start(SUBMISSION_TIMEOUT_MS)
        try:
            task_id = self._submit_callback(prompt, request_id=identity)
            if task_id is not None:
                if not isinstance(task_id, str) or not task_id:
                    raise ValueError("Task admission did not return an identity")
                self._finish_submission(identity, accepted=True)
        except Exception as exc:
            self._finish_submission(identity, accepted=False,
                                    error=f"Could not queue task: {redact(str(exc))}")

    def _cancel_selected(self) -> None:
        item = self.task_list.currentItem()
        if item is None:
            return
        task_id = item.data(Qt.ItemDataRole.UserRole)
        try:
            self._cancel_callback(str(task_id))
        except Exception as exc:
            self.details.setPlainText(f"❌ Could not cancel task: {redact(str(exc))}")

    def _show_task_details(self, current, _previous) -> None:
        if current is None:
            self.details.setPlainText("Select a task to view its result.")
            return
        task = self._tasks.get(str(current.data(Qt.ItemDataRole.UserRole)), {})
        status = _STATUS_LABELS.get(task.get("status"), "❔ Unknown")
        text = f"{status}\n\n{task.get('prompt', '')}"
        result = task.get("result") or task.get("error")
        if result:
            text += f"\n\n{result}"
        if text != self.details.toPlainText():
            self.details.setPlainText(text)
