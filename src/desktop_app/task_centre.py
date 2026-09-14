"""Interactive desktop control centre for local Jarvis tasks."""

from __future__ import annotations

import json
from typing import Callable, Optional

from PyQt6.QtCore import Qt
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
        submit_callback: Callable[[str], object],
        cancel_callback: Callable[[str], object],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("🧭 Jarvis Task Centre")
        self.setMinimumSize(720, 520)
        self.setStyleSheet(JARVIS_THEME_STYLESHEET)
        self._submit_callback = submit_callback
        self._cancel_callback = cancel_callback
        self._tasks: dict[str, dict] = {}

        layout = QVBoxLayout(self)
        title = QLabel("🧭 Task Centre")
        title.setObjectName("title")
        layout.addWidget(title)
        subtitle = QLabel("Queue prompts for Jarvis and follow their progress.")
        subtitle.setObjectName("subtitle")
        layout.addWidget(subtitle)

        self.task_list = QListWidget()
        self.task_list.setAlternatingRowColors(True)
        self.task_list.currentItemChanged.connect(self._show_task_details)
        layout.addWidget(self.task_list, 1)

        self.details = QLabel("Select a task to view its result.")
        self.details.setWordWrap(True)
        self.details.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.details)

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

    def add_or_update_task(self, event: dict) -> None:
        task_id = str(event.get("id", ""))
        if not task_id:
            return
        self._tasks[task_id] = dict(event)
        item = self._find_item(task_id)
        label = _STATUS_LABELS.get(event.get("status"), "❔ Unknown")
        prompt = str(event.get("prompt", "")).replace("\n", " ")
        text = f"{label}  {prompt[:100]}"
        if item is None:
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, task_id)
            self.task_list.insertItem(0, item)
        else:
            item.setText(text)
        if self.task_list.currentItem() is None:
            self.task_list.setCurrentItem(item)
        self._show_task_details(self.task_list.currentItem(), None)

    def process_log_line(self, line: str) -> bool:
        prefix = "__TASK__:"
        if prefix not in line:
            return False
        try:
            event = json.loads(line.split(prefix, 1)[1].strip())
        except (TypeError, ValueError):
            return False
        self.add_or_update_task(event)
        return True

    def _submit(self) -> None:
        prompt = self.prompt.toPlainText().strip()
        if not prompt:
            self.details.setText("⚠️ Enter a task before running it.")
            return
        try:
            self._submit_callback(prompt)
        except Exception as exc:
            self.details.setText(f"❌ Could not queue task: {exc}")
            return
        self.prompt.clear()

    def _cancel_selected(self) -> None:
        item = self.task_list.currentItem()
        if item is None:
            return
        task_id = item.data(Qt.ItemDataRole.UserRole)
        try:
            self._cancel_callback(str(task_id))
        except Exception as exc:
            self.details.setText(f"❌ Could not cancel task: {exc}")

    def _find_item(self, task_id: str) -> Optional[QListWidgetItem]:
        for index in range(self.task_list.count()):
            item = self.task_list.item(index)
            if item.data(Qt.ItemDataRole.UserRole) == task_id:
                return item
        return None

    def _show_task_details(self, current, _previous) -> None:
        if current is None:
            return
        task = self._tasks.get(str(current.data(Qt.ItemDataRole.UserRole)), {})
        status = _STATUS_LABELS.get(task.get("status"), "❔ Unknown")
        text = f"{status}\n\n{task.get('prompt', '')}"
        result = task.get("result") or task.get("error")
        if result:
            text += f"\n\n{result}"
        self.details.setText(text)
