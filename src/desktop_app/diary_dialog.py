"""Diary update dialog shown during shutdown."""

from __future__ import annotations
from typing import Optional, List
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QTextEdit, QProgressBar, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal, QObject
from PyQt6.QtGui import QFont

from .themes import JARVIS_THEME_STYLESHEET, COLORS

# IPC protocol prefix - must match daemon.py
DIARY_IPC_PREFIX = "__DIARY__:"


class DiarySignals(QObject):
    """Signals for diary update progress."""
    # Emitted when a new token is received from LLM
    token_received = pyqtSignal(str)
    # Emitted when status changes (e.g., "Analyzing conversations...")
    status_changed = pyqtSignal(str)
    # Emitted when conversation chunks are available
    chunks_received = pyqtSignal(list)
    # Emitted when the diary update completes
    completed = pyqtSignal(bool)  # True = success, False = failed/skipped


class DiaryUpdateDialog(QDialog):
    """
    Dialog shown during shutdown diary update.

    Shows:
    - The conversation chunks being processed
    - Live streaming of the diary entry being written
    - Progress indication
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.signals = DiarySignals()
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Set up the dialog UI."""
        self.setWindowTitle("Saving Your Diary")
        self.setMinimumSize(550, 450)
        self.setWindowFlags(
            Qt.WindowType.Dialog |
            Qt.WindowType.CustomizeWindowHint |
            Qt.WindowType.WindowTitleHint
        )

        # Apply the shared Jarvis theme
        self.setStyleSheet(JARVIS_THEME_STYLESHEET)

        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        # Title
        title = QLabel("Updating Your Diary")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Status label
        self.status_label = QLabel("Preparing to save...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setObjectName("subtitle")
        layout.addWidget(self.status_label)

        # Progress bar (indeterminate)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(6)
        layout.addWidget(self.progress_bar)

        # Conversations section
        conv_label = QLabel("Today's Conversations")
        conv_label.setObjectName("section_title")
        layout.addWidget(conv_label)

        self.conversations_text = QTextEdit()
        self.conversations_text.setReadOnly(True)
        self.conversations_text.setMaximumHeight(100)
        self.conversations_text.setPlaceholderText("Loading conversations...")
        layout.addWidget(self.conversations_text)

        # Diary entry section
        diary_label = QLabel("Diary Entry")
        diary_label.setObjectName("section_title")
        layout.addWidget(diary_label)

        self.diary_text = QTextEdit()
        self.diary_text.setReadOnly(True)
        self.diary_text.setPlaceholderText("Writing diary entry...")
        layout.addWidget(self.diary_text, stretch=1)

        # Hint at bottom
        hint = QLabel("Please wait while Jarvis saves your conversations...")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setObjectName("subtitle")
        layout.addWidget(hint)

    def _connect_signals(self):
        """Connect internal signals."""
        self.signals.token_received.connect(self._on_token)
        self.signals.status_changed.connect(self._on_status_changed)
        self.signals.chunks_received.connect(self._on_chunks_received)
        self.signals.completed.connect(self._on_completed)

    def _on_chunks_received(self, chunks: list):
        """Handle receiving conversation chunks."""
        self.set_conversations(chunks)

    def _on_token(self, token: str):
        """Handle receiving a token from the LLM."""
        # Append token to diary text
        cursor = self.diary_text.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(token)
        self.diary_text.setTextCursor(cursor)
        # Auto-scroll to bottom
        scrollbar = self.diary_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _on_status_changed(self, status: str):
        """Handle status change."""
        self.status_label.setText(status)

    def _on_completed(self, success: bool):
        """Handle completion."""
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(100)
        if success:
            self.status_label.setText("Diary saved successfully!")
            self.status_label.setStyleSheet(f"color: {COLORS['success']};")
        else:
            self.status_label.setText("No new entries to save")
            self.status_label.setStyleSheet(f"color: {COLORS['text_muted']};")
            # Clear placeholders if nothing was populated
            if not self.conversations_text.toPlainText():
                self.conversations_text.setPlainText("(No conversations to save)")
            if not self.diary_text.toPlainText():
                self.diary_text.setPlainText("(Nothing to write)")

    def set_conversations(self, chunks: List[str]):
        """Set the conversation chunks being processed."""
        if not chunks:
            self.conversations_text.setPlainText("(No conversations to save)")
            return

        # Format chunks nicely
        formatted = []
        for i, chunk in enumerate(chunks[-5:], 1):  # Show last 5 chunks
            # Truncate long chunks
            preview = chunk[:200] + "..." if len(chunk) > 200 else chunk
            # Clean up whitespace
            preview = " ".join(preview.split())
            formatted.append(f"{i}. {preview}")

        self.conversations_text.setPlainText("\n\n".join(formatted))

    def set_diary_content(self, content: str):
        """Set the diary content (for non-streaming updates)."""
        self.diary_text.setPlainText(content)

    def append_diary_token(self, token: str):
        """Append a token to the diary content (for streaming)."""
        self.signals.token_received.emit(token)

    def set_status(self, status: str):
        """Update the status message."""
        self.signals.status_changed.emit(status)

    def mark_completed(self, success: bool = True):
        """Mark the update as completed."""
        self.signals.completed.emit(success)

    def process_log_line(self, line: str) -> bool:
        """
        Process a log line, checking if it contains an IPC event.

        Used in subprocess mode where the daemon emits diary events via stdout.

        Args:
            line: A log line from the daemon

        Returns:
            True if the line was an IPC event and was processed, False otherwise
        """
        line = line.strip()
        if not line.startswith(DIARY_IPC_PREFIX):
            return False

        try:
            import json
            json_str = line[len(DIARY_IPC_PREFIX):]
            event = json.loads(json_str)
            event_type = event.get("type")
            data = event.get("data")

            if event_type == "chunks":
                self.signals.chunks_received.emit(data)
            elif event_type == "token":
                self.signals.token_received.emit(data)
            elif event_type == "status":
                self.signals.status_changed.emit(data)
            elif event_type == "complete":
                self.signals.completed.emit(data)

            return True
        except Exception:
            return False

    def set_subprocess_mode(self):
        """
        Configure dialog for subprocess mode.

        In subprocess mode, the daemon emits IPC events via stdout which are
        intercepted and forwarded to this dialog via process_log_line().
        """
        # Initial state - will be updated when IPC events arrive
        self.conversations_text.setPlaceholderText("Waiting for daemon...")
        self.diary_text.setPlaceholderText("Waiting for diary generation...")
