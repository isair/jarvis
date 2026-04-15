"""
🎙️ Dictation History Window

Displays past dictation results in a scrollable list with copy and delete
actions. Follows the same visual pattern as the Log Viewer.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QScrollArea, QFrame, QApplication,
    QMessageBox,
)
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QTimer
from PyQt6.QtGui import QFont

from desktop_app.themes import JARVIS_THEME_STYLESHEET, COLORS


# ---------------------------------------------------------------------------
# Signals for thread-safe updates from the dictation engine
# ---------------------------------------------------------------------------

class DictationHistorySignals(QObject):
    """Signals emitted when a new dictation entry arrives."""
    new_entry = pyqtSignal(dict)


# ---------------------------------------------------------------------------
# Individual history card widget
# ---------------------------------------------------------------------------

_CARD_STYLE = f"""
    QFrame#dictation_card {{
        background-color: {COLORS['bg_card']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        padding: 12px;
    }}
    QFrame#dictation_card:hover {{
        border-color: {COLORS['accent_primary']};
    }}
"""

_BTN_STYLE = """
    QPushButton {
        background-color: #27272a;
        color: #fafafa;
        border: 1px solid #3f3f46;
        border-radius: 6px;
        padding: 6px 12px;
        font-weight: 500;
        font-size: 12px;
    }
    QPushButton:hover {
        background-color: #3f3f46;
        border-color: #f59e0b;
    }
"""

_DELETE_BTN_STYLE = """
    QPushButton {
        background-color: #27272a;
        color: #ef4444;
        border: 1px solid #3f3f46;
        border-radius: 6px;
        padding: 6px 12px;
        font-weight: 500;
        font-size: 12px;
    }
    QPushButton:hover {
        background-color: #3f3f46;
        border-color: #ef4444;
    }
"""


class _DictationCard(QFrame):
    """A single dictation history entry."""

    deleted = pyqtSignal(str)  # entry ID

    def __init__(self, entry: Dict[str, Any], parent=None):
        super().__init__(parent)
        self._entry = entry
        self.setObjectName("dictation_card")
        self.setStyleSheet(_CARD_STYLE)
        self.setFrameShape(QFrame.Shape.StyledPanel)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        # Top row: timestamp + duration
        top_row = QHBoxLayout()
        top_row.setSpacing(12)

        ts = entry.get("timestamp", 0)
        dt = datetime.fromtimestamp(ts)
        time_label = QLabel(dt.strftime("📅 %Y-%m-%d  🕐 %H:%M:%S"))
        time_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
        top_row.addWidget(time_label)

        duration = entry.get("duration", 0)
        if duration > 0:
            dur_label = QLabel(f"⏱️ {duration:.1f}s")
            dur_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 12px;")
            top_row.addWidget(dur_label)

        top_row.addStretch()
        layout.addLayout(top_row)

        # Text content
        text = entry.get("text", "")
        text_label = QLabel(text)
        text_label.setWordWrap(True)
        text_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        text_label.setStyleSheet(
            f"color: {COLORS['text_primary']}; font-size: 14px; "
            f"line-height: 1.5; padding: 4px 0;"
        )
        layout.addWidget(text_label)

        # Action buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        btn_row.addStretch()

        copy_btn = QPushButton("📋 Copy")
        copy_btn.setStyleSheet(_BTN_STYLE)
        copy_btn.setToolTip("Copy text to clipboard")
        copy_btn.clicked.connect(lambda: self._copy_text(text))
        btn_row.addWidget(copy_btn)

        delete_btn = QPushButton("🗑️ Delete")
        delete_btn.setStyleSheet(_DELETE_BTN_STYLE)
        delete_btn.setToolTip("Remove this entry")
        delete_btn.clicked.connect(self._delete)
        btn_row.addWidget(delete_btn)

        layout.addLayout(btn_row)

    def _copy_text(self, text: str) -> None:
        clipboard = QApplication.clipboard()
        if clipboard:
            clipboard.setText(text)

    def _delete(self) -> None:
        self.deleted.emit(self._entry["id"])


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class DictationHistoryWindow(QMainWindow):
    """Window showing all past dictation entries with copy/delete actions."""

    def __init__(self, history=None):
        super().__init__()
        self._history = history  # DictationHistory instance (set later via set_history)
        self.signals = DictationHistorySignals()
        self.signals.new_entry.connect(self._on_new_entry)

        self.setWindowTitle("🎙️ Dictation History")
        self.setGeometry(100, 100, 700, 600)
        self.setStyleSheet(JARVIS_THEME_STYLESHEET)

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setSpacing(12)

        # Header
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 8)
        header_layout.setSpacing(12)

        title_section = QWidget()
        title_layout = QVBoxLayout(title_section)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(4)

        title = QLabel("🎙️ Dictation History")
        title.setStyleSheet(
            f"font-size: 20px; font-weight: 600; color: {COLORS['accent_secondary']};"
        )
        title_layout.addWidget(title)

        self._subtitle = QLabel("No dictations yet")
        self._subtitle.setObjectName("subtitle")
        title_layout.addWidget(self._subtitle)

        header_layout.addWidget(title_section)
        header_layout.addStretch()

        # Clear all button
        clear_btn = QPushButton("🗑️ Clear All")
        clear_btn.setToolTip("Delete all dictation history")
        clear_btn.setStyleSheet(_DELETE_BTN_STYLE)
        clear_btn.clicked.connect(self._clear_all)
        header_layout.addWidget(clear_btn)

        root_layout.addWidget(header)

        # Scrollable list of cards
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._scroll.setStyleSheet(
            f"QScrollArea {{ border: none; background: {COLORS['bg_primary']}; }}"
        )

        self._list_widget = QWidget()
        self._list_layout = QVBoxLayout(self._list_widget)
        self._list_layout.setContentsMargins(0, 0, 0, 0)
        self._list_layout.setSpacing(8)
        self._list_layout.addStretch()  # Push cards to top

        self._scroll.setWidget(self._list_widget)
        root_layout.addWidget(self._scroll)

        # Empty state label (shown when no entries)
        self._empty_label = QLabel("Hold your dictation hotkey to start.\nTranscriptions will appear here.")
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 14px; padding: 40px;"
        )
        self._list_layout.insertWidget(0, self._empty_label)

        # File-watch timer: poll the history file for changes so the window
        # updates even when the daemon runs in a separate process.
        self._last_file_mtime: float = 0.0
        self._file_watch_timer = QTimer(self)
        self._file_watch_timer.setInterval(1500)  # 1.5 s
        self._file_watch_timer.timeout.connect(self._check_file_changed)
        # Timer starts/stops with window visibility (see showEvent/hideEvent)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_history(self, history) -> None:
        """Set the DictationHistory backend and load existing entries."""
        self._history = history
        self._reload()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def showEvent(self, event) -> None:
        """Refresh the list each time the window is shown."""
        super().showEvent(event)
        self._reload()
        self._last_file_mtime = self._get_history_file_mtime()
        self._file_watch_timer.start()

    def hideEvent(self, event) -> None:
        """Stop polling when the window is hidden."""
        super().hideEvent(event)
        self._file_watch_timer.stop()

    def _is_dictation_enabled(self) -> bool:
        """Check whether dictation is enabled in config."""
        try:
            from jarvis.config import default_config_path, _load_json, get_default_config
            config = _load_json(default_config_path()) or {}
            defaults = get_default_config()
            return bool(config.get("dictation_enabled", defaults.get("dictation_enabled", True)))
        except Exception:
            return True

    def _reload(self) -> None:
        """Rebuild the card list from history."""
        # Remove all existing cards (but keep the stretch at the end).
        # We must hide + setParent(None) before deleteLater to avoid
        # Qt accessing a half-deleted widget still referenced by the layout.
        while self._list_layout.count() > 1:
            item = self._list_layout.takeAt(0)
            w = item.widget()
            if w:
                w.hide()
                w.setParent(None)
                w.deleteLater()

        if self._history is None:
            self._empty_label = self._make_empty_label()
            self._list_layout.insertWidget(0, self._empty_label)
            self._subtitle.setText("No dictations yet")
            return

        entries = self._history.get_all()

        if not entries:
            # Show a contextual hint depending on whether dictation is enabled
            if not self._is_dictation_enabled():
                placeholder = QLabel(
                    "Dictation mode is currently disabled.\n\n"
                    "Enable it in Settings \u2192 Features \u2192 Dictation Mode."
                )
            else:
                placeholder = self._make_empty_label()
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setStyleSheet(
                f"color: {COLORS['text_muted']}; font-size: 14px; padding: 40px;"
            )
            self._list_layout.insertWidget(0, placeholder)
            self._subtitle.setText("No dictations yet")
            return

        self._subtitle.setText(f"{len(entries)} dictation(s)")
        for entry in entries:
            card = _DictationCard(entry)
            card.deleted.connect(self._on_delete)
            # Insert before the stretch
            self._list_layout.insertWidget(self._list_layout.count() - 1, card)

    def _get_history_file_mtime(self) -> float:
        """Return the mtime of the history JSON file, or 0 if missing."""
        try:
            from jarvis.dictation.history import _default_history_path
            p = _default_history_path()
            return p.stat().st_mtime if p.exists() else 0.0
        except Exception:
            return 0.0

    def _check_file_changed(self) -> None:
        """Called by the timer — reload if the history file was modified."""
        mtime = self._get_history_file_mtime()
        if mtime > self._last_file_mtime:
            self._last_file_mtime = mtime
            # Re-read from disk via the public, lock-safe method
            if self._history is not None:
                self._history.reload_from_disk()
            self._reload()

    def _make_empty_label(self) -> QLabel:
        label = QLabel("Hold your dictation hotkey to start.\nTranscriptions will appear here.")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet(
            f"color: {COLORS['text_muted']}; font-size: 14px; padding: 40px;"
        )
        return label

    def _on_new_entry(self, entry: dict) -> None:
        """Slot: called (via signal) when a new dictation completes."""
        if self._history is None:
            return
        # Remove any placeholder labels (empty state / disabled state).
        # Collect indices first, then remove in reverse order so that
        # indices stay valid.  Also detach from parent before deleteLater.
        indices_to_remove = []
        for i in range(self._list_layout.count()):
            item = self._list_layout.itemAt(i)
            w = item.widget() if item else None
            if isinstance(w, QLabel):
                indices_to_remove.append(i)
        for i in reversed(indices_to_remove):
            item = self._list_layout.takeAt(i)
            w = item.widget()
            if w:
                w.hide()
                w.setParent(None)
                w.deleteLater()

        card = _DictationCard(entry)
        card.deleted.connect(self._on_delete)
        # Insert at the top (index 0) — newest first
        self._list_layout.insertWidget(0, card)

        count = self._history.count
        self._subtitle.setText(f"{count} dictation(s)")

    def _on_delete(self, entry_id: str) -> None:
        """Delete a single entry."""
        if self._history:
            self._history.delete(entry_id)
        self._reload()

    def _clear_all(self) -> None:
        """Delete all entries after confirmation."""
        if self._history is None or self._history.count == 0:
            return
        reply = QMessageBox.question(
            self,
            "Clear Dictation History",
            "Delete all dictation history entries?\nThis cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._history.clear()
            self._reload()
