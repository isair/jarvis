"""
💬 Chat Window

A text chat interface for Jarvis, alongside the existing voice path. Voice
and text share one conversation (the daemon's global dialogue memory). See
``chat_window.spec.md`` for the full contract.

The window is created lazily by the system tray and kept alive for the
session. Daemon callback signals are marshalled onto the Qt main thread via
``ChatSignals`` so UI updates never touch the worker thread directly.

Sessions (in-memory only): the chat window keeps a list of past sessions in
memory — nothing is written to disk. A session maps to the shared voice+text
conversation: starting a new session clears the daemon's dialogue memory, and
switching sessions restores an archived conversation into it. Every sent
message carries a rewind button that rolls the conversation back to that
message and regenerates a fresh reply.
"""

from __future__ import annotations

from typing import Callable, Optional

from PyQt6.QtCore import Qt, pyqtSignal, QObject
from PyQt6.QtGui import QCloseEvent, QShowEvent
from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from jarvis.debug import debug_log
from desktop_app.themes import COLORS, JARVIS_THEME_STYLESHEET


def get_hot_window_messages() -> list:
    """Thin wrapper around ``jarvis.daemon.get_hot_window_messages``.

    Lives at module scope so tests can monkeypatch
    ``desktop_app.chat_window.get_hot_window_messages`` without touching the
    daemon module (which the bundled and subprocess paths resolve
    differently).
    """
    from jarvis import daemon
    return daemon.get_hot_window_messages()


# ---------------------------------------------------------------------------
# Thread-safe signal bridge
# ---------------------------------------------------------------------------


class ChatSignals(QObject):
    """Marshals daemon-worker-thread callbacks onto the Qt main thread.

    The daemon fires ``on_start`` / ``on_complete`` / ``on_busy`` from its
    worker thread. The window connects these signals to slots so the actual
    UI mutation happens on the main thread.
    """

    started = pyqtSignal(str)
    completed = pyqtSignal(object)  # Optional[str]
    busy = pyqtSignal()


class ChatIpcSignals(QObject):
    """Marshals a raw ``__CHAT__:`` log line from the log-reader worker thread
    onto the Qt main thread.

    The desktop app's log reader runs on a plain ``threading.Thread`` and must
    not create widgets or parse IPC into widget mutations directly. It emits
    ``line_received`` (a queued cross-thread connection) and the main-thread
    slot calls ``ChatWindow.process_ipc_line``.
    """

    line_received = pyqtSignal(str)


# ---------------------------------------------------------------------------
# Window
# ---------------------------------------------------------------------------

_TRANSCRIPT_AREA_STYLE = f"""
    QScrollArea {{
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        background-color: {COLORS['bg_secondary']};
    }}
"""

_INPUT_STYLE = f"""
    QPlainTextEdit {{
        background-color: {COLORS['bg_tertiary']};
        color: {COLORS['text_primary']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        padding: 8px;
        font-family: '.AppleSystemUIFont', 'Segoe UI', sans-serif;
        font-size: 14px;
    }}
    QPlainTextEdit:focus {{
        border-color: {COLORS['accent_primary']};
    }}
"""

_SEND_BTN_STYLE = f"""
    QPushButton {{
        background-color: {COLORS['accent_primary']};
        color: #0a0b0f;
        border: none;
        border-radius: 8px;
        padding: 10px 18px;
        font-weight: 600;
        font-size: 14px;
    }}
    QPushButton:hover {{
        background-color: {COLORS['accent_secondary']};
    }}
    QPushButton:disabled {{
        background-color: {COLORS['accent_muted']};
        color: {COLORS['text_muted']};
    }}
"""

_STOP_BTN_STYLE = f"""
    QPushButton {{
        background-color: {COLORS['error']};
        color: #ffffff;
        border: none;
        border-radius: 8px;
        padding: 10px 18px;
        font-weight: 600;
        font-size: 14px;
    }}
    QPushButton:hover {{
        background-color: {COLORS['error_light']};
    }}
"""

_REWIND_BTN_STYLE = f"""
    QPushButton {{
        background-color: {COLORS['bg_tertiary']};
        color: {COLORS['text_secondary']};
        border: 1px solid {COLORS['border']};
        border-radius: 6px;
        font-size: 13px;
        padding: 2px;
    }}
    QPushButton:hover {{
        border-color: {COLORS['accent_primary']};
        color: {COLORS['accent_secondary']};
    }}
    QPushButton:disabled {{
        color: {COLORS['text_muted']};
        border-color: {COLORS['border']};
    }}
"""

_STATUS_STYLE = f"""
    QLabel {{
        color: {COLORS['text_secondary']};
        font-size: 12px;
        padding: 2px 4px;
    }}
"""

_SESSION_SIDEBAR_STYLE = f"""
    QListWidget {{
        background-color: {COLORS['bg_tertiary']};
        border: 1px solid {COLORS['border']};
        border-radius: 8px;
        color: {COLORS['text_secondary']};
        font-size: 13px;
        padding: 4px;
    }}
    QListWidget::item {{
        padding: 6px 8px;
        border-radius: 6px;
    }}
    QListWidget::item:selected {{
        background-color: {COLORS['bg_hover']};
        color: {COLORS['accent_secondary']};
    }}
"""

_DAEMON_STATUS_MESSAGES = {
    "starting": "Starting Jarvis...",
    "stopping": "Stopping Jarvis...",
    "stopped": "Start Listening from the tray to use chat.",
    "crashed": "Jarvis stopped unexpectedly. Start Listening to reconnect.",
}

_DAEMON_STATUS_PLACEHOLDERS = {
    "starting": "Jarvis is starting",
    "stopping": "Jarvis is stopping",
    "stopped": "Start Listening from the tray to use chat",
    "crashed": "Start Listening from the tray to reconnect chat",
    "running": "Type a message to Jarvis... (Enter to send, Shift+Enter for newline)",
}

_MESSAGE_TEXT_STYLES = {
    "user": f"color: {COLORS['accent_secondary']};",
    "assistant": f"color: {COLORS['text_primary']};",
    "system": f"color: {COLORS['text_muted']}; font-size: 12px;",
}


class ChatWindow(QMainWindow):
    """Text chat window. Sends via ``jarvis.daemon.submit_text_query``.

    In subprocess mode the desktop app sets ``submit_fn`` to a callable that
    writes a ``__CHAT_QUERY__:`` line to the daemon's stdin, ``cancel_fn`` to
    the cancel line writer, and ``control_fn`` to a callable that writes the
    session-control lines (new session / rewind / restore). In bundled mode
    the window calls the daemon directly.

    Sessions: an in-memory list of past conversations (never persisted).
    The active session maps 1:1 to the daemon's shared dialogue memory, so
    starting a new session or switching to an archived one rewires the
    conversation the voice path also sees. Every sent message carries a
    rewind button that truncates the conversation to before that message
    and regenerates a fresh reply to it.
    """

    def __init__(
        self,
        submit_fn=None,
        daemon_available: bool = True,
        cancel_fn=None,
        control_fn: Optional[Callable[[str, Optional[dict]], None]] = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("Jarvis Chat")
        self.setMinimumSize(760, 560)
        self.setStyleSheet(JARVIS_THEME_STYLESHEET)
        self._submit_fn = submit_fn
        # Subprocess mode routes cancellation to the daemon the same way it
        # routes a submission. Without it, Stop sets a flag in this
        # process while the query runs in the other one.
        self._cancel_fn = cancel_fn
        # Subprocess mode routes session control (new session / rewind /
        # restore) to the daemon's stdin. Bundled mode calls the daemon
        # module directly and leaves this None.
        self._control_fn = control_fn
        # Set by Stop, cleared by the next send. The engine keeps running
        # after a cancel and its reply still arrives, so the window has to
        # decline the answer to an exchange the user walked away from.
        self._query_cancelled = False
        self._daemon_available = daemon_available
        self._daemon_status = "running" if daemon_available else "stopped"

        # In-memory session store. Each session is
        # {"title": str, "messages": [{"kind", "text", "user_index"}], "active": bool}.
        # Nothing here is written to disk; a fresh app run starts with a new
        # session (there is no "last chat session" to restore).
        self._sessions: list[dict] = [
            {"title": "Session 1", "messages": [], "active": True}
        ]
        self._session_counter = 1

        # Signal bridge: daemon worker -> Qt main thread.
        self.signals = ChatSignals()
        self.signals.started.connect(self._on_start)
        self.signals.completed.connect(self._on_complete)
        self.signals.busy.connect(self._on_busy)

        # --- Layout -----------------------------------------------------
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        # Sessions sidebar (in-memory list + new session button)
        sidebar = QVBoxLayout()
        sidebar.setSpacing(8)
        sidebar_label = QLabel("Sessions")
        sidebar_label.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-size: 12px;"
            " font-weight: 600;"
        )
        sidebar.addWidget(sidebar_label)
        self.session_list = QListWidget()
        self.session_list.setStyleSheet(_SESSION_SIDEBAR_STYLE)
        self.session_list.setFixedWidth(170)
        self.session_list.itemClicked.connect(self._on_session_clicked)
        sidebar.addWidget(self.session_list, stretch=1)
        self.new_session_button = QPushButton("＋ New session")
        self.new_session_button.setStyleSheet(_SEND_BTN_STYLE)
        self.new_session_button.clicked.connect(self._new_session)
        sidebar.addWidget(self.new_session_button)
        root.addLayout(sidebar)

        # Right column: transcript + status + input row
        right = QVBoxLayout()
        right.setSpacing(8)

        # Transcript: a scroll area whose container holds one row widget per
        # message, so sent messages can carry a rewind button. Rebuilt
        # atomically on session switch / rewind (see _render_transcript).
        self.transcript_widget = QScrollArea()
        self.transcript_widget.setWidgetResizable(True)
        self.transcript_widget.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.transcript_widget.setStyleSheet(_TRANSCRIPT_AREA_STYLE)
        self._transcript_container = QWidget()
        self._transcript_layout = QVBoxLayout(self._transcript_container)
        self._transcript_layout.setContentsMargins(10, 10, 10, 10)
        self._transcript_layout.setSpacing(6)
        self._transcript_layout.addStretch(1)
        self.transcript_widget.setWidget(self._transcript_container)
        right.addWidget(self.transcript_widget, stretch=1)

        # Status indicator (display-only label)
        self._status_label = QLabel("")
        self._status_label.setStyleSheet(_STATUS_STYLE)
        self._status_label.setVisible(False)
        right.addWidget(self._status_label)

        # Input row: input box + send + stop
        row = QHBoxLayout()
        row.setSpacing(8)

        self.input_widget = QPlainTextEdit()
        self.input_widget.setPlaceholderText(_DAEMON_STATUS_PLACEHOLDERS["running"])
        self.input_widget.setFixedHeight(64)
        self.input_widget.setStyleSheet(_INPUT_STYLE)
        self.input_widget.keyPressEvent = self._input_key_press  # type: ignore[method-assign]
        row.addWidget(self.input_widget, stretch=1)

        self.send_button = QPushButton("Send")
        self.send_button.setStyleSheet(_SEND_BTN_STYLE)
        self.send_button.clicked.connect(self._send)
        row.addWidget(self.send_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.setStyleSheet(_STOP_BTN_STYLE)
        self.stop_button.clicked.connect(self._stop)
        self.stop_button.setVisible(False)
        row.addWidget(self.stop_button)

        right.addLayout(row)
        root.addLayout(right, stretch=1)

        self._query_in_flight = False
        # Whether the transcript has been seeded from the daemon's hot window.
        # Seeded once on first show so re-opening never duplicates turns.
        self._hot_window_seeded = False
        self._refresh_session_list()
        self.set_daemon_available(daemon_available)

    # --- Sessions -------------------------------------------------------

    def _active_session(self) -> dict:
        return next(s for s in self._sessions if s["active"])

    def _refresh_session_list(self) -> None:
        """Rebuild the sidebar from the in-memory session store."""
        self.session_list.clear()
        for index, session in enumerate(self._sessions):
            title = session["title"]
            if session["active"]:
                title = f"● {title}"
            item = QListWidgetItem(title)
            # Store the index, not the dict: PyQt6 marshals plain dicts
            # through QVariant on setData/data, returning a copy on read,
            # so mutating the returned object would not touch the session.
            item.setData(Qt.ItemDataRole.UserRole, index)
            self.session_list.addItem(item)
            if session["active"]:
                self.session_list.setCurrentItem(item)

    def _on_session_clicked(self, item: QListWidgetItem) -> None:
        """Switch to a past session: archive the live one, restore the
        clicked one into the shared daemon memory, re-render its transcript."""
        index = item.data(Qt.ItemDataRole.UserRole)
        if index is None or not isinstance(index, int):
            return
        target = self._sessions[index]
        if target["active"] or self._query_in_flight:
            return
        # Restore the daemon memory first: if a query is in flight the
        # daemon refuses, and the UI must not switch to a conversation the
        # memory does not hold.
        if not self._restore_session_memory(target):
            debug_log("session switch rejected: a query is in flight", "chat")
            return
        current = self._active_session()
        current["active"] = False
        target["active"] = True
        self._render_transcript(target["messages"])
        self._refresh_session_list()
        self._scroll_to_bottom()

    def _restore_session_memory(self, session: dict) -> bool:
        """Push an archived session's turns into the shared daemon memory.

        Only user/assistant turns are restored (system notices are local
        UI text). The daemon re-redacts everything on restore, so the
        diary never sees raw user text even though the window's archive
        holds it. Returns False when the daemon refused (query in flight).
        In subprocess mode the control line is fire-and-forget: the daemon
        enforces its own lock guard, and the window is optimistic.
        """
        turns = [
            {"role": m["kind"], "content": m["text"]}
            for m in session["messages"]
            if m["kind"] in ("user", "assistant")
        ]
        if self._control_fn is not None:
            self._control_fn("restore", {"messages": turns})
            return True
        from jarvis import daemon
        return daemon.set_chat_messages(turns)

    def _new_session(self) -> None:
        """Archive the current session and start a fresh one.

        The shared dialogue memory is cleared, so the voice path also
        starts a new conversation. The previous session stays in the
        in-memory list; nothing is written to disk. The daemon is asked
        first: if a query is in flight it refuses, and the UI stays put.
        """
        if self._query_in_flight:
            return
        if self._control_fn is not None:
            self._control_fn("new_session", None)
            applied = True
        else:
            from jarvis import daemon
            applied = daemon.new_chat_session()
        if not applied:
            debug_log("new session rejected: a query is in flight", "chat")
            return
        current = self._active_session()
        current["active"] = False
        self._session_counter += 1
        self._sessions.append(
            {"title": f"Session {self._session_counter}",
             "messages": [], "active": True}
        )
        self._render_transcript([])
        self._refresh_session_list()
        self._scroll_to_bottom()

    # --- Sending --------------------------------------------------------

    def _send(self) -> None:
        text = self.input_widget.toPlainText().strip()
        if not text:
            return
        if not self._daemon_available:
            self._append_system("Start Listening to use chat.")
            return

        # Echo the user message into the transcript immediately.
        self._append_user(text)
        self.input_widget.setPlainText("")

        self._query_cancelled = False
        self._set_thinking(True)

        if self._submit_fn is not None:
            # Subprocess mode: the desktop app routes the query to the daemon's
            # stdin and feeds __CHAT__: events back via the signals.
            self._submit_fn(text)
        else:
            # Bundled mode: call the daemon directly with our signal emitters.
            from jarvis import daemon

            daemon.submit_text_query(
                text,
                on_start=self.signals.started.emit,
                on_complete=self.signals.completed.emit,
                on_busy=self.signals.busy.emit,
            )

    def _stop(self) -> None:
        """Abandon the in-flight query. Never request_stop, which would
        tear down the whole voice assistant."""
        # Refuse the reply locally first: cancellation cannot unwind a
        # request already inside the engine, so the answer arrives either
        # way and this is what keeps it out of the transcript.
        self._query_cancelled = True

        if self._cancel_fn is not None:
            # Subprocess mode: the query lives in the daemon process.
            self._cancel_fn()
        else:
            from jarvis import daemon

            daemon.cancel_active_chat_query()

        # Reset the thinking indicator immediately so the user sees
        # feedback without waiting for the engine to finish.
        self._set_thinking(False)

    def _rewind_to_user(self, user_index: int, text: str) -> None:
        """Roll the conversation back to before ``user_index``-th user
        message and regenerate a fresh reply to it.

        The transcript is truncated to keep the message itself; the daemon
        memory is rewound past it and the same text is re-submitted, so the
        old reply (and everything after it) is replaced. Rewinding is
        disabled while a query is in flight.
        """
        if self._query_in_flight or not self._daemon_available:
            return
        messages = self._active_session()["messages"]
        keep_until = None
        for i, m in enumerate(messages):
            if m.get("kind") == "user" and m.get("user_index") == user_index:
                keep_until = i + 1
                break
        if keep_until is None:
            return
        # Ask the daemon first. Bundled mode: a refusal (query in flight)
        # leaves both transcript and memory untouched. Subprocess mode is
        # fire-and-forget; the daemon enforces its own lock guard.
        if self._control_fn is not None:
            self._control_fn("rewind", {"user_index": user_index})
        else:
            from jarvis import daemon
            if not daemon.rewind_chat_to_user(user_index):
                debug_log(
                    f"chat rewind rejected for user message {user_index}", "chat"
                )
                return
        self._active_session()["messages"] = messages[:keep_until]
        self._render_transcript(messages[:keep_until])

        # Regenerate: re-submit the same message for a fresh reply. The
        # message is already displayed, so no new echo is added.
        self._query_cancelled = False
        self._set_thinking(True)
        if self._submit_fn is not None:
            self._submit_fn(text)
        else:
            from jarvis import daemon

            daemon.submit_text_query(
                text,
                on_start=self.signals.started.emit,
                on_complete=self.signals.completed.emit,
                on_busy=self.signals.busy.emit,
            )

    def set_daemon_available(self, available: bool) -> None:
        """Enable or disable chat submission based on daemon availability."""
        self.set_daemon_status("running" if available else "stopped")

    def set_daemon_status(self, status: str) -> None:
        """Reflect daemon lifecycle state in chat controls and status text."""
        if status != "running" and status not in _DAEMON_STATUS_MESSAGES:
            debug_log(f"unknown chat daemon status ignored: {status}", "chat")
            status = "stopped"

        self._daemon_status = status
        self._daemon_available = status == "running"
        if not self._daemon_available:
            self._query_in_flight = False
            self.stop_button.setVisible(False)
        self.input_widget.setEnabled(self._daemon_available)
        self.input_widget.setPlaceholderText(
            _DAEMON_STATUS_PLACEHOLDERS.get(
                status,
                _DAEMON_STATUS_PLACEHOLDERS["stopped"],
            )
        )
        self._refresh_status_label()
        self._refresh_send_button()

    # --- Daemon callback slots (run on the main thread via signals) -----

    def _on_start(self, _query: str) -> None:
        # The user message is already echoed in _send. We keep the thinking
        # indicator on; nothing extra to render for the start event in the MVP.
        self._set_thinking(True)

    def _on_complete(self, reply: Optional[str]) -> None:
        self._set_thinking(False)
        if self._query_cancelled:
            debug_log("chat reply dropped: the query was cancelled", "chat")
            self._query_cancelled = False
            return
        if reply:
            self._append_assistant(reply)

    def _on_busy(self) -> None:
        self._set_thinking(False)
        self._append_system("Jarvis is busy with another query already.")

    # --- Subprocess IPC entry point --------------------------------------

    def process_ipc_line(self, line: str) -> bool:
        """Parse a ``__CHAT__:`` event line and emit the matching signal.

        Mirrors ``DiaryUpdateDialog.process_log_line``: the caller (the log
        reader thread) forwards the raw line, and this method owns the JSON
        parse + signal emit. Returns True if the line was a chat event (even
        if malformed), False otherwise. Must be called on the Qt main thread
        (the caller marshals via a main-thread-owned signal).
        """
        from jarvis.daemon import CHAT_IPC_PREFIX
        if not line.startswith(CHAT_IPC_PREFIX):
            return False
        import json as _json
        try:
            payload = _json.loads(line[len(CHAT_IPC_PREFIX):])
        except Exception:
            debug_log(f"malformed {CHAT_IPC_PREFIX} line ignored", "chat")
            return True
        kind = payload.get("type")
        data = payload.get("data")
        if kind == "start":
            self.signals.started.emit(str(data) if data is not None else "")
        elif kind == "complete":
            self.signals.completed.emit(data)
        elif kind == "busy":
            self.signals.busy.emit()
        return True

    # --- Rendering helpers ----------------------------------------------

    def _append_user(self, text: str) -> None:
        user_index = 1 + sum(
            1 for m in self._active_session()["messages"] if m.get("kind") == "user"
        )
        self._append_message("user", text, user_index=user_index)

    def _append_assistant(self, text: str) -> None:
        self._append_message("assistant", text)

    def _append_system(self, text: str) -> None:
        self._append_message("system", text)

    def _append_message(self, kind: str, text: str, user_index: Optional[int] = None) -> None:
        """Add one message row to the transcript and the active session."""
        self._active_session()["messages"].append(
            {"kind": kind, "text": text, "user_index": user_index}
        )
        # Append a single row (before the trailing stretch) instead of
        # rebuilding the whole transcript, so long sessions stay O(n).
        row = self._make_message_row(
            {"kind": kind, "text": text, "user_index": user_index}
        )
        self._transcript_layout.insertWidget(
            self._transcript_layout.count() - 1, row
        )
        self._scroll_to_bottom()

    def _render_transcript(self, messages: list) -> None:
        """Rebuild the transcript rows atomically from ``messages``.

        Rebuilding (instead of incrementally appending) keeps rewind,
        session switch, and new-session truncation trivially correct: the
        rendered rows always mirror the session's message list.
        """
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)
        for m in messages:
            layout.addWidget(self._make_message_row(m))
        layout.addStretch(1)
        old = self.transcript_widget.takeWidget()
        self.transcript_widget.setWidget(container)
        self._transcript_container = container
        self._transcript_layout = layout
        if old is not None:
            old.hide()
            old.deleteLater()
        self._scroll_to_bottom()

    def _make_message_row(self, m: dict) -> QWidget:
        kind = m.get("kind", "system")
        text = m.get("text", "")
        row_widget = QWidget()
        row = QHBoxLayout(row_widget)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        if kind == "user":
            label = QLabel(f"👤 You: {text}")
        elif kind == "assistant":
            label = QLabel(f"🤖 Jarvis: {text}")
        else:
            label = QLabel(f"  ⏳ {text}")
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        label.setStyleSheet(_MESSAGE_TEXT_STYLES.get(kind, _MESSAGE_TEXT_STYLES["system"]))
        row.addWidget(label, stretch=1)
        if kind == "user":
            user_index = m.get("user_index")
            rewind_btn = QPushButton("⟲")
            rewind_btn.setObjectName(f"rewind_{user_index}")
            rewind_btn.setToolTip("Rewind to this message and regenerate")
            rewind_btn.setStyleSheet(_REWIND_BTN_STYLE)
            rewind_btn.setFixedSize(30, 30)
            rewind_btn.setEnabled(
                self._daemon_available and not self._query_in_flight
            )
            if user_index is not None:
                rewind_btn.clicked.connect(
                    lambda _checked=False, idx=user_index, txt=text: self._rewind_to_user(idx, txt)
                )
            row.addWidget(rewind_btn, alignment=Qt.AlignmentFlag.AlignTop)
        return row_widget

    def _scroll_to_bottom(self) -> None:
        bar = self.transcript_widget.verticalScrollBar()
        bar.setValue(bar.maximum())

    def transcript_text(self) -> str:
        """Plain-text rendering of the active transcript (testing + copy)."""
        return "\n".join(
            f"👤 You: {m['text']}" if m["kind"] == "user"
            else f"🤖 Jarvis: {m['text']}" if m["kind"] == "assistant"
            else f"  ⏳ {m['text']}"
            for m in self._active_session()["messages"]
        )

    def _set_thinking(self, thinking: bool) -> None:
        self._query_in_flight = thinking and self._daemon_available
        self.stop_button.setVisible(self._query_in_flight)
        self._refresh_status_label()
        self._refresh_send_button()
        self._refresh_rewind_buttons()

    def _refresh_rewind_buttons(self) -> None:
        """Disable rewind while a query is in flight or the daemon is down."""
        enabled = self._daemon_available and not self._query_in_flight
        for btn in self.transcript_widget.findChildren(QPushButton):
            if btn.objectName().startswith("rewind_"):
                btn.setEnabled(enabled)

    def _refresh_send_button(self) -> None:
        self.send_button.setEnabled(self._daemon_available and not self._query_in_flight)

    def _refresh_status_label(self) -> None:
        if self._query_in_flight:
            self._status_label.setText("  Jarvis is thinking…")
            self._status_label.setVisible(True)
            return

        if self._daemon_status == "running":
            self._status_label.setText("")
            self._status_label.setVisible(False)
            return

        message = _DAEMON_STATUS_MESSAGES.get(
            self._daemon_status,
            _DAEMON_STATUS_MESSAGES["stopped"],
        )
        self._status_label.setText(f"  {message}")
        self._status_label.setVisible(True)

    # --- Input key handling ---------------------------------------------

    def _input_key_press(self, event) -> None:
        from PyQt6.QtGui import QKeyEvent
        from PyQt6.QtCore import Qt as _Qt

        if (
            isinstance(event, QKeyEvent)
            and event.key() in (_Qt.Key.Key_Return, _Qt.Key.Key_Enter)
            and not (event.modifiers() & _Qt.KeyboardModifier.ShiftModifier)
        ):
            # Enter (or numpad Enter) sends; Shift+Enter inserts a newline.
            self._send()
            return
        # Default handling for all other keys.
        QPlainTextEdit.keyPressEvent(self.input_widget, event)

    # --- Lifecycle ------------------------------------------------------

    def showEvent(self, event: QShowEvent) -> None:
        # A fresh app run has no previous session in memory (sessions are
        # never persisted), so the window always starts with a new session —
        # "Session 1" is created in __init__. On first show we additionally
        # seed the transcript from the daemon's hot window, so a user who has
        # been talking by voice sees their recent turns instead of a blank
        # panel. Seeding runs only once per instance: re-showing (from the
        # tray or after a hide) must never duplicate turns. Fails silently
        # when the daemon accessor is unavailable (e.g. subprocess mode) —
        # the window just opens with the new session.
        if not self._hot_window_seeded:
            self._hot_window_seeded = True
            try:
                for msg in get_hot_window_messages():
                    role = msg.get("role")
                    content = msg.get("content")
                    if role == "user" and content:
                        self._append_user(content)
                    elif role == "assistant" and content:
                        self._append_assistant(content)
            except Exception as exc:
                debug_log(f"hot window replay failed: {exc}", "chat")
        super().showEvent(event)

    def closeEvent(self, event: QCloseEvent) -> None:
        # Hide instead of destroying; the tray re-shows the same instance.
        # We intentionally do NOT call request_stop here — closing the chat
        # window does not stop the daemon or end the conversation. The explicit
        # hide() guarantees the window disappears regardless of how the close
        # is triggered (title bar button, ESC, tray toggle) and keeps the
        # instance alive so a reply that lands while hidden still lands here.
        self.hide()
        event.accept()
