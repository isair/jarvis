"""💬 Cora Chat — unified text+voice chat window (Phase 2, feature-flagged).

A typed message goes through the SAME reply core as the voice path:

    user types → new_turn(source="text") → daemon.submit_user_text
    → run_reply_engine (worker thread) → text in chat → optional local TTS

The logic lives in ``ChatController`` (no widgets, signal-driven, unit-testable);
``ChatReplyWorker`` runs the blocking reply+TTS on a ``QThread``; ``ChatWindow``
only renders and forwards user actions. TTS is a secondary output: a TTS failure
never turns a completed reply into a failed turn.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Optional

from PyQt6.QtCore import QObject, QThread, Qt, pyqtSignal
from PyQt6.QtGui import QKeyEvent, QTextCursor
from PyQt6.QtWidgets import (
    QCheckBox, QHBoxLayout, QLabel, QMainWindow, QPushButton, QTextEdit,
    QVBoxLayout, QWidget,
)

from jarvis.core.turn import TurnStatus, new_turn


# ---------------------------------------------------------------------------
# Pure helpers (no Qt state) — directly unit-testable
# ---------------------------------------------------------------------------

def is_submittable(text: Optional[str]) -> bool:
    """A message is submittable only if it has non-whitespace content."""
    return bool(text and text.strip())


def enter_sends(key: int, modifiers: Qt.KeyboardModifier) -> bool:
    """Enter/Return sends; Shift+Enter inserts a newline (returns False)."""
    if key not in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
        return False
    return not bool(modifiers & Qt.KeyboardModifier.ShiftModifier)


# UI state labels
STATE_READY = "Ready"
STATE_THINKING = "Thinking"
STATE_SPEAKING = "Speaking"
STATE_FAILED = "Failed"
STATE_CANCELLED = "Cancelled"


# ---------------------------------------------------------------------------
# Worker — runs the blocking reply + TTS off the UI thread
# ---------------------------------------------------------------------------

class ChatReplyWorker(QObject):
    """Runs one turn: submit_fn(...) then optional local TTS. Lives on a QThread."""

    text_ready = pyqtSignal(object)      # TurnResult
    speaking_started = pyqtSignal()
    speaking_finished = pyqtSignal()
    tts_failed = pyqtSignal()
    finished = pyqtSignal()

    def __init__(
        self,
        user_message: Any,
        turn_context: Any,
        speak_response: bool,
        submit_fn: Optional[Callable] = None,
        tts_getter: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self._user_message = user_message
        self._turn_context = turn_context
        self._speak_response = speak_response
        self._submit_fn = submit_fn
        self._tts_getter = tts_getter

    def _cancelled(self) -> bool:
        return bool(getattr(self._turn_context, "cancellation", None)
                    and self._turn_context.cancellation.cancelled)

    def run(self) -> None:
        try:
            submit_fn = self._submit_fn
            if submit_fn is None:  # default binding resolved lazily to avoid import cycles
                from jarvis import daemon
                submit_fn = daemon.submit_user_text
            result = submit_fn(self._user_message, self._turn_context)
            self.text_ready.emit(result)

            speak = (
                self._speak_response
                and getattr(result, "status", None) == TurnStatus.COMPLETED
                and not self._cancelled()
                and getattr(result, "assistant_message", None) is not None
                and bool(result.assistant_message.text.strip())
            )
            if speak:
                self._speak(result.assistant_message.text)
        finally:
            self.finished.emit()

    def _speak(self, text: str) -> None:
        tts = None
        try:
            getter = self._tts_getter
            if getter is None:
                from jarvis import daemon
                getter = daemon.get_tts_engine
            tts = getter()
        except Exception:
            tts = None
        if tts is None or not getattr(tts, "enabled", False):
            return  # nothing to speak with; not an error
        try:
            done = threading.Event()
            tts.speak(text, completion_callback=done.set)
            self.speaking_started.emit()
            t0 = time.monotonic()
            while time.monotonic() - t0 < 120.0:
                if done.wait(0.05):
                    break
                if self._cancelled():
                    try:
                        tts.interrupt()
                    except Exception:
                        pass
                    break
                # speaking ended without a completion callback (e.g. interrupted)
                if time.monotonic() - t0 > 1.0 and not tts.is_speaking():
                    break
            else:
                # 120s safety cap reached while still speaking: stop the audio so
                # the UI state (about to become READY) matches what the user hears.
                try:
                    tts.interrupt()
                except Exception:
                    pass
            self.speaking_finished.emit()
        except Exception:
            self.tts_failed.emit()


# ---------------------------------------------------------------------------
# Controller — all chat logic, no widgets (fully mockable)
# ---------------------------------------------------------------------------

class ChatController(QObject):
    """Owns turn identity, single-flight, worker lifecycle and the state machine.

    Emits render signals; a view (ChatWindow) draws them. No widget access here,
    so it is unit-testable with a mock submit_fn / tts_getter.
    """

    state_changed = pyqtSignal(str)
    user_message_added = pyqtSignal(str)
    assistant_message_added = pyqtSignal(str)
    error_message = pyqtSignal(str)
    tts_unavailable = pyqtSignal()
    input_cleared = pyqtSignal()
    input_restore = pyqtSignal(str)

    def __init__(
        self,
        conversation_id: Optional[str] = None,
        submit_fn: Optional[Callable] = None,
        tts_getter: Optional[Callable] = None,
        runner: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.conversation_id = conversation_id
        self._submit_fn = submit_fn
        self._tts_getter = tts_getter
        # runner(worker) executes worker.run(); default = a dedicated QThread.
        # Injected in tests for deterministic execution without a live QThread.
        self._runner = runner
        self._busy = False
        self._state = STATE_READY
        self._thread: Optional[QThread] = None
        self._worker: Optional[ChatReplyWorker] = None
        self._current_ctx = None
        self._pending_text = ""

    # -- queries -----------------------------------------------------------
    @property
    def busy(self) -> bool:
        return self._busy

    @property
    def state(self) -> str:
        return self._state

    def _set_state(self, state: str) -> None:
        self._state = state
        self.state_changed.emit(state)

    # -- actions -----------------------------------------------------------
    def submit(self, text: str, language: Optional[str], speak_response: bool) -> bool:
        """Start a turn. Returns False (no turn) if blank or already busy."""
        if not is_submittable(text):
            return False
        if self._busy:  # double-submit / mid-Thinking guard
            return False

        self._busy = True
        self._pending_text = text
        self.user_message_added.emit(text)
        self._set_state(STATE_THINKING)

        # If TTS is mid-sentence, stop it before the new turn (keep prior text).
        self._interrupt_tts_if_speaking()

        turn = new_turn(
            source="text", text=text, language=language,
            conversation_id=self.conversation_id, speak_response=speak_response,
        )
        # keep the conversation stable across the window's lifetime
        if self.conversation_id is None:
            self.conversation_id = turn.context.conversation_id
        self._current_ctx = turn.context

        self.input_cleared.emit()  # user text is now shown as a bubble

        self._start_worker(turn.user_message, turn.context, speak_response)
        return True

    def stop(self) -> None:
        """Stop: halt TTS, mark cancellation. A running reply can't be safely
        cancelled mid-flight, so its late result is ignored (not shown as success)."""
        if self._current_ctx is not None:
            try:
                self._current_ctx.cancellation.cancel("user stop")
            except Exception:
                pass
        self._interrupt_tts_if_speaking()
        if self._busy:
            self._set_state(STATE_CANCELLED)

    # -- worker plumbing ---------------------------------------------------
    def _start_worker(self, user_message, turn_context, speak_response) -> None:
        worker = ChatReplyWorker(
            user_message, turn_context, speak_response,
            submit_fn=self._submit_fn, tts_getter=self._tts_getter,
        )
        worker.text_ready.connect(self._on_text_ready)
        worker.speaking_started.connect(self._on_speaking_started)
        worker.speaking_finished.connect(self._on_speaking_finished)
        worker.tts_failed.connect(self._on_tts_failed)
        worker.finished.connect(self._on_worker_finished)
        self._worker = worker
        if self._runner is not None:
            self._runner(worker)          # custom/test execution
        else:
            self._run_on_qthread(worker)  # production: dedicated QThread

    def _run_on_qthread(self, worker: "ChatReplyWorker") -> None:
        thread = QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_thread_finished)
        self._thread = thread
        thread.start()

    def _on_text_ready(self, result) -> None:
        # Cancelled mid-flight: ignore the late result, do not claim success.
        if self._current_ctx is not None and self._current_ctx.cancellation.cancelled:
            self._set_state(STATE_CANCELLED)
            return
        status = getattr(result, "status", None)
        if status == TurnStatus.COMPLETED and result.assistant_message is not None:
            self.assistant_message_added.emit(result.assistant_message.text)
        elif status == TurnStatus.BLOCKED:
            # single-flight rejected (voice busy): keep the user's text to retry
            self.error_message.emit("Cora is busy — try again in a moment.")
            self.input_restore.emit(self._pending_text)
            self._set_state(STATE_READY)
        else:  # FAILED
            msg = "Cora couldn't respond right now."
            err = getattr(result, "structured_error", None)
            if err is not None and getattr(err, "kind", "") == "daemon_not_ready":
                msg = err.message
            self.error_message.emit(msg)
            self._set_state(STATE_FAILED)

    def _on_speaking_started(self) -> None:
        if self._state not in (STATE_CANCELLED, STATE_FAILED):
            self._set_state(STATE_SPEAKING)

    def _on_speaking_finished(self) -> None:
        if self._state == STATE_SPEAKING:
            self._set_state(STATE_READY)

    def _on_tts_failed(self) -> None:
        # Voice failed but the text reply stands — never a failed turn.
        self.tts_unavailable.emit()
        if self._state == STATE_SPEAKING:
            self._set_state(STATE_READY)

    def _on_worker_finished(self) -> None:
        self._busy = False
        self._current_ctx = None
        # Do NOT drop the QThread/worker here: worker.finished is delivered
        # before thread.quit() is processed, so the thread may still be running.
        # References are cleared only in _on_thread_finished (after it stops).
        if self._thread is None:  # injected-runner path (tests): no QThread to reap
            self._worker = None
        if self._state == STATE_THINKING:  # completed with no speaking step
            self._set_state(STATE_READY)
        else:
            # busy just flipped to False — refresh views (e.g. re-enable Send).
            self.state_changed.emit(self._state)

    def _on_thread_finished(self) -> None:
        # The production QThread has actually stopped; drop references — but ONLY
        # if it is still the current thread. A stale thread from a previous turn
        # finishing after a new turn has started must not null the new
        # thread/worker (that would drop the last ref to a running QThread and
        # abort with 'destroyed while running'). The stale thread/worker are still
        # cleaned by their own thread.finished -> deleteLater connections.
        if self.sender() is self._thread:
            self._thread = None
            self._worker = None

    def _interrupt_tts_if_speaking(self) -> None:
        try:
            getter = self._tts_getter
            if getter is None:
                from jarvis import daemon
                getter = daemon.get_tts_engine
            tts = getter()
            if tts is not None and getattr(tts, "enabled", False) and tts.is_speaking():
                tts.interrupt()
        except Exception:
            pass

    def shutdown(self) -> None:
        """Non-blocking cleanup on window close.

        A running reply cannot be aborted mid-flight (run_reply_engine takes no
        cancellation token), so we mark cancellation and stop TTS, then DETACH:
        the worker's late result is ignored (cancellation check in _on_text_ready)
        and the QThread reaps itself via finished -> deleteLater/_on_thread_finished.
        We never block the UI thread with wait(), and never drop the reference to a
        still-running thread (which would abort with 'destroyed while running').
        """
        if self._current_ctx is not None:
            try:
                self._current_ctx.cancellation.cancel("window closed")
            except Exception:
                pass
        self._interrupt_tts_if_speaking()
        thread = self._thread
        if thread is not None:
            try:
                thread.quit()  # ask the (idle) event loop to exit; do not wait
            except Exception:
                pass
        # References are retained until _on_thread_finished; a running thread must
        # not be garbage-collected while executing.


# ---------------------------------------------------------------------------
# Input box — Enter sends, Shift+Enter newline
# ---------------------------------------------------------------------------

class ChatInput(QTextEdit):
    send_requested = pyqtSignal()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 (Qt override)
        if enter_sends(event.key(), event.modifiers()):
            self.send_requested.emit()
            event.accept()
            return
        super().keyPressEvent(event)


# ---------------------------------------------------------------------------
# Window — rendering only
# ---------------------------------------------------------------------------

class ChatWindow(QMainWindow):
    """Thin view over a ChatController. Renders history, forwards user actions."""

    def __init__(self, cfg: Any = None, controller: Optional[ChatController] = None) -> None:
        super().__init__()
        self._cfg = cfg
        self._language = self._resolve_language(cfg)
        self.controller = controller or ChatController()

        self.setWindowTitle("💬 Cora Chat")
        self.resize(560, 640)

        central = QWidget()
        layout = QVBoxLayout(central)

        self.history = QTextEdit()
        self.history.setReadOnly(True)
        self.history.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        layout.addWidget(self.history, stretch=1)

        self.state_label = QLabel(STATE_READY)
        layout.addWidget(self.state_label)

        self.input = ChatInput()
        self.input.setPlaceholderText("Message Cora…  (Enter to send, Shift+Enter for a new line)")
        self.input.setFixedHeight(90)
        layout.addWidget(self.input)

        row = QHBoxLayout()
        self.speak_checkbox = QCheckBox("Speak responses")
        self.speak_checkbox.setChecked(True)  # default ON
        row.addWidget(self.speak_checkbox)
        row.addStretch(1)
        self.stop_button = QPushButton("Stop")
        self.send_button = QPushButton("Send")
        row.addWidget(self.stop_button)
        row.addWidget(self.send_button)
        layout.addLayout(row)

        self.setCentralWidget(central)

        # wiring
        self.send_button.clicked.connect(self._on_send)
        self.input.send_requested.connect(self._on_send)
        self.stop_button.clicked.connect(self.controller.stop)

        c = self.controller
        c.user_message_added.connect(lambda t: self._append("You", t))
        c.assistant_message_added.connect(lambda t: self._append("Cora", t))
        c.error_message.connect(lambda t: self._append("Cora", t, error=True))
        c.tts_unavailable.connect(lambda: self._append("Cora", "Voice unavailable.", error=True))
        c.state_changed.connect(self._on_state_changed)
        c.input_cleared.connect(self.input.clear)
        c.input_restore.connect(self._restore_input)

    @staticmethod
    def _resolve_language(cfg: Any) -> str:
        lang = getattr(cfg, "response_language", None) if cfg is not None else None
        return (str(lang).strip() or "ro") if lang else "ro"

    def _on_send(self) -> None:
        text = self.input.toPlainText()
        if not is_submittable(text) or self.controller.busy:
            return
        self.controller.submit(text, self._language, self.speak_checkbox.isChecked())

    def _restore_input(self, text: str) -> None:
        if not self.input.toPlainText().strip():
            self.input.setPlainText(text)

    def _on_state_changed(self, state: str) -> None:
        self.state_label.setText(state)
        # Send is disabled whenever a turn is in flight (busy) OR the state is
        # Thinking/Speaking — so it stays disabled through a Cancelled turn until
        # the worker actually finishes and flips busy back to False.
        busy = self.controller.busy or state in (STATE_THINKING, STATE_SPEAKING)
        self.send_button.setEnabled(not busy)

    def _append(self, who: str, text: str, error: bool = False) -> None:
        near_bottom = self._is_near_bottom()
        colour = "#c0392b" if error else ("#2d7dd2" if who == "You" else "#27ae60")
        safe = (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        safe = safe.replace("\n", "<br>")
        self.history.append(f'<b style="color:{colour}">{who}:</b> {safe}')
        if near_bottom:
            self._scroll_to_bottom()

    def _is_near_bottom(self) -> bool:
        bar = self.history.verticalScrollBar()
        return bar.value() >= bar.maximum() - 24

    def _scroll_to_bottom(self) -> None:
        bar = self.history.verticalScrollBar()
        bar.setValue(bar.maximum())
        self.history.moveCursor(QTextCursor.MoveOperation.End)

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        try:
            self.controller.shutdown()
        finally:
            super().closeEvent(event)
