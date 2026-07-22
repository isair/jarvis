"""💬 Cora — modern full-size chat window (Phase 3A, feature-flagged).

A native PyQt6 chat experience (NO web engine, NO HTTP server): a sidebar of
persistent conversations, a header with state, a scrollable transcript of
message cards, and a composer. It reuses the Phase 2 reply pipeline exactly:

    user types -> new_turn(source="text") -> TurnQueue.admit
    -> ChatReplyWorker -> daemon.submit_user_text -> run_reply_engine
    -> text card in the active conversation -> optional local TTS via TTSCoordinator

Design constraints honoured here:
  * ``ModernChatController`` subclasses the classic ``ChatController`` so the
    proven QThread reaping / late-result guards are reused verbatim; it only
    overrides gating (via ``TurnQueue``), TTS routing (via ``TTSCoordinator``,
    CHAT channel), and adds conversation management + persistence.
  * Persistence lives in ``ConversationStore`` (separate tables, additive) and
    is best-effort — a store failure never breaks a turn.
  * The classic ``ChatWindow`` is untouched and remains the default fallback.
  * No change to daemon.py / listener.py / tts.py: the coordinator wraps the
    global engine locally and only manages the chat channel, so the live voice
    path is left byte-for-byte unchanged.

Untrusted assistant text is rendered through the safe Markdown renderer
(``jarvis.utils.markdown_render``): all HTML is escaped, no JS, no remote
resources, only http/https links (and even those never auto-open).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Callable, List, Optional

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import (
    QCheckBox, QFrame, QHBoxLayout, QInputDialog, QLabel, QLineEdit,
    QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPushButton,
    QScrollArea, QToolButton, QVBoxLayout, QWidget,
)

from jarvis.core.turn import TurnStatus, new_turn
from jarvis.core.turn_queue import (
    AdmitOutcome, CompleteOutcome, StopOutcome, TurnQueue,
)
from jarvis.output.tts_coordinator import TTSCoordinator, TtsChannel

from desktop_app.chat_window import (
    ChatController, ChatInput, ChatReplyWorker, is_submittable,
    STATE_READY, STATE_THINKING, STATE_SPEAKING, STATE_FAILED, STATE_CANCELLED,
)

try:
    from jarvis.utils.markdown_render import render_markdown
except Exception:  # pragma: no cover - defensive
    def render_markdown(text: str) -> str:  # type: ignore
        import html as _html
        return _html.escape(text or "").replace("\n", "<br>")

try:
    from jarvis.debug import debug_log
except Exception:  # pragma: no cover
    def debug_log(*_a, **_k):  # type: ignore
        return None


# ---------------------------------------------------------------------------
# palette (Cora's own amber/gold family — deliberately NOT ChatGPT's teal)
# ---------------------------------------------------------------------------
BG_WINDOW = "#0a0b0f"
BG_SIDEBAR = "#0c0d12"
BG_HEADER = "#0d0e13"
BG_COMPOSER = "#0d0e13"
BG_CARD = "#161920"
BG_USER = "rgba(245, 158, 11, 0.10)"
BORDER = "#27272a"
BORDER_USER = "rgba(251, 191, 36, 0.35)"
GOLD = "#fbbf24"
ORANGE = "#f59e0b"
GLOW = "#fcd34d"
TEXT = "#f4f4f5"
TEXT_DIM = "#a1a1aa"
TEXT_MUTE = "#71717a"
OK = "#22c55e"
FAIL = "#ef4444"

MODERN_MIN_W, MODERN_MIN_H = 900, 620
MODERN_DEF_W, MODERN_DEF_H = 1100, 760

WELCOME_TEXT = "Salut! Sunt Cora. Întreabă-mă orice — scris sau vocal."
PLACEHOLDER = "Scrie un mesaj lui Cora…  (Enter trimite, Shift+Enter rând nou)"

_PILL = {
    STATE_READY: (OK, "Ready"),
    STATE_THINKING: (GOLD, "Thinking"),
    STATE_SPEAKING: (ORANGE, "Speaking"),
    STATE_CANCELLED: (TEXT_DIM, "Cancelled"),
    STATE_FAILED: (FAIL, "Failed"),
}

MODERN_CHAT_QSS = f"""
  QMainWindow, QWidget {{ background:{BG_WINDOW}; color:{TEXT};
      font-family:'Segoe UI','.AppleSystemUIFont',sans-serif; font-size:13px; }}

  QFrame#sidebar {{ background:{BG_SIDEBAR}; border-right:1px solid {BORDER}; }}
  QLineEdit#search {{ background:{BG_CARD}; border:1px solid {BORDER};
      border-radius:8px; padding:7px 12px; color:{TEXT}; }}
  QLineEdit#search:focus {{ border-color:{ORANGE}; }}
  QListWidget {{ background:transparent; border:none; }}
  QListWidget::item {{ padding:9px 10px; border-radius:8px; color:{TEXT_DIM}; }}
  QListWidget::item:hover {{ background:{BG_CARD}; color:{TEXT}; }}
  QListWidget::item:selected {{ background:rgba(245,158,11,0.14); color:{GOLD}; }}
  QLabel#emptyConv, QLabel#welcome {{ color:{TEXT_MUTE}; font-size:13px; }}

  QFrame#header {{ background:{BG_HEADER}; border-bottom:1px solid {BORDER}; }}
  QLabel#convTitle {{ font-size:15px; font-weight:600; color:{TEXT}; }}
  QLabel#modelBadge {{ color:{TEXT_MUTE}; font-size:11px;
      border:1px solid {BORDER}; border-radius:10px; padding:2px 10px; }}
  QLabel#statePill {{ font-size:12px; font-weight:600; padding:2px 10px; }}

  QScrollArea#messages {{ border:none; background:{BG_WINDOW}; }}
  QFrame#coraCard {{ background:{BG_CARD}; border:1px solid {BORDER};
      border-radius:14px; }}
  QFrame#userCard {{ background:{BG_USER}; border:1px solid {BORDER_USER};
      border-radius:14px; }}
  QLabel.msgRole {{ font-weight:600; color:{GOLD}; }}
  QLabel.msgMeta {{ color:{TEXT_MUTE}; font-size:11px; }}
  QLabel.msgBody {{ color:{TEXT}; }}

  QFrame#composer {{ background:{BG_COMPOSER}; border-top:1px solid {BORDER}; }}
  QTextEdit#composerEdit {{ background:{BG_CARD}; border:1px solid {BORDER};
      border-radius:12px; padding:10px 12px; color:{TEXT};
      selection-background-color:rgba(245,158,11,0.30); }}
  QTextEdit#composerEdit:focus {{ border-color:{ORANGE}; }}
  QLabel#hint {{ color:{TEXT_MUTE}; font-size:11px; }}

  QPushButton {{ background:{BG_HEADER}; color:{TEXT}; border:1px solid {BORDER};
      border-radius:10px; padding:8px 16px; font-weight:500; }}
  QPushButton:hover {{ border-color:{ORANGE}; color:{GOLD}; }}
  QPushButton:disabled {{ background:{BG_CARD}; color:{TEXT_MUTE};
      border-color:{BG_HEADER}; }}
  QPushButton#send, QPushButton#newConv {{
      background:{ORANGE}; color:{BG_WINDOW}; border:none; font-weight:600; }}
  QPushButton#send:hover, QPushButton#newConv:hover {{ background:{GOLD}; }}
  QPushButton#delete {{ border-color:{FAIL}; color:{FAIL}; }}
  QToolButton {{ background:transparent; color:{TEXT_DIM}; border:none;
      padding:4px 8px; border-radius:6px; }}
  QToolButton:hover {{ color:{GOLD}; }}

  QScrollBar:vertical {{ background:{BG_HEADER}; width:10px; border-radius:5px; }}
  QScrollBar::handle:vertical {{ background:{BORDER}; border-radius:5px; min-height:30px; }}
  QScrollBar::handle:vertical:hover {{ background:{ORANGE}; }}
  QScrollBar::add-line, QScrollBar::sub-line {{ height:0; width:0; }}
"""


def _now_hm() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%H:%M")


# ---------------------------------------------------------------------------
# Coordinator adapter — makes TTSCoordinator look like the engine that
# ChatReplyWorker._speak expects, bound to one chat turn (CHAT channel).
# ---------------------------------------------------------------------------

class _ChatTTSAdapter:
    def __init__(self, coordinator: TTSCoordinator, turn_id: str) -> None:
        self._coord = coordinator
        self._turn_id = turn_id

    @property
    def enabled(self) -> bool:
        # Report NOT-enabled while an unowned (voice) utterance is live, so
        # ChatReplyWorker._speak skips TTS entirely (rather than clobbering the
        # shared Piper slot or polling is_speaking() for up to 120s). Chat text
        # still shows; it just isn't spoken over the live voice reply.
        c = self._coord
        if c is None or not c.enabled:
            return False
        try:
            if c.is_speaking() and not c.owns_active_playback():
                return False
        except Exception:
            pass
        return True

    def speak(self, text: str, completion_callback=None, duration_callback=None) -> None:
        self._coord.speak(
            turn_id=self._turn_id, channel=TtsChannel.CHAT, text=text,
            on_complete=completion_callback, on_duration=duration_callback,
            speak_enabled=True,
        )

    def is_speaking(self) -> bool:
        return bool(self._coord is not None and self._coord.is_speaking())

    def interrupt(self) -> None:
        self._coord.interrupt(turn_id=self._turn_id, channel=TtsChannel.CHAT)


# ---------------------------------------------------------------------------
# Controller — gating via TurnQueue, TTS via TTSCoordinator, + persistence
# ---------------------------------------------------------------------------

class ModernChatController(ChatController):
    conversations_changed = pyqtSignal()

    def __init__(
        self,
        conversation_id: Optional[str] = None,
        submit_fn: Optional[Callable] = None,
        runner: Optional[Callable] = None,
        *,
        store: Any = None,
        coordinator: Optional[TTSCoordinator] = None,
        queue: Optional[TurnQueue] = None,
    ) -> None:
        super().__init__(conversation_id=conversation_id, submit_fn=submit_fn,
                         tts_getter=None, runner=runner)
        self._store = store
        self._coordinator = coordinator
        self._queue = queue or TurnQueue()
        self.active_conversation_id = conversation_id
        self._active_turn_id: Optional[str] = None
        self._active_generation: int = 0
        self._active_turn_conv: Optional[str] = None
        self._active_speak: bool = False  # speak decision of the in-flight turn

    # -- conversation management ------------------------------------------
    def list_conversations(self) -> List[dict]:
        if self._store is None:
            return []
        try:
            return self._store.list_conversations()
        except Exception:
            return []

    def get_messages(self, conversation_id: str) -> List[dict]:
        if self._store is None or not conversation_id:
            return []
        try:
            return self._store.get_messages(conversation_id)
        except Exception:
            return []

    def select_active_on_start(self) -> Optional[str]:
        cid = None
        if self._store is not None:
            try:
                cid = self._store.select_active()
            except Exception:
                cid = None
        if cid is None:
            cid = self._mint_conversation()
        self.switch_conversation(cid)
        return cid

    def new_conversation(self) -> str:
        cid = self._mint_conversation()
        self.switch_conversation(cid)
        self.conversations_changed.emit()
        return cid

    def switch_conversation(self, conversation_id: str) -> None:
        # Does NOT cancel any in-flight turn (per scope); only re-points the view.
        self.active_conversation_id = conversation_id
        self.conversation_id = conversation_id

    def rename_conversation(self, conversation_id: str, title: str) -> bool:
        if self._store is None:
            return False
        try:
            ok = self._store.rename(conversation_id, title)
            if ok:
                self.conversations_changed.emit()
            return ok
        except Exception:
            return False

    def delete_conversation(self, conversation_id: str) -> bool:
        if self._store is None:
            return False
        try:
            ok = self._store.delete(conversation_id, confirm=True)
            if ok:
                self.conversations_changed.emit()
            return ok
        except Exception:
            return False

    def clear_active(self) -> bool:
        if self._store is None or not self.active_conversation_id:
            return False
        try:
            self._store.clear_messages(self.active_conversation_id, confirm=True)
            self.conversations_changed.emit()
            return True
        except Exception:
            return False

    def search(self, query: str) -> List[dict]:
        if self._store is None:
            return []
        try:
            return self._store.search_titles(query)
        except Exception:
            return []

    def _mint_conversation(self) -> str:
        return str(uuid.uuid4())

    # -- submit / stop (gated by the queue) -------------------------------
    def submit(self, text: str, language: Optional[str], speak_response: bool) -> bool:
        if not is_submittable(text):
            return False
        conv_id = self.active_conversation_id
        if conv_id is None:
            conv_id = self._mint_conversation()
            self.switch_conversation(conv_id)

        turn = new_turn(
            source="text", text=text, language=language,
            conversation_id=conv_id, speak_response=speak_response,
        )
        outcome, rec = self._queue.admit(turn)
        if outcome is AdmitOutcome.REJECTED_BUSY:
            self.input_restore.emit(text)  # keep composer text
            return False
        if outcome is AdmitOutcome.REJECTED_DUPLICATE or rec is None:
            return False

        self._busy = True
        self._pending_text = text
        self._active_turn_id = turn.context.turn_id
        self._active_generation = rec.generation
        self._active_turn_conv = conv_id
        self._active_speak = bool(speak_response)
        self._current_ctx = turn.context

        self._persist_user(conv_id, turn.user_message, text)
        self.user_message_added.emit(text)
        self._set_state(STATE_THINKING)
        self._interrupt_tts_if_speaking()
        self.input_cleared.emit()
        self._queue.mark_running(turn.context.turn_id)
        self._start_worker(turn.user_message, turn.context, speak_response)
        self.conversations_changed.emit()
        return True

    def stop(self) -> None:
        tid = self._active_turn_id
        if tid is None:
            self._interrupt_tts_if_speaking()
            return
        outcome = self._queue.stop(tid)
        self._interrupt_active_audio(tid)
        if outcome is StopOutcome.CANCELLED_THINKING:
            if self._busy:
                self._set_state(STATE_CANCELLED)
        elif outcome is StopOutcome.STOP_AUDIO_ONLY:
            self._set_state(STATE_READY)

    # -- worker plumbing overrides ----------------------------------------
    def _start_worker(self, user_message, turn_context, speak_response) -> None:
        if self._coordinator is not None:
            tid = turn_context.turn_id
            self._tts_getter = lambda: _ChatTTSAdapter(self._coordinator, tid)
        super()._start_worker(user_message, turn_context, speak_response)

    def _interrupt_tts_if_speaking(self) -> None:
        if self._coordinator is not None:
            try:
                self._coordinator.interrupt(channel=TtsChannel.CHAT)
            except Exception:
                pass
        else:
            super()._interrupt_tts_if_speaking()

    def _interrupt_active_audio(self, turn_id: str) -> None:
        if self._coordinator is not None:
            try:
                self._coordinator.interrupt(turn_id=turn_id, channel=TtsChannel.CHAT)
            except Exception:
                pass
        else:
            super()._interrupt_tts_if_speaking()

    def _on_text_ready(self, result) -> None:
        tid = self._active_turn_id
        gen = self._active_generation
        conv = self._active_turn_conv
        outcome, _rec = (CompleteOutcome.STALE, None)
        if tid is not None:
            outcome, _rec = self._queue.complete(tid, result, gen)

        if outcome is CompleteOutcome.IGNORED_CANCELLED:
            self._persist_terminal(conv, result, "cancelled")
            self._set_state(STATE_CANCELLED)
            return

        status = getattr(result, "status", None)
        if status == TurnStatus.COMPLETED and getattr(result, "assistant_message", None) is not None:
            am = result.assistant_message
            self._persist_assistant(conv, am, "completed")
            if conv == self.active_conversation_id and outcome is CompleteOutcome.ACCEPTED:
                self.assistant_message_added.emit(am.text)
            self.conversations_changed.emit()
        elif status == TurnStatus.BLOCKED:
            self.error_message.emit("Cora is busy — try again in a moment.")
            self.input_restore.emit(self._pending_text)
            self._set_state(STATE_READY)
        else:  # FAILED
            msg = "Cora couldn't respond right now."
            err = getattr(result, "structured_error", None)
            if err is not None and getattr(err, "kind", "") == "daemon_not_ready":
                msg = err.message
            self._persist_terminal(conv, result, "failed")
            if conv == self.active_conversation_id:
                self.error_message.emit(msg)
            self._set_state(STATE_FAILED)

    def _on_worker_finished(self) -> None:
        tid = self._active_turn_id
        gen = self._active_generation
        if tid is not None:
            self._queue.release(tid, gen)
            self._queue.forget(tid)
            # Reclaim any lingering TTS ownership WITHOUT interrupting audio, so a
            # clobbered (never-fired) guarded callback cannot leave the coordinator
            # stranded and force-interrupt a later/voice playback.
            if self._coordinator is not None:
                try:
                    self._coordinator.release_if_owned(tid)
                except Exception:
                    pass
        self._active_turn_id = None
        self._active_generation = 0
        self._active_turn_conv = None
        super()._on_worker_finished()

    def shutdown(self) -> None:
        # Cancel the in-flight turn in the QUEUE too (base only flips the token),
        # so a result landing after the window closes is treated as cancelled by
        # _on_text_ready (queue.complete -> IGNORED_CANCELLED) instead of being
        # rendered into the closed window and persisted as completed.
        tid = self._active_turn_id
        if tid is not None:
            try:
                self._queue.stop(tid)
            except Exception:
                pass
        super().shutdown()

    # -- persistence (best-effort; never breaks a turn) -------------------
    def _persist_user(self, conv_id, user_message, text) -> None:
        if self._store is None:
            return
        try:
            self._store.ensure_conversation(conv_id, title_seed=text)
            corr = self._current_ctx.correlation_id if self._current_ctx else None
            self._store.append_user_message(
                conv_id, user_message.message_id, text, source="text",
                turn_id=user_message.turn_id, correlation_id=corr,
            )
        except Exception as e:
            debug_log(f"modern chat: persist user failed: {type(e).__name__}", "desktop")

    def _persist_assistant(self, conv_id, assistant_message, status) -> None:
        if self._store is None or conv_id is None or assistant_message is None:
            return
        try:
            err = getattr(assistant_message, "error", None)
            source = "voice" if self._active_speak else "text"
            self._store.append_or_update_assistant_message(
                conv_id, assistant_message.message_id,
                assistant_message.text or "", status=status,
                turn_id=getattr(assistant_message, "turn_id", None), source=source,
                error_code=(getattr(err, "kind", None) if err else None),
            )
        except Exception as e:
            debug_log(f"modern chat: persist assistant failed: {type(e).__name__}", "desktop")

    def _persist_terminal(self, conv_id, result, status) -> None:
        am = getattr(result, "assistant_message", None)
        if am is not None:
            self._persist_assistant(conv_id, am, status)


# ---------------------------------------------------------------------------
# Message card
# ---------------------------------------------------------------------------

class MessageCard(QFrame):
    def __init__(self, role: str, text: str, *, source: str = "text",
                 status: str = "completed", timestamp: Optional[str] = None,
                 parent=None) -> None:
        super().__init__(parent)
        is_user = role == "user"
        self.setObjectName("userCard" if is_user else "coraCard")
        self._raw_text = text or ""

        outer = QVBoxLayout(self)
        outer.setContentsMargins(14, 10, 14, 12)
        outer.setSpacing(6)

        header = QHBoxLayout()
        header.setSpacing(8)
        role_label = QLabel("Tu" if is_user else "Cora")
        role_label.setProperty("class", "msgRole")
        glyph = "⌨" if source == "text" else "🎙"
        meta = QLabel(f"{glyph}  {timestamp or _now_hm()}")
        meta.setProperty("class", "msgMeta")
        header.addWidget(role_label)
        header.addWidget(meta)
        header.addStretch(1)
        copy_btn = QToolButton()
        copy_btn.setText("⧉")
        copy_btn.setToolTip("Copiază")
        copy_btn.clicked.connect(self._copy)
        header.addWidget(copy_btn)
        outer.addLayout(header)

        body = QLabel()
        body.setProperty("class", "msgBody")
        body.setTextFormat(Qt.TextFormat.RichText)
        body.setWordWrap(True)
        body.setOpenExternalLinks(False)  # untrusted links never auto-open
        body.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard
            | Qt.TextInteractionFlag.LinksAccessibleByMouse
        )
        body.setText(render_markdown(text or ""))
        outer.addWidget(body)

        if status in ("failed", "cancelled"):
            footer = QLabel("Eșuat" if status == "failed" else "Anulat")
            footer.setProperty("class", "msgMeta")
            footer.setStyleSheet(f"color:{FAIL if status == 'failed' else TEXT_MUTE};")
            outer.addWidget(footer)
            self.setStyleSheet(
                f"QFrame#{self.objectName()} {{ border-color:"
                f"{FAIL if status == 'failed' else TEXT_MUTE}; }}"
            )

    def _copy(self) -> None:
        try:
            QGuiApplication.clipboard().setText(self._raw_text)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Window
# ---------------------------------------------------------------------------

class ModernChatWindow(QMainWindow):
    def __init__(self, cfg: Any = None, controller: Optional[ModernChatController] = None,
                 *, store: Any = None, coordinator: Optional[TTSCoordinator] = None,
                 submit_fn: Optional[Callable] = None) -> None:
        super().__init__()
        self._cfg = cfg
        self._language = self._resolve_language(cfg)

        if controller is None:
            store = store if store is not None else self._default_store()
            coordinator = coordinator if coordinator is not None else self._default_coordinator()
            controller = ModernChatController(
                submit_fn=submit_fn, store=store, coordinator=coordinator,
            )
        self.controller = controller

        self.setWindowTitle("💬 Cora")
        self.resize(MODERN_DEF_W, MODERN_DEF_H)
        self.setMinimumSize(MODERN_MIN_W, MODERN_MIN_H)

        self._build_ui()
        self._wire_controller()
        self.setStyleSheet(MODERN_CHAT_QSS)

        # resume last conversation (or mint a fresh one)
        cid = self.controller.select_active_on_start()
        self._reload_conversations()
        self._select_in_sidebar(cid)
        self._load_active_messages()
        # Correct the initial control enablement (Send on / Stop off) — no turn
        # is in flight yet, and _on_state_changed is otherwise not fired at build.
        self._on_state_changed(STATE_READY)

    # -- construction -----------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._sidebar = self._build_sidebar()
        self._sidebar.setFixedWidth(280)

        main = QWidget()
        mcol = QVBoxLayout(main)
        mcol.setContentsMargins(0, 0, 0, 0)
        mcol.setSpacing(0)
        mcol.addWidget(self._build_header())
        mcol.addWidget(self._build_messages(), 1)
        mcol.addWidget(self._build_composer())

        root.addWidget(self._sidebar)
        root.addWidget(main, 1)
        self.setCentralWidget(central)

    def _build_sidebar(self) -> QWidget:
        frame = QFrame()
        frame.setObjectName("sidebar")
        col = QVBoxLayout(frame)
        col.setContentsMargins(12, 12, 12, 12)
        col.setSpacing(10)

        new_btn = QPushButton("＋  Conversație nouă")
        new_btn.setObjectName("newConv")
        new_btn.clicked.connect(self._on_new_conversation)
        col.addWidget(new_btn)

        self._search = QLineEdit()
        self._search.setObjectName("search")
        self._search.setPlaceholderText("Caută conversații…")
        self._search.textChanged.connect(self._on_search)
        col.addWidget(self._search)

        self._conv_list = QListWidget()
        self._conv_list.itemClicked.connect(self._on_conversation_clicked)
        col.addWidget(self._conv_list, 1)

        self._empty_conv = QLabel("Nicio conversație încă")
        self._empty_conv.setObjectName("emptyConv")
        self._empty_conv.setAlignment(Qt.AlignmentFlag.AlignCenter)
        col.addWidget(self._empty_conv)

        row = QHBoxLayout()
        rename_btn = QPushButton("Redenumește")
        rename_btn.clicked.connect(self._on_rename)
        del_btn = QPushButton("Șterge")
        del_btn.setObjectName("delete")
        del_btn.clicked.connect(self._on_delete)
        row.addWidget(rename_btn)
        row.addWidget(del_btn)
        col.addLayout(row)
        return frame

    def _build_header(self) -> QWidget:
        frame = QFrame()
        frame.setObjectName("header")
        frame.setFixedHeight(56)
        row = QHBoxLayout(frame)
        row.setContentsMargins(16, 8, 16, 8)
        row.setSpacing(10)

        collapse = QToolButton()
        collapse.setText("☰")
        collapse.setToolTip("Ascunde/arată bara laterală")
        collapse.clicked.connect(self._toggle_sidebar)
        row.addWidget(collapse)

        self._conv_title = QLabel("Cora")
        self._conv_title.setObjectName("convTitle")
        row.addWidget(self._conv_title, 1)

        self._model_badge = QLabel(self._model_label())
        self._model_badge.setObjectName("modelBadge")
        row.addWidget(self._model_badge)

        self._state_pill = QLabel("● Ready")
        self._state_pill.setObjectName("statePill")
        self._state_pill.setStyleSheet(f"color:{OK};")
        row.addWidget(self._state_pill)

        self._speak_checkbox = QCheckBox("Speak responses")
        self._speak_checkbox.setChecked(True)
        row.addWidget(self._speak_checkbox)

        clear_btn = QPushButton("Golește")
        clear_btn.clicked.connect(self._on_clear_active)
        row.addWidget(clear_btn)
        return frame

    def _build_messages(self) -> QWidget:
        self._scroll = QScrollArea()
        self._scroll.setObjectName("messages")
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)

        self._msg_container = QWidget()
        self._msg_container.setObjectName("msgContainer")
        self._msg_layout = QVBoxLayout(self._msg_container)
        self._msg_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._msg_layout.setContentsMargins(24, 20, 24, 20)
        self._msg_layout.setSpacing(12)

        self._welcome = QLabel(WELCOME_TEXT)
        self._welcome.setObjectName("welcome")
        self._welcome.setWordWrap(True)
        self._welcome.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._msg_layout.addWidget(self._welcome)
        self._msg_layout.addStretch(1)

        self._scroll.setWidget(self._msg_container)
        return self._scroll

    def _build_composer(self) -> QWidget:
        frame = QFrame()
        frame.setObjectName("composer")
        col = QVBoxLayout(frame)
        col.setContentsMargins(16, 10, 16, 12)
        col.setSpacing(8)

        self._input = ChatInput()
        self._input.setObjectName("composerEdit")
        self._input.setPlaceholderText(PLACEHOLDER)
        self._input.setFixedHeight(84)
        col.addWidget(self._input)

        row = QHBoxLayout()
        hint = QLabel("Enter trimite · Shift+Enter rând nou")
        hint.setObjectName("hint")
        row.addWidget(hint)
        row.addStretch(1)
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.clicked.connect(self.controller.stop)
        self._send_btn = QPushButton("Trimite")
        self._send_btn.setObjectName("send")
        self._send_btn.clicked.connect(self._on_send)
        row.addWidget(self._stop_btn)
        row.addWidget(self._send_btn)
        col.addLayout(row)

        self._input.send_requested.connect(self._on_send)
        return frame

    # -- controller wiring ------------------------------------------------
    def _wire_controller(self) -> None:
        c = self.controller
        c.user_message_added.connect(self._on_user_added)
        c.assistant_message_added.connect(self._on_assistant_added)
        c.error_message.connect(self._on_error_added)
        c.tts_unavailable.connect(self._on_tts_unavailable)
        c.state_changed.connect(self._on_state_changed)
        c.input_cleared.connect(self._input.clear)
        c.input_restore.connect(self._restore_input)
        if isinstance(c, ModernChatController):
            c.conversations_changed.connect(self._reload_conversations)

    # -- rendering slots (UI thread) --------------------------------------
    def _on_user_added(self, text: str) -> None:
        self._add_card(MessageCard("user", text, source="text", status="completed"))

    def _on_assistant_added(self, text: str) -> None:
        src = "voice" if self._speak_checkbox.isChecked() else "text"
        self._add_card(MessageCard("assistant", text, source=src, status="completed"))

    def _on_error_added(self, text: str) -> None:
        self._add_card(MessageCard("assistant", text, source="text", status="failed"))

    def _on_tts_unavailable(self) -> None:
        # discreet inline notice; never a failed card
        self._add_card(MessageCard("assistant", "Voce indisponibilă.",
                                   source="text", status="completed"))

    def _on_state_changed(self, state: str) -> None:
        colour, label = _PILL.get(state, (OK, state))
        self._state_pill.setText(f"● {label}")
        self._state_pill.setStyleSheet(f"color:{colour}; font-weight:600;")
        busy = self.controller.busy or state in (STATE_THINKING, STATE_SPEAKING)
        self._send_btn.setEnabled(not busy)
        self._stop_btn.setEnabled(busy)

    def _restore_input(self, text: str) -> None:
        if not self._input.toPlainText().strip():
            self._input.setPlainText(text)

    def _add_card(self, card: MessageCard) -> None:
        near = self._is_near_bottom()
        self._welcome.hide()
        # insert before the trailing stretch (last item)
        self._msg_layout.insertWidget(self._msg_layout.count() - 1, card)
        if near:
            QTimer.singleShot(0, self._scroll_to_bottom)

    def _clear_cards(self) -> None:
        # remove every widget except the trailing stretch; re-show welcome
        while self._msg_layout.count() > 1:
            item = self._msg_layout.takeAt(0)
            w = item.widget()
            if w is not None and w is not self._welcome:
                w.setParent(None)
                w.deleteLater()
        # re-insert welcome at top if it was removed
        if self._msg_layout.indexOf(self._welcome) == -1:
            self._msg_layout.insertWidget(0, self._welcome)
        self._welcome.show()

    def _is_near_bottom(self) -> bool:
        bar = self._scroll.verticalScrollBar()
        return bar.value() >= bar.maximum() - 24

    def _scroll_to_bottom(self) -> None:
        bar = self._scroll.verticalScrollBar()
        bar.setValue(bar.maximum())

    # -- conversation management ------------------------------------------
    def _reload_conversations(self) -> None:
        query = self._search.text().strip() if hasattr(self, "_search") else ""
        try:
            convs = self.controller.search(query) if query else self.controller.list_conversations()
        except Exception:
            convs = []
        self._conv_list.blockSignals(True)
        self._conv_list.clear()
        for conv in convs:
            item = QListWidgetItem(conv.get("title") or "New chat")
            item.setData(Qt.ItemDataRole.UserRole, conv.get("id"))
            self._conv_list.addItem(item)
        self._conv_list.blockSignals(False)
        self._empty_conv.setVisible(len(convs) == 0)
        self._select_in_sidebar(self.controller.active_conversation_id)
        self._refresh_title()

    def _select_in_sidebar(self, conversation_id: Optional[str]) -> None:
        if not conversation_id:
            return
        for i in range(self._conv_list.count()):
            it = self._conv_list.item(i)
            if it.data(Qt.ItemDataRole.UserRole) == conversation_id:
                self._conv_list.setCurrentItem(it)
                break

    def _refresh_title(self) -> None:
        cid = self.controller.active_conversation_id
        title = "Cora"
        for i in range(self._conv_list.count()):
            it = self._conv_list.item(i)
            if it.data(Qt.ItemDataRole.UserRole) == cid:
                title = it.text()
                break
        self._conv_title.setText(title)

    def _load_active_messages(self) -> None:
        self._clear_cards()
        cid = self.controller.active_conversation_id
        msgs = self.controller.get_messages(cid) if cid else []
        for m in msgs:
            self._add_card(MessageCard(
                m.get("role", "assistant"), m.get("content", ""),
                source=(m.get("source") or "text"),
                status=(m.get("status") or "completed"),
                timestamp=self._fmt_ts(m.get("created_at")),
            ))
        self._refresh_title()

    @staticmethod
    def _fmt_ts(iso: Optional[str]) -> Optional[str]:
        if not iso:
            return None
        try:
            dt = datetime.fromisoformat(iso)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone().strftime("%H:%M")
        except Exception:
            return None

    def _on_new_conversation(self) -> None:
        self.controller.new_conversation()
        self._load_active_messages()

    def _on_conversation_clicked(self, item: QListWidgetItem) -> None:
        cid = item.data(Qt.ItemDataRole.UserRole)
        if cid and cid != self.controller.active_conversation_id:
            self.controller.switch_conversation(cid)
            self._load_active_messages()

    def _on_rename(self) -> None:
        cid = self.controller.active_conversation_id
        if not cid:
            return
        title, ok = QInputDialog.getText(self, "Redenumește conversația", "Titlu nou:")
        if ok and title.strip():
            self.controller.rename_conversation(cid, title.strip())

    def _on_delete(self) -> None:
        cid = self.controller.active_conversation_id
        if not cid:
            return
        resp = QMessageBox.question(
            self, "Șterge conversația",
            "Sigur ștergi această conversație? (reversibil intern)",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if resp == QMessageBox.StandardButton.Yes:
            self.controller.delete_conversation(cid)
            new_cid = self.controller.select_active_on_start()
            self._reload_conversations()
            self._select_in_sidebar(new_cid)
            self._load_active_messages()

    def _on_clear_active(self) -> None:
        cid = self.controller.active_conversation_id
        if not cid:
            return
        resp = QMessageBox.question(
            self, "Golește conversația",
            "Ștergi toate mesajele din această conversație?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if resp == QMessageBox.StandardButton.Yes:
            self.controller.clear_active()
            self._load_active_messages()

    def _on_search(self, _text: str) -> None:
        self._reload_conversations()

    def _toggle_sidebar(self) -> None:
        self._sidebar.setVisible(not self._sidebar.isVisible())

    # -- composer ---------------------------------------------------------
    def _on_send(self) -> None:
        text = self._input.toPlainText()
        if not is_submittable(text) or self.controller.busy:
            return
        self.controller.submit(text, self._language, self._speak_checkbox.isChecked())

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _resolve_language(cfg: Any) -> str:
        lang = getattr(cfg, "response_language", None) if cfg is not None else None
        return (str(lang).strip() or "ro") if lang else "ro"

    def _model_label(self) -> str:
        model = getattr(self._cfg, "ollama_chat_model", None) if self._cfg else None
        return f"◍ {model} (local)" if model else "◍ local"

    def _default_store(self):
        try:
            from jarvis import daemon as _daemon
            from jarvis.memory.conversation_store import ConversationStore
            db = _daemon.get_db()
            return ConversationStore(db) if db is not None else None
        except Exception:
            return None

    def _default_coordinator(self):
        try:
            from jarvis import daemon as _daemon
            engine = _daemon.get_tts_engine()
            return TTSCoordinator(engine) if engine is not None else None
        except Exception:
            return None

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt override)
        try:
            self.controller.shutdown()
        finally:
            super().closeEvent(event)
