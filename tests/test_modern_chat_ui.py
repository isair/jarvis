"""Phase 3A: offline tests for the modern chat window + ModernChatController.

Offscreen Qt, synchronous worker runner (no live QThread), stub submit_fn, temp
DB store, mock coordinator. Verifies: send -> cards + persistence, Stop, TTS via
coordinator, multi-conversation switch does not cancel a running turn, safe
Markdown in cards.
"""

from __future__ import annotations

import os
import threading

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtWidgets import QApplication

from jarvis.core.turn import (
    AssistantStatus, TurnResult, TurnStatus, new_assistant_message,
)
from jarvis.memory.db import Database
from jarvis.memory.conversation_store import ConversationStore
from jarvis.output.tts_coordinator import TTSCoordinator, TtsChannel
from desktop_app.modern_chat_window import (
    ModernChatController, ModernChatWindow, MessageCard, STATE_CANCELLED,
)

SYNC = lambda w: w.run()
NORUN = lambda w: None


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def store(tmp_path):
    db = Database(str(tmp_path / "jarvis.db"))
    yield ConversationStore(db)
    db.close()


class CfgStub:
    response_language = "ro"
    ollama_chat_model = "gemma4:e2b"


class MockEngine:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self._speaking = False
        self.spoken = []
        self.interrupts = 0

    def is_speaking(self):
        return self._speaking

    def speak(self, text, completion_callback=None, duration_callback=None):
        self.spoken.append(text)
        self._speaking = False
        if completion_callback:
            completion_callback()

    def interrupt(self):
        self.interrupts += 1
        self._speaking = False


def _ok_submit(idents=None):
    def _submit(um, ctx):
        if idents is not None:
            idents.append(threading.get_ident())
        am = new_assistant_message(ctx)
        am.text = "Salut! Răspuns la: " + um.text
        am.status = AssistantStatus.COMPLETED
        return TurnResult(status=TurnStatus.COMPLETED, assistant_message=am)
    return _submit


def _window(app, store, submit_fn, coordinator=None, runner=SYNC):
    ctrl = ModernChatController(submit_fn=submit_fn, runner=runner,
                               store=store, coordinator=coordinator)
    win = ModernChatWindow(cfg=CfgStub(), controller=ctrl)
    win.show()
    return win, ctrl


def _card_count(win):
    return sum(
        1 for i in range(win._msg_layout.count())
        if isinstance(win._msg_layout.itemAt(i).widget(), MessageCard)
    )


# --- send / render / persist ------------------------------------------------

def test_send_adds_user_and_assistant_cards(app, store):
    win, ctrl = _window(app, store, _ok_submit())
    win._input.setPlainText("cât e ceasul?")
    win._speak_checkbox.setChecked(False)
    win._on_send()
    app.processEvents()
    assert _card_count(win) == 2
    win.close()


def test_send_persists_to_store(app, store):
    win, ctrl = _window(app, store, _ok_submit())
    win._input.setPlainText("întrebare")
    win._speak_checkbox.setChecked(False)
    win._on_send()
    app.processEvents()
    msgs = store.get_messages(ctrl.active_conversation_id)
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    win.close()


def test_worker_ran_via_runner(app, store):
    idents = []
    win, ctrl = _window(app, store, _ok_submit(idents))
    win._input.setPlainText("hi")
    win._speak_checkbox.setChecked(False)
    win._on_send()
    app.processEvents()
    assert len(idents) == 1  # submit_fn called exactly once
    win.close()


def test_double_submit_while_busy_rejected(app, store):
    # NORUN leaves the turn in flight (busy); a 2nd submit must be rejected + kept
    win, ctrl = _window(app, store, _ok_submit(), runner=NORUN)
    win._input.setPlainText("first")
    win._speak_checkbox.setChecked(False)
    assert win.controller.submit("first", "ro", False) is True
    assert win.controller.submit("second", "ro", False) is False  # busy
    win.close()


# --- TTS via coordinator ----------------------------------------------------

def test_speak_on_routes_through_coordinator(app, store):
    eng = MockEngine()
    coord = TTSCoordinator(eng)
    win, ctrl = _window(app, store, _ok_submit(), coordinator=coord)
    win._input.setPlainText("vorbeste")
    win._speak_checkbox.setChecked(True)
    win._on_send()
    app.processEvents()
    assert eng.spoken and eng.spoken[0].startswith("Salut! Răspuns")
    win.close()


def test_speak_off_no_tts(app, store):
    eng = MockEngine()
    coord = TTSCoordinator(eng)
    win, ctrl = _window(app, store, _ok_submit(), coordinator=coord)
    win._input.setPlainText("taci")
    win._speak_checkbox.setChecked(False)
    win._on_send()
    app.processEvents()
    assert eng.spoken == []
    win.close()


# --- stop -------------------------------------------------------------------

def test_stop_while_thinking_sets_cancelled(app, store):
    win, ctrl = _window(app, store, _ok_submit(), runner=NORUN)
    win._input.setPlainText("lent")
    win._speak_checkbox.setChecked(False)
    ctrl.submit("lent", "ro", False)
    ctrl.stop()
    assert ctrl.state == STATE_CANCELLED
    win.close()


# --- multi-conversation -----------------------------------------------------

def test_new_conversation_switches_active(app, store):
    win, ctrl = _window(app, store, _ok_submit())
    first = ctrl.active_conversation_id
    ctrl.new_conversation()
    app.processEvents()
    assert ctrl.active_conversation_id != first
    win.close()


def test_switch_conversation_does_not_cancel_running_turn(app, store):
    win, ctrl = _window(app, store, _ok_submit(), runner=NORUN)
    win._input.setPlainText("in A")
    win._speak_checkbox.setChecked(False)
    ctrl.submit("in A", "ro", False)   # in flight in conversation A
    origin = ctrl._active_turn_conv
    ctrl.switch_conversation("some-other-conv")
    # the in-flight turn's cancellation must NOT be flipped by a view switch
    assert ctrl._current_ctx.cancellation.cancelled is False
    assert ctrl._active_turn_conv == origin
    win.close()


def test_reload_shows_persisted_messages_of_active(app, store):
    # seed a conversation directly in the store, then open a window on it
    store.ensure_conversation("seeded", title_seed="Titlu prestabilit")
    store.append_user_message("seeded", "m1", "salut", source="text", turn_id="t1")
    store.append_or_update_assistant_message("seeded", "m2", "buna", status="completed", turn_id="t1")
    win, ctrl = _window(app, store, _ok_submit())
    ctrl.switch_conversation("seeded")
    win._load_active_messages()
    app.processEvents()
    assert _card_count(win) == 2
    win.close()


# --- rendering safety in the card ------------------------------------------

def test_message_card_renders_safe_markdown(app):
    card = MessageCard("assistant", "**bold** <script>alert(1)</script>")
    # find the body label
    from PyQt6.QtWidgets import QLabel
    bodies = [w for w in card.findChildren(QLabel)
              if w.property("class") == "msgBody"]
    assert bodies, "no body label found"
    html = bodies[0].text()
    assert "<b>bold</b>" in html
    assert "<script>" not in html and "&lt;script&gt;" in html


def test_message_card_stores_raw_text_for_copy(app):
    raw = "răspuns cu **markdown** și diacritice"
    card = MessageCard("assistant", raw)
    assert card._raw_text == raw
