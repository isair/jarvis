"""Offline tests for the Phase 2 chat UI (desktop_app.chat_window) + submit path.

No real model, microphone, GPU, network, or TTS is used. Qt runs on the
offscreen platform; all reply/TTS dependencies are mocked. The worker executor
is injected so tests run deterministically without a live QThread. Tests verify
behaviours: input rules, turn wiring, result/TTS handling, threading, feature flag.
"""

from __future__ import annotations

import os
import threading
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from desktop_app.chat_window import (
    ChatController, ChatWindow, enter_sends, is_submittable,
    STATE_CANCELLED, STATE_FAILED, STATE_READY, STATE_SPEAKING, STATE_THINKING,
)
from jarvis.core.turn import (
    AssistantStatus, MemoryStatus, StructuredError, TurnResult, TurnStatus,
    VerificationStatus, new_assistant_message, new_turn,
)

# Runners: how a ChatReplyWorker is executed in a test.
SYNC = lambda w: w.run()          # inline: signals fire directly, deterministic
NORUN = lambda w: None            # never runs: leaves the controller "busy"


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


class FakeTTS:
    def __init__(self, enabled=True, speaking=False, raise_on_speak=False):
        self.enabled = enabled
        self._speaking = speaking
        self.raise_on_speak = raise_on_speak
        self.spoke = []
        self.interrupted = 0

    def is_speaking(self):
        return self._speaking

    def speak(self, text, completion_callback=None, duration_callback=None):
        if self.raise_on_speak:
            raise RuntimeError("tts boom")
        self.spoke.append(text)
        self._speaking = False
        if completion_callback:
            completion_callback()

    def interrupt(self):
        self.interrupted += 1
        self._speaking = False


def _ok_result(ctx, text="buna, cu ce te pot ajuta?"):
    am = new_assistant_message(ctx)
    am.text = text
    am.status = AssistantStatus.COMPLETED
    return TurnResult(status=TurnStatus.COMPLETED, assistant_message=am,
                      verification=VerificationStatus.SKIPPED,
                      memory_status=MemoryStatus.WRITTEN)


def _pump_until(app, predicate, timeout=5.0):
    t0 = time.monotonic()
    while not predicate() and time.monotonic() - t0 < timeout:
        app.processEvents()
        time.sleep(0.005)
    return predicate()


class Recorder:
    def __init__(self, controller):
        self.events = []
        controller.user_message_added.connect(lambda t: self.events.append(("user", t)))
        controller.assistant_message_added.connect(lambda t: self.events.append(("assistant", t)))
        controller.error_message.connect(lambda t: self.events.append(("error", t)))
        controller.tts_unavailable.connect(lambda: self.events.append(("tts_unavailable", None)))
        controller.state_changed.connect(lambda s: self.events.append(("state", s)))
        controller.input_restore.connect(lambda t: self.events.append(("restore", t)))

    def kinds(self):
        return [k for k, _ in self.events]


# ===========================================================================
# A. UI / input
# ===========================================================================

def test_enter_sends_shift_enter_newline():
    assert enter_sends(Qt.Key.Key_Return, Qt.KeyboardModifier.NoModifier) is True
    assert enter_sends(Qt.Key.Key_Enter, Qt.KeyboardModifier.NoModifier) is True
    assert enter_sends(Qt.Key.Key_Return, Qt.KeyboardModifier.ShiftModifier) is False
    assert enter_sends(Qt.Key.Key_A, Qt.KeyboardModifier.NoModifier) is False


def test_whitespace_only_rejected(app):
    ctrl = ChatController(submit_fn=lambda u, c: _ok_result(c), runner=SYNC)
    assert ctrl.submit("   \n\t", "ro", False) is False
    assert is_submittable("   ") is False and is_submittable(None) is False


def test_busy_blocks_second_submit_one_turn(app):
    calls = []
    ctrl = ChatController(submit_fn=lambda u, c: calls.append(u.text) or _ok_result(c),
                          runner=NORUN)  # worker never completes -> stays busy
    assert ctrl.submit("first", "ro", False) is True
    assert ctrl.busy is True
    assert ctrl.submit("second", "ro", False) is False   # blocked while busy
    assert calls == []                                    # NORUN: engine not reached


def test_one_turn_calls_submit_once(app):
    calls = []
    ctrl = ChatController(submit_fn=lambda u, c: calls.append(u.text) or _ok_result(c),
                          tts_getter=lambda: FakeTTS(enabled=False), runner=SYNC)
    ctrl.submit("hello", "ro", False)
    assert calls == ["hello"] and ctrl.busy is False


def test_speak_default_on_window(app):
    win = ChatWindow(cfg=None, controller=ChatController())
    assert win.speak_checkbox.isChecked() is True
    win.close()


def test_send_disabled_during_thinking(app):
    win = ChatWindow(cfg=None, controller=ChatController())
    win.controller.state_changed.emit(STATE_THINKING)
    assert win.send_button.isEnabled() is False
    win.controller.state_changed.emit(STATE_READY)
    assert win.send_button.isEnabled() is True
    win.close()


def test_history_is_selectable_copyable(app):
    win = ChatWindow(cfg=None, controller=ChatController())
    flags = win.history.textInteractionFlags()
    assert flags & Qt.TextInteractionFlag.TextSelectableByMouse
    win.close()


def test_shutdown_cancels_active_turn(app):
    ctrl = ChatController(submit_fn=lambda u, c: _ok_result(c), runner=NORUN)
    ctrl.submit("hi", "ro", False)
    ctx = ctrl._current_ctx
    ctrl.shutdown()
    assert ctx.cancellation.cancelled is True


# ===========================================================================
# B. Turn wiring
# ===========================================================================

def test_turn_wiring_source_text_stable_conversation_unique_ids(app):
    seen = []
    ctrl = ChatController(conversation_id="conv-fixed",
                          submit_fn=lambda u, c: seen.append(
                              (u.source.value, u.conversation_id, c.turn_id, c.correlation_id)) or _ok_result(c),
                          tts_getter=lambda: FakeTTS(enabled=False), runner=SYNC)
    ctrl.submit("one", "ro", False)
    ctrl.submit("two", "ro", False)
    assert len(seen) == 2
    assert all(s[0] == "text" for s in seen)                 # source=text
    assert seen[0][1] == seen[1][1] == "conv-fixed"          # stable conversation_id
    assert seen[0][2] != seen[1][2]                          # unique turn_id
    assert seen[0][3] != seen[1][3]                          # unique correlation_id


def test_submit_reports_memory_truthfully_and_never_double_writes():
    """submit_user_text never writes memory itself; memory_status reflects whether
    the engine actually wrote (WRITTEN) vs a no-write local-answer path (SKIPPED)."""
    import jarvis.daemon as d
    import jarvis.reply.engine as eng
    added = []
    class DM:
        def __init__(self): self._messages = []
        def add_message(self, role, content): added.append((role, content))
    dm = DM()
    d._global_cfg = object(); d._global_db = object(); d._global_dialogue_memory = dm
    orig = eng.run_reply_engine
    try:
        # (a) engine persists the turn (appends user+assistant) -> WRITTEN
        def writing_engine(db, cfg, tts, text, dmx, language=None):
            dmx._messages.append((0.0, "user", text))
            dmx._messages.append((0.0, "assistant", "hi"))
            return "hi"
        eng.run_reply_engine = writing_engine
        t = new_turn("text", "salut", "ro")
        res = d.submit_user_text(t.user_message, t.context)
        assert res.status == TurnStatus.COMPLETED
        assert res.memory_status == MemoryStatus.WRITTEN

        # (b) engine returns text WITHOUT writing (local-answer) -> SKIPPED, not WRITTEN
        eng.run_reply_engine = lambda db, cfg, tts, text, dmx, language=None: "2 plus 2 fac 4."
        t2 = new_turn("text", "cat fac 2+2", "ro")
        res2 = d.submit_user_text(t2.user_message, t2.context)
        assert res2.status == TurnStatus.COMPLETED
        assert res2.memory_status == MemoryStatus.SKIPPED
    finally:
        eng.run_reply_engine = orig
        d._global_cfg = d._global_db = d._global_dialogue_memory = None
    assert added == []          # submit_user_text itself never called add_message


def test_submit_user_text_passes_none_tts_and_language():
    import jarvis.daemon as d
    import jarvis.reply.engine as eng
    captured = {}
    d._global_cfg = object(); d._global_db = object()
    class DM:
        def add_message(self, *a, **k): pass
    d._global_dialogue_memory = DM()
    orig = eng.run_reply_engine
    def fake(db, cfg, tts, text, dm, language=None):
        captured.update(tts=tts, text=text, language=language)
        return "ok"
    eng.run_reply_engine = fake
    try:
        t = new_turn("text", "buna", "ro")
        d.submit_user_text(t.user_message, t.context)
    finally:
        eng.run_reply_engine = orig
        d._global_cfg = d._global_db = d._global_dialogue_memory = None
    assert captured["tts"] is None and captured["text"] == "buna" and captured["language"] == "ro"


# ===========================================================================
# C. Result / TTS
# ===========================================================================

def test_reply_text_before_tts(app):
    tts = FakeTTS(enabled=True)
    ctrl = ChatController(submit_fn=lambda u, c: _ok_result(c), tts_getter=lambda: tts, runner=SYNC)
    rec = Recorder(ctrl)
    ctrl.submit("hi", "ro", True)
    kinds = rec.kinds()
    assert "assistant" in kinds
    speaking_states = [i for i, (k, v) in enumerate(rec.events) if k == "state" and v == STATE_SPEAKING]
    if speaking_states:
        assert kinds.index("assistant") < speaking_states[0]   # text before Speaking
    assert tts.spoke == ["buna, cu ce te pot ajuta?"]


def test_tts_failure_keeps_text_not_failed(app):
    tts = FakeTTS(enabled=True, raise_on_speak=True)
    ctrl = ChatController(submit_fn=lambda u, c: _ok_result(c), tts_getter=lambda: tts, runner=SYNC)
    rec = Recorder(ctrl)
    ctrl.submit("hi", "ro", True)
    assert ("assistant", "buna, cu ce te pot ajuta?") in rec.events   # text kept
    assert ("tts_unavailable", None) in rec.events
    assert ctrl.state == STATE_READY                                  # NOT failed
    assert tts.spoke == []


def test_speak_off_does_not_call_tts(app):
    tts = FakeTTS(enabled=True)
    ctrl = ChatController(submit_fn=lambda u, c: _ok_result(c), tts_getter=lambda: tts, runner=SYNC)
    ctrl.submit("hi", "ro", False)          # speak_response=False
    assert tts.spoke == [] and ctrl.state == STATE_READY


def test_late_result_after_stop_not_success(app):
    ctrl = ChatController()
    turn = new_turn("text", "hi", "ro")
    ctrl._current_ctx = turn.context
    rec = Recorder(ctrl)
    turn.context.cancellation.cancel("user stop")
    ctrl._on_text_ready(_ok_result(turn.context))
    assert ("assistant", "buna, cu ce te pot ajuta?") not in rec.events   # ignored
    assert ctrl.state == STATE_CANCELLED


def test_stop_interrupts_active_tts(app):
    tts = FakeTTS(enabled=True, speaking=True)
    ctrl = ChatController(tts_getter=lambda: tts)
    turn = new_turn("text", "hi", "ro")
    ctrl._current_ctx = turn.context
    ctrl._busy = True
    ctrl.stop()
    assert tts.interrupted == 1
    assert turn.context.cancellation.cancelled is True
    assert ctrl.state == STATE_CANCELLED


def test_blocked_result_restores_input(app):
    blocked = TurnResult(status=TurnStatus.BLOCKED, structured_error=StructuredError("busy", "busy"))
    ctrl = ChatController(submit_fn=lambda u, c: blocked, runner=SYNC)
    rec = Recorder(ctrl)
    ctrl.submit("keep me", "ro", False)
    assert ("restore", "keep me") in rec.events and ctrl.state == STATE_READY


# ===========================================================================
# D. Feature flag
# ===========================================================================

def test_feature_flag_defaults_off_others_untouched():
    # Assert the CODE default (hermetic) — not the machine's merged config.json.
    from jarvis.config import get_default_config
    d = get_default_config()
    assert d["chat_ui_enabled"] is False               # OFF by default
    assert d["openai_realtime_enabled"] is False        # Realtime untouched
    assert d["conversation_learning_enabled"] is False  # Learning untouched


def test_feature_flag_in_settings_metadata():
    from desktop_app.settings_window import FIELD_METADATA
    fm = [f for f in FIELD_METADATA if f.key == "chat_ui_enabled"]
    assert len(fm) == 1 and fm[0].field_type == "bool" and fm[0].category == "features"


# ===========================================================================
# E. Threading
# ===========================================================================

def test_reply_runs_off_ui_thread_single_worker(app):
    main_ident = threading.get_ident()
    idents = []
    def submit_fn(u, c):
        idents.append(threading.get_ident())
        time.sleep(0.02)
        return _ok_result(c)
    def thread_runner(worker):
        threading.Thread(target=worker.run, daemon=True).start()
    ctrl = ChatController(submit_fn=submit_fn, tts_getter=lambda: FakeTTS(enabled=False),
                          runner=thread_runner)
    assert ctrl.submit("hi", "ro", False) is True
    assert ctrl.submit("again", "ro", False) is False    # no second worker while busy
    assert _pump_until(app, lambda: not ctrl.busy)
    assert len(idents) == 1 and idents[0] != main_ident  # one worker, off the UI thread


def test_qthread_runner_reaps_thread_no_orphan(app):
    """Production path (default runner = real QThread): the thread is reaped and
    references cleared only after it finishes — no orphan / 'destroyed while running'."""
    ctrl = ChatController(submit_fn=lambda u, c: _ok_result(c),
                          tts_getter=lambda: FakeTTS(enabled=False))  # runner=None -> QThread
    assert ctrl.submit("hi", "ro", False) is True
    assert _pump_until(app, lambda: ctrl._thread is None, timeout=8.0)  # reaped
    assert ctrl.busy is False and ctrl._worker is None


def test_qthread_shutdown_is_non_blocking_and_reaps(app):
    """shutdown() must not block the UI thread on a still-running reply, must mark
    cancellation, and the QThread must still reap itself once the reply completes."""
    gate = threading.Event()
    def slow_submit(u, c):
        gate.wait(2.0)
        return _ok_result(c)
    ctrl = ChatController(submit_fn=slow_submit, tts_getter=lambda: FakeTTS(enabled=False))
    ctrl.submit("hi", "ro", False)
    ctx = ctrl._current_ctx
    t0 = time.monotonic()
    ctrl.shutdown()
    assert time.monotonic() - t0 < 0.5           # did NOT block on wait()
    assert ctx.cancellation.cancelled is True     # marked cancellation
    gate.set()                                    # let the reply finish
    assert _pump_until(app, lambda: ctrl._thread is None, timeout=8.0)  # reaped cleanly


def test_stale_thread_finish_does_not_null_current_thread(app):
    """Identity guard: a stale thread finishing must NOT null the current
    thread/worker (which would drop the last ref to a running QThread)."""
    from PyQt6.QtCore import QThread
    ctrl = ChatController()
    current = QThread()
    sentinel_worker = object()
    ctrl._thread = current
    ctrl._worker = sentinel_worker
    # Direct call => sender() is None, which is NOT the current thread, so the
    # guard must refuse to null the still-current references.
    ctrl._on_thread_finished()
    assert ctrl._thread is current and ctrl._worker is sentinel_worker
