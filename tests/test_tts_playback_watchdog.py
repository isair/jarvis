"""Playback watchdog for PiperTTS (stuck-output recovery).

A wireless output device can drop mid-playback so the OutputStream never reports
`active == False`; the single TTS worker would then spin forever in the playback
wait-loop, stranding `_is_speaking == True` and blocking ALL further voice/chat
TTS. The watchdog bounds the wait to (audio duration + margin), aborts the stuck
stream, guarantees `_audio_stream=None` / `_is_speaking=False`, fires the
completion callback exactly once (recovery), and lets the next queued item run.

These tests fake audio (no device/model) and shrink the margin so a stuck stream
times out in a fraction of a second.
"""

from __future__ import annotations

import sys
import threading
import time
import types

import pytest

import jarvis.output.tts as ttsmod
from jarvis.output.tts import PiperTTS, _TTSItem


class _FakeChunk:
    def __init__(self, arr):
        self.audio_int16_array = arr


class _FakeVoice:
    def synthesize(self, text, syn_config):
        import numpy as np
        yield _FakeChunk(np.zeros(256, dtype=np.int16))


class _StuckStream:
    """An OutputStream that NEVER goes inactive (the wireless-drop symptom)."""
    KEEP_ACTIVE = True

    def __init__(self, **kw):
        self.active = _StuckStream.KEEP_ACTIVE
        self.aborted = False

    def start(self):
        pass

    def abort(self):
        self.aborted = True
        self.active = False

    def close(self):
        self.active = False


class _FakeSD:
    CallbackAbort = type("CallbackAbort", (Exception,), {})
    CallbackStop = type("CallbackStop", (Exception,), {})
    OutputStream = _StuckStream


@pytest.fixture
def fake_audio(monkeypatch):
    _StuckStream.KEEP_ACTIVE = True
    monkeypatch.setitem(sys.modules, "sounddevice", _FakeSD)
    pcfg = types.ModuleType("piper.config")
    pcfg.SynthesisConfig = type("SynthesisConfig", (), {"__init__": lambda self, **k: None})
    monkeypatch.setitem(sys.modules, "piper.config", pcfg)
    # Shrink the watchdog margin so a stuck stream times out fast.
    monkeypatch.setattr(ttsmod, "_PLAYBACK_TIMEOUT_MARGIN_SEC", 0.3, raising=True)
    # Capture the timeout marker.
    seen = []
    orig = ttsmod.debug_log
    monkeypatch.setattr(ttsmod, "debug_log", lambda m, c="debug": seen.append(str(m)))
    yield seen


def _stub(per_item: bool) -> PiperTTS:
    tts = PiperTTS(enabled=True, per_item_callbacks=per_item)
    tts._ensure_initialized = lambda: True
    tts._voice = _FakeVoice()
    tts._thread = object()  # don't spawn a lazy worker on speak()
    return tts


def _run_one(tts, item):
    t = threading.Thread(target=tts._speak_once, args=(item,), daemon=True)
    t.start()
    return t


def test_stuck_stream_times_out_and_unblocks(fake_audio):
    tts = _stub(per_item=True)
    fired = []
    item = _TTSItem("hello", completion_callback=lambda: fired.append(1))
    t = _run_one(tts, item)
    t.join(timeout=3.0)
    assert not t.is_alive()                 # watchdog returned (no infinite hang)
    assert tts.is_speaking() is False        # _is_speaking cleared
    assert tts._audio_stream is None         # stream released
    assert fired == [1]                      # completion fired exactly once (recovery)
    assert any("timed out" in m and "aborted" in m for m in fake_audio)


def test_next_item_processed_after_timeout(fake_audio):
    """The worker must recover and play the next queued item after a timeout."""
    tts = _stub(per_item=True)
    order = []
    a = _TTSItem("a", completion_callback=lambda: order.append("a"))
    b = _TTSItem("b", completion_callback=lambda: order.append("b"))
    t1 = _run_one(tts, a); t1.join(timeout=3.0)
    t2 = _run_one(tts, b); t2.join(timeout=3.0)
    assert not t1.is_alive() and not t2.is_alive()
    assert order == ["a", "b"]               # both recovered, in order


@pytest.mark.parametrize("per_item", [True, False])
def test_completion_exactly_once_on_timeout(fake_audio, per_item):
    tts = _stub(per_item=per_item)
    n = []
    if per_item:
        item = _TTSItem("x", completion_callback=lambda: n.append(1))
    else:
        tts._completion_callback = lambda: n.append(1)  # OFF reads the instance slot
        item = _TTSItem("x", completion_callback=None)
    t = _run_one(tts, item); t.join(timeout=3.0)
    assert not t.is_alive()
    assert n == [1]                          # exactly once, both flag states


def test_after_timeout_is_speaking_false_can_speak_again(fake_audio):
    """coordinator.is_speaking() delegates to engine.is_speaking(); it must be
    False after a timeout so chat/voice can dispatch speak() again."""
    tts = _stub(per_item=True)
    t = _run_one(tts, _TTSItem("first", completion_callback=lambda: None))
    t.join(timeout=3.0)
    assert tts.is_speaking() is False
    # a second playback still works (recovers again)
    fired = []
    t2 = _run_one(tts, _TTSItem("second", completion_callback=lambda: fired.append(1)))
    t2.join(timeout=3.0)
    assert not t2.is_alive() and fired == [1]


def test_normal_flow_unchanged_no_timeout(fake_audio):
    """A stream that completes promptly must NOT trip the watchdog."""
    _StuckStream.KEEP_ACTIVE = False          # completes instantly
    tts = _stub(per_item=True)
    fired = []
    t = _run_one(tts, _TTSItem("ok", completion_callback=lambda: fired.append(1)))
    t.join(timeout=3.0)
    assert not t.is_alive() and fired == [1]
    assert not any("timed out" in m for m in fake_audio)   # watchdog silent


def test_interrupt_during_stuck_stream_no_deadlock(fake_audio):
    tts = _stub(per_item=True)
    fired = []
    t = _run_one(tts, _TTSItem("x", completion_callback=lambda: fired.append(1)))
    t0 = time.monotonic()
    while tts._audio_stream is None and time.monotonic() - t0 < 2.0:
        time.sleep(0.005)
    tts._should_interrupt.set()               # user barge-in before the deadline
    t.join(timeout=3.0)
    assert not t.is_alive()
    assert fired == []                        # interrupt suppresses completion (unchanged)


def test_shutdown_with_stuck_stream_no_hang(fake_audio):
    tts = PiperTTS(enabled=True, per_item_callbacks=True)
    tts._ensure_initialized = lambda: True
    tts._voice = _FakeVoice()
    tts.start()
    tts.speak("x", completion_callback=lambda: None)
    t0 = time.monotonic()
    tts.stop()                                 # must not hang even mid-stuck-playback
    assert time.monotonic() - t0 < 3.0
    assert tts._thread is None
