"""Phase 3B.1 — per-item TTS callbacks (T-F1 root fix).

Verifies that, with the flag ON, each queued TTS item carries and fires its OWN
completion/duration callbacks (so a later speak() cannot clobber an earlier
item's callback), while with the flag OFF behaviour is byte-for-byte Phase-3A
(instance-slot callbacks, incl. the documented T-F1 clobber).

No real audio device / Piper model: sounddevice is faked, synthesis is stubbed,
and _speak_once is driven directly for determinism. A couple of worker-thread
tests cover FIFO order + shutdown.
"""

from __future__ import annotations

import sys
import threading
import time
import types

import pytest

from jarvis.output.tts import PiperTTS, _TTSItem


# --- fakes ------------------------------------------------------------------

class _FakeChunk:
    def __init__(self, arr):
        self.audio_int16_array = arr


class _FakeVoice:
    def synthesize(self, text, syn_config):
        import numpy as np
        yield _FakeChunk(np.zeros(256, dtype=np.int16))


class _FakeOutputStream:
    KEEP_ACTIVE = False  # False => playback "completes" instantly

    def __init__(self, **kw):
        self.active = _FakeOutputStream.KEEP_ACTIVE

    def start(self):
        pass

    def abort(self):
        self.active = False

    def close(self):
        self.active = False


class _FakeSD:
    CallbackAbort = type("CallbackAbort", (Exception,), {})
    CallbackStop = type("CallbackStop", (Exception,), {})
    OutputStream = _FakeOutputStream


@pytest.fixture
def fake_audio(monkeypatch):
    """Fake sounddevice so _speak_once needs no audio device."""
    _FakeOutputStream.KEEP_ACTIVE = False
    monkeypatch.setitem(sys.modules, "sounddevice", _FakeSD)
    # Lightweight piper.config stub so the SynthesisConfig import never touches
    # the real library.
    pcfg = types.ModuleType("piper.config")

    class SynthesisConfig:
        def __init__(self, **kw):
            pass

    pcfg.SynthesisConfig = SynthesisConfig
    monkeypatch.setitem(sys.modules, "piper.config", pcfg)
    yield


def _stub_tts(per_item: bool) -> PiperTTS:
    tts = PiperTTS(enabled=True, per_item_callbacks=per_item)
    tts._ensure_initialized = lambda: True   # no model load
    tts._voice = _FakeVoice()
    tts._thread = object()                    # prevent speak() from spawning a worker
    return tts


def _drain(tts):
    items = []
    while not tts._q.empty():
        items.append(tts._q.get_nowait())
    return items


# --- per-item dispatch (flag ON) -------------------------------------------

def test_per_item_completion_three_consecutive(fake_audio):
    tts = _stub_tts(per_item=True)
    calls = []
    for name in ("a", "b", "c"):
        tts.speak(name, completion_callback=lambda n=name: calls.append(n))
    for item in _drain(tts):
        tts._speak_once(item)
    assert calls == ["a", "b", "c"]  # each item fired its OWN callback, in order


def test_two_callers_interleaved_no_crossfire(fake_audio):
    """Simulate voice + chat callbacks interleaved on the shared engine."""
    tts = _stub_tts(per_item=True)
    voice, chat = [], []
    tts.speak("v1", completion_callback=lambda: voice.append(1))   # voice
    tts.speak("c1", completion_callback=lambda: chat.append(1))    # chat
    tts.speak("v2", completion_callback=lambda: voice.append(2))   # voice
    for item in _drain(tts):
        tts._speak_once(item)
    assert voice == [1, 2] and chat == [1]  # zero cross-fire


def test_completion_fires_exactly_once(fake_audio):
    tts = _stub_tts(per_item=True)
    n = []
    tts.speak("x", completion_callback=lambda: n.append(1))
    (item,) = _drain(tts)
    tts._speak_once(item)
    assert n == [1]  # exactly once


def test_per_item_duration_dispatch(fake_audio):
    tts = _stub_tts(per_item=True)
    dur = {}
    tts.speak("a", completion_callback=None, duration_callback=lambda d: dur.setdefault("a", d))
    tts.speak("b", completion_callback=None, duration_callback=lambda d: dur.setdefault("b", d))
    for item in _drain(tts):
        tts._speak_once(item)
    assert set(dur) == {"a", "b"}
    assert dur["a"] > 0 and dur["b"] > 0  # each got its own item's duration


def test_callback_exception_does_not_kill_worker(fake_audio):
    tts = _stub_tts(per_item=True)
    ok = []

    def boom():
        raise RuntimeError("callback boom")

    tts.speak("a", completion_callback=boom)
    tts.speak("b", completion_callback=lambda: ok.append(1))
    items = _drain(tts)
    tts._speak_once(items[0])   # must NOT raise
    tts._speak_once(items[1])
    assert ok == [1]  # the throwing callback did not stop later items


# --- the T-F1 regression witness -------------------------------------------

def test_flag_off_reproduces_tf1_clobber(fake_audio):
    """Flag OFF = Phase-3A: the 2nd speak() clobbers the shared slot, so item A's
    completion fires B's callback and A's callback is lost. This documents T-F1
    and proves the flag toggles it."""
    tts = _stub_tts(per_item=False)
    a, b = [], []
    tts.speak("a", completion_callback=lambda: a.append(1))
    tts.speak("b", completion_callback=lambda: b.append(1))  # clobbers the slot
    items = _drain(tts)
    tts._speak_once(items[0])  # fires whatever is in the slot now => B
    tts._speak_once(items[1])  # slot already cleared => nothing
    assert a == [] and b == [1]  # the clobber: A lost, B fired at the wrong time


def test_flag_on_fixes_tf1(fake_audio):
    tts = _stub_tts(per_item=True)
    a, b = [], []
    tts.speak("a", completion_callback=lambda: a.append(1))
    tts.speak("b", completion_callback=lambda: b.append(1))
    items = _drain(tts)
    tts._speak_once(items[0])
    tts._speak_once(items[1])
    assert a == [1] and b == [1]  # each callback fires for its own item


def test_per_item_none_callback_does_not_leak_prior_slot(fake_audio):
    """Under flag ON, an item whose OWN completion_callback is None must fire
    NOTHING — even though a prior speak() left a callback in the instance-slot
    mirror. Guards against a future refactor re-introducing the slot fallback."""
    tts = _stub_tts(per_item=True)
    a, b = [], []
    tts.speak("a", completion_callback=lambda: a.append(1))
    tts.speak("b", completion_callback=None)   # own cb is None; slot may hold a's
    items = _drain(tts)
    tts._speak_once(items[0])
    tts._speak_once(items[1])
    assert a == [1] and b == []  # b fired nothing; a not double-fired via the slot


# --- flag-OFF parity (single item unchanged) --------------------------------

def test_flag_off_single_item_completion_fires(fake_audio):
    tts = _stub_tts(per_item=False)
    n = []
    tts.speak("x", completion_callback=lambda: n.append(1))
    (item,) = _drain(tts)
    tts._speak_once(item)
    assert n == [1]  # Phase-3A single-item behaviour preserved


def test_flag_off_duration_fires(fake_audio):
    tts = _stub_tts(per_item=False)
    got = []
    tts.speak("x", completion_callback=None, duration_callback=lambda d: got.append(d))
    (item,) = _drain(tts)
    tts._speak_once(item)
    assert len(got) == 1 and got[0] > 0


# --- interrupt suppresses the item's callback -------------------------------

def test_stale_callback_suppressed_on_interrupt(fake_audio):
    """An interrupt DURING playback must suppress that item's completion
    callback (the `not interrupted` guard), under per-item dispatch."""
    _FakeOutputStream.KEEP_ACTIVE = True     # stream stays 'active' until aborted
    tts = _stub_tts(per_item=True)
    fired = []
    item = _TTSItem("hello", completion_callback=lambda: fired.append(1))
    t = threading.Thread(target=tts._speak_once, args=(item,))
    t.start()
    # Sync on ACTUAL playback start rather than a fixed sleep: _audio_stream is
    # assigned AFTER _should_interrupt.clear() at _speak_once entry, so once it
    # is non-None the clear has run and our interrupt cannot be swallowed.
    t0 = time.monotonic()
    while tts._audio_stream is None and time.monotonic() - t0 < 2.0:
        time.sleep(0.005)
    tts._should_interrupt.set()  # interrupt mid-playback
    t.join(timeout=3.0)
    assert not t.is_alive()
    assert fired == []           # interrupted => callback NOT fired


# --- worker-thread FIFO + shutdown -----------------------------------------

def test_fifo_order_via_worker(fake_audio):
    tts = PiperTTS(enabled=True, per_item_callbacks=True)
    tts._ensure_initialized = lambda: True
    tts._voice = _FakeVoice()
    order = []
    tts.start()  # real worker thread
    try:
        for name in ("a", "b", "c"):
            tts.speak(name, completion_callback=lambda n=name: order.append(n))
        t0 = time.monotonic()
        while len(order) < 3 and time.monotonic() - t0 < 5.0:
            time.sleep(0.02)
    finally:
        tts.stop()
    assert order == ["a", "b", "c"]


def test_shutdown_with_active_and_queued_items(fake_audio):
    tts = PiperTTS(enabled=True, per_item_callbacks=True)
    tts._ensure_initialized = lambda: True
    tts._voice = _FakeVoice()
    tts.start()
    for name in ("a", "b", "c"):
        tts.speak(name, completion_callback=lambda: None)
    t0 = time.monotonic()
    tts.stop()                       # must not hang
    assert time.monotonic() - t0 < 3.0
    assert tts._thread is None       # worker reaped


# --- public API + config wiring --------------------------------------------

def test_speak_signature_unchanged(fake_audio):
    tts = _stub_tts(per_item=True)
    # positional + keyword both accepted (backward-compatible signature)
    tts.speak("a", (lambda: None), (lambda d: None))
    tts.speak("b", completion_callback=lambda: None, duration_callback=lambda d: None)
    assert len(_drain(tts)) == 2


def test_default_is_off():
    tts = PiperTTS(enabled=False)          # no per_item arg
    assert tts._per_item_callbacks is False


def test_factory_passes_flag():
    from jarvis.output.tts import create_tts_engine
    tts = create_tts_engine(engine="piper", enabled=False, piper_per_item_callbacks=True)
    assert tts._per_item_callbacks is True
    tts_off = create_tts_engine(engine="piper", enabled=False)
    assert tts_off._per_item_callbacks is False


def test_config_has_flag_default_false(tmp_path):
    import json, os
    from jarvis.config import load_settings
    p = tmp_path / "config.json"
    p.write_text(json.dumps({}), encoding="utf-8")
    old = os.environ.get("JARVIS_CONFIG_PATH")
    os.environ["JARVIS_CONFIG_PATH"] = str(p)
    try:
        s = load_settings()
        assert s.tts_per_item_callbacks is False
        p.write_text(json.dumps({"tts_per_item_callbacks": True}), encoding="utf-8")
        assert load_settings().tts_per_item_callbacks is True
    finally:
        if old is None:
            os.environ.pop("JARVIS_CONFIG_PATH", None)
        else:
            os.environ["JARVIS_CONFIG_PATH"] = old
