"""Mandatory tests for the Supertonic 3 (F5) TTS engine integration.

Covers the owner's must-preserve contract WITHOUT a real device, model, or child
process: Piper stays intact and selectable; Supertonic is a PiperTTS subclass so
the whole sacred playback lifecycle (queue / single worker / single-flight /
is_speaking / interrupt / shutdown / per-item callbacks / playback watchdog /
exactly-one-completion) is REUSED unchanged; long replies split only at complete
sentences; and any Supertonic failure yields exactly ONE Piper fallback for that
one response (never both voices, never a half-and-half reply, never during
shutdown/interrupt).
"""
from __future__ import annotations

import json
import os
import sys
import threading
import types
import wave

import numpy as np
import pytest

import jarvis.output.tts as ttsmod
from jarvis.output.tts import (
    ChatterboxTTS,
    PiperTTS,
    SupertonicTTS,
    _SupertonicService,
    _TTSItem,
    create_tts_engine,
    _SUPERTONIC_WHOLE_TEXT_MAX_CHARS as BUDGET,
)


# --------------------------------------------------------------------------- #
# Fakes                                                                        #
# --------------------------------------------------------------------------- #
class _FakeService:
    """Stand-in for the persistent pipe worker: writes a tiny real WAV per synth."""

    def __init__(self, tmpdir, sr=44100, fail_on=()):
        self.sr = sr
        self.tmpdir = str(tmpdir)
        self.calls = []
        self.fail_on = set(fail_on)  # indices (0-based) or exact texts that fail
        self.stopped = False

    def is_alive(self):
        return True

    def synth(self, text, voice, lang, steps, speed, timeout):
        idx = len(self.calls)
        self.calls.append((text, voice, lang, steps, speed, timeout))
        if idx in self.fail_on or text in self.fail_on:
            return None
        path = os.path.join(self.tmpdir, f"fake_{idx}.wav")
        n = int(self.sr * 0.1)
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.sr)
            w.writeframes(np.zeros(n, dtype="<i2").tobytes())
        return path

    def stop(self, *a, **k):
        self.stopped = True


class _StreamOK:
    """OutputStream that completes immediately (no watchdog trip)."""

    def __init__(self, **kw):
        self.active = False

    def start(self):
        pass

    def abort(self):
        self.active = False

    def close(self):
        self.active = False


class _FakeSD:
    CallbackAbort = type("CallbackAbort", (Exception,), {})
    CallbackStop = type("CallbackStop", (Exception,), {})
    OutputStream = _StreamOK


def _make(tmp_path, **kw):
    """A SupertonicTTS whose service is a fake (no real subprocess)."""
    tts = SupertonicTTS(runtime_path=str(tmp_path), enabled=True, **kw)
    return tts


def _attach_service(tts, svc):
    tts._ensure_service = lambda: True
    tts._service = svc


# --------------------------------------------------------------------------- #
# 1-3  Factory: Supertonic selectable, Piper & Chatterbox preserved            #
# --------------------------------------------------------------------------- #
def test_factory_supertonic_returns_supertonic(tmp_path):
    eng = create_tts_engine(engine="supertonic", supertonic_runtime_path=str(tmp_path))
    assert isinstance(eng, SupertonicTTS)
    assert isinstance(eng, PiperTTS)          # inherits the whole lifecycle
    assert eng._engine_label == "Supertonic TTS"


def test_factory_piper_still_default_and_selectable():
    assert isinstance(create_tts_engine(engine="piper"), PiperTTS)
    assert not isinstance(create_tts_engine(engine="piper"), SupertonicTTS)
    assert isinstance(create_tts_engine(), PiperTTS)   # default unchanged


def test_factory_chatterbox_preserved():
    assert isinstance(create_tts_engine(engine="chatterbox"), ChatterboxTTS)


# --------------------------------------------------------------------------- #
# 4-7  Long-text handling: whole text normally, sentence-only splitting        #
# --------------------------------------------------------------------------- #
def test_short_reply_is_one_whole_request():
    txt = "Bună ziua, maestre. Cu ce vă pot ajuta astăzi?"
    assert SupertonicTTS._split_for_supertonic(txt) == [txt]


def test_long_reply_splits_into_multiple_groups_within_budget():
    txt = " ".join(f"Aceasta este propoziția numărul {i} din test." for i in range(40))
    groups = SupertonicTTS._split_for_supertonic(txt)
    assert len(groups) > 1
    multi = [g for g in groups if len(_sents(g)) > 1]
    assert all(len(g) <= BUDGET for g in multi)   # every multi-sentence group fits


def test_split_preserves_every_sentence_in_order():
    txt = " ".join(f"Propoziția {i} aici." for i in range(40))
    groups = SupertonicTTS._split_for_supertonic(txt)
    joined = []
    for g in groups:
        joined += _sents(g)
    assert joined == _sents(txt)


def test_single_oversized_sentence_is_not_cut():
    long_sentence = "Cora " + "foarte " * 200 + "importantă."   # one sentence > budget
    groups = SupertonicTTS._split_for_supertonic(long_sentence)
    assert groups == [long_sentence.strip()]       # kept whole, never split mid-sentence


def _sents(s):
    import re
    return [x.strip() for x in re.split(r"(?<=[.!?])\s+", s.strip()) if x.strip()]


# --------------------------------------------------------------------------- #
# 8  WAV reading                                                               #
# --------------------------------------------------------------------------- #
def test_read_wav_int16_roundtrip(tmp_path):
    p = str(tmp_path / "x.wav")
    data = (np.arange(-5, 5, dtype="<i2"))
    with wave.open(p, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(44100)
        w.writeframes(data.tobytes())
    samples, sr = SupertonicTTS._read_wav_int16(p)
    assert sr == 44100
    assert samples.dtype == np.dtype("<i2")
    assert list(samples) == list(data)


# --------------------------------------------------------------------------- #
# 9-13  Render + fallback semantics                                            #
# --------------------------------------------------------------------------- #
def test_render_supertonic_success_returns_audio_and_sr(tmp_path):
    tts = _make(tmp_path)
    _attach_service(tts, _FakeService(tmp_path, sr=44100))
    out = tts._render_supertonic("Bună ziua, maestre.")
    assert out is not None
    audio, sr = out
    assert sr == 44100 and len(audio) > 0 and audio.dtype == np.dtype("<i2")


def test_success_path_never_calls_piper(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(ttsmod.PiperTTS, "_render_audio",
                        lambda self, text: calls.append(text) or (np.zeros(4, "<i2"), 22050))
    tts = _make(tmp_path)
    _attach_service(tts, _FakeService(tmp_path))
    out = tts._render_audio("Salut.")            # Supertonic succeeds
    assert out is not None and out[1] == 44100
    assert calls == []                            # Piper fallback NOT invoked (no double voice)


def test_service_failure_triggers_exactly_one_piper_fallback(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(ttsmod.PiperTTS, "_render_audio",
                        lambda self, text: calls.append(text) or (np.zeros(4, "<i2"), 22050))
    tts = _make(tmp_path)
    tts._ensure_service = lambda: False           # service unavailable
    out = tts._render_audio("Salut.")
    assert out is not None and out[1] == 22050    # Piper's sr (the fallback rendered)
    assert calls == ["Salut."]                    # exactly ONE Piper attempt


def test_chunk_failure_fails_whole_render_then_one_fallback(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(ttsmod.PiperTTS, "_render_audio",
                        lambda self, text: calls.append(text) or (np.zeros(4, "<i2"), 22050))
    tts = _make(tmp_path)
    # 2nd sentence-group fails -> whole Supertonic render aborts (no half reply).
    svc = _FakeService(tmp_path, fail_on={1})
    _attach_service(tts, svc)
    long_txt = " ".join(f"Propoziția numărul {i} din testul lung al vocii." for i in range(60))
    assert len(SupertonicTTS._split_for_supertonic(long_txt)) > 1
    out = tts._render_audio(long_txt)
    assert out[1] == 22050                         # fell back to Piper
    assert calls == [long_txt]                     # ONE Piper attempt on the FULL text


def test_no_fallback_during_interrupt(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(ttsmod.PiperTTS, "_render_audio",
                        lambda self, text: calls.append(text) or (np.zeros(4, "<i2"), 22050))
    tts = _make(tmp_path)
    tts._ensure_service = lambda: False
    tts._should_interrupt.set()                    # barge-in / stopping
    assert tts._render_audio("x") is None
    assert calls == []                             # no stray Piper clip during interrupt


def test_no_fallback_during_shutdown(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(ttsmod.PiperTTS, "_render_audio",
                        lambda self, text: calls.append(text) or (np.zeros(4, "<i2"), 22050))
    tts = _make(tmp_path)
    tts._ensure_service = lambda: False
    tts._stop.set()
    assert tts._render_audio("x") is None
    assert calls == []


# --------------------------------------------------------------------------- #
# 14-15  Service is fail-closed and safe                                       #
# --------------------------------------------------------------------------- #
def test_service_start_fail_closed_on_missing_runtime(tmp_path):
    svc = _SupertonicService(
        python_exe=str(tmp_path / "nope.exe"),
        service_py=str(tmp_path / "nope.py"),
        model_dir=str(tmp_path / "nomodel"),
        tmp_dir=str(tmp_path / "tmp"),
        warmup_voice="F5",
        log_path=str(tmp_path / "log.txt"),
    )
    assert svc.start() is False       # missing runtime -> fail closed, no exception
    assert svc.is_alive() is False


def test_service_stop_idempotent_when_never_started(tmp_path):
    svc = _SupertonicService("a", "b", "c", str(tmp_path), "F5", str(tmp_path / "l.txt"))
    svc.stop()                        # must not raise
    svc.stop()
    assert svc.is_alive() is False


# --------------------------------------------------------------------------- #
# 16-18  Lifecycle inheritance preserved end-to-end                            #
# --------------------------------------------------------------------------- #
def test_supertonic_inherits_full_interface(tmp_path):
    tts = _make(tmp_path)
    for attr in ("speak", "stop", "interrupt", "is_speaking",
                 "get_last_spoken_text", "_run", "_q", "_should_interrupt"):
        assert hasattr(tts, attr)
    assert tts.is_speaking() is False


@pytest.mark.parametrize("per_item", [True, False])
def test_speak_once_fires_completion_exactly_once(tmp_path, monkeypatch, per_item):
    monkeypatch.setitem(sys.modules, "sounddevice", _FakeSD)
    tts = _make(tmp_path, per_item_callbacks=per_item)
    _attach_service(tts, _FakeService(tmp_path))
    tts._thread = object()                        # don't spawn a worker on speak()
    n = []
    if per_item:
        item = _TTSItem("Salut.", completion_callback=lambda: n.append(1))
    else:
        tts._completion_callback = lambda: n.append(1)
        item = _TTSItem("Salut.", completion_callback=None)
    t = threading.Thread(target=tts._speak_once, args=(item,), daemon=True)
    t.start()
    t.join(timeout=3.0)
    assert not t.is_alive()
    assert n == [1]                               # exactly one completion, both flag states
    assert tts.is_speaking() is False


def test_speak_once_uses_service_sample_rate_for_playback(tmp_path, monkeypatch):
    """Playback must open the OutputStream at the ENGINE's 44.1 kHz, not Piper's."""
    seen = {}

    class _CapStream(_StreamOK):
        def __init__(self, **kw):
            super().__init__(**kw)
            seen["sr"] = kw.get("samplerate")

    fake = types.SimpleNamespace(
        CallbackAbort=_FakeSD.CallbackAbort,
        CallbackStop=_FakeSD.CallbackStop,
        OutputStream=_CapStream,
    )
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    tts = _make(tmp_path)
    _attach_service(tts, _FakeService(tmp_path, sr=44100))
    tts._thread = object()
    t = threading.Thread(target=tts._speak_once,
                         args=(_TTSItem("Salut.", completion_callback=lambda: None),), daemon=True)
    t.start()
    t.join(timeout=3.0)
    assert seen.get("sr") == 44100


# --------------------------------------------------------------------------- #
# Remediation of the adversarial review (desync / wedge / cooldown / shutdown) #
# --------------------------------------------------------------------------- #
class _FakeStdin:
    def __init__(self, on_write=None):
        self._on_write = on_write

    def write(self, s):
        if self._on_write:
            self._on_write(s)

    def flush(self):
        pass


class _FakeProc:
    """Minimal Popen stand-in for exercising _SupertonicService.request()."""

    def __init__(self, on_write=None):
        self._alive = True
        self.killed = False
        self.stdin = _FakeStdin(on_write)

    def poll(self):
        return None if self._alive else -9

    def kill(self):
        self.killed = True
        self._alive = False

    def wait(self, t=None):
        self._alive = False
        return -9


def test_request_timeout_kills_child_for_restart(tmp_path):
    """#1/#2: a synth that never replies must KILL the child (so a late reply
    can't desync a future request) and flip is_alive() False for a fresh restart."""
    svc = _SupertonicService("a", "b", "c", str(tmp_path), "F5", str(tmp_path / "l"))
    fp = _FakeProc()                      # writes go nowhere; queue stays empty
    svc._proc = fp
    rep = svc.request({"cmd": "health"}, timeout=0.2)
    assert rep is None
    assert fp.killed is True
    assert svc.is_alive() is False        # -> next _ensure_service restarts fresh


def test_request_id_correlation_discards_stale_reply(tmp_path):
    """#1: a late reply from a prior (timed-out) request carries an OLD id and is
    discarded; request() returns only the reply whose id matches THIS request."""
    svc = _SupertonicService("a", "b", "c", str(tmp_path), "F5", str(tmp_path / "l"))

    def on_write(s):
        rid = json.loads(s).get("id")
        svc._replies.put({"ok": True, "stale": True, "id": rid - 1})          # stale
        svc._replies.put({"ok": True, "status": "ready", "id": rid})          # ours

    svc._proc = _FakeProc(on_write)
    rep = svc.request({"cmd": "health"}, timeout=2.0)
    assert rep is not None
    assert rep.get("id") == 1 and rep.get("stale") is None   # got OURS, not the stale one


def test_empty_runtime_path_disabled_always_piper(monkeypatch):
    """#6: empty runtime_path disables the service entirely (never abspath('')==CWD)."""
    calls = []
    monkeypatch.setattr(ttsmod.PiperTTS, "_render_audio",
                        lambda self, text: calls.append(text) or (np.zeros(4, "<i2"), 22050))
    tts = SupertonicTTS(runtime_path="", enabled=True)
    assert tts._st_enabled is False
    assert tts._ensure_service() is False
    out = tts._render_audio("Salut.")
    assert out[1] == 22050 and calls == ["Salut."]   # went straight to Piper


def test_failed_start_opens_cooldown_no_respawn(tmp_path, monkeypatch):
    """#4: after a failed start, stay on Piper for a cooldown without respawning."""
    tts = SupertonicTTS(runtime_path=str(tmp_path), enabled=True)   # no .venv/service.py here
    assert tts._st_enabled is True
    assert tts._ensure_service() is False
    assert tts._st_fail_until > 0                    # cooldown opened
    spawned = []
    monkeypatch.setattr(ttsmod, "_SupertonicService",
                        lambda *a, **k: spawned.append(1))
    assert tts._ensure_service() is False            # within cooldown
    assert spawned == []                             # did NOT construct/spawn a new child


def test_completion_suppressed_during_shutdown(tmp_path, monkeypatch):
    """#5 + Finding 1: a render finishing AFTER stop() returned must NOT fire a
    late completion (mic-reopen/turn-release). stop() sets _closing and clears
    _stop after the join, so suppression must key off the PERSISTENT _closing
    flag — not the transient _stop (which is already cleared by then)."""
    monkeypatch.setitem(sys.modules, "sounddevice", _FakeSD)
    tts = _make(tmp_path, per_item_callbacks=True)
    _attach_service(tts, _FakeService(tmp_path))
    tts._thread = object()
    tts._closing.set()                               # stop() sets this and keeps it set
    tts._stop.clear()                                # stop() clears _stop after the join
    fired = []
    item = _TTSItem("Salut.", completion_callback=lambda: fired.append(1))
    t = threading.Thread(target=tts._speak_once, args=(item,), daemon=True)
    t.start()
    t.join(timeout=3.0)
    assert not t.is_alive()
    assert fired == []                               # suppressed via _closing (not _stop)


def test_consecutive_fallbacks_open_cooldown_and_teardown(tmp_path, monkeypatch):
    """#10: repeated real Supertonic failures pause it (cooldown) + tear the
    service down, so we stop paying the synth timeout on every response."""
    calls = []
    monkeypatch.setattr(ttsmod.PiperTTS, "_render_audio",
                        lambda self, text: calls.append(text) or (np.zeros(4, "<i2"), 22050))
    tts = _make(tmp_path)
    svc = _FakeService(tmp_path, fail_on=set(range(100)))   # every synth fails
    tts._ensure_service = lambda: True                       # service "alive"
    tts._service = svc
    tts._st_enabled = True
    n = ttsmod._SUPERTONIC_MAX_CONSECUTIVE_FAIL
    for _ in range(n):
        out = tts._render_audio("x")                         # attempts Supertonic, falls back
        assert out[1] == 22050
    assert tts._st_fail_until > 0                            # cooldown opened at the threshold
    assert svc.stopped is True                               # wedged service torn down
    assert tts._service is None
    assert len(calls) == n                                   # one Piper fallback per response


def test_healthy_render_resets_consecutive_fail(tmp_path):
    """A successful Supertonic render clears the consecutive-fallback counter."""
    tts = _make(tmp_path)
    _attach_service(tts, _FakeService(tmp_path))             # synth succeeds
    tts._st_consecutive_fail = 1
    out = tts._render_audio("Salut.")
    assert out is not None and out[1] == 44100
    assert tts._st_consecutive_fail == 0


def test_stop_does_not_hang_with_service_present(tmp_path):
    """#3: stop() tears down the service quickly (no blocking on a startup wait)."""
    tts = _make(tmp_path)
    svc = _FakeService(tmp_path)
    tts._service = svc
    tts._thread = None                               # nothing to join
    import time as _t
    t0 = _t.monotonic()
    tts.stop()
    assert _t.monotonic() - t0 < 2.0
    assert svc.stopped is True
    assert tts._st_stopping.is_set()
