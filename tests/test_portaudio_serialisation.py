"""
Regression tests for the Windows "Fatal Python error: Aborted" family
(#462, #401, #422): PortAudio stream lifecycle calls made concurrently
from independent threads.

PortAudio documents stream open/close as not thread safe; on Windows an
internal assertion failure aborts the whole process, which cannot be
caught from Python. The defence is behavioural and testable on any
platform:

1. Every stream lifecycle call goes through the process-wide
   ``portaudio_lock``, so two threads can never be inside PortAudio
   lifecycle code at the same time.
2. The dictation engine never opens streams on the pynput hook thread —
   the hotkey callback returns immediately and stream work happens on a
   worker thread (Windows silently unhooks slow hook callbacks, and the
   hook thread has no COM/PortAudio affinity guarantees).

These tests install a fake ``sounddevice`` whose lifecycle entry points
record concurrency and calling thread, then drive the real engine code.
"""

import threading
import time

import jarvis.dictation.dictation_engine as de
from jarvis.utils.audio_lock import portaudio_lock


class _ConcurrencyProbe:
    """Counts how many threads are inside a lifecycle call simultaneously."""

    def __init__(self):
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.calls = []

    def enter(self, name):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.calls.append((name, threading.get_ident()))
        time.sleep(0.02)  # widen the overlap window

    def exit(self):
        with self._lock:
            self.active -= 1


class _FakeStream:
    def __init__(self, probe, **kwargs):
        probe.enter("open")
        self._probe = probe
        self.closed = False
        probe.exit()

    def start(self):
        self._probe.enter("start")
        self._probe.exit()

    def stop(self):
        self._probe.enter("stop")
        self._probe.exit()

    def close(self):
        self._probe.enter("close")
        self.closed = True
        self._probe.exit()


class _FakeSounddevice:
    def __init__(self, probe):
        self._probe = probe

    def InputStream(self, **kwargs):
        return _FakeStream(self._probe, **kwargs)

    def query_devices(self, *args, **kwargs):
        return {"default_samplerate": 16000}


def _make_engine(monkeypatch, probe):
    fake_sd = _FakeSounddevice(probe)
    monkeypatch.setattr(de, "sd", fake_sd)
    engine = de.DictationEngine(
        whisper_model_ref=lambda: object(),
        whisper_backend_ref=lambda: "faster-whisper",
        mlx_repo_ref=lambda: None,
    )
    return engine


def test_hotkey_press_does_not_open_stream_on_calling_thread(monkeypatch):
    """The (pynput) calling thread must return without touching PortAudio."""
    probe = _ConcurrencyProbe()
    engine = _make_engine(monkeypatch, probe)

    hook_thread_ident = [None]

    def press():
        hook_thread_ident[0] = threading.get_ident()
        engine._start_recording()

    t = threading.Thread(target=press)
    t.start()
    t.join(timeout=5)

    # Wait for the worker to open the stream.
    deadline = time.time() + 5
    while time.time() < deadline and not any(n == "open" for n, _ in probe.calls):
        time.sleep(0.01)

    open_threads = [ident for name, ident in probe.calls if name in ("open", "start")]
    assert open_threads, "stream was never opened"
    assert hook_thread_ident[0] not in open_threads, (
        "stream lifecycle ran on the hotkey callback thread"
    )
    engine._stop_recording(discard=True)


def test_stop_before_stream_opens_still_closes_stream(monkeypatch):
    """Press/release faster than the stream opens: no leaked live stream."""
    probe = _ConcurrencyProbe()
    engine = _make_engine(monkeypatch, probe)

    engine._start_recording()
    engine._stop_recording(discard=True)  # may run before the worker opened it

    # Whatever the interleaving, the engine must settle on: not recording,
    # no stored stream, and any opened stream closed.
    deadline = time.time() + 5
    while time.time() < deadline:
        opened = [c for c in probe.calls if c[0] == "open"]
        closed = [c for c in probe.calls if c[0] == "close"]
        if not engine.is_recording and engine._stream is None and (not opened or closed):
            break
        time.sleep(0.01)

    assert not engine.is_recording
    assert engine._stream is None
    opened = [c for c in probe.calls if c[0] == "open"]
    closed = [c for c in probe.calls if c[0] == "close"]
    assert not opened or closed, "stream opened after stop was never closed"


def test_lifecycle_calls_are_serialised_with_portaudio_lock(monkeypatch):
    """No lifecycle call may overlap another thread holding portaudio_lock."""
    probe = _ConcurrencyProbe()
    engine = _make_engine(monkeypatch, probe)

    stop = threading.Event()

    def competitor():
        # Simulates any other subsystem (listener, TTS, tune player) doing
        # guarded lifecycle work in a tight loop.
        while not stop.is_set():
            with portaudio_lock:
                probe.enter("competitor-lifecycle")
                probe.exit()

    t = threading.Thread(target=competitor, daemon=True)
    t.start()
    try:
        for _ in range(5):
            engine._start_recording()
            deadline = time.time() + 5
            while time.time() < deadline and engine._stream is None:
                time.sleep(0.005)
            engine._stop_recording(discard=True)
    finally:
        stop.set()
        t.join(timeout=5)

    assert probe.max_active == 1, (
        f"PortAudio lifecycle calls overlapped (max concurrency {probe.max_active})"
    )
