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
    def __init__(self, probe, gate=None, **kwargs):
        probe.enter("open")
        if gate is not None:
            gate.wait(timeout=5)  # let tests pin the open/stop interleaving
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
    def __init__(self, probe, open_gate=None):
        self._probe = probe
        self._open_gate = open_gate

    def InputStream(self, **kwargs):
        return _FakeStream(self._probe, gate=self._open_gate, **kwargs)

    def query_devices(self, *args, **kwargs):
        return {"default_samplerate": 16000}


def _drain_worker_threads(baseline):
    """Join threads spawned during a test so a late worker can't run against
    the next test's monkeypatched fake sounddevice."""
    for t in threading.enumerate():
        if t not in baseline and t is not threading.current_thread():
            t.join(timeout=5)


def _make_engine(monkeypatch, probe, open_gate=None):
    fake_sd = _FakeSounddevice(probe, open_gate=open_gate)
    monkeypatch.setattr(de, "sd", fake_sd)
    engine = de.DictationEngine(
        whisper_model_ref=lambda: object(),
        whisper_backend_ref=lambda: "faster-whisper",
        mlx_repo_ref=lambda: None,
    )
    return engine


def test_hotkey_press_does_not_open_stream_on_calling_thread(monkeypatch):
    """The (pynput) calling thread must return without touching PortAudio."""
    baseline = set(threading.enumerate())
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
    _drain_worker_threads(baseline)


def test_stop_before_stream_opens_still_closes_stream(monkeypatch):
    """Press/release faster than the stream opens: no leaked live stream."""
    baseline = set(threading.enumerate())
    probe = _ConcurrencyProbe()
    open_gate = threading.Event()
    engine = _make_engine(monkeypatch, probe, open_gate=open_gate)

    engine._start_recording()
    engine._stop_recording(discard=True)  # guaranteed before the open finishes
    open_gate.set()  # now let the worker's stream open complete

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
    _drain_worker_threads(baseline)


def test_stale_worker_from_superseded_press_never_stores_its_stream(monkeypatch):
    """press → release → press again while the first open is stalled.

    The first (stale) worker must not attach its stream to the second
    session — that would leak a live microphone capture (#462 family).
    """
    baseline = set(threading.enumerate())
    probe = _ConcurrencyProbe()
    open_gate = threading.Event()
    engine = _make_engine(monkeypatch, probe, open_gate=open_gate)

    engine._start_recording()          # session 1, worker A stalls in open
    deadline = time.time() + 5         # wait until A is pinned inside the open
    while time.time() < deadline and not any(c[0] == "open" for c in probe.calls):
        time.sleep(0.005)
    assert any(c[0] == "open" for c in probe.calls), "worker A never reached open"
    engine._stop_recording(discard=True)   # release while A is mid-open

    # Second press with instant opens.
    open_gate.set()
    engine._start_recording()          # session 2

    deadline = time.time() + 5
    while time.time() < deadline and engine._stream is None:
        time.sleep(0.01)
    session_stream = engine._stream
    assert session_stream is not None, "second session never got its stream"

    # Wait for worker A to finish; it must have closed its own stream and
    # left session 2's stream in place.
    deadline = time.time() + 5
    while time.time() < deadline and not any(c[0] == "close" for c in probe.calls):
        time.sleep(0.01)
    assert engine._stream is session_stream
    opened = [c for c in probe.calls if c[0] == "open"]
    closed = [c for c in probe.calls if c[0] == "close"]
    assert len(opened) == 2
    assert len(closed) >= 1, "stale worker's stream was never closed"

    engine._stop_recording(discard=True)
    _drain_worker_threads(baseline)


def test_lifecycle_calls_are_serialised_with_portaudio_lock(monkeypatch):
    """No lifecycle call may overlap another thread holding portaudio_lock."""
    baseline = set(threading.enumerate())
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
    _drain_worker_threads(baseline)
