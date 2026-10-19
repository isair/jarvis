"""Behavioural tests for the thinking-tune player.

Covers:
- Sample generation: right format/size, seam is effectively seamless.
- TunePlayer lifecycle: idempotent start/stop, is_playing state, prompt
  stop even when a "stream" is running.
- Sounddevice dispatch: stop_tune closes the stream cleanly from the
  owning thread (no cross-thread abort — that races with close on
  macOS CoreAudio and logs a spurious !obj error).

The sounddevice stream is exercised via a fake `sounddevice` module
injected into sys.modules — works headlessly in CI.
"""
from __future__ import annotations

import sys
import threading
import time
import types

import pytest

from jarvis.output.tune_player import (
    TunePlayer,
    _generate_thinking_pad_samples,
    _get_thinking_pad_samples,
)


pytestmark = pytest.mark.unit


# --- Sample generation -----------------------------------------------

def test_thinking_pad_samples_have_expected_shape():
    samples, rate = _generate_thinking_pad_samples()
    assert rate == 44100
    assert samples.dtype.name == "int16"
    assert samples.ndim == 1
    # Long enough to contain several pulse-silence cycles.
    assert samples.size / rate >= 5.0


def test_thinking_pad_samples_cached():
    a = _get_thinking_pad_samples()
    b = _get_thinking_pad_samples()
    assert a is b


def test_thinking_pad_seam_is_effectively_seamless():
    samples, _ = _generate_thinking_pad_samples()
    first = int(samples[0])
    last = int(samples[-1])
    # Seam step must be well under full-scale; observed ≈ 500.
    assert abs(first - last) < 0.05 * 32767


def test_thinking_pad_breathes():
    """The pad is intentionally not continuous — it has a short audible
    breath followed by a silent pause each loop so long thinking runs
    aren't fatiguing. Verify both extremes exist."""
    samples, rate = _generate_thinking_pad_samples()
    win = rate // 10  # 100ms windows
    peaks = [
        int(abs(samples[i : i + win]).max())
        for i in range(0, samples.size - win, win)
    ]
    # At least one window is clearly audible (the hold section).
    assert max(peaks) > 0.10 * 32767
    # At least one window is effectively silent (the rest pause).
    assert min(peaks) < 0.005 * 32767


# --- TunePlayer lifecycle --------------------------------------------------

class _FakeStream:
    """Minimal sounddevice.OutputStream stand-in."""

    def __init__(self, *args, **kwargs):
        self.started = False
        self.aborted = False
        self.closed = False
        self._callback = kwargs.get("callback")

    def start(self):
        self.started = True

    def abort(self):
        self.aborted = True

    def close(self):
        self.closed = True


def _install_fake_sounddevice(monkeypatch, stream_factory=None):
    """Inject a fake sounddevice module that records the created stream."""
    created = {}

    def _OutputStream(*args, **kwargs):
        stream = (stream_factory or _FakeStream)(*args, **kwargs)
        created["stream"] = stream
        return stream

    fake_sd = types.ModuleType("sounddevice")
    fake_sd.OutputStream = _OutputStream
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)
    return created


def test_disabled_player_never_starts():
    tp = TunePlayer(enabled=False)
    tp.start_tune()
    try:
        assert not tp.is_playing()
        assert tp._thread is None
    finally:
        tp.stop_tune()


def test_stop_is_idempotent():
    tp = TunePlayer(enabled=False)
    tp.stop_tune()
    tp.stop_tune()


def test_double_start_is_ignored(monkeypatch):
    _install_fake_sounddevice(monkeypatch)
    tp = TunePlayer(enabled=True)
    tp.start_tune()
    first = tp._thread
    try:
        tp.start_tune()
        assert tp._thread is first
    finally:
        tp.stop_tune()


def test_stop_closes_the_stream_and_returns_quickly(monkeypatch):
    created = _install_fake_sounddevice(monkeypatch)
    tp = TunePlayer(enabled=True)
    tp.start_tune()

    # Wait until the stream is actually started.
    for _ in range(100):
        stream = created.get("stream")
        if stream is not None and stream.started:
            break
        time.sleep(0.01)
    stream = created.get("stream")
    assert stream is not None and stream.started

    t0 = time.time()
    tp.stop_tune()
    elapsed = time.time() - t0

    # Only the tune thread closes the stream; stop_tune must NOT abort
    # from the caller's thread — that races with close() on macOS.
    assert stream.closed
    assert not stream.aborted
    assert elapsed < 1.0
    assert not tp.is_playing()


def test_fallback_when_sounddevice_unavailable(monkeypatch):
    # Force the sounddevice import inside _play_tune to fail.
    fake_sd = types.ModuleType("sounddevice_broken")

    def _raise(*a, **kw):
        raise RuntimeError("no audio here")

    fake_sd.OutputStream = _raise
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sd)

    tp = TunePlayer(enabled=True)
    tp.start_tune()
    # Give the thread a moment to reach the fallback loop.
    for _ in range(50):
        if tp.is_playing():
            break
        time.sleep(0.01)
    assert tp.is_playing()

    t0 = time.time()
    tp.stop_tune()
    elapsed = time.time() - t0
    assert elapsed < 1.5
    assert not tp.is_playing()


def test_stream_callback_wraps_seamlessly(monkeypatch):
    """Every frame matches the looping PCM buffer, including the seam."""
    import numpy as np

    captured = {}
    callback_ready = threading.Event()

    class _SpyStream(_FakeStream):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            captured["callback"] = kwargs["callback"]
            callback_ready.set()

    _install_fake_sounddevice(monkeypatch, stream_factory=_SpyStream)
    player = TunePlayer(enabled=True)
    player.start_tune()
    try:
        assert callback_ready.wait(timeout=1.0), "Playback callback did not become available"
        samples, _ = _get_thinking_pad_samples()
        frames = 1024
        out = np.empty((frames, 1), dtype=np.int16)
        for offset in range(0, samples.size + frames, frames):
            captured["callback"](out, frames, None, None)
            expected = samples[np.arange(offset, offset + frames) % samples.size]
            np.testing.assert_array_equal(out[:, 0], expected)
    finally:
        player.stop_tune()


def test_restart_waits_for_the_previous_audio_stream_to_close(monkeypatch):
    closing = threading.Event()
    allow_close = threading.Event()
    closed = threading.Event()
    started = threading.Event()
    next_started = threading.Event()
    streams = []

    class _DelayedCloseStream(_FakeStream):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.index = len(streams)
            streams.append(self)

        def start(self):
            super().start()
            (started if self.index == 0 else next_started).set()

        def close(self):
            if self.index == 0:
                closing.set()
                assert allow_close.wait(timeout=5.0), 'Audio teardown was not released'
            super().close()
            if self.index == 0:
                closed.set()

    _install_fake_sounddevice(monkeypatch, stream_factory=_DelayedCloseStream)
    player = TunePlayer(enabled=True)
    player.start_tune()
    try:
        assert started.wait(timeout=1.0)
        player.stop_tune()
        assert closing.is_set()
        player.start_tune()
        allow_close.set()
        assert closed.wait(timeout=1.0)
        assert not next_started.wait(timeout=0.2), 'Restart during teardown opened another audio stream'
        player.start_tune()
        assert next_started.wait(timeout=1.0), 'Restart after teardown did not play'
    finally:
        allow_close.set()
        player.stop_tune()


def test_failed_playback_releases_worker_ownership_for_a_fresh_start(monkeypatch):
    first_closed = threading.Event()
    restarted = threading.Event()
    streams = []

    class _FailFirstStartStream(_FakeStream):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.index = len(streams)
            streams.append(self)

        def start(self):
            if self.index == 0:
                raise RuntimeError('Audio output unavailable')
            super().start()
            restarted.set()

        def close(self):
            super().close()
            if self.index == 0:
                first_closed.set()

    _install_fake_sounddevice(monkeypatch, stream_factory=_FailFirstStartStream)
    player = TunePlayer(enabled=True)
    player.start_tune()
    try:
        assert first_closed.wait(timeout=1.0)
        deadline = time.monotonic() + 1.0
        while player.is_playing() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not player.is_playing()
        player.start_tune()
        assert restarted.wait(timeout=1.0), 'A finished worker blocked playback retry'
    finally:
        player.stop_tune()


def test_worker_creation_failure_allows_stop_and_retry(monkeypatch):
    original_start = threading.Thread.start
    refused = threading.Event()
    started = threading.Event()

    def refuse_once(thread):
        if not refused.is_set():
            refused.set()
            raise RuntimeError('Cannot start output worker')
        return original_start(thread)

    class _ReadyStream(_FakeStream):
        def start(self):
            super().start()
            started.set()

    _install_fake_sounddevice(monkeypatch, stream_factory=_ReadyStream)
    monkeypatch.setattr(threading.Thread, 'start', refuse_once)
    player = TunePlayer(enabled=True)
    try:
        with pytest.raises(RuntimeError, match='Cannot start output worker'):
            player.start_tune()
        player.stop_tune()
        player.start_tune()
        assert started.wait(timeout=1.0), 'Worker creation failure prevented playback retry'
    finally:
        if started.is_set():
            player.stop_tune()
