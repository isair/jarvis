"""Behavioural tests for the thinking-tune player.

Covers:
- WAV generation: right format/size, seam is effectively seamless.
- TunePlayer lifecycle: idempotent start/stop, is_playing state, prompt
  stop even when a "player" is running.
- Platform-player dispatch: the macOS/Linux loop respects stop_event.

Platform code paths are exercised via the Linux `ffplay`-style branch
using a fake player binary, which works headlessly in CI.
"""
from __future__ import annotations

import io
import struct
import sys
import time
import wave
from pathlib import Path
from unittest.mock import patch

import pytest

from jarvis.output import tune_player
from jarvis.output.tune_player import (
    TunePlayer,
    _generate_thinking_pad_wav,
    _get_thinking_pad_wav,
)


# --- WAV generation --------------------------------------------------------

def test_thinking_pad_wav_is_well_formed_and_long():
    data = _generate_thinking_pad_wav()
    with wave.open(io.BytesIO(data)) as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2  # 16-bit PCM
        assert w.getframerate() == 44100
        n = w.getnframes()
    # Long enough that subprocess relaunch gaps are rare.
    assert n / 44100 >= 30.0


def test_thinking_pad_wav_is_cached():
    a = _get_thinking_pad_wav()
    b = _get_thinking_pad_wav()
    assert a is b  # same object, not regenerated


def test_thinking_pad_wav_seam_is_effectively_seamless():
    """First and last PCM samples should be within a small fraction of
    full-scale — the loop wrap produces a step no larger than a normal
    sample-to-sample delta, i.e. inaudible."""
    data = _generate_thinking_pad_wav()
    with wave.open(io.BytesIO(data)) as w:
        frames = w.readframes(w.getnframes())
    first = struct.unpack("<h", frames[:2])[0]
    last = struct.unpack("<h", frames[-2:])[0]
    # Full scale is 32767; observed real seam ≈ 500. Allow 5% as generous
    # headroom so this doesn't fight future tweaks.
    assert abs(first - last) < 0.05 * 32767


def test_thinking_pad_wav_stays_loud_throughout():
    """Amplitude must never drop to silence — it's meant to be a
    continuous flowing texture."""
    data = _generate_thinking_pad_wav()
    with wave.open(io.BytesIO(data)) as w:
        n = w.getnframes()
        rate = w.getframerate()
        frames = w.readframes(n)
    pcm = struct.unpack(f"<{n}h", frames)
    # Check every ~100ms window — none should be silent.
    win = rate // 10
    min_peak = min(
        max(abs(v) for v in pcm[i : i + win])
        for i in range(0, n - win, win)
    )
    # Observed min ≈ 2800; require a meaningful floor so any future
    # change that lets the texture drop to silence would be caught.
    assert min_peak > 0.05 * 32767


# --- TunePlayer lifecycle --------------------------------------------------

def test_disabled_player_never_starts():
    tp = TunePlayer(enabled=False)
    tp.start_tune()
    try:
        assert not tp.is_playing()
        assert tp._thread is None
    finally:
        tp.stop_tune()


def test_stop_is_idempotent():
    tp = TunePlayer(enabled=False)  # fast path, no real playback
    tp.stop_tune()  # no thread running; must not raise
    tp.stop_tune()


def test_lifecycle_with_fallback_player(monkeypatch):
    """Force the fallback path (no platform players found) and verify
    is_playing flips correctly around start/stop, and stop returns promptly."""
    # Neutralise every platform so _play_tune routes nowhere and falls
    # through _play_fallback_tune, which exits on stop_event.
    monkeypatch.setattr(tune_player.platform, "system", lambda: "Other")

    tp = TunePlayer(enabled=True)
    tp.start_tune()
    # Give the thread a moment to enter _play_tune.
    for _ in range(20):
        if tp.is_playing():
            break
        time.sleep(0.01)
    assert tp.is_playing()

    t0 = time.time()
    tp.stop_tune()
    elapsed = time.time() - t0
    # Should stop almost instantly — well under the join timeout of 2s.
    assert elapsed < 1.5
    assert not tp.is_playing()
    assert tp._thread is None


def test_double_start_is_ignored(monkeypatch):
    monkeypatch.setattr(tune_player.platform, "system", lambda: "Other")
    tp = TunePlayer(enabled=True)
    tp.start_tune()
    first_thread = tp._thread
    try:
        tp.start_tune()
        assert tp._thread is first_thread  # second start is a no-op
    finally:
        tp.stop_tune()


def test_stop_terminates_running_subprocess(monkeypatch, tmp_path):
    """If a player subprocess is running, stop_tune must terminate it
    rather than waiting for it to exit on its own."""
    # Build a fake 'ffplay' that sleeps indefinitely so we can observe
    # whether stop actually kills it.
    fake_player = tmp_path / "ffplay"
    fake_player.write_text(
        "#!/usr/bin/env python3\n"
        "import time\n"
        "while True: time.sleep(1)\n"
    )
    fake_player.chmod(0o755)

    # Point the Linux path at our fake ffplay, and route platform to Linux.
    monkeypatch.setattr(tune_player.platform, "system", lambda: "Linux")

    def fake_which(name):
        if name == "ffplay":
            return str(fake_player)
        return None

    monkeypatch.setattr(tune_player.shutil, "which", fake_which)

    tp = TunePlayer(enabled=True)
    tp.start_tune()

    # Wait for the subprocess to actually be spawned.
    for _ in range(50):
        if tp._current_process is not None:
            break
        time.sleep(0.02)
    assert tp._current_process is not None
    proc = tp._current_process

    t0 = time.time()
    tp.stop_tune()
    elapsed = time.time() - t0

    # Process should be dead, and stop should return quickly.
    assert proc.poll() is not None
    assert elapsed < 2.0
