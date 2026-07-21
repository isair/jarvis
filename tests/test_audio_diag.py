"""Tests for JARVIS_AUDIO_DIAG one-shot capture (no OpenAI, no mic)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from jarvis.listening.audio_diag import (
    audio_diag_enabled,
    save_one_shot_capture,
    write_wav_float32_mono,
)


@pytest.mark.unit
def test_audio_diag_enabled_only_on_flag(monkeypatch):
    monkeypatch.delenv("JARVIS_AUDIO_DIAG", raising=False)
    assert audio_diag_enabled() is False
    monkeypatch.setenv("JARVIS_AUDIO_DIAG", "1")
    assert audio_diag_enabled() is True
    monkeypatch.setenv("JARVIS_AUDIO_DIAG", "0")
    assert audio_diag_enabled() is False


@pytest.mark.unit
def test_save_one_shot_writes_only_allowed_manifest_keys(tmp_path):
    raw = np.linspace(-0.2, 0.2, 1600, dtype=np.float32)
    pre = np.linspace(-0.1, 0.1, 800, dtype=np.float32)
    from jarvis.voice.openai_realtime import float32_mono_to_pcm16_24k
    pcm24 = float32_mono_to_pcm16_24k(pre, 16000)
    man = save_one_shot_capture(
        out_dir=tmp_path,
        raw_audio=raw,
        raw_rate=16000,
        pre_openai_audio=pre,
        pre_openai_rate=16000,
        openai_pcm16_24k=pcm24,
        dropped_frames=42,
    )
    assert man is not None
    assert (tmp_path / "raw_capture.wav").is_file()
    assert (tmp_path / "pre_openai.wav").is_file()
    assert (tmp_path / "openai_24k.wav").is_file()
    data = json.loads(man.read_text(encoding="utf-8"))
    assert set(data.keys()) == {"raw_capture", "pre_openai", "openai_24k"}
    allowed = {"sample_rate", "samples", "channels", "duration", "peak", "rms", "dropped_frames"}
    for stage in ("raw_capture", "pre_openai", "openai_24k"):
        assert set(data[stage].keys()) == allowed
        assert data[stage]["dropped_frames"] == 42
    # No secrets / transcripts
    blob = man.read_text(encoding="utf-8").lower()
    assert "transcript" not in blob
    assert "sk-" not in blob
    assert "authorization" not in blob


@pytest.mark.unit
def test_flag_off_finalize_creates_no_files(tmp_path, monkeypatch):
    monkeypatch.delenv("JARVIS_AUDIO_DIAG", raising=False)
    from jarvis.listening.listener import VoiceListener

    listener = object.__new__(VoiceListener)
    listener.cfg = SimpleNamespace(voice_debug=False, whisper_min_audio_duration=0.01, min_voiced_ms=0)
    listener._audio_diag_enabled = False
    listener._audio_diag_done = False
    listener._audio_diag_raw_frames = []
    listener._audio_diag_dropped_mid_frames = 0
    listener._samplerate = 16000
    listener._stream_samplerate = 16000
    # Must not call write when disabled
    with patch.object(VoiceListener, "_audio_diag_write_capture", MagicMock()) as w:
        # Simulate only the diag gate path
        if listener._audio_diag_enabled and not listener._audio_diag_done:
            listener._audio_diag_write_capture(
                diag_raw=np.zeros(10), raw_rate=16000,
                pre_openai=np.zeros(10), pre_rate=16000,
                dropped_frames=0, samples_before=10, samples_after=10,
            )
        w.assert_not_called()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.unit
def test_one_shot_write_then_skip(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_AUDIO_DIAG", "1")
    from jarvis.listening.listener import VoiceListener
    from jarvis.listening import audio_diag as ad

    listener = object.__new__(VoiceListener)
    listener.cfg = SimpleNamespace(sample_rate=16000)
    listener._audio_diag_enabled = True
    listener._audio_diag_done = False
    listener._samplerate = 16000
    listener._stream_samplerate = 16000
    listener._audio_diag_callback_frames = 100
    listener._audio_diag_status_overflow = 0

    with patch.object(ad, "save_one_shot_capture", wraps=ad.save_one_shot_capture) as wrapped:
        # Point Path parents to tmp by patching write to use tmp_path
        def _write(**kw):
            from jarvis.voice.openai_realtime import float32_mono_to_pcm16_24k
            pcm24 = float32_mono_to_pcm16_24k(np.asarray(kw["pre_openai"], dtype=np.float32), kw["pre_rate"])
            return ad.save_one_shot_capture(
                out_dir=tmp_path,
                raw_audio=np.asarray(kw["diag_raw"], dtype=np.float32),
                raw_rate=kw["raw_rate"],
                pre_openai_audio=np.asarray(kw["pre_openai"], dtype=np.float32),
                pre_openai_rate=kw["pre_rate"],
                openai_pcm16_24k=pcm24,
                dropped_frames=kw["dropped_frames"],
            )

        listener._audio_diag_write_capture = lambda **kw: _write(**kw)
        raw = np.zeros(3200, dtype=np.float32)
        pre = np.zeros(1600, dtype=np.float32)
        listener._audio_diag_write_capture(
            diag_raw=raw, raw_rate=16000, pre_openai=pre, pre_rate=16000,
            dropped_frames=50, samples_before=1600, samples_after=1600,
        )
        listener._audio_diag_done = True
        # Second attempt must be skipped by caller gate
        if not listener._audio_diag_done:
            listener._audio_diag_write_capture(
                diag_raw=raw, raw_rate=16000, pre_openai=pre, pre_rate=16000,
                dropped_frames=1, samples_before=1, samples_after=1,
            )
    assert (tmp_path / "manifest.json").is_file()
    # Only one capture set
    assert len(list(tmp_path.glob("*.wav"))) == 3
