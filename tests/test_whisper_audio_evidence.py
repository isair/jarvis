"""Quiet noise cannot reach recognition as a confident spoken utterance."""

from dataclasses import replace

import numpy as np
import pytest

from jarvis.config import load_settings
from jarvis.listening import listener as capture

pytestmark = pytest.mark.unit


@pytest.fixture
def listener(monkeypatch):
    monkeypatch.setattr(capture, "create_intent_judge", lambda cfg: None)
    cfg = replace(load_settings(), voice_debug=False, voice_min_energy=.04,
                  sample_rate=16000, whisper_min_audio_duration=.1)
    return capture.VoiceListener(None, cfg, None, None)


@pytest.mark.parametrize("energy_ratio", [0, .25, .99])
def test_below_floor_capture_cannot_be_submitted_for_recognition(listener, energy_ratio):
    frame = np.full(320, listener.cfg.voice_min_energy * energy_ratio, dtype=np.float32)
    listener._utterance_frames = [frame.copy() for _ in range(30)]
    listener._finalize_utterance()
    assert listener._transcription_jobs_q.empty()


def test_energy_evidence_preserves_the_entire_utterance_including_quiet_syllables(listener):
    floor = listener.cfg.voice_min_energy
    quiet = np.full(320, floor / 4, dtype=np.float32)
    voiced = np.full(320, floor * 2, dtype=np.float32)
    frames = [quiet.copy() for _ in range(12)] + [voiced] + [quiet.copy() for _ in range(25)]
    listener._utterance_frames = frames
    listener._finalize_utterance()
    audio = listener._transcription_jobs_q.get_nowait().audio
    np.testing.assert_array_equal(audio, np.concatenate(frames))


def test_energy_floor_remains_configurable_including_zero(listener):
    audio = np.full(16000, listener.cfg.voice_min_energy / 4, dtype=np.float32)
    listener.cfg = replace(listener.cfg, voice_min_energy=0)
    listener._utterance_frames = [audio]
    listener._finalize_utterance()
    np.testing.assert_array_equal(listener._transcription_jobs_q.get_nowait().audio, audio)


def test_quiet_capture_does_not_block_the_next_spoken_request(listener):
    listener.is_speech_active = True
    listener._utterance_frames = [np.zeros(16000, dtype=np.float32)]
    listener._finalize_utterance()
    speech = np.full(16000, listener.cfg.voice_min_energy * 2, dtype=np.float32)
    listener._utterance_frames = [speech]
    listener._finalize_utterance()
    np.testing.assert_array_equal(listener._transcription_jobs_q.get_nowait().audio, speech)
    assert listener._transcription_jobs_q.empty()
