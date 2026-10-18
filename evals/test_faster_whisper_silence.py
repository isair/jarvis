"""Offline speech guard qualification with explicit local model and voice paths.

Set JARVIS_EVAL_FASTER_WHISPER_MODEL and JARVIS_EVAL_PIPER_VOICE to cached
local files. This eval never downloads models, records audio or plays sound.
"""
import os
from pathlib import Path

import numpy as np
import pytest

from tests.test_whisper_silence_guard import listener_with_decoder

pytestmark = pytest.mark.eval


@pytest.fixture(scope='module')
def local_decoder():
    model_path = os.environ.get('JARVIS_EVAL_FASTER_WHISPER_MODEL')
    if not model_path:
        pytest.skip('Explicit local faster-whisper model required')
    assert Path(model_path).is_dir()
    from faster_whisper import WhisperModel
    return WhisperModel(model_path, device='cpu', compute_type='int8',
                        local_files_only=True, cpu_threads=2)


@pytest.mark.parametrize('noise', ['silence', 'quiet-noise', 'loud-noise', 'hum'])
def test_non_speech_is_removed_before_recognition(monkeypatch, local_decoder, noise):
    rate = 16000
    samples = np.zeros(rate * 3, dtype=np.float32)
    if noise.endswith('noise'):
        level = 0.003 if noise == 'quiet-noise' else 0.03
        samples = np.random.default_rng(1141).normal(0, level, samples.size).astype(np.float32)
    elif noise == 'hum':
        samples = (0.05 * np.sin(2 * np.pi * 60 * np.arange(samples.size) / rate)).astype(np.float32)
    listener = listener_with_decoder(monkeypatch, 'cpu', local_decoder.transcribe)
    segments, info = listener._decode_faster_whisper(samples)
    assert segments == []
    assert info.duration_after_vad == 0, 'Non-speech must not reach acoustic decoding'


@pytest.mark.parametrize('spoken,keyword', [
    ('Jarvis, what time is it?', 'time'),
    ('Stop speaking.', 'stop'),
    ('Thank you.', 'thank'),
])
def test_short_speech_and_caption_like_real_words_survive(monkeypatch, local_decoder, spoken, keyword):
    voice_path = os.environ.get('JARVIS_EVAL_PIPER_VOICE')
    assert voice_path and Path(voice_path).is_file(), 'Explicit local Piper voice required'
    from piper import PiperVoice
    from piper.config import SynthesisConfig
    from jarvis.listening.listener import _resample
    voice = PiperVoice.load(voice_path)
    chunks = list(voice.synthesize(spoken, SynthesisConfig(noise_scale=0, noise_w_scale=0)))
    samples = _resample(np.concatenate([chunk.audio_float_array for chunk in chunks]),
                        chunks[0].sample_rate, 16000)
    listener = listener_with_decoder(monkeypatch, 'cpu', local_decoder.transcribe)
    text, _, events = listener._transcribe_audio(samples)
    assert keyword in text.casefold(), text
    assert events == ()
