"""Offline native capture and Whisper qualification with generated audio.

Run with JARVIS_RUN_WHISPER_AUDIO_EVAL=1 on Apple Silicon with cached MLX
large-v3-turbo and the default Piper voice. No microphone or playback is used.
"""

from dataclasses import replace
import os
from pathlib import Path
import re

import pytest

pytestmark = pytest.mark.eval


@pytest.fixture(scope="module")
def local_audio_assets():
    if os.environ.get("JARVIS_RUN_WHISPER_AUDIO_EVAL") != "1":
        pytest.skip("Explicit offline audio qualification required")
    import numpy as np
    from piper import PiperVoice
    from piper.config import SynthesisConfig
    from huggingface_hub import snapshot_download
    from jarvis.output.tts import PIPER_DEFAULT_VOICE
    from jarvis.listening.listener import _resample

    voice_path = Path.home() / ".local/share/jarvis/models/piper" / f"{PIPER_DEFAULT_VOICE}.onnx"
    assert voice_path.is_file(), "Cache the default local Piper voice before this eval"
    model = snapshot_download("mlx-community/whisper-large-v3-turbo", local_files_only=True)
    voice = PiperVoice.load(voice_path)
    audio = {}
    for name, text in (("weather", "How is the weather tomorrow, Jarvis?"), ("stop", "Stop speaking.")):
        chunks = list(voice.synthesize(text, SynthesisConfig(noise_scale=0, noise_w_scale=0)))
        audio[name] = _resample(np.concatenate([chunk.audio_float_array for chunk in chunks]),
                                chunks[0].sample_rate, 16000)
    return model, audio


def captured_jobs(audio, monkeypatch):
    import numpy as np
    from jarvis.config import load_settings, get_default_config
    from jarvis.listening import listener as capture

    monkeypatch.setattr(capture, "create_intent_judge", lambda cfg: None)
    cfg = replace(load_settings(), voice_debug=False, sample_rate=16000, vad_enabled=True,
                  vad_aggressiveness=2, voice_min_energy=get_default_config()["voice_min_energy"])
    obj = capture.VoiceListener(None, cfg, None, None)
    assert obj._vad is not None, "Native WebRTC VAD is required"
    obj._frame_samples = 320
    obj._check_audio_health = lambda: None
    padded = np.concatenate((np.zeros(16000, dtype=np.float32), audio,
                             np.zeros(16000, dtype=np.float32)))

    class Feed:
        ready = True

        def get(self, **kwargs):
            if self.ready:
                self.ready = False
                return capture._CapturedAudio(padded, obj._audio_generation)
            obj._should_stop = True
            return None

    obj._audio_q = Feed()
    obj._consume_audio_frames(20)
    return list(obj._transcription_jobs_q.queue)


@pytest.mark.parametrize("kind", ["silence", "quiet_noise", "quiet_tone"])
def test_non_speech_cannot_reach_native_recognition(local_audio_assets, monkeypatch, kind):
    import numpy as np
    from jarvis.config import get_default_config

    level = get_default_config()["voice_min_energy"] / 4
    audio = {
        "silence": np.zeros(32000, dtype=np.float32),
        "quiet_noise": np.random.default_rng(731).normal(0, level, 32000).astype(np.float32),
        "quiet_tone": (level * np.sin(2 * np.pi * 220 * np.arange(32000) / 16000)).astype(np.float32),
    }[kind]
    assert captured_jobs(audio, monkeypatch) == []


@pytest.mark.parametrize("kind,peak_ratio,words", [
    ("weather", 8, ["how", "is", "the", "weather", "tomorrow", "jarvis"]),
    ("weather", 4, ["how", "is", "the", "weather", "tomorrow", "jarvis"]),
    ("weather", 2, ["how", "is", "the", "weather", "tomorrow", "jarvis"]),
    ("weather", 1.05, ["how", "is", "the", "weather", "tomorrow", "jarvis"]),
    ("stop", 8, ["stop", "speaking"]),
    ("stop", 2, ["stop", "speaking"]),
])
def test_native_recognition_preserves_quiet_requests(local_audio_assets, monkeypatch, kind, peak_ratio, words):
    import mlx_whisper
    import numpy as np
    from jarvis.config import get_default_config

    model, audio = local_audio_assets
    samples = audio[kind]
    frames = samples[:len(samples) // 320 * 320].reshape(-1, 320)
    peak = float(np.sqrt(np.mean(frames * frames, axis=1)).max())
    gain = get_default_config()["voice_min_energy"] * peak_ratio / peak
    jobs = captured_jobs(samples * gain, monkeypatch)
    assert len(jobs) == 1
    result = mlx_whisper.transcribe(jobs[0].audio, path_or_hf_repo=str(model), language=None)
    assert re.findall(r"\w+", result["text"].casefold()) == words
