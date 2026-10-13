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


@pytest.fixture(autouse=True)
def private_audio_configuration(tmp_path, monkeypatch):
    """Qualification uses default settings without reading or migrating personal config."""
    monkeypatch.setenv("JARVIS_CONFIG_PATH", str(tmp_path / "config.json"))


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


@pytest.fixture
def local_french_audio(local_audio_assets, tmp_path):
    import subprocess
    import numpy as np
    from scipy.io import wavfile

    voices = subprocess.run(["/usr/bin/say", "-v", "?"], check=True, capture_output=True, text=True)
    assert any(line.startswith("Thomas ") and "fr_FR" in line for line in voices.stdout.splitlines())
    path = tmp_path / "french-echo.wav"
    subprocess.run(["/usr/bin/say", "-v", "Thomas", "-o", str(path),
                    "--file-format=WAVE", "--data-format=LEI16@16000",
                    "Quel temps fera-t-il demain à Londres ?"], check=True, capture_output=True, timeout=20)
    rate, audio = wavfile.read(path)
    assert rate == 16000
    return audio.astype(np.float32) / 32768


@pytest.mark.parametrize("rejection", ["empty", "echo"])
def test_native_request_language_survives_rejected_recognition(local_audio_assets, monkeypatch, request, rejection):
    from jarvis.config import load_settings
    from jarvis.listening import listener as capture
    from jarvis.reply import engine

    model, audio = local_audio_assets
    jobs = captured_jobs(audio["weather"], monkeypatch)
    assert len(jobs) == 1
    cfg = replace(load_settings(), sample_rate=16000)
    obj = capture.VoiceListener(None, cfg, None, None)
    obj._whisper_backend = "mlx"
    obj._mlx_model_repo = str(model)
    obj._transcription_jobs_q.put(jobs[0])
    obj._transcription_jobs_q.put(None)
    obj._run_transcription_worker()
    accepted = obj._transcription_results_q.get_nowait()
    assert re.findall(r"\w+", accepted.text.casefold()) == [
        "how", "is", "the", "weather", "tomorrow", "jarvis",
    ]
    assert accepted.language == "en"
    obj._start_thinking_tune = lambda: None
    obj._stop_thinking_tune = lambda: None
    obj._handle_transcription_result(accepted)
    assert obj.state_manager.get_pending_query() == accepted.text
    if rejection == "empty":
        obj._handle_transcription_result(replace(accepted, text="", language="ko"))
    else:
        from types import SimpleNamespace
        foreign_audio = request.getfixturevalue("local_french_audio")
        obj._transcription_jobs_q.put(replace(jobs[0], audio=foreign_audio, captured_during_tts=True))
        obj._transcription_jobs_q.put(None)
        obj._run_transcription_worker()
        echo = obj._transcription_results_q.get_nowait()
        assert echo.text.strip() and echo.language == "fr"
        obj.tts = SimpleNamespace(enabled=True, is_speaking=lambda: True)
        obj.echo_detector._last_tts_text = echo.text
        monkeypatch.setattr(obj.echo_detector, "should_reject_as_echo", lambda *args, **kwargs: True)
        obj._handle_transcription_result(echo)
    collected = obj.state_manager.clear_collection()
    assert collected.text == accepted.text
    observed = []
    def reply(*args, language, **kwargs):
        observed.append((args[3], language))
        return None
    monkeypatch.setattr(engine, "run_reply_engine", reply)
    obj._dispatch_query(collected.text, language=collected.language, generation=obj._capture_generation)
    assert observed == [(accepted.text, accepted.language)]
