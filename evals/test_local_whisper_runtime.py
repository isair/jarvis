"""Offline speech-runtime qualification with cached, generated audio."""
import os
from pathlib import Path
import re

import pytest

pytestmark = pytest.mark.eval

PHRASES = [
    ("How is the weather tomorrow, Jarvis?", ratio) for ratio in (8, 2, 1.05)
] + [("Stop speaking.", ratio) for ratio in (8, 2, 1.05)] + [
    ("Jarvis.", ratio) for ratio in (8, 2, 1.05)
] + [("Yes.", ratio) for ratio in (8, 2, 1.05)] + [("No.", ratio) for ratio in (8, 2)]


@pytest.fixture(scope="module")
def generated_audio():
    if os.environ.get("JARVIS_RUN_WHISPER_AUDIO_EVAL") != "1":
        pytest.skip("Explicit offline speech qualification required")
    import numpy as np
    from piper import PiperVoice
    from piper.config import SynthesisConfig
    from jarvis.output.tts import PIPER_DEFAULT_VOICE
    from jarvis.listening.listener import _resample

    path = Path.home()/".local/share/jarvis/models/piper"/(PIPER_DEFAULT_VOICE+".onnx")
    assert path.is_file(), "Cache the local default Piper voice before this eval"
    voice = PiperVoice.load(path)
    samples = {}
    for phrase, _ in PHRASES:
        if phrase in samples:
            continue
        chunks = list(voice.synthesize(phrase, SynthesisConfig(noise_scale=0, noise_w_scale=0)))
        samples[phrase] = _resample(np.concatenate([c.audio_float_array for c in chunks]),
                                   chunks[0].sample_rate, 16000)
    return samples


def calibrated(samples, ratio):
    import numpy as np
    from jarvis.config import get_default_config

    frames = samples[:len(samples)//320*320].reshape(-1, 320)
    peak = float(np.sqrt(np.mean(frames*frames, axis=1)).max())
    return samples * (get_default_config()["voice_min_energy"] * ratio / peak)


@pytest.fixture(scope="module")
def cpu_model(generated_audio):
    from faster_whisper import WhisperModel

    path = os.environ.get("JARVIS_FW_EVAL_MODEL_DIR")
    assert path and Path(path).is_dir(), "Select a cached CPU model with JARVIS_FW_EVAL_MODEL_DIR"
    return WhisperModel(path, device="cpu", compute_type="int8", cpu_threads=2)


@pytest.mark.parametrize("phrase,ratio", PHRASES)
def test_cpu_recognition_retains_short_and_quiet_requests(cpu_model, generated_audio, phrase, ratio):
    segments, info = cpu_model.transcribe(
        calibrated(generated_audio[phrase], ratio), language=None, vad_filter=False,
        condition_on_previous_text=False, without_timestamps=True,
    )
    text = " ".join(segment.text for segment in segments)
    assert re.findall(r"\w+", text.casefold()) == re.findall(r"\w+", phrase.casefold())
    assert info.language == "en"


@pytest.mark.parametrize("phrase,ratio", PHRASES)
def test_local_detector_retains_speech_presence(generated_audio, phrase, ratio):
    from faster_whisper.vad import get_speech_timestamps

    assert get_speech_timestamps(calibrated(generated_audio[phrase], ratio), min_speech_duration_ms=0)


@pytest.mark.parametrize("kind", ["tone", "noise"])
def test_local_detector_rejects_non_speech(generated_audio, kind):
    import numpy as np
    from faster_whisper.vad import get_speech_timestamps
    from jarvis.config import get_default_config

    floor = get_default_config()["voice_min_energy"]
    audio = (floor*4*np.sin(2*np.pi*220*np.arange(32000)/16000)).astype(np.float32)
    if kind == "noise":
        audio = np.random.default_rng(731).normal(0, floor*2, 32000).astype(np.float32)
    assert not get_speech_timestamps(audio, min_speech_duration_ms=0)
