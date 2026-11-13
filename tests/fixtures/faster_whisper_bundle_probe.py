"""Exercise the bundled speech detector without models, network or audio devices."""
import numpy as np
from faster_whisper import available_models
from faster_whisper.vad import get_speech_timestamps
from jarvis.listening.whisper_capabilities import is_faster_whisper_turbo_supported

assert "large-v3-turbo" in available_models()
assert is_faster_whisper_turbo_supported()
assert not get_speech_timestamps(np.zeros(16000, dtype=np.float32))
print("✅ FROZEN_WHISPER_RUNTIME_PASSED", flush=True)
