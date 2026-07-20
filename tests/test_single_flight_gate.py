"""Single-flight capture gate: LISTENING → PROCESSING → SPEAKING → COOLDOWN.

Upstream, `_on_audio` gated only on `_should_stop` / `_dictation_active`, so the
microphone kept filling the queue while Piper was speaking. In a real session
that produced 24 hallucinated "Heard" lines out of 45 and overlapping playback.
These tests pin the gate shut in every non-listening state — and, just as
importantly, pin that it always reopens.
"""

import queue
import threading
import time
from unittest.mock import Mock

import numpy as np
import pytest

from src.jarvis.listening.listener import (
    FLIGHT_COOLDOWN,
    FLIGHT_LISTENING,
    FLIGHT_PROCESSING,
    FLIGHT_SPEAKING,
    VoiceListener,
)


def _listener(cooldown=0.05, min_voiced_ms=250):
    vl = VoiceListener.__new__(VoiceListener)
    vl.cfg = Mock()
    vl.cfg.vad_aggressiveness = 2
    vl.cfg.min_voiced_ms = min_voiced_ms
    vl.cfg.voice_debug = False
    vl._should_stop = False
    vl._dictation_active = False
    vl._callback_count = 0
    vl._audio_q = queue.Queue()
    vl._flight_lock = threading.Lock()
    vl._flight_state = FLIGHT_LISTENING
    vl._utterance_seq = 0
    vl._active_utterance_id = None
    vl._cooldown_timer = None
    vl._post_tts_cooldown_sec = cooldown
    vl._spoken_utterance_ids = set()
    vl._samplerate = 16000
    vl._stream_samplerate = 16000
    vl.is_speech_active = False
    vl._silence_frames = 0
    vl._utterance_frames = []
    vl._pre_roll = __import__("collections").deque()
    return vl


def _feed(vl, n=3):
    for _ in range(n):
        vl._on_audio(np.zeros((320, 1), dtype=np.float32), 320, None, None)


# ------------------------------------------------------------- capture gate

@pytest.mark.unit
def test_audio_accepted_only_in_listening():
    vl = _listener()
    _feed(vl, 3)
    assert vl._audio_q.qsize() == 3, "LISTENING must accept audio"

    for state in (FLIGHT_PROCESSING, FLIGHT_SPEAKING, FLIGHT_COOLDOWN):
        vl._flight_state = state
        before = vl._audio_q.qsize()
        _feed(vl, 5)
        assert vl._audio_q.qsize() == before, f"{state} must drop audio"


@pytest.mark.unit
def test_callback_count_not_incremented_when_gated():
    """The health check keys off _callback_count — gated frames must not inflate it."""
    vl = _listener()
    vl._flight_state = FLIGHT_SPEAKING
    _feed(vl, 10)
    assert vl._callback_count == 0


@pytest.mark.unit
def test_dictation_and_stop_flags_still_gate():
    vl = _listener()
    vl._dictation_active = True
    _feed(vl, 5)
    assert vl._audio_q.qsize() == 0
    vl._dictation_active = False
    vl._should_stop = True
    _feed(vl, 5)
    assert vl._audio_q.qsize() == 0


# ----------------------------------------------------------- state machine

@pytest.mark.unit
def test_begin_utterance_claims_slot_with_unique_id():
    vl = _listener()
    first = vl._begin_utterance()
    assert vl.get_flight_state() == FLIGHT_PROCESSING
    assert first == "u0001"
    vl._flight_state = FLIGHT_LISTENING
    assert vl._begin_utterance() == "u0002", "ids must be unique per utterance"


@pytest.mark.unit
def test_cooldown_returns_to_listening_and_flushes():
    vl = _listener(cooldown=0.05)
    vl._begin_utterance()
    vl._set_flight_state(FLIGHT_SPEAKING)

    # Audio captured before the gate closed must not survive the cooldown.
    vl._audio_q.put(np.zeros((320, 1), dtype=np.float32))
    vl._clear_audio_buffers = Mock()

    vl._enter_cooldown()
    assert vl.get_flight_state() == FLIGHT_COOLDOWN

    deadline = time.time() + 3.0
    while time.time() < deadline and vl.get_flight_state() != FLIGHT_LISTENING:
        time.sleep(0.01)

    assert vl.get_flight_state() == FLIGHT_LISTENING, "must reopen the microphone"
    assert vl._audio_q.qsize() == 0, "queue must be flushed at end of cooldown"
    assert vl._clear_audio_buffers.called
    assert vl._active_utterance_id is None


@pytest.mark.unit
def test_cooldown_keeps_gate_shut_for_its_duration():
    vl = _listener(cooldown=0.4)
    vl._clear_audio_buffers = Mock()
    vl._begin_utterance()
    vl._enter_cooldown()

    _feed(vl, 5)
    assert vl._audio_q.qsize() == 0, "audio during cooldown must be dropped"
    assert vl.get_flight_state() == FLIGHT_COOLDOWN

    time.sleep(0.6)
    assert vl.get_flight_state() == FLIGHT_LISTENING


@pytest.mark.unit
def test_watchdog_releases_a_stuck_speaking_state():
    """A stuck SPEAKING state means a permanently deaf assistant."""
    vl = _listener(cooldown=0.02)
    vl._clear_audio_buffers = Mock()
    utt = vl._begin_utterance()
    vl._set_flight_state(FLIGHT_SPEAKING)

    vl._arm_speaking_watchdog(utt, max_sec=0.05)

    deadline = time.time() + 3.0
    while time.time() < deadline and vl.get_flight_state() != FLIGHT_LISTENING:
        time.sleep(0.01)
    assert vl.get_flight_state() == FLIGHT_LISTENING, "watchdog must reopen the mic"


@pytest.mark.unit
def test_watchdog_does_not_fire_for_a_finished_utterance():
    vl = _listener(cooldown=0.02)
    vl._clear_audio_buffers = Mock()
    utt = vl._begin_utterance()
    vl._set_flight_state(FLIGHT_SPEAKING)
    vl._arm_speaking_watchdog(utt, max_sec=0.05)

    # Playback completes normally before the watchdog deadline.
    vl._enter_cooldown()
    time.sleep(0.3)
    assert vl.get_flight_state() == FLIGHT_LISTENING
    # A second utterance must not be disturbed by the stale watchdog.
    vl._begin_utterance()
    vl._set_flight_state(FLIGHT_SPEAKING)
    time.sleep(0.1)
    assert vl.get_flight_state() == FLIGHT_SPEAKING


# --------------------------------------------------------------- VAD gate

@pytest.mark.unit
def test_vad_gate_rejects_silence_and_noise():
    vl = _listener(min_voiced_ms=250)
    rng = np.random.default_rng(0)
    silence = np.zeros(16000 * 2, dtype=np.float32)
    noise = (rng.standard_normal(16000 * 2) * 0.02).astype(np.float32)

    assert vl._has_real_speech([silence]) is False, "silence must be rejected"
    assert vl._has_real_speech([noise]) is False, "white noise must be rejected"


@pytest.mark.unit
def test_vad_gate_is_disabled_by_zero_threshold():
    vl = _listener(min_voiced_ms=0)
    assert vl._has_real_speech([np.zeros(16000, dtype=np.float32)]) is True


@pytest.mark.unit
def test_vad_gate_fails_open_on_empty_input():
    """The gate may reject audio; it must never block the pipeline by erroring."""
    vl = _listener()
    assert vl._has_real_speech([]) is True
    assert vl._has_real_speech([np.zeros(8, dtype=np.float32)]) is True


@pytest.mark.unit
def test_vad_gate_is_independent_of_whisper_settings():
    """The whole point: it must not consult Whisper's own signals."""
    import ast
    import inspect
    from pathlib import Path

    src = Path("src/jarvis/listening/listener.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_has_real_speech")

    # Compare executable code only — the docstring deliberately *names* these
    # signals to explain why they are not used, so a raw substring check would
    # match its own rationale.
    body = list(fn.body)
    if (body and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        body = body[1:]
    code = "\n".join(ast.dump(stmt) for stmt in body)

    for forbidden in ("no_speech_prob", "whisper_initial_prompt", "avg_logprob"):
        assert forbidden not in code, f"VAD gate must not depend on {forbidden}"
    # It must genuinely use WebRTC VAD on raw PCM.
    assert "webrtcvad" in code and "is_speech" in code


# ------------------------------------------------------------ config wiring

@pytest.mark.unit
def test_gate_settings_reach_settings_object():
    from src.jarvis.config import get_default_config, load_settings
    d = get_default_config()
    assert d["post_tts_cooldown_sec"] == 0.5
    assert d["min_voiced_ms"] == 250
    s = load_settings()
    assert isinstance(s.post_tts_cooldown_sec, float)
    assert isinstance(s.min_voiced_ms, float)


@pytest.mark.unit
def test_initial_prompt_currently_disabled():
    """BLOCKED_ASR_PROMPT_SAFETY — the key stays supported but must be off."""
    from src.jarvis.config import load_settings
    assert load_settings().whisper_initial_prompt is None


@pytest.mark.unit
def test_audio_pipeline_settings_untouched():
    from src.jarvis.config import load_settings
    s = load_settings()
    assert s.sample_rate == 16000
    assert s.voice_device == "12"
    assert s.voice_min_energy == 0.005
    assert s.whisper_model == "large-v3"
    assert s.whisper_language == "ro"
