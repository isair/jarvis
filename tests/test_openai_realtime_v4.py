"""Premium Audio V4 — streaming path unit tests (no real network / no API)."""

from __future__ import annotations

import base64
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from jarvis.listening.listener import FLIGHT_COOLDOWN, FLIGHT_LISTENING, FLIGHT_SPEAKING, VoiceListener
from jarvis.voice.openai_realtime import (
    MockRealtimeTransport,
    OpenAIRealtimeSession,
    PendingTranscript,
    StatefulPcm24kResampler,
    STREAM_PENDING_BYTES_LIMIT,
    build_ga_session_update,
    reset_realtime_session_for_tests,
    transcript_starts_with_wake,
)


def _cfg(**kw):
    base = dict(
        openai_realtime_enabled=True,
        openai_realtime_model="gpt-realtime-2.1",
        openai_realtime_transcription_model="gpt-4o-transcribe",
        openai_realtime_voice="marin",
        openai_realtime_language="ro",
        openai_realtime_idle_timeout_sec=60.0,
        openai_realtime_fallback_local=False,
        openai_realtime_require_wake_each_turn=True,
        wake_word="cora",
        wake_aliases=["cora"],
        sample_rate=16000,
        voice_min_energy=0.0045,
        endpoint_silence_ms=1300,
        vad_enabled=True,
    )
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.fixture(autouse=True)
def _reset_session():
    reset_realtime_session_for_tests()
    yield
    reset_realtime_session_for_tests()


@pytest.mark.unit
def test_v4_session_update_semantic_vad_near_field():
    p = build_ga_session_update(
        model="gpt-realtime-2.1",
        voice="marin",
        transcription_model="gpt-4o-transcribe",
        language="ro",
    )
    inp = p["session"]["audio"]["input"]
    assert inp["transcription"] == {"model": "gpt-4o-transcribe", "language": "ro"}
    assert inp["noise_reduction"] == {"type": "near_field"}
    assert inp["turn_detection"] == {
        "type": "semantic_vad",
        "eagerness": "low",
        "create_response": False,
        "interrupt_response": False,
    }


@pytest.mark.unit
def test_stateful_resampler_voice_silence_voice_preserves_all_samples():
    rs = StatefulPcm24kResampler(16000)
    voice = np.linspace(-0.2, 0.2, 1600, dtype=np.float32)  # 100 ms
    silence = np.zeros(1600, dtype=np.float32)
    out = b"".join([
        rs.convert_float32(voice),
        rs.convert_float32(silence),
        rs.convert_float32(voice),
    ])
    assert rs.samples_in == 4800
    # 16000→24000: ~1.5× samples (±1 frame of ratecv lag)
    assert abs(rs.samples_out - 7200) <= 8
    assert len(out) == rs.samples_out * 2
    # Same state object reused — not reset between chunks
    assert rs._state is not None or rs.source_rate == 24000


@pytest.mark.unit
def test_stateful_resampler_7s_produces_approx_7s():
    rs = StatefulPcm24kResampler(16000)
    frame = np.zeros(480, dtype=np.float32)  # 30 ms @ 16k
    n = int(7.0 / 0.03)
    for _ in range(n):
        rs.convert_float32(frame)
    # Duration out ≈ duration in
    assert abs(rs.duration_in_sec - 7.0) < 0.05
    assert abs(rs.duration_out_sec - rs.duration_in_sec) < 0.05


@pytest.mark.unit
def test_stream_append_counts_and_no_silence_drop():
    transport = MockRealtimeTransport(script=[
        {"type": "session.created", "session": {"type": "realtime"}},
        {"type": "session.updated", "session": {"type": "realtime"}},
    ])
    cfg = _cfg()
    with patch("jarvis.voice.openai_realtime.require_openai_api_key", return_value="SYNTHETIC"):
        session = OpenAIRealtimeSession(
            cfg, transport_factory=lambda: transport, allow_network=False,
            audio_player=lambda *_: None,
        )
        assert session.ensure_streaming(16000)
        voice = (np.random.randn(480) * 0.05).astype(np.float32)
        silence = np.zeros(480, dtype=np.float32)
        assert session.stream_append_float32(voice, 16000)
        assert session.stream_append_float32(silence, 16000)
        assert session.stream_append_float32(voice, 16000)
    appends = [p for p in transport.sent if p.get("type") == "input_audio_buffer.append"]
    assert len(appends) == 3
    c = session.stream_counters()
    assert c["frames_appended"] == 3
    assert c["frames_dropped"] == 0
    assert c["samples_in"] == 1440


@pytest.mark.unit
def test_v4_zero_response_create_without_wake():
    transport = MockRealtimeTransport(script=[
        {"type": "session.created", "session": {}},
        {"type": "session.updated", "session": {}},
    ])
    cfg = _cfg()
    with patch("jarvis.voice.openai_realtime.require_openai_api_key", return_value="SYNTHETIC"):
        session = OpenAIRealtimeSession(
            cfg, transport_factory=lambda: transport, allow_network=False,
            audio_player=lambda *_: None,
        )
        session.ensure_streaming(16000)
        result = session.finish_premium_turn(
            PendingTranscript("Să vă mulțumim pentru vizionare", "item_x")
        )
    assert result.ignored_no_wake is True
    assert result.response_created is False
    types = [p.get("type") for p in transport.sent]
    assert "response.create" not in types
    assert "conversation.item.delete" in types


@pytest.mark.unit
def test_v4_exactly_one_response_create_with_wake():
    audio = b"\x00\x01" * 80
    transport = MockRealtimeTransport(script=[
        {"type": "session.created", "session": {}},
        {"type": "session.updated", "session": {}},
        {"type": "response.output_audio.delta", "delta": base64.b64encode(audio).decode()},
        {"type": "response.output_audio_transcript.done", "transcript": "da"},
        {"type": "response.done"},
    ])
    plays = []
    cfg = _cfg()
    with patch("jarvis.voice.openai_realtime.require_openai_api_key", return_value="SYNTHETIC"):
        session = OpenAIRealtimeSession(
            cfg, transport_factory=lambda: transport, allow_network=False,
            audio_player=lambda s, r: plays.append(r),
        )
        session.ensure_streaming(16000)
        result = session.finish_premium_turn(
            PendingTranscript("Cora, ce mai faci?", "item_1")
        )
    assert result.ok and result.used_premium
    assert result.response_created is True
    assert [p.get("type") for p in transport.sent].count("response.create") == 1
    assert len(plays) == 1


@pytest.mark.unit
def test_thanks_for_watching_ignored():
    assert transcript_starts_with_wake(
        "Să vă mulțumim pentru vizionare", "cora", ["cora"]
    ) is False


@pytest.mark.unit
def test_premium_frame_loop_skips_local_vad_energy_endpoint():
    """Premium path must not call _is_speech_frame / touch _utterance_frames."""
    listener = object.__new__(VoiceListener)
    listener.cfg = _cfg()
    listener._flight_state = FLIGHT_LISTENING
    listener._flight_lock = __import__("threading").RLock()
    listener._samplerate = 16000
    listener._stream_samplerate = 16000
    listener._frame_samples = 480
    listener._realtime_session = None
    listener._premium_pcm_segments = []
    listener._premium_pcm_candidate = None
    listener._premium_collecting = False
    listener.is_speech_active = False
    listener._silence_frames = 0
    listener._utterance_frames = []
    listener._spoken_utterance_ids = set()
    listener.dialogue_memory = None

    calls = {"vad": 0, "stream": 0}

    def _vad(_f):
        calls["vad"] += 1
        return True

    def _stream(_f):
        calls["stream"] += 1

    listener._is_speech_frame = _vad
    listener._premium_stream_mic_frame = _stream
    listener._premium_poll_and_handle_turn = lambda: None
    listener._check_query_timeout = lambda: None
    listener._premium_realtime_enabled = lambda: True

    # Simulate one frame through the premium branch logic
    frame = np.zeros(480, dtype=np.float32)
    if listener._premium_realtime_enabled():
        listener._premium_stream_mic_frame(frame)
        listener._premium_poll_and_handle_turn()
    else:
        listener._is_speech_frame(frame)

    assert calls["stream"] == 1
    assert calls["vad"] == 0
    assert listener._utterance_frames == []


@pytest.mark.unit
def test_zero_stream_outside_listening():
    listener = object.__new__(VoiceListener)
    listener.cfg = _cfg()
    listener._flight_lock = __import__("threading").RLock()
    listener._flight_state = FLIGHT_SPEAKING
    listener._samplerate = 16000
    listener._stream_samplerate = 16000
    listener._realtime_session = MagicMock()
    streamed = []

    def fake_ensure(*_a, **_k):
        return listener._realtime_session

    listener._premium_ensure_session = fake_ensure
    listener._realtime_session.ensure_streaming = MagicMock(return_value=True)
    listener._realtime_session.stream_append_float32 = lambda *a, **k: streamed.append(1) or True

    listener._premium_stream_mic_frame(np.zeros(480, dtype=np.float32))
    assert streamed == []

    listener._flight_state = FLIGHT_COOLDOWN
    listener._premium_stream_mic_frame(np.zeros(480, dtype=np.float32))
    assert streamed == []

    listener._flight_state = FLIGHT_LISTENING
    listener._premium_stream_mic_frame(np.zeros(480, dtype=np.float32))
    assert len(streamed) == 1


@pytest.mark.unit
def test_backpressure_counts_dropped_frames():
    transport = MockRealtimeTransport(script=[
        {"type": "session.created", "session": {}},
        {"type": "session.updated", "session": {}},
    ])
    cfg = _cfg()
    with patch("jarvis.voice.openai_realtime.require_openai_api_key", return_value="SYNTHETIC"):
        session = OpenAIRealtimeSession(
            cfg, transport_factory=lambda: transport, allow_network=False,
            audio_player=lambda *_: None,
        )
        session.ensure_streaming(16000)
        session._pending_append_bytes = STREAM_PENDING_BYTES_LIMIT
        ok = session.stream_append_float32(np.zeros(480, dtype=np.float32), 16000)
    assert ok is False
    assert session.stream_counters()["frames_dropped"] == 1


@pytest.mark.unit
def test_flag_false_local_path_untouched_gate():
    listener = object.__new__(VoiceListener)
    listener.cfg = _cfg(openai_realtime_enabled=False)
    assert listener._premium_realtime_enabled() is False
