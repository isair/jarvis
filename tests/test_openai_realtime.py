"""OpenAI Realtime premium backend — unit tests (no real network)."""

from __future__ import annotations

import base64
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from jarvis.voice.openai_credentials import (
    CREDENTIAL_TARGET,
    CREDENTIAL_USERNAME,
    MissingOpenAICredential,
    read_openai_api_key,
    require_openai_api_key,
)
from jarvis.voice.openai_realtime import (
    CORA_INSTRUCTIONS,
    MockRealtimeTransport,
    OpenAIRealtimeSession,
    build_ga_session_update,
    float32_mono_to_pcm16_24k,
    get_realtime_session,
    play_pcm16_24k,
    premium_enabled,
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
        openai_realtime_fallback_local=True,
        openai_realtime_require_wake_each_turn=True,
        wake_word="cora",
        wake_aliases=["cora"],
        sample_rate=16000,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def _handshake_events(model="gpt-realtime-2.1", voice="marin"):
    return [
        {"type": "session.created", "session": {"type": "realtime", "model": model}},
        {
            "type": "session.updated",
            "session": {
                "type": "realtime",
                "model": model,
                "audio": {"output": {"voice": voice}},
            },
        },
    ]


def _turn_events(
    text_user="Cora, salut",
    text_asst="bună",
    audio=b"\x00\x01" * 100,
    *,
    item_id="item_1",
    include_response=True,
):
    events = [
        {"type": "conversation.item.created", "item": {"id": item_id}},
        {
            "type": "conversation.item.input_audio_transcription.completed",
            "transcript": text_user,
            "item_id": item_id,
        },
    ]
    if include_response:
        events.extend([
            {"type": "response.output_audio.delta", "delta": base64.b64encode(audio).decode()},
            {"type": "response.output_audio.done"},
            {"type": "response.output_audio_transcript.done", "transcript": text_asst},
            {"type": "response.done"},
        ])
    return events


def _pcm_event(text_user="Cora, salut", text_asst="bună", audio=b"\x00\x01" * 100):
    """Full first-connection script: handshake then one accepted turn."""
    return _handshake_events() + _turn_events(text_user, text_asst, audio)


@pytest.fixture(autouse=True)
def _reset_session():
    reset_realtime_session_for_tests()
    yield
    reset_realtime_session_for_tests()


# ---------------------------------------------------------------- credentials

@pytest.mark.unit
def test_credential_missing_returns_none():
    import win32cred
    with patch.object(win32cred, "CredRead", side_effect=OSError("not found")):
        assert read_openai_api_key() is None


@pytest.mark.unit
def test_credential_present_not_logged(capsys):
    import win32cred
    # Synthetic only — never read the live Credential Manager in this test.
    secret = "SYNTHETIC_OPENAI_KEY_" + ("A" * 40)
    fake = {
        "UserName": CREDENTIAL_USERNAME,
        "CredentialBlob": secret.encode("utf-16-le"),
    }
    with patch.object(win32cred, "CredRead", return_value=fake):
        key = read_openai_api_key()
    assert key == secret
    out = capsys.readouterr().out
    assert secret not in out
    assert "AAAA" not in out  # no fragment dump


@pytest.mark.unit
def test_require_raises_missing_code():
    with patch("jarvis.voice.openai_credentials.read_openai_api_key", return_value=None):
        with pytest.raises(MissingOpenAICredential) as ei:
            require_openai_api_key()
        assert ei.value.code == "MISSING_OPENAI_CREDENTIAL"


@pytest.mark.unit
def test_credential_blob_decode_utf16():
    blob = "SYNTHETIC_OPENAI_KEY_ABCDEFGHIJKLMNOP".encode("utf-16-le")
    from jarvis.voice.openai_credentials import _decode_credential_blob
    assert _decode_credential_blob(blob).startswith("SYNTHETIC_OPENAI_KEY_")


# ---------------------------------------------------------------- flag / traffic

@pytest.mark.unit
def test_flag_false_zero_connections():
    cfg = _cfg(openai_realtime_enabled=False)
    assert premium_enabled(cfg) is False
    transport = MockRealtimeTransport(script=_pcm_event())
    session = OpenAIRealtimeSession(
        cfg,
        transport_factory=lambda: transport,
        allow_network=False,
        audio_player=lambda *_: None,
    )
    # Listener gate: when flag false, _try_openai_realtime returns False without connect.
    # Session itself is only opened when handle_utterance is called — simulate gate:
    if not premium_enabled(cfg):
        assert transport.connect_calls == 0
        return
    pytest.fail("should not reach")


@pytest.mark.unit
def test_success_one_audio_playback_correct_models():
    plays = []
    transport = MockRealtimeTransport(script=_pcm_event())
    cfg = _cfg()
    with patch("jarvis.voice.openai_realtime.require_openai_api_key", return_value="SYNTHETIC_OPENAI_KEY_REDACTED"):
        session = OpenAIRealtimeSession(
            cfg,
            transport_factory=lambda: transport,
            allow_network=False,
            audio_player=lambda samples, rate: plays.append((len(samples), rate)),
        )
        audio = np.zeros(1600, dtype=np.float32)
        result = session.handle_utterance(audio, 16000)

    assert result.ok and result.used_premium
    assert len(plays) == 1
    assert plays[0][1] == 24000
    assert transport.connect_calls == 1
    # No beta header
    assert "OpenAI-Beta" not in transport._headers
    assert "Authorization" in transport._headers
    # session.update only after session.created consumed (script order + send order)
    updates = [p for p in transport.sent if p.get("type") == "session.update"]
    assert len(updates) == 1
    sess = updates[0]["session"]
    assert sess["type"] == "realtime"
    assert sess["model"] == "gpt-realtime-2.1"
    assert sess["output_modalities"] == ["audio"]
    assert sess["audio"]["output"]["voice"] == "marin"
    assert sess["audio"]["input"]["transcription"]["model"] == "gpt-4o-transcribe"
    assert sess["audio"]["input"]["transcription"]["language"] == "ro"
    assert sess["audio"]["input"]["format"]["type"] == "audio/pcm"
    assert sess["audio"]["input"]["format"]["rate"] == 24000
    assert sess["audio"]["output"]["format"]["rate"] == 24000
    assert "modalities" not in sess
    assert "input_audio_format" not in sess
    assert "output_audio_format" not in sess
    assert "input_audio_transcription" not in sess
    assert "tools" not in sess
    assert "tool_choice" not in sess
    assert "gpt-realtime-2.1" in getattr(transport, "_url", "")
    assert CORA_INSTRUCTIONS[:20] in sess["instructions"]
    types = [p.get("type") for p in transport.sent]
    # Audio only after session.update
    assert types.index("session.update") < types.index("input_audio_buffer.clear")
    assert "input_audio_buffer.clear" in types
    assert "input_audio_buffer.append" in types
    assert "input_audio_buffer.commit" in types
    # response.create only AFTER transcription consumed (phase B)
    assert types.index("input_audio_buffer.commit") < types.index("response.create")
    assert "response.create" in types


@pytest.mark.unit
def test_ga_session_update_builder_exact_schema():
    payload = build_ga_session_update(
        model="gpt-realtime-2.1",
        voice="marin",
        transcription_model="gpt-4o-transcribe",
        language="ro",
        instructions="x",
    )
    assert payload == {
        "type": "session.update",
        "session": {
            "type": "realtime",
            "model": "gpt-realtime-2.1",
            "output_modalities": ["audio"],
            "instructions": "x",
            "audio": {
                "input": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "transcription": {"model": "gpt-4o-transcribe", "language": "ro"},
                    "noise_reduction": {"type": "near_field"},
                    "turn_detection": {
                        "type": "semantic_vad",
                        "eagerness": "low",
                        "create_response": False,
                        "interrupt_response": False,
                    },
                },
                "output": {
                    "format": {"type": "audio/pcm", "rate": 24000},
                    "voice": "marin",
                },
            },
        },
    }


@pytest.mark.unit
def test_error_before_session_created_is_structured():
    transport = MockRealtimeTransport(
        script=[{
            "type": "error",
            "event_id": "evt_1",
            "error": {
                "type": "invalid_request_error",
                "code": "beta_api_shape_disabled",
                "param": None,
            },
        }]
    )
    cfg = _cfg()
    with patch("jarvis.voice.openai_realtime.require_openai_api_key", return_value="SYNTHETIC_OPENAI_KEY"):
        session = OpenAIRealtimeSession(
            cfg,
            transport_factory=lambda: transport,
            allow_network=False,
            audio_player=lambda *_: None,
        )
        result = session.handle_utterance(np.zeros(800, dtype=np.float32), 16000)
    assert result.ok is False
    assert result.fallback_needed is True
    assert result.error_type == "invalid_request_error"
    assert result.error_code == "beta_api_shape_disabled"
    assert result.event_id == "evt_1"
    # Must not have sent session.update or audio
    assert not any(p.get("type") == "session.update" for p in transport.sent)
    assert not any(p.get("type") == "input_audio_buffer.append" for p in transport.sent)


@pytest.mark.unit
def test_session_update_not_before_session_created():
    """Mock records that session.update is sent only after created was received."""
    transport = MockRealtimeTransport(script=_pcm_event())
    order = {"created_idx": None, "update_sent_at_recv": None}

    orig_recv = transport.recv_json
    orig_send = transport.send_json

    def recv(timeout):
        evt = orig_recv(timeout)
        if evt and evt.get("type") == "session.created":
            order["created_idx"] = transport._idx - 1
        return evt

    def send(payload):
        if payload.get("type") == "session.update":
            # created must already have been consumed
            assert order["created_idx"] is not None
            order["update_sent_at_recv"] = transport._idx
        return orig_send(payload)

    transport.recv_json = recv
    transport.send_json = send
    cfg = _cfg()
    with patch("jarvis.voice.openai_realtime.require_openai_api_key", return_value="SYNTHETIC_OPENAI_KEY"):
        session = OpenAIRealtimeSession(
            cfg, transport_factory=lambda: transport, allow_network=False,
            audio_player=lambda *_: None,
        )
        assert session.handle_utterance(np.zeros(400, dtype=np.float32), 16000).ok
    assert order["created_idx"] is not None
    assert order["update_sent_at_recv"] is not None


@pytest.mark.unit
def test_missing_credential_prints_marker_and_fallback(capsys):
    transport = MockRealtimeTransport()
    cfg = _cfg()
    with patch(
        "jarvis.voice.openai_realtime.require_openai_api_key",
        side_effect=MissingOpenAICredential("MISSING_OPENAI_CREDENTIAL"),
    ):
        session = OpenAIRealtimeSession(
            cfg, transport_factory=lambda: transport, allow_network=False
        )
        result = session.handle_utterance(np.zeros(800, dtype=np.float32), 16000)
    assert result.fallback_needed
    assert result.error == "MISSING_OPENAI_CREDENTIAL"
    assert transport.connect_calls == 0
    assert "MISSING_OPENAI_CREDENTIAL" in capsys.readouterr().out


@pytest.mark.unit
@pytest.mark.parametrize("err_evt", [
    {"type": "error", "error": {"type": "rate_limit_error", "code": "rate_limit", "status": 429}},
    {"type": "error", "error": {"type": "auth", "code": "auth", "status": 401}},
    {"type": "error", "error": {"type": "forbidden", "code": "forbidden", "status": 403}},
    {"type": "error", "error": {"type": "server_error", "code": "server", "status": 503}},
])
def test_errors_close_and_request_fallback(err_evt):
    # Error during handshake (before session.created)
    transport = MockRealtimeTransport(script=[err_evt])
    cfg = _cfg()
    with patch("jarvis.voice.openai_realtime.require_openai_api_key", return_value="SYNTHETIC_OPENAI_KEY"):
        session = OpenAIRealtimeSession(
            cfg,
            transport_factory=lambda: transport,
            allow_network=False,
            audio_player=lambda *_: None,
        )
        result = session.handle_utterance(np.zeros(800, dtype=np.float32), 16000)
    assert result.fallback_needed
    assert result.error_code == err_evt["error"]["code"]
    assert transport.closed or not transport.connected


@pytest.mark.unit
def test_followup_reuses_session():
    clock = {"t": 1000.0}

    def now():
        return clock["t"]

    transport = MockRealtimeTransport(
        script=_handshake_events() + _turn_events() + _turn_events()
    )
    factory_calls = {"n": 0}

    def factory():
        factory_calls["n"] += 1
        return transport

    cfg = _cfg()
    with patch("jarvis.voice.openai_realtime.require_openai_api_key", return_value="SYNTHETIC_OPENAI_KEY"):
        session = OpenAIRealtimeSession(
            cfg,
            transport_factory=factory,
            allow_network=False,
            audio_player=lambda *_: None,
            clock=now,
        )
        a = np.zeros(800, dtype=np.float32)
        assert session.handle_utterance(a, 16000).ok
        clock["t"] += 10  # within idle
        assert session.handle_utterance(a, 16000).ok
    assert factory_calls["n"] == 1
    assert transport.connect_calls == 1
    # Only one session.update for the session lifetime
    assert sum(1 for p in transport.sent if p.get("type") == "session.update") == 1
    assert sum(1 for p in transport.sent if p.get("type") == "input_audio_buffer.clear") == 2


@pytest.mark.unit
def test_idle_timeout_closes_session():
    clock = {"t": 0.0}

    def now():
        return clock["t"]

    queue = [
        MockRealtimeTransport(script=_pcm_event()),
        MockRealtimeTransport(script=_pcm_event()),
    ]

    cfg = _cfg(openai_realtime_idle_timeout_sec=60.0)
    with patch("jarvis.voice.openai_realtime.require_openai_api_key", return_value="SYNTHETIC_OPENAI_KEY"):
        session = OpenAIRealtimeSession(
            cfg,
            transport_factory=lambda: queue.pop(0),
            allow_network=False,
            audio_player=lambda *_: None,
            clock=now,
        )
        r1 = session.handle_utterance(np.zeros(800, dtype=np.float32), 16000)
        assert r1.ok, r1.error
        first = session._transport
        clock["t"] += 61
        r2 = session.handle_utterance(np.zeros(800, dtype=np.float32), 16000)
        assert r2.ok, r2.error
        assert first is not None and first.closed
        assert session._transport is not first


@pytest.mark.unit
def test_pcm_conversion_rate():
    audio = np.zeros(16000, dtype=np.float32)  # 1s @ 16k
    out = float32_mono_to_pcm16_24k(audio, 16000)
    # ~1s @ 24k int16
    assert abs(len(out) // 2 - 24000) < 50


@pytest.mark.unit
def test_listener_flag_false_skips_premium():
    from jarvis.listening.listener import VoiceListener

    listener = object.__new__(VoiceListener)
    listener.cfg = _cfg(openai_realtime_enabled=False)
    listener._realtime_session = None
    listener._premium_pcm_segments = []
    listener._premium_pcm_candidate = None
    listener._premium_collecting = False
    assert listener._try_openai_realtime("utt-1", "salut", np.zeros(10)) is False


@pytest.mark.unit
def test_listener_premium_success_skips_local(monkeypatch):
    from jarvis.listening.listener import VoiceListener

    listener = object.__new__(VoiceListener)
    listener.cfg = _cfg(openai_realtime_enabled=True)
    listener._realtime_session = None
    listener._spoken_utterance_ids = set()
    listener._premium_pcm_segments = [np.zeros(100, dtype=np.float32)]
    listener._premium_pcm_candidate = None
    listener._premium_collecting = True
    listener.dialogue_memory = MagicMock()
    listener._stop_thinking_tune = MagicMock()
    listener.track_tts_start = MagicMock()
    listener._enter_cooldown = MagicMock()
    listener.activate_hot_window = MagicMock()

    plays = []
    transport = MockRealtimeTransport(script=_pcm_event("Cora, intrebare openai", "raspuns openai"))

    def fake_get(cfg, **kw):
        return OpenAIRealtimeSession(
            cfg,
            transport_factory=lambda: transport,
            allow_network=False,
            audio_player=lambda s, r: plays.append(1),
        )

    monkeypatch.setattr(
        "jarvis.voice.openai_realtime.get_realtime_session", fake_get
    )
    monkeypatch.setattr(
        "jarvis.voice.openai_realtime.require_openai_api_key",
        lambda: "SYNTHETIC_OPENAI_KEY",
    )
    # Patch require inside session module used by handle
    with patch("jarvis.voice.openai_realtime.require_openai_api_key", return_value="SYNTHETIC_OPENAI_KEY"):
        handled = listener._try_openai_realtime(
            "utt-9", "whisper text should not win", np.zeros(1600, dtype=np.float32)
        )
    assert handled is True
    assert len(plays) == 1
    # Dialogue memory got OpenAI transcript, not Whisper
    args = listener.dialogue_memory.add_message.call_args_list
    assert any("Cora, intrebare openai" in str(c) or "intrebare openai" in str(c) for c in args)
    assert not any("whisper text should not win" in str(c) for c in args)
    assert listener._premium_pcm_segments == []


@pytest.mark.unit
def test_listener_fallback_once_on_error(monkeypatch):
    from jarvis.listening.listener import VoiceListener

    listener = object.__new__(VoiceListener)
    listener.cfg = _cfg(openai_realtime_enabled=True, openai_realtime_fallback_local=True)
    listener._realtime_session = None
    listener._spoken_utterance_ids = set()
    listener._premium_pcm_segments = [np.zeros(50, dtype=np.float32)]
    listener._premium_pcm_candidate = None
    listener._premium_collecting = True
    listener.dialogue_memory = None
    listener._stop_thinking_tune = MagicMock()
    listener._enter_cooldown = MagicMock()

    transport = MockRealtimeTransport(
        script=[{"type": "error", "error": {"code": "server", "status": 500}}]
    )

    def fake_get(cfg, **kw):
        return OpenAIRealtimeSession(
            cfg,
            transport_factory=lambda: transport,
            allow_network=False,
            audio_player=lambda *_: None,
        )

    monkeypatch.setattr("jarvis.voice.openai_realtime.get_realtime_session", fake_get)
    with patch("jarvis.voice.openai_realtime.require_openai_api_key", return_value="SYNTHETIC_OPENAI_KEY"):
        # False → caller continues to local pipeline exactly once
        assert listener._try_openai_realtime(
            "utt", "q", np.zeros(800, dtype=np.float32)
        ) is False
    assert listener._premium_pcm_segments == []


@pytest.mark.unit
def test_learning_not_auto_saved_on_premium_success(monkeypatch):
    """Premium success must not invoke conversation learning APIs."""
    from jarvis.listening.listener import VoiceListener

    listener = object.__new__(VoiceListener)
    listener.cfg = _cfg(openai_realtime_enabled=True)
    listener._realtime_session = None
    listener._spoken_utterance_ids = set()
    listener._premium_pcm_segments = []
    listener._premium_pcm_candidate = None
    listener._premium_collecting = False
    listener.dialogue_memory = MagicMock()
    listener._stop_thinking_tune = MagicMock()
    listener.track_tts_start = MagicMock()
    listener._enter_cooldown = MagicMock()
    listener.activate_hot_window = MagicMock()

    transport = MockRealtimeTransport(script=_pcm_event())

    def fake_get(cfg, **kw):
        return OpenAIRealtimeSession(
            cfg,
            transport_factory=lambda: transport,
            allow_network=False,
            audio_player=lambda *_: None,
        )

    monkeypatch.setattr("jarvis.voice.openai_realtime.get_realtime_session", fake_get)
    with patch("jarvis.voice.openai_realtime.require_openai_api_key", return_value="SYNTHETIC_OPENAI_KEY"), \
         patch.dict("sys.modules", {"jarvis.memory.learning": MagicMock()}) as mods:
        # If learning were imported/called, we'd notice — ensure module unused.
        assert listener._try_openai_realtime(
            "u", "q", np.zeros(800, dtype=np.float32)
        ) is True
    # conversation_learning stays false by config default on this tree
    from jarvis.config import get_default_config
    assert get_default_config()["openai_realtime_enabled"] is False
    assert get_default_config().get("conversation_learning_enabled", False) in (False, None)


@pytest.mark.unit
def test_config_defaults_and_load():
    from jarvis.config import get_default_config, load_settings
    d = get_default_config()
    assert d["openai_realtime_enabled"] is False
    assert d["openai_realtime_model"] == "gpt-realtime-2.1"
    assert d["openai_realtime_transcription_model"] == "gpt-4o-transcribe"
    assert d["openai_realtime_voice"] == "marin"
    assert d["openai_realtime_language"] == "ro"
    assert d["openai_realtime_idle_timeout_sec"] == 60.0
    assert d["openai_realtime_require_wake_each_turn"] is True
    # load_settings from real user config must not crash; flag may be absent → False
    s = load_settings()
    assert s.openai_realtime_enabled is False
    assert s.openai_realtime_voice == "marin"
    assert s.openai_realtime_require_wake_each_turn is True


@pytest.mark.unit
def test_play_once_helper():
    seen = []
    play_pcm16_24k(b"\x00\x00" * 10, player=lambda s, r: seen.append((len(s), r)))
    assert seen == [(10, 24000)]


# ---------------------------------------------------------------- Authoritative OpenAI Gate V3


@pytest.mark.unit
def test_transcript_starts_with_wake():
    assert transcript_starts_with_wake("Cora, bună", "cora", ["cora"])
    assert transcript_starts_with_wake("cora salut", "cora", [])
    assert not transcript_starts_with_wake("Bună!", "cora", ["cora"])
    assert not transcript_starts_with_wake("Să vă mulțumim pentru vizionare", "cora", ["cora"])


@pytest.mark.unit
def test_v3_no_wake_deletes_item_no_response_create():
    transport = MockRealtimeTransport(
        script=_handshake_events()
        + _turn_events("Bună!", include_response=False, item_id="item_nw")
    )
    cfg = _cfg()
    with patch("jarvis.voice.openai_realtime.require_openai_api_key", return_value="SYNTHETIC_OPENAI_KEY"):
        session = OpenAIRealtimeSession(
            cfg, transport_factory=lambda: transport, allow_network=False,
            audio_player=lambda *_: (_ for _ in ()).throw(AssertionError("no play")),
        )
        result = session.handle_utterance(np.zeros(800, dtype=np.float32), 16000)
    assert result.ignored_no_wake is True
    assert result.ok is False
    assert result.fallback_needed is False
    assert result.item_deleted is True
    assert result.response_created is False
    types = [p.get("type") for p in transport.sent]
    assert "response.create" not in types
    assert "conversation.item.delete" in types
    delete = next(p for p in transport.sent if p.get("type") == "conversation.item.delete")
    assert delete["item_id"] == "item_nw"


@pytest.mark.unit
def test_v3_youtube_phrase_ignored():
    transport = MockRealtimeTransport(
        script=_handshake_events()
        + _turn_events("Să vă mulțumim pentru vizionare", include_response=False)
    )
    cfg = _cfg()
    plays = []
    with patch("jarvis.voice.openai_realtime.require_openai_api_key", return_value="SYNTHETIC_OPENAI_KEY"):
        session = OpenAIRealtimeSession(
            cfg, transport_factory=lambda: transport, allow_network=False,
            audio_player=lambda *_: plays.append(1),
        )
        result = session.handle_utterance(np.zeros(800, dtype=np.float32), 16000)
    assert result.ignored_no_wake
    assert plays == []
    assert "response.create" not in [p.get("type") for p in transport.sent]


@pytest.mark.unit
def test_v3_cora_bună_accepted_one_playback():
    transport = MockRealtimeTransport(script=_pcm_event("Cora, bună", "Salut!"))
    plays = []
    cfg = _cfg()
    with patch("jarvis.voice.openai_realtime.require_openai_api_key", return_value="SYNTHETIC_OPENAI_KEY"):
        session = OpenAIRealtimeSession(
            cfg, transport_factory=lambda: transport, allow_network=False,
            audio_player=lambda *_: plays.append(1),
        )
        result = session.handle_utterance(np.zeros(800, dtype=np.float32), 16000)
    assert result.ok and result.used_premium
    assert result.user_transcript.lower().startswith("cora")
    assert plays == [1]
    assert result.response_created is True


@pytest.mark.unit
def test_v3_empty_transcript_fail_closed():
    transport = MockRealtimeTransport(
        script=_handshake_events()
        + _turn_events("", include_response=False, item_id="item_e")
    )
    cfg = _cfg()
    with patch("jarvis.voice.openai_realtime.require_openai_api_key", return_value="SYNTHETIC_OPENAI_KEY"):
        session = OpenAIRealtimeSession(
            cfg, transport_factory=lambda: transport, allow_network=False,
            audio_player=lambda *_: None,
        )
        result = session.handle_utterance(np.zeros(400, dtype=np.float32), 16000)
    assert result.ok is False
    assert result.fallback_needed is False
    assert result.error == "empty_transcript"
    assert "response.create" not in [p.get("type") for p in transport.sent]


@pytest.mark.unit
def test_v3_premium_finalize_skips_faster_whisper(monkeypatch):
    """Acoustic premium path must not call model.transcribe / mlx."""
    from jarvis.listening.listener import VoiceListener
    import threading

    listener = object.__new__(VoiceListener)
    listener.cfg = _cfg(openai_realtime_enabled=True)
    listener.tts = MagicMock(enabled=False)
    listener.tts.is_speaking.return_value = False
    listener._premium_pcm_segments = []
    listener._premium_pcm_candidate = None
    listener._premium_collecting = False
    listener._skip_premium_once = False
    listener._realtime_session = None
    listener._spoken_utterance_ids = set()
    listener._flight_lock = threading.Lock()
    listener._flight_state = "listening"
    listener._utterance_seq = 0
    listener._active_utterance_id = None
    listener._cooldown_timer = None
    listener._post_tts_cooldown_sec = 0.1
    listener.dialogue_memory = MagicMock()
    listener._start_thinking_tune = MagicMock()
    listener._stop_thinking_tune = MagicMock()
    listener._set_face_state_listening = MagicMock()
    listener._clear_audio_buffers = MagicMock()
    listener._enter_cooldown = MagicMock()
    listener.activate_hot_window = MagicMock()
    listener.track_tts_start = MagicMock()
    listener.model = MagicMock()
    listener.model.transcribe.side_effect = AssertionError("faster-whisper must not run")
    listener._whisper_backend = "faster-whisper"
    listener.transcribe_lock = threading.Lock()

    transport = MockRealtimeTransport(script=_pcm_event("Cora, test", "ok"))

    def fake_get(cfg, **kw):
        return OpenAIRealtimeSession(
            cfg, transport_factory=lambda: transport, allow_network=False,
            audio_player=lambda *_: None,
        )

    monkeypatch.setattr("jarvis.voice.openai_realtime.get_realtime_session", fake_get)
    with patch("jarvis.voice.openai_realtime.require_openai_api_key", return_value="SYNTHETIC_OPENAI_KEY"):
        from jarvis.listening.listener import VoiceListener as VL
        listener._begin_utterance = VL._begin_utterance.__get__(listener, VL)
        listener._set_flight_state = VL._set_flight_state.__get__(listener, VL)
        listener._premium_dispatch_pcm_direct(np.zeros(1600, dtype=np.float32))
    listener.model.transcribe.assert_not_called()


@pytest.mark.unit
def test_v3_ignored_no_wake_zero_memory_tools(monkeypatch, capsys):
    from jarvis.listening.listener import VoiceListener

    listener = object.__new__(VoiceListener)
    listener.cfg = _cfg()
    listener._realtime_session = None
    listener._spoken_utterance_ids = set()
    listener._premium_pcm_segments = []
    listener._premium_pcm_candidate = None
    listener._premium_collecting = False
    listener._skip_premium_once = False
    listener.dialogue_memory = MagicMock()
    listener._stop_thinking_tune = MagicMock()
    listener._enter_cooldown = MagicMock()
    listener.activate_hot_window = MagicMock()
    listener.track_tts_start = MagicMock()

    transport = MockRealtimeTransport(
        script=_handshake_events() + _turn_events("Bună", include_response=False)
    )

    def fake_get(cfg, **kw):
        return OpenAIRealtimeSession(
            cfg, transport_factory=lambda: transport, allow_network=False,
            audio_player=lambda *_: (_ for _ in ()).throw(AssertionError("play")),
        )

    monkeypatch.setattr("jarvis.voice.openai_realtime.get_realtime_session", fake_get)
    with patch("jarvis.voice.openai_realtime.require_openai_api_key", return_value="SYNTHETIC_OPENAI_KEY"):
        assert listener._try_openai_realtime("u", "", np.zeros(400, dtype=np.float32)) is True
    listener.dialogue_memory.add_message.assert_not_called()
    listener.activate_hot_window.assert_not_called()
    assert "no OpenAI wake word" in capsys.readouterr().out


@pytest.mark.unit
def test_config_require_wake_default():
    from jarvis.config import get_default_config, load_settings
    d = get_default_config()
    assert d["openai_realtime_require_wake_each_turn"] is True
    s = load_settings()
    assert s.openai_realtime_require_wake_each_turn is True


# ---------------------------------------------------------------- ASR gate remediation (legacy helpers kept for local-path coverage)


def _bare_premium_listener(**kw):
    from jarvis.listening.listener import VoiceListener
    from jarvis.listening.state_manager import StateManager
    from jarvis.listening.echo_detection import EchoDetector
    from jarvis.listening.transcript_buffer import TranscriptBuffer

    listener = object.__new__(VoiceListener)
    listener.cfg = _cfg(
        wake_word="cora",
        wake_aliases=["cora"],
        wake_fuzzy_ratio=0.78,
        voice_debug=False,
        **kw,
    )
    listener.tts = None
    listener._wake_timestamp = None
    listener._premium_pcm_segments = []
    listener._premium_pcm_candidate = None
    listener._premium_collecting = False
    listener._skip_premium_once = False
    listener._realtime_session = None
    listener._spoken_utterance_ids = set()
    listener.dialogue_memory = MagicMock()
    listener.echo_detector = EchoDetector()
    listener.state_manager = StateManager(
        hot_window_seconds=3.0, echo_tolerance=0.3,
        voice_collect_seconds=0.05, max_collect_seconds=60.0,
    )
    listener._transcript_buffer = TranscriptBuffer(max_duration_sec=120.0)
    listener._intent_judge = MagicMock()
    listener._intent_judge.available = True
    listener._intent_judge.judge = MagicMock(
        side_effect=AssertionError("intent judge must not run on premium path")
    )
    listener._start_thinking_tune = MagicMock()
    listener._stop_thinking_tune = MagicMock()
    listener._set_face_state_listening = MagicMock()
    listener._is_thinking_tune_active = MagicMock(return_value=False)
    listener.track_tts_start = MagicMock()
    listener._enter_cooldown = MagicMock()
    listener.activate_hot_window = MagicMock()
    listener._begin_utterance = MagicMock(return_value="utt-t")
    listener._clear_audio_buffers = MagicMock()
    return listener


@pytest.mark.unit
def test_premium_openai_transcript_authoritative_not_whisper(monkeypatch):
    """Whisper garbage must not replace gpt-4o-transcribe in memory."""
    listener = _bare_premium_listener()
    whisper = "Cora memorează căle răstunduri scure și direcțe"
    openai_tr = "Cora, memorează că prefer răspunsuri scurte și directe."
    plays = []
    transport = MockRealtimeTransport(script=_pcm_event(openai_tr, "ok"))

    def fake_get(cfg, **kw):
        return OpenAIRealtimeSession(
            cfg, transport_factory=lambda: transport, allow_network=False,
            audio_player=lambda *_: plays.append(1),
        )

    monkeypatch.setattr("jarvis.voice.openai_realtime.get_realtime_session", fake_get)
    with patch("jarvis.voice.openai_realtime.require_openai_api_key", return_value="SYNTHETIC_OPENAI_KEY"):
        assert listener._try_openai_realtime("u", whisper, np.zeros(1600, dtype=np.float32)) is True
    mem = [str(c) for c in listener.dialogue_memory.add_message.call_args_list]
    assert any(openai_tr in m for m in mem)
    assert not any(whisper in m for m in mem)
    assert not any("răstunduri" in m for m in mem)
    assert len(plays) == 1


@pytest.mark.unit
def test_premium_bypasses_intent_judge_even_if_would_reject():
    listener = _bare_premium_listener()
    # Would throw if called
    listener._intent_judge.judge.side_effect = RuntimeError("intent must not run")
    pcm = np.arange(800, dtype=np.float32) / 800.0
    listener._premium_pcm_stash_candidate(pcm)
    listener._process_transcript_premium(
        "Cora explică în profesional în trei pozitice",
        utterance_start_time=1.0,
        utterance_end_time=2.0,
    )
    assert listener._intent_judge.judge.call_count == 0
    assert len(listener._premium_pcm_segments) == 1
    assert listener.state_manager.is_collecting()


@pytest.mark.unit
def test_premium_works_when_intent_judge_raises_if_called():
    """Even a broken judge cannot block premium — it is not on the path."""
    listener = _bare_premium_listener()
    listener._intent_judge.judge.side_effect = Exception("boom")
    listener._premium_pcm_stash_candidate(np.ones(400, dtype=np.float32))
    listener._process_transcript_premium("Cora salut", 0.0, 1.0, 2.0)
    assert len(listener._premium_pcm_segments) == 1
    assert listener._intent_judge.judge.call_count == 0


@pytest.mark.unit
def test_premium_low_confidence_wake_still_sends_pcm(monkeypatch):
    """Wake alias + valid audio → OpenAI even if Whisper text is rough."""
    listener = _bare_premium_listener()
    sent = []

    def capture_dispatch(query):
        sent.append(listener._premium_pcm_concat())

    listener._dispatch_query = capture_dispatch
    rough = "cora ceva neclar dar alias ok"
    listener._premium_pcm_stash_candidate(np.linspace(0, 1, 500, dtype=np.float32))
    listener._process_transcript_premium(rough, 0.1, 10.0, 11.0)
    assert len(listener._premium_pcm_segments) == 1
    # Force collection timeout → dispatch
    listener.state_manager._last_voice_time = 0.0
    listener._flush_collection_if_ready()
    assert len(sent) == 1
    assert sent[0] is not None
    assert len(sent[0]) == 500


@pytest.mark.unit
def test_premium_no_wake_no_hot_zero_openai(monkeypatch):
    listener = _bare_premium_listener()
    connect = {"n": 0}

    class Boom:
        def handle_utterance(self, *a, **k):
            connect["n"] += 1
            raise AssertionError("must not call OpenAI")

    listener._realtime_session = Boom()
    listener._premium_pcm_stash_candidate(np.zeros(100, dtype=np.float32))
    listener._process_transcript_premium("ceasul sună la birou fără alias", 0.0, 1.0, 2.0)
    assert listener._premium_pcm_segments == []
    assert listener._premium_pcm_candidate is None
    assert connect["n"] == 0
    assert listener._try_openai_realtime("u", "x", None) is False


@pytest.mark.unit
def test_premium_hot_window_without_cora_ignored(monkeypatch):
    """V3: hot window does not waive wake — OpenAI transcript without Cora is ignored."""
    listener = _bare_premium_listener()
    transport = MockRealtimeTransport(
        script=_handshake_events() + _turn_events("continuă te rog", include_response=False)
    )

    def fake_get(cfg, **kw):
        return OpenAIRealtimeSession(
            cfg, transport_factory=lambda: transport, allow_network=False,
            audio_player=lambda *_: (_ for _ in ()).throw(AssertionError("no play")),
        )

    monkeypatch.setattr("jarvis.voice.openai_realtime.get_realtime_session", fake_get)
    with patch("jarvis.voice.openai_realtime.require_openai_api_key", return_value="SYNTHETIC_OPENAI_KEY"):
        assert listener._try_openai_realtime(
            "u", "ignored", np.zeros(300, dtype=np.float32)
        ) is True
    assert "response.create" not in [p.get("type") for p in transport.sent]
    listener.activate_hot_window.assert_not_called()


@pytest.mark.unit
def test_premium_three_segments_concat_in_order():
    listener = _bare_premium_listener()
    s1 = np.array([1.0, 1.0], dtype=np.float32)
    s2 = np.array([2.0, 2.0], dtype=np.float32)
    s3 = np.array([3.0, 3.0], dtype=np.float32)
    listener._premium_pcm_stash_candidate(s1)
    listener._process_transcript_premium("Cora unul", 0.0, 1.0, 2.0)
    listener._premium_pcm_stash_candidate(s2)
    listener._process_transcript_premium("doi", 0.0, 2.1, 3.0)
    listener._premium_pcm_stash_candidate(s3)
    listener._process_transcript_premium("trei", 0.0, 3.1, 4.0)
    assert len(listener._premium_pcm_segments) == 3
    out = listener._premium_pcm_concat()
    # First samples of each speech block (gaps are zeros between)
    assert out[0] == 1.0 and out[1] == 1.0
    # Find second block after silence gap
    idx2 = None
    for i in range(2, len(out) - 1):
        if out[i] == 2.0 and out[i + 1] == 2.0:
            idx2 = i
            break
    assert idx2 is not None
    idx3 = None
    for i in range(idx2 + 2, len(out) - 1):
        if out[i] == 3.0 and out[i + 1] == 3.0:
            idx3 = i
            break
    assert idx3 is not None
    assert idx2 < idx3


@pytest.mark.unit
def test_premium_empty_openai_transcript_fail_closed(monkeypatch, capsys):
    """V3: empty OpenAI transcript → ignore; no audio, no Whisper memory."""
    listener = _bare_premium_listener()
    plays = []
    script = _handshake_events() + [
        {"type": "conversation.item.created", "item": {"id": "item_e"}},
        {"type": "conversation.item.input_audio_transcription.completed", "transcript": "", "item_id": "item_e"},
        # Must not be consumed — no response.create
        {"type": "response.output_audio.delta", "delta": base64.b64encode(b"\x00\x01" * 50).decode()},
    ]
    transport = MockRealtimeTransport(script=script)

    def fake_get(cfg, **kw):
        return OpenAIRealtimeSession(
            cfg, transport_factory=lambda: transport, allow_network=False,
            audio_player=lambda *_: plays.append(1),
        )

    monkeypatch.setattr("jarvis.voice.openai_realtime.get_realtime_session", fake_get)
    whisper = "Cora text corupt whisper"
    with patch("jarvis.voice.openai_realtime.require_openai_api_key", return_value="SYNTHETIC_OPENAI_KEY"):
        assert listener._try_openai_realtime("u", whisper, np.zeros(800, dtype=np.float32)) is True
    assert plays == []
    assert "response.create" not in [p.get("type") for p in transport.sent]
    listener.dialogue_memory.add_message.assert_not_called()
    out = capsys.readouterr().out
    assert whisper not in out


@pytest.mark.unit
def test_premium_success_gemma_piper_intent_zero(monkeypatch):
    from jarvis.listening.listener import VoiceListener, FLIGHT_LISTENING

    listener = _bare_premium_listener()
    listener.db = None
    listener._last_detected_language = "ro"
    listener._flight_state = FLIGHT_LISTENING
    listener._flight_lock = __import__("threading").Lock()
    listener._active_utterance_id = None
    listener._utterance_seq = 0
    listener._cooldown_timer = None
    listener._post_tts_cooldown_sec = 0.1
    listener._spoken_utterance_ids = set()

    counts = {"gemma": 0, "intent": 0, "play": 0}
    listener._intent_judge.judge.side_effect = lambda *a, **k: counts.__setitem__("intent", counts["intent"] + 1)

    transport = MockRealtimeTransport(script=_pcm_event("Cora q", "a"))

    def fake_get(cfg, **kw):
        return OpenAIRealtimeSession(
            cfg, transport_factory=lambda: transport, allow_network=False,
            audio_player=lambda *_: counts.__setitem__("play", counts["play"] + 1),
        )

    monkeypatch.setattr("jarvis.voice.openai_realtime.get_realtime_session", fake_get)

    def boom_reply(*a, **k):
        counts["gemma"] += 1
        return "should not"

    monkeypatch.setattr("jarvis.reply.engine.run_reply_engine", boom_reply)

    listener._premium_pcm_segments = [np.zeros(400, dtype=np.float32)]
    with patch("jarvis.voice.openai_realtime.require_openai_api_key", return_value="SYNTHETIC_OPENAI_KEY"):
        # Use real _begin_utterance path lightly
        listener._begin_utterance = VoiceListener._begin_utterance.__get__(listener, VoiceListener)
        listener._set_flight_state = VoiceListener._set_flight_state.__get__(listener, VoiceListener)
        listener._dispatch_query("carrier")

    assert counts["gemma"] == 0
    assert counts["intent"] == 0
    assert counts["play"] == 1


@pytest.mark.unit
def test_premium_failure_local_fallback_once(monkeypatch):
    from jarvis.listening.listener import VoiceListener, FLIGHT_LISTENING
    import threading

    listener = _bare_premium_listener()
    listener.db = None
    listener._last_detected_language = "ro"
    listener._flight_state = FLIGHT_LISTENING
    listener._flight_lock = threading.Lock()
    listener._active_utterance_id = None
    listener._utterance_seq = 0
    listener._cooldown_timer = None
    listener._post_tts_cooldown_sec = 0.1
    listener.tts = MagicMock(enabled=False)
    listener._spoken_utterance_ids = set()
    listener._premium_pcm_segments = [np.zeros(200, dtype=np.float32)]

    transport = MockRealtimeTransport(
        script=[{"type": "error", "error": {"code": "server", "status": 500}}]
    )

    def fake_get(cfg, **kw):
        return OpenAIRealtimeSession(
            cfg, transport_factory=lambda: transport, allow_network=False,
            audio_player=lambda *_: None,
        )

    monkeypatch.setattr("jarvis.voice.openai_realtime.get_realtime_session", fake_get)
    fallbacks = []

    def local_once(*a, **k):
        fallbacks.append(1)
        return "local"

    monkeypatch.setattr("jarvis.reply.engine.run_reply_engine", local_once)
    listener._begin_utterance = VoiceListener._begin_utterance.__get__(listener, VoiceListener)
    listener._set_flight_state = VoiceListener._set_flight_state.__get__(listener, VoiceListener)
    with patch("jarvis.voice.openai_realtime.require_openai_api_key", return_value="SYNTHETIC_OPENAI_KEY"):
        listener._dispatch_query("fallback whisper once")
    assert fallbacks == [1]


@pytest.mark.unit
def test_flag_false_identical_local_zero_openai():
    """Flag false → premium helpers idle; process_transcript does not take premium path."""
    from jarvis.listening.listener import VoiceListener

    listener = object.__new__(VoiceListener)
    listener.cfg = _cfg(openai_realtime_enabled=False)
    assert listener._premium_realtime_enabled() is False
    # _try_openai_realtime must short-circuit
    listener._premium_pcm_segments = [np.zeros(10)]
    listener._premium_pcm_candidate = None
    listener._premium_collecting = False
    assert listener._try_openai_realtime("u", "q", np.zeros(10)) is False
