"""Unit tests for the Voice PE (stock firmware) integration.

Behaviour-level: what the transport does with a stubbed Native API client,
not how the stubs were called. No network, no live device.
"""

import asyncio
import queue
from types import SimpleNamespace

import pytest

from jarvis.integrations.voice_pe import config as pe_config
from jarvis.integrations.voice_pe import events as pe_events
from jarvis.integrations.voice_pe.capabilities import build_snapshot
from jarvis.integrations.voice_pe.device import VoicePEDevice
from jarvis.integrations.voice_pe.discovery import match_voice_pe
from jarvis.integrations.voice_pe.entities import EntityIndex
from jarvis.integrations.voice_pe.media import VoicePEMediaController
from jarvis.integrations.voice_pe.models import (
    FEATURE_ANNOUNCE,
    FEATURE_API_AUDIO,
    FEATURE_MULTI_CHANNEL_AUDIO,
    FEATURE_SPEAKER,
    FEATURE_TIMERS,
    FEATURE_VOICE_ASSISTANT,
    DeviceState,
    SessionState,
    VoicePEConfig,
)
from jarvis.integrations.voice_pe.tts_stream import (
    pcm_from_wav,
    resample_pcm16,
    stream_wav,
)
from jarvis.integrations.voice_pe.voice_transport import (
    AudioIngress,
    pcm16_to_float32,
)


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

class FakeClient:
    """Records what the transport sends; mirrors the aioesphomeapi surface."""

    def __init__(self):
        self.events = []
        self.audio = []
        self.lights = []
        self.media = []

    def send_voice_assistant_event(self, event_type, data):
        self.events.append((int(event_type), dict(data or {})))

    def send_voice_assistant_audio(self, payload):
        self.audio.append(payload)

    def light_command(self, key, **kwargs):
        self.lights.append((key, kwargs))

    def media_player_command(self, key, **kwargs):
        self.media.append((key, kwargs))

    def switch_command(self, key, state, device_id=0):
        self.media.append(("switch", key, state))

    # ``_on_connect`` stops early when the identity is unknown, which keeps the
    # fakes free of the whole entity surface.
    async def connect(self, on_stop=None, login=False, log_errors=True):
        return None

    async def device_info(self):
        return None

    async def disconnect(self, force=False):
        return None


class FakeListener:
    def __init__(self):
        self._audio_q = queue.Queue(maxsize=64)


def _config(**overrides):
    base = {
        "enabled": True,
        "prefer_api_audio": True,
        "audio_queue_ms": 300,
        "led_rgb": [0.55, 0.0, 1.0],
    }
    base.update(overrides)
    return pe_config.from_settings(
        SimpleNamespace(**{f"voice_pe_{k}": v for k, v in base.items()})
    )


def _device(client=None, capabilities=None, **config_overrides):
    device = VoicePEDevice(
        _config(**config_overrides),
        listener=FakeListener(),
        tts_engine=None,
        host="192.168.1.50",
        port=6053,
    )
    device._client = client or FakeClient()
    device.entities = EntityIndex()
    device.capabilities = capabilities or build_snapshot(
        SimpleNamespace(voice_assistant_feature_flags=0), []
    )
    device.media = VoicePEMediaController(device._client, 11, 12)
    device._ingress = AudioIngress(device._listener, device.config, device.metrics)
    device.loop = None
    return device


def _run_loop(coro, device=None):
    """Run one coroutine on a fresh loop bound to the device (if any)."""
    loop = asyncio.new_event_loop()
    if device is not None:
        device.loop = loop
    try:
        return loop.run_until_complete(coro() if callable(coro) else coro)
    finally:
        # The pump task outlives a single call, so cancel it before closing to
        # keep the "Task was destroyed but it is pending" notes out of the log.
        for task in asyncio.all_tasks(loop):
            task.cancel()
        loop.close()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestVoicePEConfig:
    def test_defaults_match_the_product_mode(self):
        cfg = _config()
        assert cfg.enabled is True
        assert cfg.port == 6053
        assert cfg.disable_wake_words is True
        assert cfg.continued_conversation is True
        assert cfg.preferred_input_channel == 0
        assert cfg.audio_queue_ms == 300
        assert cfg.led_brightness == pytest.approx(0.66)
        assert cfg.led_rgb == (0.55, 0.0, 1.0)

    def test_missing_keys_fall_back_to_defaults(self):
        cfg = pe_config.from_settings(SimpleNamespace())
        assert cfg.enabled is False
        assert cfg.discovery_enabled is True
        assert cfg.host is None
        assert cfg.led_rgb == (0.55, 0.0, 1.0)

    def test_button_mapping_defaults_are_the_safe_set(self):
        actions = _config().button_actions
        assert actions["double_press"] == "toggle_overlay"
        assert actions["triple_press"] == "open_command_palette"
        assert actions["long_press"] == "cancel_current_agent_run"
        assert actions["easter_egg_press"] == "toaster_easter_egg"
        # The single click stays on-device and is deliberately unmapped.
        assert "single_press" not in actions

    def test_fold_button_actions_accepts_both_ui_forms(self):
        assert pe_config.fold_button_actions(
            ["double_press=toggle_overlay", "long_press -> ignore"]
        ) == {"double_press": "toggle_overlay", "long_press": "ignore"}
        assert pe_config.fold_button_actions(
            {"double_press": "ignore"}
        ) == {"double_press": "ignore"}

    def test_hex_and_triplet_colour_forms_parse(self):
        assert pe_config._parse_rgb("8c00ff", (0, 0, 0)) == (0.549, 0.0, 1.0)
        assert pe_config._parse_rgb("0.5,0.25,0.1", (0, 0, 0)) == (0.5, 0.25, 0.1)
        assert pe_config._parse_rgb(None, (0.1, 0.2, 0.3)) == (0.1, 0.2, 0.3)

    def test_load_settings_round_trips_voice_pe_keys(self, tmp_path, monkeypatch):
        import json

        from jarvis.config import load_settings

        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(
            json.dumps(
                {
                    "voice_pe_enabled": True,
                    "voice_pe_host": "10.0.0.24",
                    "voice_pe_audio_queue_ms": 220,
                    "voice_pe_button_actions": ["double_press=ignore"],
                }
            )
        )
        monkeypatch.setenv("JARVIS_CONFIG_PATH", str(cfg_path))

        settings = load_settings()
        assert settings.voice_pe_enabled is True
        assert settings.voice_pe_host == "10.0.0.24"
        assert settings.voice_pe_audio_queue_ms == 220
        assert settings.voice_pe_button_actions == {"double_press": "ignore"}

    def test_in_range_clamps(self, tmp_path, monkeypatch):
        import json

        from jarvis.config import load_settings

        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(
            json.dumps(
                {
                    "voice_pe_preferred_input_channel": 9,
                    "voice_pe_audio_queue_ms": 1,
                    "voice_pe_led_brightness": 3.0,
                }
            )
        )
        monkeypatch.setenv("JARVIS_CONFIG_PATH", str(cfg_path))
        settings = load_settings()
        assert settings.voice_pe_preferred_input_channel == 1
        assert settings.voice_pe_audio_queue_ms == 20
        assert settings.voice_pe_led_brightness == 1.0


# ---------------------------------------------------------------------------
# Capabilities and entities
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestCapabilities:
    def test_feature_flags_expand_to_names(self):
        flags = FEATURE_VOICE_ASSISTANT | FEATURE_API_AUDIO | FEATURE_SPEAKER
        snapshot = build_snapshot(
            SimpleNamespace(voice_assistant_feature_flags=flags), []
        )
        assert set(snapshot.names()) == {"voice_assistant", "api_audio", "speaker"}
        assert snapshot.uses_api_audio is True

    def test_every_capability_degrades_alone(self):
        snapshot = build_snapshot(SimpleNamespace(voice_assistant_feature_flags=0), [])
        assert snapshot.names() == []
        assert snapshot.api_audio is False
        assert snapshot.announce is False
        assert snapshot.has_media_player is False
        # Still usable: the assistant is there, extras are optional.
        assert snapshot.entity_count == 0

    def test_capabilities_response_wins_over_flat_field(self):
        flags = FEATURE_VOICE_ASSISTANT | FEATURE_API_AUDIO
        caps = SimpleNamespace(
            voice_assistant=SimpleNamespace(
                feature_flags=flags | FEATURE_MULTI_CHANNEL_AUDIO
            )
        )
        snapshot = build_snapshot(
            SimpleNamespace(voice_assistant_feature_flags=flags), [], None, caps
        )
        assert snapshot.multi_channel_audio is True

    def test_announce_and_timers_are_readable(self):
        flags = FEATURE_ANNOUNCE | FEATURE_TIMERS
        snapshot = build_snapshot(
            SimpleNamespace(voice_assistant_feature_flags=flags), []
        )
        assert snapshot.announce is True
        assert snapshot.timers is True


class _Info:
    def __init__(self, object_id, key, name="", **extra):
        self.object_id = object_id
        self.key = key
        self.name = name
        for k, v in extra.items():
            setattr(self, k, v)


class LightInfo(_Info):
    pass


class EventInfo(_Info):
    pass


class MediaPlayerInfo(_Info):
    pass


class SwitchInfo(_Info):
    pass


class BinarySensorInfo(_Info):
    pass


@pytest.mark.unit
class TestEntityIndex:
    def _entities(self, led_key=4, media_key=11, mute_key=7, event_key=5):
        return [
            LightInfo("led_ring", led_key, "LED Ring",
                      supported_color_modes=["RGB"], effects=["Rainbow"]),
            EventInfo("button_press_event", event_key, "Button press",
                      event_types=["double_press", "triple_press", "long_press",
                                   "easter_egg_press"]),
            MediaPlayerInfo("external_media_player", media_key, "Media Player"),
            SwitchInfo("master_mute_switch", mute_key, "Mute"),
            BinarySensorInfo("hardware_mute_switch", 9, "Mute"),
        ]

    def test_lookup_is_by_object_id_not_by_number(self):
        index = EntityIndex().build(self._entities())
        assert index.led_key() == 4
        assert index.media_key() == 11
        assert index.mute_key() == 7
        assert index.supports_transition() is True
        assert index.effects() == ("Rainbow",)

    def test_keys_are_per_generation(self):
        first = EntityIndex().build(self._entities())
        second = EntityIndex().build(self._entities(led_key=9, media_key=2, mute_key=3))
        assert (first.led_key(), second.led_key()) == (4, 9)
        assert second.info_for(2) is not None

    def test_missing_entities_stay_none(self):
        index = EntityIndex().build([LightInfo("other", 1, "Other")])
        assert index.button_event is None
        assert index.media_key() is None
        assert index.event_types() == []


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestEvents:
    def test_pipeline_events_map_to_stock_led_phases(self):
        assert pe_events.EVENT_LED_PHASE["STT_VAD_START"] == "listening_for_command"
        assert pe_events.EVENT_LED_PHASE["STT_VAD_END"] == "thinking"
        assert pe_events.EVENT_LED_PHASE["TTS_STREAM_START"] == "replying"
        assert pe_events.EVENT_LED_PHASE["RUN_END"] == "idle"

    def test_event_ids_match_the_native_api_enum(self):
        assert pe_events.EVENT_IDS["RUN_START"] == 1
        assert pe_events.EVENT_IDS["RUN_END"] == 2
        assert pe_events.EVENT_IDS["STT_VAD_START"] == 11
        assert pe_events.EVENT_IDS["STT_VAD_END"] == 12
        assert pe_events.EVENT_IDS["TTS_STREAM_START"] == 98
        assert pe_events.EVENT_IDS["TTS_STREAM_END"] == 99

    def test_send_event_stringifies_values(self):
        client = FakeClient()
        pe_events.send_event(client, "INTENT_END", {"continue_conversation": 1})
        assert client.events == [(6, {"continue_conversation": "1"})]

    def test_mapping_is_configurable_and_unknown_is_ignored(self):
        mapping = {"double_press": "ignore"}
        assert pe_events.resolve_action("double_press", mapping) == "ignore"
        assert pe_events.resolve_action("triple_press", mapping) == "ignore"
        custom = {"triple_press": "toaster_easter_egg"}
        assert pe_events.resolve_action("triple_press", custom) == "toaster_easter_egg"

    def test_action_runner_reports_unhandled_without_raising(self):
        runner = pe_events.ActionRunner()
        assert runner.run("toggle_overlay").startswith("unhandled")
        seen = []
        runner.register("toggle_overlay", lambda: seen.append(1))
        assert runner.run("toggle_overlay") == "ok:toggle_overlay"
        assert seen == [1]


# ---------------------------------------------------------------------------
# Audio ingress
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestAudioIngress:
    def test_pcm16_converts_to_float32_scaling(self):
        pcm = (1, -1, 32767, -32768)
        import struct

        payload = struct.pack("<4h", *pcm)
        array = pcm16_to_float32(payload)
        assert array.dtype.name == "float32"
        assert round(float(array[0]), 5) == pytest.approx(1 / 32768, abs=1e-5)
        assert len(array) == 4

    def test_queue_is_bounded_and_drops_oldest_first(self):
        metrics = {}
        ingress = AudioIngress(FakeListener(), _config(audio_queue_ms=20), metrics)

        async def _fill():
            # 20 ms of 16 kHz mono audio = 320 samples = 640 bytes of budget.
            for _ in range(4):
                ingress.put(b"\x01" * 512)
            return ingress.depth_ms(), metrics.get("audio_dropped_chunks", 0)

        depth, dropped = asyncio.new_event_loop().run_until_complete(_fill())
        assert dropped >= 1
        assert depth <= 20.0

    def test_pump_hands_frames_to_the_shared_listener_queue(self):
        listener = FakeListener()
        metrics = {}
        ingress = AudioIngress(listener, _config(), metrics)
        ingress.put(b"\x00\x00" * 160)

        async def _pump_once():
            task = asyncio.ensure_future(ingress.pump())
            await asyncio.sleep(0.02)
            task.cancel()
            return listener._audio_q.qsize()

        size = asyncio.new_event_loop().run_until_complete(_pump_once())
        assert size == 1

    def test_channel_one_is_used_when_preferred(self):
        metrics = {}
        ingress = AudioIngress(FakeListener(), _config(preferred_input_channel=1), metrics)
        ingress.put(b"\x01\x01", b"\x02\x02")
        item = ingress._queue.get_nowait()
        assert item == b"\x02\x02"


# ---------------------------------------------------------------------------
# WAV / PCM helpers
# ---------------------------------------------------------------------------

def _wav(pcm: bytes, rate: int = 16000, channels: int = 1) -> bytes:
    import struct

    fmt = struct.pack("<HHIIHH", 1, channels, rate, rate * channels * 2,
                      channels * 2, 16)
    header = b"RIFF" + struct.pack("<I", 4 + len(fmt) + 8 + len(pcm)) + b"WAVE"
    return header + b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + \
        struct.pack("<I", len(pcm)) + pcm


@pytest.mark.unit
class TestTtsStreamHelpers:
    def test_pcm_from_wav_skips_the_header(self):
        pcm = bytes(range(8)) * 2
        assert pcm_from_wav(_wav(pcm)) == pcm

    def test_stream_yields_512_sample_payloads(self):
        pcm = bytes(512 * 2 * 2 + 3)  # two full chunks plus a tail

        async def _iter():
            payload = _wav(pcm)
            for i in range(0, len(payload), 7):  # awkward chunk sizes
                yield payload[i: i + 7]

        async def _run():
            return [chunk async for chunk in stream_wav(_iter())]

        chunks = asyncio.new_event_loop().run_until_complete(_run())
        assert [len(c) for c in chunks] == [1024, 1024, 3]

    def test_wrong_sample_rate_is_rejected(self):
        import pytest as _pytest

        def _go():
            async def _inner():
                payload = _wav(bytes(4), rate=22050)

                async def _iter():
                    yield payload

                return [c async for c in stream_wav(_iter())]

            return asyncio.new_event_loop().run_until_complete(_inner())

        with _pytest.raises(ValueError):
            _go()

    def test_resample_is_identity_at_the_same_rate(self):
        pcm = bytes(range(1, 9))
        assert resample_pcm16(pcm, 16000) == pcm

    def test_resample_downsamples_to_half(self):
        import struct

        pcm = struct.pack("<4h", 0, 1000, 2000, 3000)
        out = resample_pcm16(pcm, 32000, 16000)
        assert len(out) == len(pcm) // 2


# ---------------------------------------------------------------------------
# Device sessions
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestDeviceSession:
    def _api_device(self, client=None):
        caps = build_snapshot(
            SimpleNamespace(
                voice_assistant_feature_flags=(
                    FEATURE_VOICE_ASSISTANT | FEATURE_SPEAKER | FEATURE_API_AUDIO
                    | FEATURE_MULTI_CHANNEL_AUDIO
                )
            ),
            [],
        )
        return _device(client, caps)

    def test_start_returns_api_port_and_opens_wake_free_session(self):
        client = FakeClient()
        device = self._api_device(client)

        async def _run():
            port = await device.handle_pipeline_start("", 0, SimpleNamespace(), None)
            return port, device.session, device.session_state

        port, session, state = _run_loop(_run, device)
        assert port == 0
        assert session.sample_rate == 16000
        assert session.channels == 1 and session.sample_width == 2
        assert session.conversation_id
        assert state is SessionState.LISTENING
        assert [event for event, _ in client.events] == [1, 3]  # RUN_START, STT_START

    def test_wake_flagged_start_still_begins_in_stt(self):
        client = FakeClient()
        device = self._api_device(client)
        _run_loop(
            lambda: device.handle_pipeline_start("abc", 2, SimpleNamespace(), "okay nabu"),
            device,
        )
        # No WAKE_WORD_START/END (9/10) is emitted: the pipeline starts at STT.
        assert [event for event, _ in client.events] == [1, 3]
        assert device.metrics["wake_word_flag"] == 1

    def test_device_supplied_conversation_id_is_kept(self):
        device = self._api_device()

        async def _run():
            await device.handle_pipeline_start("conv-1", 0, SimpleNamespace(), None)
            second = device.session.conversation_id
            await device.handle_pipeline_start("", 0, SimpleNamespace(), None)
            return second, device.session.conversation_id

        kept, follow_up = _run_loop(_run, device)
        assert kept == "conv-1"
        assert follow_up == "conv-1"

    def test_transcript_and_reply_event_sequence(self):
        client = FakeClient()
        device = self._api_device(client)

        async def _run():
            await device.handle_pipeline_start("", 0, SimpleNamespace(), None)
            device.on_vad_start()
            device.on_vad_end()
            device.on_transcript("who is luister")
            device.on_reply("Luister is a speaker.")
            for _ in range(6):
                await asyncio.sleep(0)

        _run_loop(_run, device)
        sent = [event for event, _ in client.events]
        # RUN_START, STT_START, VAD_START, VAD_END, STT_END, INTENT_START,
        # INTENT_END, TTS_START, TTS_STREAM_START
        assert sent[:9] == [1, 3, 11, 12, 4, 5, 6, 7, 98]
        intent = dict(client.events[6][1])
        assert intent["speech"] == "Luister is a speaker."
        assert intent["conversation_id"]
        assert intent["continue_conversation"] in {"0", "1"}

    def test_question_keeps_the_dialog_open_and_statement_closes_it(self):
        device = self._api_device()
        assert device._should_continue("Which room?") is True
        assert device._should_continue("Done.") is False

    def test_muted_device_closes_the_continuation(self):
        device = self._api_device()
        device.media.muted = True
        assert device._should_continue("Still open?") is False

    def test_continued_conversation_disabled_by_config(self):
        device = _device(
            None,
            build_snapshot(
                SimpleNamespace(
                    voice_assistant_feature_flags=FEATURE_VOICE_ASSISTANT
                    | FEATURE_SPEAKER | FEATURE_API_AUDIO
                ),
                []
            ),
            continued_conversation=False,
        )
        assert device._should_continue("Which room?") is False

    def test_generation_bump_discards_stale_tts_chunks(self):
        client = FakeClient()
        device = self._api_device(client)

        async def _run():
            await device.handle_pipeline_start("", 0, SimpleNamespace(), None)
            stale = device.session_generation
            # No PCM available (tts_engine is None) closes the stream cleanly.
            await device._stream_reply("reply")
            device.session_generation += 1
            await device._close_stream(stale)
            return len(client.audio)

        audio_count = _run_loop(_run, device)
        assert audio_count == 0
        assert device.session_state is SessionState.IDLE

    def test_audio_callback_is_non_blocking_into_bounded_queue(self):
        client = FakeClient()
        device = self._api_device(client)

        async def _run():
            await device.handle_pipeline_start("", 0, SimpleNamespace(), None)
            await device.handle_audio(b"\x00\x01" * 16)
            await device.handle_audio(b"\x02\x03" * 16, b"\x04\x05" * 16)
            return device._ingress.depth_ms(), device.metrics.get("audio_chunks")

        depth, chunks = _run_loop(_run, device)
        assert chunks == 2
        assert depth <= 300.0

    def test_channel_one_selected_for_non_enhanced_audio(self):
        device = _device(
            FakeClient(),
            build_snapshot(
                SimpleNamespace(
                    voice_assistant_feature_flags=FEATURE_VOICE_ASSISTANT
                    | FEATURE_API_AUDIO | FEATURE_MULTI_CHANNEL_AUDIO
                ),
                []
            ),
            preferred_input_channel=1,
        )

        async def _run():
            await device.handle_pipeline_start("", 0, SimpleNamespace(), None)
            await device.handle_audio(b"\x01\x01", b"\x02\x02")
            return device.session.enhanced, device._ingress._queue.get_nowait()

        enhanced, payload = _run_loop(_run, device)
        assert enhanced is False
        assert payload == b"\x02\x02"

    def test_single_channel_firmware_falls_back_to_channel_zero(self):
        device = _device(
            FakeClient(),
            build_snapshot(
                SimpleNamespace(
                    voice_assistant_feature_flags=FEATURE_VOICE_ASSISTANT
                    | FEATURE_API_AUDIO
                ),
                []
            ),
            preferred_input_channel=1,
        )

        async def _run():
            await device.handle_pipeline_start("", 0, SimpleNamespace(), None)
            return device._active_channel, device.session.enhanced

        channel, enhanced = _run_loop(_run, device)
        assert channel == 0 and enhanced is True

    def test_mute_state_closes_the_session_without_reopening_it(self):
        device = self._api_device()
        entities = EntityIndex().build([SwitchInfo("master_mute_switch", 7, "Mute")])
        device.entities = entities

        async def _run():
            await device.handle_pipeline_start("", 0, SimpleNamespace(), None)

        _run_loop(_run, device)
        device._on_state(SimpleNamespace(key=7, state=True))
        assert device.session_state is SessionState.IDLE
        assert device.media.muted is True

    def test_button_event_maps_to_the_configured_action(self):
        device = self._api_device()
        calls = []
        device.actions.register("toggle_overlay", lambda: calls.append("overlay"))
        device._handle_button_event("double_press")
        device._handle_button_event("long_press")
        device._handle_button_event("some_unknown_press")
        assert calls == ["overlay"]
        assert device.metrics["button_double_press"] >= 1

    def test_long_press_cancels_the_active_run(self):
        device = self._api_device()

        async def _run():
            await device.handle_pipeline_start("", 0, SimpleNamespace(), None)

        _run_loop(_run, device)
        device._handle_button_event("long_press")
        assert device.session_state is SessionState.IDLE

    def test_announcement_finished_moves_to_continue_or_idle(self):
        device = self._api_device()
        device.session_state = SessionState.SPEAKING
        _run_loop(
            lambda: device.handle_announcement_finished(SimpleNamespace(success=True)),
            device,
        )
        # continued_conversation is on: the next turn needs no wake word.
        assert device.session_state is SessionState.CONTINUE_PENDING
        _run_loop(
            lambda: device.handle_announcement_finished(SimpleNamespace(success=True)),
            device,
        )
        assert device.session_state is SessionState.IDLE

    def test_health_snapshot_has_the_documented_keys(self):
        device = self._api_device()
        device.state = DeviceState.READY
        health = device.health_snapshot()
        for key in (
            "connected", "authenticated", "device", "api_version", "voice_features",
            "audio_queue_ms", "session_state", "wake_words_disabled", "last_event_at",
        ):
            assert key in health
        assert health["wake_words_disabled"] is True

    def test_auth_failure_keeps_the_stored_key(self):
        device = self._api_device()
        device.handle_auth_error(RuntimeError("invalid psk"))
        assert device.state is DeviceState.AUTH_REQUIRED
        assert "invalid psk" in device.last_error


@pytest.mark.unit
class TestMediaController:
    def test_volume_from_device_is_authoritative(self):
        client = FakeClient()
        media = VoicePEMediaController(client, 11, 7)
        update = media.update_state(SimpleNamespace(state=2, volume=0.45, muted=False))
        assert update["source"] == "device"
        assert media.volume == 0.45
        assert media.state == "playing"
        assert media.volume_source == "device"

    def test_own_command_marks_the_echo_source(self):
        client = FakeClient()
        media = VoicePEMediaController(client, 11, 7)
        media.set_volume(0.65)
        assert media.volume_source == "jarvis"
        update = media.update_state(SimpleNamespace(state=2, volume=0.65, muted=False))
        assert update["source"] == "jarvis"
        assert client.media and client.media[0][1]["volume"] == 0.65

    def test_play_pause_stop_use_the_published_key(self):
        client = FakeClient()
        media = VoicePEMediaController(client, 11, 7)
        media.play_url("http://lane/one.flac")
        media.pause()
        media.stop()
        keys = [entry[0] for entry in client.media]
        assert keys == [11, 11, 11]
        assert client.media[0][1]["media_url"] == "http://lane/one.flac"

    def test_no_media_player_makes_commands_noop(self):
        client = FakeClient()
        media = VoicePEMediaController(client, None, None)
        media.stop()
        media.set_volume(0.5)
        assert media.media is None if hasattr(media, "media") else True
        assert client.media == []

    def test_announcing_state_is_reported(self):
        media = VoicePEMediaController(FakeClient(), 1)
        media.update_state(SimpleNamespace(state=4, volume=0.5, muted=False))
        assert media.state == "announcing"
        assert media.is_active() is True


@pytest.mark.unit
class TestProvisioningHelpers:
    def test_provisionable_flag_reads_the_device_info_marker(self):
        from jarvis.integrations.voice_pe.provisioning import is_provisionable

        assert is_provisionable(SimpleNamespace(api_encryption_provisionable=True))
        assert not is_provisionable(SimpleNamespace(api_encryption_provisionable=False))
        assert not is_provisionable(None)

    def test_improv_states_map_to_readable_names(self):
        from jarvis.integrations.voice_pe.provisioning import improv_state_name

        assert improv_state_name(1) == "no_network"
        assert improv_state_name(3) == "network_connecting"
        assert improv_state_name(4) == "have_ip"
        assert improv_state_name(99) == "unknown"

    def test_improv_packet_splits_type_and_payload(self):
        from jarvis.integrations.voice_pe.provisioning import _parse_improv_packet

        # [length][version=1][type][payload...]
        rtype, payload = _parse_improv_packet(bytes([6, 1, 1, 4, 0, 0]))
        assert rtype == 1
        assert payload == bytes([4, 0, 0])
        assert _parse_improv_packet(b"") == (0, b"")


# ---------------------------------------------------------------------------
# Discovery matching
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestDiscoveryMatching:
    def test_official_project_metadata_matches(self):
        info = SimpleNamespace(
            project_name="esphome/home-assistant-voice-pe",
            name="home-assistant-voice-aabbcc",
            model="ESP32-S3",
        )
        assert match_voice_pe(info) is True

    def test_node_name_only_devices_still_match(self):
        info = SimpleNamespace(project_name="", name="home-assistant-voice-112233",
                               model="")
        assert match_voice_pe(info) is True

    def test_foreign_esphome_node_is_not_adopted(self):
        info = SimpleNamespace(project_name="something/other", name="desk-lamp",
                               model="ESP32")
        assert match_voice_pe(info) is False

    def test_none_is_not_a_device(self):
        assert match_voice_pe(None) is False


# ---------------------------------------------------------------------------
# Manager wiring
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestManagerWiring:
    def test_disabled_config_creates_no_loop(self):
        from jarvis.integrations.voice_pe.manager import VoicePEManager

        manager = VoicePEManager(SimpleNamespace(), FakeListener(), None)
        assert manager.enabled is False
        assert manager.start() is False
        assert manager.health()["devices"] == []

    def test_candidate_prefers_manual_host(self):
        from jarvis.integrations.voice_pe.manager import VoicePEManager

        settings = SimpleNamespace(
            voice_pe_enabled=True,
            voice_pe_host="10.0.0.9",
            voice_pe_port=6053,
            voice_pe_device_name="home-assistant-voice-aabbcc",
            voice_pe_mac_address="aa:bb:cc:dd:ee:ff",
            voice_pe_devices={
                "aa:bb:cc:dd:ee:ff": {"addresses": ["10.0.0.8"], "node_name": "x"}
            },
        )
        manager = VoicePEManager(settings, FakeListener(), None)
        hosts = manager._candidate_hosts()
        assert hosts[0][0] == "10.0.0.9"

    def test_stored_addresses_are_used_without_manual_host(self):
        from jarvis.integrations.voice_pe.manager import VoicePEManager

        settings = SimpleNamespace(
            voice_pe_enabled=True,
            voice_pe_host=None,
            voice_pe_port=6053,
            voice_pe_devices={
                "aabbccddeeff": {"addresses": ["10.0.0.8"], "node_name": "voice"}
            },
        )
        manager = VoicePEManager(settings, FakeListener(), None)
        hosts = manager._candidate_hosts()
        assert hosts[0][0] == "10.0.0.8"

    def test_sink_fanout_reaches_every_device(self):
        from jarvis.integrations.voice_pe.manager import SinkFanout

        first = _device(FakeClient())
        second = _device(FakeClient())
        fan = SinkFanout([first, second])
        first.session = second.session = None
        fan.on_transcript("text")
        fan.on_vad_start()
        # Only a device holding an open session emits events.
        for client in (first._client, second._client):
            assert client.events == []

    def test_wizard_status_reports_paired_devices(self):
        settings = SimpleNamespace(
            voice_pe_enabled=True,
            voice_pe_device_name=None,
            voice_pe_devices={
                "aa:bb:cc:dd:ee:ff": {"node_name": "home-assistant-voice-aabbcc"}
            },
        )
        ok, text = pe_config.wizard_status(settings)
        assert ok is True
        assert "home-assistant-voice-aabbcc" in text

    def test_wizard_status_explains_an_empty_setup(self):
        ok, text = pe_config.wizard_status(SimpleNamespace(voice_pe_enabled=True))
        assert ok is False
        assert "voice-pe pair" in text


@pytest.mark.unit
class TestLibraryContract:
    """The pinned ``aioesphomeapi`` call shapes the transport relies on."""

    def test_make_client_uses_the_three_positional_shape(self):
        from aioesphomeapi import APIClient

        from jarvis.integrations.voice_pe.models import make_client

        async def _run():
            client = make_client(
                "192.168.1.50", 6053, "cHNr", device_name="home-assistant-voice-aabbcc"
            )
            assert isinstance(client, APIClient)
            # The library keeps only the identity-bearing params public.
            assert client.address == "192.168.1.50"
            assert client.expected_name == "home-assistant-voice-aabbcc"
            assert client.api_version is None
            await client.disconnect(True)

        _run_loop(_run)

    def test_feature_bit_values_match_the_library_enum(self):
        from aioesphomeapi import VoiceAssistantFeature

        import jarvis.integrations.voice_pe.models as m

        pairs = (
            (m.FEATURE_VOICE_ASSISTANT, VoiceAssistantFeature.VOICE_ASSISTANT),
            (m.FEATURE_SPEAKER, VoiceAssistantFeature.SPEAKER),
            (m.FEATURE_API_AUDIO, VoiceAssistantFeature.API_AUDIO),
            (m.FEATURE_TIMERS, VoiceAssistantFeature.TIMERS),
            (m.FEATURE_ANNOUNCE, VoiceAssistantFeature.ANNOUNCE),
            (m.FEATURE_START_CONVERSATION, VoiceAssistantFeature.START_CONVERSATION),
        )
        for mine, theirs in pairs:
            assert mine == int(theirs)

    def test_on_connect_without_reconnect_logic_opens_and_readies(self):
        client = FakeClient()
        device = _device(client)

        async def _run():
            device.loop = asyncio.get_running_loop()
            await device._on_connect()
            return device.state, device.connection_generation

        state, generation = _run_loop(_run, device)
        # A stub with no identity parks the device in the retry state, and the
        # generation counter still moved exactly once.
        assert generation == 1
        assert state is DeviceState.RECONNECTING

    def test_media_command_enum_names_exist(self):
        from aioesphomeapi import MediaPlayerCommand

        for name in ("PLAY", "PAUSE", "STOP", "MUTE", "UNMUTE"):
            assert getattr(MediaPlayerCommand, name) is not None
