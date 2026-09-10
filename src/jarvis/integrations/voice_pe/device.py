"""One long-lived Voice PE connection: subscriptions and session state machine.

Exactly one Voice Assistant subscriber exists per physical device, matching
the single-client limit of the ESPHome Native API. Each session carries a
monotonic ``session_generation``; audio, STT output, agent output and TTS
chunks from an older generation are dropped.

Stock turn taking is preserved: the microphone stops after STT and the reply
is streamed afterwards (``STREAMING_MICROPHONE`` then ``STREAMING_RESPONSE``),
so the LED phases follow the pipeline events rather than manual light calls.
"""

from __future__ import annotations

import asyncio
import socket
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from . import config as pe_config
from . import events as pe_events
from .capabilities import CapabilitySnapshot, build_snapshot
from .entities import EntityIndex, describe as describe_entities
from .led import STOCK_LED_BRIGHTNESS, sync_defaults as sync_led_defaults
from .media import VoicePEMediaController
from .models import (
    ANNOUNCEMENT_TIMEOUT_S,
    COMMAND_FLAG_USE_WAKE_WORD,
    LED_PHASE_JARVIS_STATE,
    LED_PHASES,
    VA_CONFIG_TIMEOUT_S,
    DeviceState,
    SessionState,
    VoiceInputSession,
    VoicePEConfig,
    feature_list,
    make_client,
)
from .provisioning import is_provisionable, provision_noise_key
from .tts_stream import TtsHttpServer, lan_ip_for, synthesize_pcm
from .voice_transport import AudioIngress, UdpAudioServer

try:  # pragma: no cover - trivial import shim
    from jarvis.debug import debug_log
except ImportError:  # pragma: no cover
    def debug_log(message: str, category: str = "debug") -> None:  # type: ignore[misc]
        pass

#: Shared announcement timeout.
ANNOUNCE_TIMEOUT_S = ANNOUNCEMENT_TIMEOUT_S


def _kvp(pairs: dict) -> str:
    return " ".join(
        f"{key}={value}" for key, value in pairs.items() if value is not None
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _iter_payloads(pcm: bytes, chunk_bytes: int = 1024):
    for offset in range(0, len(pcm), chunk_bytes):
        yield pcm[offset: offset + chunk_bytes]


class VoicePEDevice:
    """Stock Voice PE satellite attached to the existing Jarvis pipeline."""

    def __init__(
        self,
        config: VoicePEConfig,
        *,
        listener: Any,
        tts_engine: Any,
        host: str,
        port: int,
        psk: Optional[str] = None,
        device_name: Optional[str] = None,
        expected_mac: Optional[str] = None,
        metrics: Optional[dict] = None,
    ) -> None:
        self.config = config
        self._listener = listener
        self._tts = tts_engine
        self._host = host
        self._port = port
        self._psk = psk
        self._device_name = device_name
        self._expected_mac = expected_mac
        self.metrics: dict = metrics if metrics is not None else {}

        self.state = DeviceState.DISABLED
        self.session_state = SessionState.IDLE
        self.connection_generation = 0
        self.session_generation = 0
        self.identity: dict[str, Any] = {}
        self.capabilities: CapabilitySnapshot = CapabilitySnapshot()
        self.entities = EntityIndex()
        self.media = VoicePEMediaController(None, None)
        self.session: Optional[VoiceInputSession] = None
        self.led_phase = "not_ready"
        self.wake_words_disabled = bool(config.disable_wake_words)
        self.last_event_at = ""
        self.last_error = ""

        self._client = None
        self._reconnect = None
        self._unsub_voice_assistant = None
        self._ingress: Optional[AudioIngress] = None
        self._pump_task: Optional[asyncio.Task] = None
        self._udp_server: Optional[UdpAudioServer] = None
        self._tts_task: Optional[asyncio.Task] = None
        self._http: Optional[TtsHttpServer] = None
        self._lan_ip = ""
        self._tts_media_id = ""
        self._conversation_id = ""
        self._conversation_started = 0.0
        self._active_channel = int(config.preferred_input_channel or 0)
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.actions = pe_events.ActionRunner()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Connect through ``ReconnectLogic`` (backoff, jitter, mDNS wake)."""
        if not self.config.enabled:
            self.state = DeviceState.DISABLED
            return

        from aioesphomeapi import ReconnectLogic

        self.loop = asyncio.get_running_loop()
        self._ingress = AudioIngress(self._listener, self.config, self.metrics)
        self._client = make_client(
            self._host,
            self._port,
            self._psk,
            device_name=self._device_name,
            mac=self._expected_mac,
        )
        self._reconnect = ReconnectLogic(
            client=self._client,
            on_connect=self._on_connect,
            on_disconnect=self._on_connection_stopped,
            name=self._device_name or self._host,
        )
        await self._reconnect.start()
        self._count("connections")

    async def stop(self) -> None:
        """Tear down subscriptions, tasks and the managed connection."""
        self._drop_voice_assistant_subscription()
        self._cancel_tts_task()
        self._cancel_pump_task()
        if self._ingress is not None:
            self._ingress.close()
        if self._udp_server is not None:
            try:
                self._udp_server.close()
            except Exception:
                pass
            self._udp_server = None
        if self._http is not None:
            try:
                await self._http.stop()
            except Exception:
                pass
            self._http = None
        if self._reconnect is not None:
            try:
                result = self._reconnect.stop()
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                pass
            self._reconnect = None
        if self._client is not None:
            try:
                await self._client.disconnect(True)
            except Exception:
                pass
            self._client = None
        self.session = None
        self.session_state = SessionState.IDLE
        self.led_phase = "not_ready"

    async def _on_connect(self, *_args) -> None:
        """Post-connect synchronisation in the stock contract order."""
        client = self._client
        if client is None:
            return

        self.connection_generation += 1
        if self.connection_generation > 1:
            self._count("reconnects")
        # Per-generation pump counter, restart-nulled with the generation.
        self.metrics["pump_tasks"] = 0
        self.state = DeviceState.CONNECTING

        # ``ReconnectLogic`` already ran the handshake including
        # ``finish_connection(login=True)``, so only a manual start has to open
        # the socket itself.
        if self._reconnect is None:
            try:
                await client.connect(on_stop=self._on_connection_stopped)
            except Exception as err:
                self.last_error = f"connect: {err}"
                self.state = DeviceState.ERROR
                return

        self.state = DeviceState.AUTHENTICATING
        try:
            info = await client.device_info()
        except Exception as err:
            self.last_error = f"device_info: {err}"
            self.state = DeviceState.ERROR
            return
        if info is None:
            # Unauthenticated or still booting: the next reconnect retries.
            self.state = DeviceState.RECONNECTING
            return

        api_version = getattr(client, "api_version", None)
        api_text = (
            f"{getattr(api_version, 'major', 0)}.{getattr(api_version, 'minor', 0)}"
            if api_version is not None
            else ""
        )

        # Noise provisioning, only inside the device provisioning window.
        if is_provisionable(info) and not self._psk:
            self.state = DeviceState.PROVISIONING
            ok, encoded = await provision_noise_key(client, info)
            if ok and encoded:
                pe_config.save_device_metadata(
                    str(getattr(info, "mac_address", "") or "") or self._host,
                    {"noise_psk": encoded},
                )
                self._psk = encoded
                await self._restart_managed_connection()
                return

        self.state = DeviceState.SYNCING_CAPABILITIES
        # Fail-fast sync: an entity list is the base of every later lookup, so
        # a swallowed error here would leave a ``READY`` device with no keys.
        try:
            entities, services = await client.list_entities_services()
        except Exception as err:
            await self._fail_connection(f"list_entities: {err}")
            raise
        if not entities:
            await self._fail_connection("list_entities: empty entity list")
            raise RuntimeError("voice_pe: empty entity list")

        capabilities = None
        try:
            capabilities = await client.device_capabilities_compat(info)
        except Exception:
            try:
                capabilities = await client.device_capabilities()
            except Exception:
                capabilities = None

        self.capabilities = build_snapshot(info, entities, services, capabilities)
        self.entities = EntityIndex().build(entities)
        self.media = VoicePEMediaController(
            client, self.entities.media_key(), self.entities.mute_key()
        )
        self.identity = {
            "mac_address": str(getattr(info, "mac_address", "") or ""),
            "node_name": str(getattr(info, "name", "") or ""),
            "friendly_name": str(getattr(info, "friendly_name", "") or ""),
            "project_name": str(getattr(info, "project_name", "") or ""),
            "project_version": str(getattr(info, "project_version", "") or ""),
            "model": str(getattr(info, "model", "") or ""),
            "manufacturer": str(getattr(info, "manufacturer", "") or ""),
            "esphome_version": str(getattr(info, "esphome_version", "") or ""),
            "api_version": api_text,
            "suggested_area": str(getattr(info, "suggested_area", "") or ""),
            "voice_features": feature_list(self.capabilities.feature_flags),
        }
        if not self._device_name:
            self._device_name = self.identity["node_name"] or None
        self._lan_ip = lan_ip_for(self._host, self._port)

        # One persistent subscription per connection generation: the previous
        # handle is released before the fresh one is installed.
        self._drop_voice_assistant_subscription()
        try:
            client.subscribe_states(self._on_state)
        except Exception as err:
            await self._fail_connection(f"subscribe_states: {err}")
            raise
        if not getattr(client, "is_connected", True):
            await self._fail_connection("subscribe_states: connection closed")
            raise RuntimeError("voice_pe: subscribe_states did not stick")

        try:
            self._unsub_voice_assistant = client.subscribe_voice_assistant(
                handle_start=self.handle_pipeline_start,
                handle_stop=self.handle_pipeline_stop,
                handle_audio=(
                    self.handle_audio if self.capabilities.api_audio else None
                ),
                handle_announcement_finished=self.handle_announcement_finished,
            )
        except Exception as err:
            await self._fail_connection(f"subscribe_voice_assistant: {err}")
            raise
        if not callable(self._unsub_voice_assistant):
            await self._fail_connection("subscribe_voice_assistant: no handle")
            raise RuntimeError("voice_pe: subscribe_voice_assistant did not stick")

        # One microphone pump per connection generation: it stays across the
        # runs of this connection and is only torn down with the generation.
        self._ensure_pump()

        if self.capabilities.voice_assistant:
            try:
                await client.get_voice_assistant_configuration(VA_CONFIG_TIMEOUT_S)
            except Exception:
                pass
            if self.config.disable_wake_words:
                # Reading the configuration is optional, writing it is what the
                # push-to-talk product mode depends on, so a failed write closes
                # the generation and the reconnect re-syncs it.
                try:
                    await client.set_voice_assistant_configuration([])
                    self.wake_words_disabled = True
                except Exception as err:
                    await self._fail_connection(f"set_configuration: {err}")
                    raise

        try:
            sync_led_defaults(client, self.entities, self.config)
        except Exception as err:
            self.last_error = f"led: {err}"

        self._persist_identity()
        self.state = DeviceState.READY
        self.led_phase = "idle"
        self._sync_face_state()
        self._mark_event("connected")
        debug_log(
            _kvp({
                "component": "voice_pe",
                "device_mac": self.identity.get("mac_address"),
                "device_name": self.identity.get("node_name"),
                "room": self.config.room,
                "connection_generation": self.connection_generation,
                "voice_features": ",".join(self.capabilities.names()) or "none",
                "event_type": "connected",
            }),
            "voice",
        )

    async def _restart_managed_connection(self) -> None:
        """Rebuild client + reconnect logic so a fresh PSK takes effect."""
        from aioesphomeapi import ReconnectLogic

        if self._reconnect is not None:
            try:
                result = self._reconnect.stop()
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                pass
        if self._client is not None:
            try:
                await self._client.disconnect(True)
            except Exception:
                pass
        self._drop_voice_assistant_subscription()
        self._client = make_client(
            self._host,
            self._port,
            self._psk,
            device_name=self._device_name,
            mac=self._expected_mac,
        )
        self._reconnect = ReconnectLogic(
            client=self._client,
            on_connect=self._on_connect,
            on_disconnect=self._on_connection_stopped,
            name=self._device_name or self._host,
        )
        await self._reconnect.start()

    async def _on_connection_stopped(self, *_args) -> None:
        """Generation-scoped teardown; ``ReconnectLogic`` reconnects itself.

        A coroutine because 46.x awaits the ``on_disconnect`` callback and wraps
        the ``on_stop`` one into a background task.
        """
        self._drop_voice_assistant_subscription()
        self._cancel_tts_task()
        if self._ingress is not None:
            self._ingress.reset()
        self._cancel_pump_task()
        self.session = None
        self.session_state = SessionState.IDLE
        self.led_phase = "not_ready"
        if self.state is not DeviceState.ERROR:
            self.state = DeviceState.RECONNECTING
        self._mark_event("disconnected")
        # The avatar follows the connection: ``not_ready`` is the asleep phase,
        # an error keeps the error phase.
        self._sync_face_state()

    def _drop_voice_assistant_subscription(self) -> None:
        """Release the single live Voice Assistant subscription handle."""
        unsub, self._unsub_voice_assistant = self._unsub_voice_assistant, None
        if unsub is None:
            return
        try:
            unsub()
        except Exception:
            pass

    async def _fail_connection(self, message: str) -> None:
        """Close this generation so ``ReconnectLogic`` re-syncs from scratch."""
        self.last_error = message
        self.state = DeviceState.ERROR
        self._drop_voice_assistant_subscription()
        self._cancel_tts_task()
        self._cancel_pump_task()
        if self._ingress is not None:
            self._ingress.reset()
        self.session = None
        self.session_state = SessionState.IDLE
        self.led_phase = "not_ready"
        if self._client is not None:
            try:
                # A forced disconnect runs the ``on_stop`` chain, which is what
                # schedules the next backed-off connect attempt.
                await self._client.disconnect(True)
            except Exception:
                pass
        self._mark_event("sync_failed")
        self._sync_face_state()
        debug_log(
            _kvp({
                "component": "voice_pe",
                "device_mac": self.identity.get("mac_address"),
                "device_name": self.identity.get("node_name") or self._host,
                "connection_generation": self.connection_generation,
                "error_code": message,
                "event_type": "sync_failed",
            }),
            "voice",
        )

    # ------------------------------------------------------------------
    # Voice Assistant callbacks
    # ------------------------------------------------------------------

    async def handle_pipeline_start(
        self,
        conversation_id: str,
        flags: int,
        audio_settings: Any,
        wake_word_phrase: Optional[str],
    ) -> Optional[int]:
        """Open one wake-free Jarvis session for a device-triggered run."""
        self.session_generation += 1
        self._bump_metric("sessions")

        if self._ingress is not None:
            self._ingress.reset()
        self._cancel_tts_task()

        self._active_channel = int(self.config.preferred_input_channel or 0)
        if not self.capabilities.multi_channel_audio:
            self._active_channel = 0

        self._rotate_conversation(conversation_id)
        self.session = VoiceInputSession(
            source_id=self.identity.get("node_name") or self._host,
            room=self.config.room,
            conversation_id=self._conversation_id,
            enhanced=self._active_channel == 0,
        )
        self.state = DeviceState.VOICE_ACTIVE
        # The wake-word flag is diagnostic only: with disable_wake_words the
        # pipeline always starts in the STT stage.
        self._metric_wake_flag(bool(flags & COMMAND_FLAG_USE_WAKE_WORD))

        self.session_state = SessionState.BUTTON_TRIGGERED
        self._mark_event("run_start")
        await self._event("RUN_START", {})
        self.session_state = SessionState.LISTENING
        await self._event("STT_START", {})

        # The generation's pump is usually already running from ``_on_connect``;
        # a restarted generation gets exactly one new one here.
        self._ensure_pump()

        debug_log(
            _kvp({
                "component": "voice_pe",
                "device_name": self.identity.get("node_name"),
                "connection_generation": self.connection_generation,
                "session_id": self.session_generation,
                "conversation_id": self._conversation_id,
                "voice_state": self.session_state.value,
                "audio_channel": self._active_channel,
                "event_type": "run_start",
            }),
            "voice",
        )

        if self.capabilities.uses_api_audio or self.capabilities.api_audio:
            return 0
        return await self._start_udp_server()

    async def handle_pipeline_stop(self, abort: bool) -> None:
        """Close or abort the current run without replaying old audio."""
        if self._ingress is not None:
            self._ingress.reset()
        self._cancel_tts_task()
        if abort:
            self.session_state = SessionState.IDLE
            self.session = None
            self.led_phase = "idle"
            self._sync_face_state()
            self._mark_event("aborted")
            self._ensure_listener_queue_idle()
            return
        # ``abort=False`` is the microphone end marker: the buffered frames are
        # the utterance, so they are kept and the VAD is closed by the silence
        # tail instead of by the reset marker.
        self._mark_event("microphone_end")
        pad = getattr(self._listener, "pad_until_endpoint", None)
        if callable(pad):
            try:
                pad()
                return
            except Exception:
                pass
        self._ensure_listener_queue_idle()

    async def handle_audio(self, data: bytes, data2: Optional[bytes] = None) -> None:
        """Non-blocking microphone ingress into the bounded queue."""
        started = time.monotonic()
        if self.session_state is SessionState.LISTENING:
            self.session_state = SessionState.RECORDING
        if self._ingress is not None:
            self._ingress.put(data, data2)
        self._record_latency("audio_callback_ms", started)

    async def handle_announcement_finished(self, finished: Any) -> None:
        """Announcement (and streamed TTS) completion reported by the device."""
        self._bump_metric("announcements_finished")
        success = bool(getattr(finished, "success", False))
        if self.session_state is SessionState.SPEAKING and success:
            if self.config.continued_conversation:
                self.session_state = SessionState.CONTINUE_PENDING
            else:
                self.session_state = SessionState.IDLE
        elif self.session_state is SessionState.CONTINUE_PENDING:
            self.session_state = SessionState.IDLE
        self.led_phase = "idle"
        self._mark_event("announcement_finished")
        self._sync_face_state()

    def holds_session(self) -> bool:
        """True while this device owns the one open pipeline run."""
        return (
            self.session is not None
            and self.session_state is not SessionState.IDLE
        )

    @property
    def device_id(self) -> str:
        """Stable id of this satellite for the audio lease: MAC, name, host."""
        return str(
            self.identity.get("mac_address")
            or self.identity.get("node_name")
            or self._host
        )

    async def wait_until_ready(self, timeout_s: float = 10.0) -> bool:
        """Poll until the connection reached ``READY``/``VOICE_ACTIVE``."""
        deadline = 30
        for _ in range(max(1, int(timeout_s * 20))):
            if self.state in (DeviceState.READY, DeviceState.VOICE_ACTIVE):
                return True
            if self.state is DeviceState.AUTH_REQUIRED:
                return False
            await asyncio.sleep(0.05)
        return self.state in (DeviceState.READY, DeviceState.VOICE_ACTIVE)

    def _face_state_name(self) -> str:
        """Avatar phase: the session first, the ring phase as the fallback.

        ``RUN_END`` closes the pipeline run while the satellite is still playing,
        so the media player state keeps ``speaking`` until the device reports the
        announcement finished or its media player goes idle.
        """
        if self.media is not None and self.media.muted:
            return "muted"
        if self.state is DeviceState.ERROR:
            return "error"
        if self.state in (
            DeviceState.DISABLED,
            DeviceState.RECONNECTING,
            DeviceState.AUTH_REQUIRED,
        ):
            return "asleep"
        if self.session_state in (
            SessionState.SPEAKING,
            SessionState.CONTINUE_PENDING,
        ):
            return "speaking"
        if self.session_state in (
            SessionState.BUTTON_TRIGGERED,
            SessionState.LISTENING,
            SessionState.RECORDING,
        ):
            return "listening"
        if self.session_state in (SessionState.TRANSCRIBING, SessionState.THINKING):
            return "thinking"
        if self.media is not None and self.media.is_active():
            return "speaking"
        return LED_PHASE_JARVIS_STATE.get(self.led_phase, "idle")

    def _sync_face_state(self) -> None:
        """One shared bridge: session, media and ring phase drive the avatar."""
        name = self._face_state_name()
        try:
            from desktop_app.face_widget import JarvisState, get_jarvis_state

            get_jarvis_state().set_state(JarvisState(name))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Sink callbacks, invoked from the voice listener thread
    # ------------------------------------------------------------------

    def on_vad_start(self) -> None:
        self._submit(self._event("STT_VAD_START", {}))

    def on_vad_end(self) -> None:
        if self.session_state is SessionState.RECORDING:
            self.session_state = SessionState.TRANSCRIBING
        self._submit(self._event("STT_VAD_END", {}))

    def on_transcript(self, text: str) -> None:
        self._submit(self._on_transcript_async(text or ""))

    def on_reply(self, reply: str) -> None:
        self._submit(self._on_reply_async(reply or ""))

    def on_error(self, code: str, message: str) -> None:
        self._submit(self._event("ERROR", {"code": code, "message": message}))

    # ------------------------------------------------------------------
    # Entity state handling
    # ------------------------------------------------------------------

    def _on_state(self, state: Any) -> None:
        """Dispatch media volume, mute switch and button event entities.

        Matching is by published attributes and by the entity index, so it
        survives library renames and stays version-independent.
        """
        key = int(getattr(state, "key", 0) or 0)

        # Event entities publish only ``event_type``.
        if hasattr(state, "event_type"):
            self._handle_button_event(str(getattr(state, "event_type", "") or ""))
            return

        # The media player state is the only one carrying volume + muted.
        if hasattr(state, "volume") and hasattr(state, "muted"):
            update = self.media.update_state(state)
            if update:
                self._mark_event("media_state")
                # ``playing``/``announcing`` keeps the avatar at ``speaking``,
                # the move to ``idle`` is what releases it.
                self._sync_face_state()
                debug_log(
                    _kvp({
                        "component": "voice_pe",
                        "device_name": self.identity.get("node_name"),
                        "event_type": update["source"],
                        "voice_state": update["state"],
                    }),
                    "voice",
                )
            return

        if not hasattr(state, "state"):
            return

        # Mute: the soft switch entity and the hardware binary sensor.
        if key and key == self.entities.mute_key():
            self.media.muted = bool(getattr(state, "state", False))
            if self.media.muted:
                self._handle_muted()
            return

        info = self.entities.info_for(key) if key else None
        if info is not None and str(getattr(info, "name", "")).lower().startswith("mute"):
            self.media.muted = bool(getattr(state, "state", False))
            if self.media.muted:
                self._handle_muted()


    def _handle_muted(self) -> None:
        """Hardware and soft mute win over every other source."""
        if self._ingress is not None:
            self._ingress.reset()
        self._cancel_tts_task()
        self.session_state = SessionState.IDLE
        self.session = None
        self.led_phase = "idle"
        self._ensure_listener_queue_idle()
        self._mark_event("muted")
        # Mute is its own avatar phase, not the plain idle one.
        self._sync_face_state()

    def _handle_button_event(self, event_value: str) -> None:
        action = pe_events.resolve_action(event_value, self.config.button_actions)
        self._bump_metric(f"button_{event_value}")
        result = self.actions.run(action)
        if action == "cancel_current_agent_run" and not result.startswith("error"):
            if self._ingress is not None:
                self._ingress.reset()
            self.session_state = SessionState.IDLE
        debug_log(
            _kvp({
                "component": "voice_pe",
                "device_name": self.identity.get("node_name"),
                "event_type": event_value or "unknown",
                "error_code": None if result.startswith("ok") else result,
            }),
            "voice",
        )
        self._mark_event(f"button:{action}")

    # ------------------------------------------------------------------
    # Async internals
    # ------------------------------------------------------------------

    async def _event(self, name: str, data: dict) -> None:
        if self._client is None:
            return
        self.led_phase = pe_events.EVENT_LED_PHASE.get(name, self.led_phase)
        self._mark_event(name)
        self._sync_face_state()
        try:
            pe_events.send_event(self._client, name, data)
        except Exception as err:
            self.last_error = f"{name}: {err}"

    async def _on_transcript_async(self, text: str) -> None:
        if self.session is None:
            return
        started = time.monotonic()
        await self._event("STT_END", {"text": text})
        self._bump_metric("stt_end")
        self.session_state = SessionState.THINKING
        await self._event("INTENT_START", {})
        self._record_latency("stt_end_ms", started)

    async def _on_reply_async(self, reply: str) -> None:
        # Only the device that owns the run speaks; a device without an open
        # session leaves the reply to the local microphone.
        if self._client is None or self.session is None:
            return
        self.session_state = SessionState.SPEAKING
        continue_conversation = self._should_continue(reply)
        await self._event(
            "INTENT_END",
            {
                "conversation_id": self._conversation_id,
                "continue_conversation": "1" if continue_conversation else "0",
                "speech": reply,
            },
        )
        await self._event("TTS_START", {"text": reply})
        self._cancel_tts_task()
        started = time.monotonic()
        if self.capabilities.uses_api_audio:
            # ``API_AUDIO && SPEAKER``: raw PCM frames over the Native API.
            await self._event("TTS_STREAM_START", {})
            self._tts_task = asyncio.ensure_future(self._stream_reply(reply))
        else:
            # Everything else (e.g. flags ``61``: API_AUDIO, TIMERS, ANNOUNCE,
            # START_CONVERSATION, no SPEAKER): a WAV the satellite fetches.
            self._tts_task = asyncio.ensure_future(self._deliver_tts_by_url(reply))
        await asyncio.sleep(0)
        self._record_latency("tts_first_chunk_ms", started)

    async def _stream_reply(self, reply: str) -> None:
        """Paced 512-sample PCM stream: 384 ms of stock ring buffer by design."""
        generation = self.session_generation
        try:
            pcm = synthesize_pcm(self._tts, reply) or b""
        except Exception as err:
            self.last_error = f"tts: {err}"
            await self._close_stream(generation, reply)
            return

        if not pcm:
            await self._close_stream(generation, reply)
            return

        self._tts_media_id = ""
        loop = asyncio.get_running_loop()
        seconds_in_chunk = 512 / 16000
        start_time = loop.time()
        audio_duration_sent = 0.0
        for payload in _iter_payloads(pcm):
            if self.session_generation != generation or self._client is None:
                return
            self._client.send_voice_assistant_audio(payload)
            audio_duration_sent += seconds_in_chunk
            wait_s = (audio_duration_sent - 0.384) - (loop.time() - start_time)
            if wait_s > 0:
                await asyncio.sleep(wait_s)

        if self.session_generation != generation or self._client is None:
            return
        await self._event("TTS_STREAM_END", {})
        await self._end_run(reply)

    async def _close_stream(self, generation: int, reply: str = "") -> None:
        if self.session_generation != generation or self._client is None:
            return
        await self._event("TTS_STREAM_END", {})
        await self._end_run(reply)

    async def _deliver_tts_by_url(self, reply: str) -> None:
        """Non-speaker egress: WAV served over LAN HTTP, referenced by ``TTS_END``."""
        generation = self.session_generation
        try:
            pcm = synthesize_pcm(self._tts, reply) or b""
        except Exception as err:
            self.last_error = f"tts: {err}"
            await self._end_run("")
            return

        url = ""
        if pcm:
            try:
                url = await self._serve_wav(pcm)
            except Exception as err:
                self.last_error = f"tts_http: {err}"
        if self.session_generation != generation or self._client is None:
            return
        if url:
            await self._event("TTS_END", {"url": url})
            self._bump_metric("tts_url_deliveries")
        await self._end_run(reply)

    async def _serve_wav(self, pcm: bytes) -> str:
        """Publish one PCM payload on the LAN HTTP server, return its URL."""
        if self._http is None:
            self._http = TtsHttpServer()
        if not self._http.port:
            self._lan_ip = self._lan_ip or lan_ip_for(self._host, self._port)
            await self._http.start()
        key = f"{self.connection_generation}-{self.session_generation}"
        path = self._http.put(key, pcm)
        self._tts_media_id = key
        return f"http://{self._lan_ip or self._host}:{self._http.port}{path}"

    async def tts_media_url(self, text: str) -> str:
        """Public synthesis helper: LAN URL of ``text``, ``""`` when unavailable."""
        try:
            pcm = synthesize_pcm(self._tts, text) or b""
        except Exception as err:
            self.last_error = f"tts: {err}"
            return ""
        if not pcm:
            return ""
        return await self._serve_wav(pcm)

    async def _end_run(self, reply: str) -> None:
        """Close the run; the firmware carries an open follow-up itself."""
        if self._client is None:
            return
        # ``RUN_END`` is always the last event of a run, so the ring leaves
        # ``replying`` in both cases. With ``continue_conversation`` set in
        # ``INTENT_END`` the stock firmware reopens the microphone itself and
        # calls the start callback again - no extra RPC is needed and sending
        # one would only restart the already playing reply.
        await self._event("RUN_END", {})
        self._bump_metric("run_end")
        if self._should_continue(reply):
            self.session_state = SessionState.CONTINUE_PENDING
            self._mark_event("continue_pending")
            self._sync_face_state()
            return
        self.session_state = SessionState.IDLE
        self.session = None
        self._mark_event("run_end")
        # The media player can still be ``playing`` here; the sync maps that to
        # ``speaking`` and the next media or announce event closes it.
        self._sync_face_state()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _rotate_conversation(self, device_conversation_id: str) -> None:
        now = time.monotonic()
        expired = (now - self._conversation_started) > self.config.conversation_timeout_s
        if device_conversation_id:
            self._conversation_id = device_conversation_id
            self._conversation_started = now
        elif not self._conversation_id or expired:
            self._conversation_id = uuid.uuid4().hex
            self._conversation_started = now

    def _should_continue(self, reply: str) -> bool:
        """An asked question keeps the dialog open; anything else closes it."""
        if not self.config.continued_conversation or not reply:
            return False
        if self.media.muted:
            return False
        text = reply.strip()
        return text.endswith("?")

    async def _start_udp_server(self) -> Optional[int]:
        """UDP microphone for firmware without the ``API_AUDIO`` flag."""
        if self._ingress is None:
            return None
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setblocking(False)
        sock.bind(("", 0))
        _transport, protocol = await asyncio.get_running_loop().create_datagram_endpoint(
            lambda: UdpAudioServer(self._ingress.queue()), sock=sock
        )
        self._udp_server = protocol  # type: ignore[assignment]
        return int(sock.getsockname()[1])

    def _ensure_listener_queue_idle(self) -> None:
        if self._listener is None:
            return
        try:
            self._listener._audio_q.put_nowait(None)
        except Exception:
            pass

    def _cancel_tts_task(self) -> None:
        if self._tts_task is not None:
            self._tts_task.cancel()
            self._tts_task = None

    def _ensure_pump(self) -> None:
        """Keep exactly one microphone pump per connection generation."""
        if self._ingress is None:
            return
        if self._pump_task is not None and not self._pump_task.done():
            return
        self._pump_task = asyncio.ensure_future(self._ingress.pump())
        self.metrics["pump_tasks"] = int(self.metrics.get("pump_tasks", 0)) + 1

    def _cancel_pump_task(self) -> None:
        if self._pump_task is not None:
            self._pump_task.cancel()
            self._pump_task = None

    def _submit(self, coro) -> None:
        loop = self.loop
        if loop is None:
            coro.close()
            return
        try:
            asyncio.run_coroutine_threadsafe(coro, loop)
        except Exception:
            coro.close()

    def _mark_event(self, name: str) -> None:
        self.last_event_at = _now_iso()
        self.metrics["last_event"] = name

    def _count(self, name: str) -> None:
        self.metrics[name] = int(self.metrics.get(name, 0)) + 1

    def _bump_metric(self, name: str) -> None:
        self._count(name)

    def _record_latency(self, name: str, started_mono: float) -> None:
        self.metrics[name] = round((time.monotonic() - started_mono) * 1000.0, 2)

    def _metric_wake_flag(self, flagged: bool) -> None:
        self.metrics["wake_word_flag"] = int(flagged)

    def _persist_identity(self) -> None:
        mac = self.identity.get("mac_address") or ""
        if not mac:
            return
        meta = {
            "mac_address": mac,
            "node_name": self.identity.get("node_name", ""),
            "friendly_name": self.identity.get("friendly_name", ""),
            "project_name": self.identity.get("project_name", ""),
            "project_version": self.identity.get("project_version", ""),
            "api_version": self.identity.get("api_version", ""),
            "voice_feature_flags": self.capabilities.feature_flags,
            "addresses": [self._host],
            "port": self._port,
            "last_connected": self.last_event_at,
        }
        if self._psk:
            meta["noise_psk"] = self._psk
        pe_config.save_device_metadata(mac, meta)

    def handle_auth_error(self, err: Exception) -> None:
        """Rejected PSK: keep the stored key and ask for an explicit import."""
        self.last_error = str(err)
        self.state = DeviceState.AUTH_REQUIRED

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def health_snapshot(self) -> dict:
        """Health view for the diagnostics panel and the CLI status command."""
        queue_ms = self._ingress.depth_ms() if self._ingress is not None else 0
        return {
            "connected": bool(
                self._client is not None and getattr(self._client, "is_connected", False)
            ),
            "authenticated": self.state in (DeviceState.READY, DeviceState.VOICE_ACTIVE),
            "device": self.identity.get("node_name") or self._host,
            "api_version": self.identity.get("api_version", ""),
            "voice_features": self.capabilities.names(),
            "audio_queue_ms": queue_ms,
            "session_state": self.session_state.value,
            "wake_words_disabled": self.wake_words_disabled,
            "last_event_at": self.last_event_at,
            "device_state": self.state.value,
            "led_phase_id": LED_PHASES.get(self.led_phase, 0),
            "connection_generation": self.connection_generation,
            "session_generation": self.session_generation,
            "error": self.last_error,
        }

    def ui_view(self) -> dict:
        """Everything the single ``Voice PE`` settings card renders."""
        return {
            "config": {
                "enabled": self.config.enabled,
                "room": self.config.room,
                "host": self._host,
                "port": self._port,
            },
            "connection": self.health_snapshot(),
            "firmware": {
                "esphome_version": self.identity.get("esphome_version", ""),
                "api_version": self.identity.get("api_version", ""),
                "project": self.identity.get("project_name", ""),
                "project_version": self.identity.get("project_version", ""),
            },
            "audio": {
                "capabilities": self.capabilities.names(),
                "api_audio": self.capabilities.api_audio,
                "speaker": self.capabilities.speaker,
                "multi_channel_audio": self.capabilities.multi_channel_audio,
                "input_channel": self._active_channel,
            },
            "wake_words": "disabled" if self.wake_words_disabled else "enabled",
            "led": {
                "rgb": list(self.config.led_rgb),
                "brightness": self.config.led_brightness,
                "stock_brightness": STOCK_LED_BRIGHTNESS,
                "effects": list(self.capabilities.led_effects),
            },
            "media": self.media.snapshot(),
            "buttons": dict(self.config.button_actions),
            "entities": describe_entities(self.entities),
            "metrics": dict(self.metrics),
        }
