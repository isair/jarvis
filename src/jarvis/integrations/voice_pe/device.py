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
from collections import deque as _deque
from datetime import datetime, timezone
from typing import Any, Optional

from . import config as pe_config
from . import events as pe_events
from .capabilities import CapabilitySnapshot, build_snapshot
from .entities import EntityIndex, describe as describe_entities
from .led import STOCK_LED_BRIGHTNESS, sync_defaults as sync_led_defaults
from .media import VoicePEMediaController
from .models import (
    AUDIO_SOURCE_VOICE_PE,
    ANNOUNCEMENT_TIMEOUT_S,
    COMMAND_FLAG_USE_WAKE_WORD,
    LED_PHASE_JARVIS_STATE,
    LED_PHASES,
    LOCAL_STREAM,
    VA_CONFIG_TIMEOUT_S,
    DeviceState,
    PendingPlayback,
    SessionState,
    StreamId,
    TurnContext,
    VoiceInputSession,
    VoicePEConfig,
    feature_list,
    is_current_turn,
    make_client,
)
from .provisioning import is_provisionable, provision_noise_key
from .tts_stream import (
    TtsHttpServer,
    lan_ip_for,
    split_sentences,
    synthesize_pcm_async,
)
from .voice_transport import AudioIngress, UdpAudioServer, pcm16_to_float32

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


# The library pins ``MAXIMUM_BACKOFF`` and drives the backoff ladder through
# ``ReconnectLogic._tries``; the two helpers below touch those private names.
# Version is pinned in ``requirements.txt`` and hard-checked here so a drift
# fails at start rather than silently falling back to library defaults.
_EXPECTED_AIOESPHOMEAPI = "46.3.0"


def _installed_aioesphomeapi_version() -> str:
    try:
        from importlib.metadata import version as _pkg_version

        return str(_pkg_version("aioesphomeapi"))
    except Exception:
        try:
            import aioesphomeapi as _ap

            return str(getattr(_ap, "__version__", "?"))
        except Exception:
            return "?"


def _reconnect_logic_compat_check(logic: Any) -> None:
    """Fail-fast when the pinned private attributes are not as-expected.

    Raises ``RuntimeError`` with the installed version on drift so the
    caller can surface it in ``health_snapshot`` / the CLI last-error line.
    """
    installed = _installed_aioesphomeapi_version()
    try:
        import aioesphomeapi.reconnect_logic as _rl
    except Exception as err:  # pragma: no cover - defensive
        raise RuntimeError(
            f"voice_pe: aioesphomeapi.reconnect_logic not importable "
            f"({_EXPECTED_AIOESPHOMEAPI} expected, got {installed!r}): {err}"
        ) from err
    if logic is None:
        return
    if not hasattr(logic, "_tries") or not isinstance(
        getattr(logic, "_tries", None), int
    ):
        raise RuntimeError(
            f"voice_pe: ReconnectLogic._tries missing or non-int "
            f"(expected aioesphomeapi {_EXPECTED_AIOESPHOMEAPI}, "
            f"installed {installed!r})"
        )
    max_backoff = getattr(_rl, "MAXIMUM_BACKOFF", None)
    if max_backoff is None or not isinstance(max_backoff, float):
        raise RuntimeError(
            f"voice_pe: aioesphomeapi.reconnect_logic.MAXIMUM_BACKOFF "
            f"missing or non-float (expected {_EXPECTED_AIOESPHOMEAPI}, "
            f"installed {installed!r})"
        )


def _wire_reconnect_backoff(logic: Any, cfg: Any) -> None:
    """Bind ``voice_pe_reconnect_max_s`` onto the library backoff cap.

    ``aioesphomeapi`` hard-codes ``MAXIMUM_BACKOFF = 60.0`` and applies
    ``wait = round(min(1.8 ** tries, MAXIMUM_BACKOFF))``. Only the ceiling
    is a real module constant, so write the configured ceiling there once
    per construction; the module is shared across every voice_pe device
    but their configs share a single flat ``voice_pe_*`` namespace.
    """
    _reconnect_logic_compat_check(logic)
    try:
        max_s = float(getattr(cfg, "reconnect_max_s", 0) or 0.0)
    except (TypeError, ValueError):
        max_s = 0.0
    if max_s <= 0.0:
        return
    try:
        import aioesphomeapi.reconnect_logic as _rl

        _rl.MAXIMUM_BACKOFF = max_s
    except Exception:
        pass


def _seed_reconnect_tries(logic: Any, cfg: Any) -> None:
    """Nudge ``_tries`` so the first retry respects ``reconnect_min_s``.

    ``_handle_connection_failure`` bumps ``_tries`` and the rescheduler uses
    ``round(1.8 ** _tries)``, so setting ``_tries = n`` makes the next
    failure wait ``round(1.8 ** (n + 1))``. Choose the smallest ``n`` whose
    bump meets ``min_s``; ``_on_connect`` calls this again after every
    success to counter the library's ``_tries = 0`` reset.
    """
    if logic is None:
        return
    _reconnect_logic_compat_check(logic)
    try:
        min_s = float(getattr(cfg, "reconnect_min_s", 0) or 0.0)
    except (TypeError, ValueError):
        return
    if min_s <= 0.0:
        return
    tries = 0
    while tries < 10 and round(1.8 ** (tries + 1)) < min_s:
        tries += 1
    try:
        logic._tries = tries  # noqa: SLF001 - documented in the library
    except Exception:
        pass


class DesktopMicLease:
    """Per-satellite desktop microphone lease for the Windows virtual mic.

    The stock Voice PE session machine (``SessionState``) is not changed; this
    lease only tracks the *consumer* side of the cleaned stream. The broker
    reports the physical capture-client count of the ``Toustovač Clean
    Microphone`` endpoint; a 0 -> non-zero transition arms the lease and the
    device keeps one continuous Assist capture session, while the zero-count
    idle window releases it after ``idle_release_s``. All transitions are
    idempotent, and ``generation`` increments on every arm so callers can
    recognize stale callbacks, mirroring the connection/session generations.
    """

    #: Full contract set of the lease (the UI shows exactly these values).
    STATES = ("disabled", "arming", "streaming", "reconnecting", "stopping")

    #: Finite exponential backoff of reopen attempts; the sequence is
    #: exhausted after ``len(BACKOFF_S)`` tries and the next failure then
    #: closes the lease with a named reason instead of looping forever.
    BACKOFF_S: tuple = (0.05, 0.1, 0.2, 0.4, 0.8)

    def __init__(self, idle_release_s: float = 5.0) -> None:
        self._state = "disabled"
        self._generation = 0
        self._idle_release_s = float(idle_release_s or 5.0)
        self._last_active_mono = 0.0
        self._retry_index = 0
        self._muted = False

    # -- state helpers ------------------------------------------------------

    @property
    def state(self) -> str:
        return self._state

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def idle_release_s(self) -> float:
        return self._idle_release_s

    @property
    def muted(self) -> bool:
        return self._muted

    def _stamp(self) -> None:
        self._last_active_mono = time.monotonic()

    # -- contract methods ---------------------------------------------------

    def on_client_count(self, count: Any) -> str:
        """One broker poll: arm on 0 -> non-zero, expire after the idle window."""
        try:
            n = int(count)
        except (TypeError, ValueError):
            n = 0
        if n > 0:
            self._stamp()
            if self._state == "disabled":
                self._generation += 1
                self._state = "arming"
                self._retry_index = 0
            elif self._state in ("arming", "streaming", "reconnecting"):
                self._state = "streaming"
            return self._state
        if self._state == "disabled":
            return self._state
        if self._last_active_mono and (
            time.monotonic() - self._last_active_mono >= self._idle_release_s
        ):
            self.release()
        return self._state

    def mark_streaming(self) -> None:
        """The first cleaned frame of the armed generation reached the bus."""
        if self._state in ("arming", "reconnecting", "streaming"):
            self._state = "streaming"
            self._stamp()

    def on_failure(self, reason: str) -> None:
        """Finite exponential backoff for reopen attempts."""
        if self._state == "disabled":
            return
        self._retry_index += 1
        if self._retry_index > len(self.BACKOFF_S):
            self._state = "disabled"
            return
        self._state = "reconnecting"
        self._stamp()

    def next_backoff_s(self) -> float:
        index = min(self._retry_index, len(self.BACKOFF_S) - 1)
        return float(self.BACKOFF_S[index])

    def set_muted(self, muted: bool) -> None:
        """Hardware/soft mute is a device-level mute of the virtual mic."""
        self._muted = bool(muted)

    def release(self) -> None:
        """Idempotent final release: ``stopping`` then ``disabled``."""
        if self._state == "disabled":
            return
        self._state = "stopping"
        self._state = "disabled"

    def snapshot(self) -> dict:
        return {
            "state": self._state,
            "generation": int(self._generation),
            "idle_release_s": float(self._idle_release_s),
            "muted": bool(self._muted),
            "retry_index": int(self._retry_index),
        }


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
        #: Generation whose playback is still running after ``RUN_END``.
        self._playback_latch: int = 0
        #: Generation whose active media state the device already reported.
        self._playback_seen: int = 0
        #: Delivered replies awaiting the device's own end-of-playback report,
        #: in first-in-first-out order like the finished callbacks arrive.
        self._pending_playback: "OrderedDeque" = _deque()
        #: Sent Voice Assistant events as ``(session_generation, name)``.
        self.event_ledger: list = []
        #: Per-``TurnContext`` terminal ledger of one-run close of
        #: ``ERROR``+``RUN_END``. A second ``pipeline_stop`` or a duplicate
        #: ``on_error`` is a no-op after this tuple is in here, so a run
        #: never gets two lifecycle terminals.
        self._closed_runs: set = set()
        #: Last light-entity push of the public ``led_ring``, for the ring check.
        self.last_light_push: Optional[dict] = None
        #: Last ``AnnounceFinished`` report, of the generation it closed.
        self.last_announce_success: Optional[bool] = None
        self.last_finished_generation: int = 0
        self._conversation_id = ""
        self._conversation_started = 0.0
        #: Desktop microphone lease of this satellite (virtual mic side).
        self.lease = DesktopMicLease()
        self._active_channel = int(config.preferred_input_channel or 0)
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.actions = pe_events.ActionRunner()
        #: Ordered listeners for `handle_pipeline_start`: `add_...` returns a
        #: per-call unsubscribe so a harness drops only its own listener; the
        #: production wiring and the reconnect logic are never clobbered.
        self._pipeline_start_listeners: list = []
        #: `(stage, monotonic_ns)` of the harness, naming the layer a press
        #: died on: listener_registered / ARNED_printed / physical_press_event /
        #: handle_pipeline_start_enter / queue_put_scheduled /
        #: queue_item_received / CAPTURING_started.
        self.pipeline_timeline: list = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def add_pipeline_start_listener(self, callback):
        """Register `callback(session_generation)`, return the unsubscribe.

        The unsubscribe is a plain callable: calling it removes this exact
        listener. Production callbacks registered earlier stay, reconnect stays,
        and a raising listener does not stop the rest.
        """
        self._pipeline_start_listeners.append(callback)

        def _unsubscribe():
            try:
                self._pipeline_start_listeners.remove(callback)
            except ValueError:
                pass

        return _unsubscribe

    def _timeline_stamp(self, stage: str) -> None:
        self.pipeline_timeline.append((str(stage), int(time.monotonic_ns())))

    def _dispatch_pipeline_start(self, generation: int) -> None:
        for cb in tuple(self._pipeline_start_listeners):
            try:
                cb(int(generation))
            except Exception as err:  # noqa: BLE001 - per-listener isolation
                debug_log(f"pipeline_start listener failed: {err}", "voice")

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
        _wire_reconnect_backoff(self._reconnect, self.config)
        _seed_reconnect_tries(self._reconnect, self.config)
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
        self.lease.release()

    async def _on_connect(self, *_args) -> None:
        """Post-connect synchronisation in the stock contract order."""
        client = self._client
        if client is None:
            return

        self.connection_generation += 1
        if self.connection_generation > 1:
            self._count("reconnects")
        # Re-seed so the *next* failure starts at the configured min_s,
        # because ``ReconnectLogic`` nulls ``_tries`` on every success.
        _seed_reconnect_tries(self._reconnect, self.config)
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
                except Exception as err:
                    await self._fail_connection(f"set_configuration: {err}")
                    raise
                # The flag is the device's own answer, not the write request:
                # read the configuration back and judge that.
                try:
                    readback = await client.get_voice_assistant_configuration(
                        VA_CONFIG_TIMEOUT_S
                    )
                    active = list(getattr(readback, "active_wake_words", None) or [])
                    self.wake_words_disabled = not active
                    self.metrics["wake_words_active"] = len(active)
                    if active:
                        await self._fail_connection(
                            f"set_configuration: {len(active)} wake words still active"
                        )
                        raise RuntimeError("voice_pe: wake words still active")
                except Exception as err:
                    await self._fail_connection(f"get_configuration: {err}")
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
                "psk_length": len(self._psk) if self._psk else 0,
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
        _wire_reconnect_backoff(self._reconnect, self.config)
        _seed_reconnect_tries(self._reconnect, self.config)
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
        self._release_local_state()
        # The connection ended, so any query of this device's stream is gone too.
        self._clear_listener_audio(None, cancel_pending=True)
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
        self._timeline_stamp("handle_pipeline_start_enter")
        self.session_generation += 1
        self._bump_metric("sessions")
        # A new run replaces the playback of the previous one.
        self._playback_latch = 0

        if self._ingress is not None:
            self._ingress.reset()
            self._ingress.set_stream(self._stream())
        self._cancel_tts_task()

        self._active_channel = int(self.config.preferred_input_channel or 0)
        # Host AEC modes consume the raw channel when the firmware carries it.
        if str(self.config.voice_pe_dsp_mode or "host_raw_aec") in (
            "host_raw_aec",
            "shadow_compare",
        ) and self.capabilities.multi_channel_audio:
            self._active_channel = 1
        if self._ingress is not None:
            self._ingress._multi = bool(self.capabilities.multi_channel_audio)
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
        if self.lease.state in ("arming", "streaming", "reconnecting"):
            self.lease.mark_streaming()
        # The wake-word flag is diagnostic only: with disable_wake_words the
        # pipeline always starts in the STT stage.
        self._metric_wake_flag(bool(flags & COMMAND_FLAG_USE_WAKE_WORD))

        self.session_state = SessionState.BUTTON_TRIGGERED
        self._mark_event("run_start")
        await self._event("RUN_START", {})
        self.session_state = SessionState.LISTENING
        # The STT stage of this run starts here; its single terminal is either
        # ``STT_END`` (text) or the ``ERROR`` of a skipped/filtered stage.
        self._bump_metric("stt_start")
        await self._event("STT_START", {})
        # A single click is resolved on the device and reaches Jarvis as this
        # pipeline start, so it is traced too: every press has a structured
        # ``accepted`` row, none disappears without a reason.
        self._trace_press("pipeline_start", True, "")
        # Fan out to every ``add_pipeline_start_listener`` callback; the
        # harness drops its own in ``finally``, one listener raising does not
        # block the rest, and the production callbacks stay in place.
        self._dispatch_pipeline_start(self.session_generation)
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
        if abort:
            # The firmware answers the microphone-end marker with a plain
            # pipeline close; when that end-of-stream was already stated (or
            # still waits in the FIFO), the close is the *normal* terminal of
            # the run. Drain first, then close without wiping the audio that
            # the VAD is still finalizing.
            eos = self._ingress is not None and self._ingress.eos_pending(
                getattr(self._ingress, "stream", None) or self._stream()
            )
            if eos:
                try:
                    self._ingress.drain()
                except Exception:
                    pass
                await self.abort_run(self.session_generation, "microphone_end")
                return
            await self.abort_run(self.session_generation, "aborted")
            return
        # ``abort=False`` is the microphone end marker. The EOS marker is queued
        # behind the PCM blocks of this run, so the pump delivers every block
        # first and only then states the stream end on the shared queue - the
        # buffered audio is what the VAD finalizes, nothing is dropped here.
        self._mark_event("microphone_end")
        if self._ingress is not None:
            stream = getattr(self._ingress, "stream", None) or self._stream()
            self._ingress.mark_end_of_stream(stream, AUDIO_SOURCE_VOICE_PE)

    async def abort_run(
        self, generation: int, reason: str, context: Optional[TurnContext] = None
    ) -> None:
        """One close for one run: tasks, both queues, events, lease, avatar.

        The full ``TurnContext`` is the key; a second close of the same run
        is a logged no-op so the ledger holds exactly one ``ERROR`` and one
        ``RUN_END`` per run. The generation-only path also keys off the
        current generation only, since a bare numeric close has no full
        context to compare.
        """
        if context is not None:
            if not is_current_turn(
                context,
                self.device_id,
                self.connection_generation,
                self.session_generation,
            ):
                return
        elif int(generation) != int(self.session_generation):
            return
        # Second callback for the very same run is ignored: only the first
        # writes the ERROR/RUN_END pair.
        close_key: tuple
        if context is not None:
            close_key = (
                str(getattr(context, "source", "") or ""),
                str(getattr(context, "device_id", "") or ""),
                int(getattr(context, "connection_generation", 0) or 0),
                int(getattr(context, "session_generation", 0) or 0),
            )
        else:
            close_key = ("", self.device_id, int(self.connection_generation), int(generation))
        if close_key in self._closed_runs:
            self._mark_event(f"ignored_duplicate:{close_key[-1]}")
            return
        # ``microphone_end`` is the normal terminal of a drained run: its
        # audio (plus the padded silence tail) is already on the listener
        # queue or still mid-finalisation there, so this close keeps both the
        # audio and the session that owns them. Everything else is a hard
        # close and releases first, idempotently.
        keep_audio = str(reason) == "microphone_end"
        if not keep_audio:
            self._release_local_state()
            self._cancel_tts_task()
            if self._ingress is not None:
                self._ingress.reset()
            self._clear_listener_audio()
        else:
            # Drain already delivered the whole utterance; only the per-run
            # input bookkeeping is closed here, the listener queue stays.
            self._cancel_tts_task()
            if self._ingress is not None:
                self._ingress.reset()
        try:
            if self.media is not None and self.media.is_active():
                self.media.stop()
        except Exception:
            pass
        if self._client is not None:
            await self._event(
                "ERROR", {"code": reason or "aborted", "message": reason or ""}
            )
            await self._event("RUN_END", {})
            self._bump_metric("run_end")
        self._closed_runs.add(close_key)
        # Cap the set to the last 32 runs so a long-lived device never grows
        # its dedup memory.
        if len(self._closed_runs) > 32:
            self._closed_runs = set(list(self._closed_runs)[-32:])
        self._mark_event("aborted" if not reason else f"abort:{reason}")
        self._sync_face_state()
        self._lease_after_run()

    def _clear_listener_audio(
        self, context: Optional[TurnContext] = None, cancel_pending: bool = True
    ) -> None:
        """Drain and reset the shared listener queue and this turn's query."""
        clear = getattr(self._listener, "_clear_audio_buffers", None)
        if callable(clear):
            try:
                clear()
            except Exception:
                pass
        else:
            self._ensure_listener_queue_idle()
        if not cancel_pending:
            return
        state_manager = getattr(self._listener, "state_manager", None)
        cancel = getattr(state_manager, "cancel_pending", None)
        if callable(cancel):
            try:
                cancel(context)
            except Exception:
                pass

    def _release_local_state(self) -> None:
        """Session, lease, pending deliveries and latch, released as one unit."""
        self.session = None
        self.session_state = SessionState.IDLE
        self._playback_latch = 0
        self._playback_seen = 0
        self._pending_playback.clear()
        self.led_phase = "idle"

    def _stream(self) -> StreamId:
        """Full identity of this device's microphone stream right now."""
        return StreamId(
            self.device_id,
            int(self.connection_generation),
            int(self.session_generation),
        )

    def turn_context(self) -> TurnContext:
        """The immutable identity of the open turn, for the listener to carry."""
        return TurnContext(
            source=AUDIO_SOURCE_VOICE_PE,
            device_id=self.device_id,
            connection_generation=int(self.connection_generation),
            session_generation=int(self.session_generation),
        )

    def _token_is_current(self, token: Any) -> bool:
        """Drop a milestone of an older run at the entry of every stage.

        Every field is compared; a missing identity is a stale one, so only the
        explicitly contextless local case is answered by this device itself.
        """
        if token is None:
            # Local-microphone turn: this device only answers for its own run.
            return self.session is not None
        return is_current_turn(
            token,
            self.device_id,
            self.connection_generation,
            self.session_generation,
            getattr(token, "source", None),
        )

    async def handle_audio(self, data: bytes, data2: Optional[bytes] = None) -> None:
        """Non-blocking microphone ingress into the bounded queue."""
        started = time.monotonic()
        if self.session_state is SessionState.LISTENING:
            self.session_state = SessionState.RECORDING
        if self._ingress is not None:
            self._ingress.put(data, data2, self._stream())
        self._record_latency("audio_callback_ms", started)

    async def handle_announcement_finished(self, finished: Any) -> None:
        """Announcement (and streamed TTS) completion reported by the device."""
        success = bool(getattr(finished, "success", False))
        if not self._pending_playback:
            # A finished callback with no pending entry belongs to a pre-announce
            # or to a run that is already closed: only the flag moves, the
            # generation of the last delivery stays as it was.
            self.last_announce_success = success
            self._mark_event("announcement_finished")
            return
        # Announcements complete in order on the device: the oldest delivery is
        # the one this callback closes.
        pending = self._pending_playback.popleft()
        self._bump_metric("announcements_finished")
        self._stamp("announcement_finished")
        self.last_announce_success = success
        # The generation of the delivered reply, not of this instant.
        self.last_finished_generation = int(pending.session_generation)
        if self.session_state is SessionState.SPEAKING and success:
            if self.config.continued_conversation:
                self.session_state = SessionState.CONTINUE_PENDING
            else:
                self.session_state = SessionState.IDLE
                self.session = None
        elif self.session_state is SessionState.CONTINUE_PENDING:
            self.session_state = SessionState.IDLE
        if self._playback_latch and self._playback_latch == int(pending.session_generation):
            self._playback_latch = 0
        self.led_phase = "idle"
        self._mark_event("announcement_finished")
        self._sync_face_state()

    def holds_session(self) -> bool:
        """True while this device owns the one open pipeline run."""
        return (
            self.session is not None
            and self.session_state is not SessionState.IDLE
        )

    # ------------------------------------------------------------------
    # Desktop microphone lease (Windows virtual microphone)
    # ------------------------------------------------------------------

    def notify_capture_clients(self, count: Any) -> str:
        """Feed one broker-reported capture-client count into this lease.

        The 0 -> non-zero transition arms the lease and opens the first
        continuous Assist capture session; zero clients expire it after the
        configured idle window. Idempotent: repeated values only stamp.
        """
        self.lease.on_client_count(count)
        if self.lease.state == "arming":
            self._submit(self._lease_cycle(int(self.lease.generation)))
        return self.lease.state

    async def _lease_cycle(self, lease_generation: int) -> None:
        """Open/keep one continuous Assist capture session for the lease."""
        lease = self.lease
        if (
            int(lease.generation) != int(lease_generation)
            or lease.state not in ("arming", "streaming", "reconnecting")
        ):
            return
        if self._client is None or not getattr(self._client, "is_connected", False):
            lease.on_failure("no_connection")
            if lease.state == "reconnecting":
                await asyncio.sleep(lease.next_backoff_s())
                await self._lease_cycle(lease_generation)
            return
        # A programmatic open of the *same* conversation keeps the stock
        # contract: the firmware then calls ``handle_pipeline_start`` again,
        # the session generation moves, and the ingress gets a fresh lane.
        if not self.holds_session():
            self._rotate_conversation("")
            try:
                await self._client.start_conversation(self._conversation_id)
                lease.mark_streaming()
            except Exception as err:  # noqa: BLE001 - finite backoff below
                self.last_error = f"lease: {err}"
                lease.on_failure(str(err))
                if lease.state == "reconnecting":
                    await asyncio.sleep(lease.next_backoff_s())
                    await self._lease_cycle(lease_generation)
            return
        lease.mark_streaming()

    def _lease_after_run(self) -> None:
        """Chain the next continuous-capture run after any run terminal."""
        if self.lease.state in ("arming", "streaming", "reconnecting"):
            self._submit(self._lease_cycle(int(self.lease.generation)))

    def selected_audio_channel(self, stream=None) -> Optional[int]:
        """Locked audio channel for one stream, or ``None`` before the lock."""
        if self._ingress is None:
            return None
        try:
            return self._ingress.selected_audio_channel(stream)
        except Exception:
            return None

    def packet_stats(self, stream=None, channel=None) -> list:
        """Per-packet ``PacketStats`` rows of this device's own channel log."""
        if self._ingress is None:
            return []
        try:
            return self._ingress.packet_stats(stream, channel)
        except Exception:
            return []

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
        # Playback latch: the run is closed but the satellite is still playing
        # it, so the avatar holds ``speaking`` until the device reports the end.
        if self._playback_latch and self._playback_latch == self.session_generation:
            return "speaking"
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

    def on_vad_start(self, token: Any = None) -> None:
        if not self._token_is_current(token):
            return
        self._submit(self._event("STT_VAD_START", {}))

    def on_vad_end(self, token: Any = None) -> None:
        if not self._token_is_current(token):
            return
        if self.session_state is SessionState.RECORDING:
            self.session_state = SessionState.TRANSCRIBING
        self._submit(self._event("STT_VAD_END", {}))

    def on_transcript(self, text: str, token: Any = None) -> None:
        if not self._token_is_current(token):
            return
        self._submit(self._on_transcript_async(text or "", token))

    def on_reply(self, reply: str, token: Any = None) -> None:
        if not self._token_is_current(token):
            return
        self._submit(self._on_reply_async(reply or "", token))

    def announce_reply(
        self,
        reply: str,
        token: Any = None,
        *,
        start_conversation: bool = True,
    ) -> None:
        """Deliver a late reply after its Voice Assistant run has expired."""
        if not reply or self._client is None:
            return
        self._submit(
            self._announce_reply_async(
                reply, token, start_conversation=start_conversation
            )
        )

    def on_error(self, code: str, message: str, token: Any = None) -> None:
        if not self._token_is_current(token):
            return
        # The context is carried into the coroutine, so the close it performs is
        # guarded by the identity of the turn that failed, not by the moment.
        self._submit(self._on_error_async(code, message, token))

    async def _on_error_async(
        self, code: str, message: str, token: Any = None
    ) -> None:
        """One close for a failed turn: STT_END, ERROR, RUN_END, released lease.

        STT_START is always paired with an STT_END, also on a skipped or
        filtered stage; the status is what tells the phases apart. The
        ordering keeps ``RUN_END`` as the single last event of a run.
        """
        if self.session is not None:
            self._bump_metric("stt_end")
            self._bump_metric(f"stt_end_{code}")
            await self._event("STT_END", {"text": "", "status": str(code or "error")})
        await self.abort_run(self.session_generation, str(code or "error"), token)

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
                # ``playing``/``announcing`` keeps the avatar at ``speaking``.
                # Only a push that follows such an active state releases the
                # playback latch, so a plain initial ``idle`` cannot do it.
                if self.media.is_active():
                    self._playback_seen = self._playback_latch or self._playback_seen
                elif self._playback_seen and self._playback_seen == self._playback_latch:
                    self._playback_latch = 0
                    self._playback_seen = 0
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

        # The public ``led_ring`` light pushes its own state: that is the device
        # side acknowledgement of the phase the ring is in.
        if key and self.entities.led_key() and key == self.entities.led_key():
            self.last_light_push = {
                "on": bool(getattr(state, "state", False)),
                "brightness": getattr(state, "brightness", None),
                "color_mode": getattr(state, "color_mode", None),
                "color": [
                    float(v)
                    for v in (getattr(state, "color", None) or ())[:3]
                ],
            }
            self._mark_event("led_state")
            return

        if not hasattr(state, "state"):
            return

        # Mute: the soft switch entity and the hardware binary sensor.
        if key and key == self.entities.mute_key():
            self.media.muted = bool(getattr(state, "state", False))
            self.lease.set_muted(self.media.muted)
            if self.media.muted:
                self._handle_muted(self.turn_context())
            return

        info = self.entities.info_for(key) if key else None
        if info is not None and str(getattr(info, "name", "")).lower().startswith("mute"):
            self.media.muted = bool(getattr(state, "state", False))
            self.lease.set_muted(self.media.muted)
            if self.media.muted:
                self._handle_muted(self.turn_context())


    def _handle_muted(self, context: Optional[TurnContext] = None) -> None:
        """Hardware and soft mute win over every other source."""
        # The identity is captured here, synchronously, so the close that runs
        # later is guarded by the turn that was open at the mute.
        self._submit(self._muted_close(context or self.turn_context()))

    async def _muted_close(self, context: Optional[TurnContext] = None) -> None:
        await self.abort_run(
            getattr(context, "session_generation", self.session_generation),
            "muted",
            context,
        )
        self._mark_event("muted")
        self._sync_face_state()

    def _handle_button_event(self, event_value: str) -> None:
        self._timeline_stamp("physical_press_event")
        action = pe_events.resolve_action(event_value, self.config.button_actions)
        self._bump_metric(f"button_{event_value}")
        result = self.actions.run(action)
        # Every press leaves a structured trace: accepted, or a named reason.
        # ``ok:`` and ``unhandled:`` both mean the press was consumed without an
        # error; only ``error:`` is a real rejection of the handler.
        accepted = not result.startswith("error")
        rejected_reason = ""
        if not accepted:
            rejected_reason = f"handler_error:{result.split(':', 2)[-1]}"
        elif result.startswith("unhandled"):
            rejected_reason = (
                "unmapped_event" if action == "ignore" else "no_handler_registered"
            )
        if self._client is None:
            rejected_reason = rejected_reason or "no_subscription"
        if (
            action == "cancel_current_agent_run"
            and accepted
        ):
            # One guarded close: the identity of the run is captured here, so a
            # later generation is left alone by this late callback.
            context = TurnContext(
                source=AUDIO_SOURCE_VOICE_PE,
                device_id=self.device_id,
                connection_generation=int(self.connection_generation),
                session_generation=int(self.session_generation),
            )
            self._submit(
                self.abort_run(context.session_generation, "button_cancel", context)
            )
            rejected_reason = rejected_reason or "accepted_with_close"
        trace = {
            "event": event_value or "unknown",
            "action": action,
            "result": result,
            "accepted": bool(accepted),
            "rejected_reason": rejected_reason,
        }
        self._trace_press(
            trace["event"],
            bool(accepted),
            rejected_reason,
            extra={"action": action, "result": result},
        )
        self.metrics["button_last"] = {**self.metrics.get("button_last", {}), **trace}
        debug_log(
            _kvp({
                "component": "voice_pe",
                "device_name": self.identity.get("node_name"),
                "event_type": event_value or "unknown",
                "action": action,
                "accepted": trace["accepted"],
                "rejected_reason": trace["rejected_reason"] or None,
                "connection_generation": int(self.connection_generation),
                "session_generation": int(self.session_generation),
                "error_code": None if accepted else result,
            }),
            "voice",
        )
        # Press trail in the dialog: what was pressed, which action ran, and
        # whether the press was accepted or why it was rejected.
        verdict = "accepted" if accepted else f"rejected ({rejected_reason or result})"
        print(
            f"  🔘 Voice PE button: {event_value or 'unknown'} → action "
            f"'{action}' {verdict}, gen c{self.connection_generation}"
            f"s{self.session_generation}",
            flush=True,
        )
        self._mark_event(f"button:{action}")
        self._sync_face_state()

    # ------------------------------------------------------------------
    # Async internals
    # ------------------------------------------------------------------

    async def _event(self, name: str, data: dict) -> None:
        if self._client is None:
            return
        self.led_phase = pe_events.EVENT_LED_PHASE.get(name, self.led_phase)
        # One line per Voice Assistant event in the desktop log dialog: the ring
        # phase the satellite is driven into, named the same way the LED ring
        # animates — waiting = cyan pulse, listening = fast green, thinking =
        # slow amber, replying = amber-to-white, idle = off.
        print(
            f"  💡 Voice PE {name} → ring "
            f"{self.led_phase} (phase #{LED_PHASES.get(self.led_phase, 0)}), "
            f"gen c{self.connection_generation}s{self.session_generation}",
            flush=True,
        )
        # Per-generation ledger of the events this device accepted, so a check
        # can name the generation an event belongs to instead of the moment.
        self.event_ledger.append((int(self.session_generation), name))
        del self.event_ledger[:-16]
        self._stamp(name)
        self._mark_event(name)
        self._sync_face_state()
        try:
            pe_events.send_event(self._client, name, data)
        except Exception as err:
            self.last_error = f"{name}: {err}"

    def _trace_press(
        self,
        event: str,
        accepted: bool,
        rejected_reason: str = "",
        extra: Optional[dict] = None,
    ):
        """Append one structured press record: accepted, or a named reason."""
        row = {
            "event": str(event),
            "accepted": bool(accepted),
            "rejected_reason": str(rejected_reason or ""),
            "device_id": self.device_id,
            "connection_generation": int(self.connection_generation),
            "session_generation": int(self.session_generation),
            "state": self.session_state.value,
            "at": time.time(),
        }
        if extra:
            row.update(extra)
        previous = self.metrics.get("button_last")
        self.metrics["button_last"] = row if previous is None else {
            **row,
            **{k: v for k, v in (extra or {}).items()},
        }
        self.metrics["button_trace"] = (
            [row] + list(self.metrics.get("button_trace") or [])
        )[:8]
        return row

    def _stamp(self, name: str) -> None:
        """Monotonic time of one lifecycle event, kept per generation.

        The timeline is what lets a latency be named instead of guessed: LLM
        time is ``INTENT_END - INTENT_START``, synthesis is
        ``TTS_END - TTS_START``, playback is ``announcement_finished - TTS_END``.
        """
        timeline = self.metrics.get("timeline")
        if not isinstance(timeline, list):
            timeline = []
            self.metrics["timeline"] = timeline
        timeline.append(
            {
                "generation": int(self.session_generation),
                "event": str(name),
                "at": round(time.monotonic(), 4),
            }
        )
        del timeline[:-24]

    def timeline_for(self, generation: int) -> list:
        """The stamped lifecycle events of one generation, in arrival order."""
        timeline = self.metrics.get("timeline") or []
        return [row for row in timeline if int(row.get("generation", -1)) == int(generation)]

    def latency_summary(self, generation: int) -> dict:
        """Derived latencies of one generation, from the stamped timeline."""
        moments = {
            str(row.get("event")): float(row.get("at") or 0.0)
            for row in self.timeline_for(generation)
        }
        summary = {"generation": int(generation)}
        for base, first, last in (
            ("stt_ms", "STT_START", "STT_END"),
            ("llm_ms", "INTENT_START", "INTENT_END"),
            ("tts_ms", "TTS_START", "TTS_END"),
            ("run_ms", "RUN_START", "RUN_END"),
            ("playback_ms", "TTS_END", "announcement_finished"),
        ):
            if first in moments and last in moments:
                summary[base] = round((moments[last] - moments[first]) * 1000.0, 2)
        return summary

    async def _on_transcript_async(self, text: str, token: Any = None) -> None:
        # Generation is checked at the STT entry, not only in the TTS transport:
        # a transcript of an older run must not open a new thinking phase.
        if not self._token_is_current(token) or self.session is None:
            return
        started = time.monotonic()
        # This turn's STT closes with a text: the only status the transcript
        # entry can carry is ``success``, one-to-one with the ``stt_start`` of
        # this run. A skipped or filtered stage answers through ``on_error``.
        await self._event("STT_END", {"text": text, "status": "success"})
        # The human-readable answer of the capture phase, so the dialog shows
        # what the satellite's mic produced between the two ring switches.
        preview = text if len(text) <= 60 else text[:60] + "…"
        print(f"  📝 Voice PE heard: \"{preview}\"", flush=True)
        self._bump_metric("stt_end")
        self._bump_metric("stt_end_success")
        self.session_state = SessionState.THINKING
        await self._event("INTENT_START", {})
        self._record_latency("stt_end_ms", started)

    async def _on_reply_async(self, reply: str, token: Any = None) -> None:
        # Only the device that owns the run speaks; a device without an open
        # session leaves the reply to the local microphone. The token of the
        # turn is what ties a late reply to the run it belongs to.
        if not self._token_is_current(token):
            return
        if self._client is None or self.session is None:
            return
        # Which stage produced this text: the LLM or the env-gated diagnostic
        # reply. Both travel with the same events, so the ledger alone cannot
        # tell them apart and the source is named explicitly.
        reply_source = "llm"
        for source in (
            getattr(self._listener, "metrics", None),
            getattr(self._listener, "state_manager", None),
        ):
            values = source if isinstance(source, dict) else None
            if values and values.get("last_reply_source"):
                reply_source = str(values["last_reply_source"])
                break
        self.metrics["reply_source"] = reply_source
        generation = self.session_generation
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
            print("  🔊 Voice PE output: Native API PCM", flush=True)
            await self._event("TTS_STREAM_START", {})
            self._tts_task = asyncio.ensure_future(self._stream_reply(reply, generation))
        else:
            # Everything else (e.g. flags ``61``: API_AUDIO, TIMERS, ANNOUNCE,
            # START_CONVERSATION, no SPEAKER): a WAV the satellite fetches.
            self._tts_task = asyncio.ensure_future(
                self._deliver_tts_by_url(reply, generation)
            )
            print("  🔊 Voice PE output: LAN WAV announcement", flush=True)
        await asyncio.sleep(0)
        self._record_latency("tts_first_chunk_ms", started)

    async def _announce_reply_async(
        self,
        reply: str,
        token: Any = None,
        *,
        start_conversation: bool = True,
    ) -> None:
        """Out-of-run TTS fallback tied to the same physical satellite.

        This path is intentionally separate from the generation ledger: the
        old run is already terminal, so reopening its TTS events would violate
        the one-terminal-per-run invariant. The announcement RPC owns its own
        completion response and may open the next conversation when configured.
        """
        if self._client is None or not reply:
            return
        # A newer button press wins; an old answer must not talk over it.
        if self.holds_session():
            debug_log("late Voice PE reply suppressed: newer run is active", "voice")
            return
        media_id = await self.tts_media_url(reply)
        if not media_id:
            self.last_error = "tts_announcement: empty media URL"
            return
        started = time.monotonic()
        try:
            print("  🔊 Voice PE output: late-reply announcement fallback", flush=True)
            success = await self.media.announce(
                media_id,
                text=reply,
                timeout=max(30.0, float(self.config.conversation_timeout_s)),
                start_conversation=(
                    start_conversation
                    and self.config.continued_conversation
                    and self.capabilities.start_conversation
                ),
            )
            self._bump_metric("late_reply_announcements")
            self.metrics["late_reply_announcement_success"] = bool(success)
            self._record_latency("late_reply_announcement_ms", started)
            debug_log(
                f"late Voice PE reply announcement success={bool(success)} "
                f"source_generation={getattr(token, 'session_generation', 0)}",
                "voice",
            )
        except Exception as err:
            self.last_error = f"tts_announcement: {type(err).__name__}"
            debug_log(
                f"late Voice PE reply announcement failed: {type(err).__name__}",
                "voice",
            )

    async def _stream_reply(self, reply: str, generation: Optional[int] = None) -> None:
        """Paced 512-sample PCM stream: 384 ms of stock ring buffer by design.

        Sentences are synthesized one at a time on the dedicated executor and
        streamed as each one is ready, so the first chunk is on the wire while
        the second sentence is still being synthesized.
        """
        generation = (
            self.session_generation if generation is None else int(generation)
        )
        loop = asyncio.get_running_loop()
        seconds_in_chunk = 512 / 16000
        start_time: Optional[float] = None
        audio_duration_sent = 0.0
        sentences = split_sentences(reply)
        if not sentences:
            await self._close_stream(generation, reply)
            return
        self._pending_playback.append(PendingPlayback(
            generation=int(self.connection_generation),
            session_generation=generation,
            media_id="",
            egress="pcm",
        ))

        for sentence in sentences:
            try:
                pcm = await synthesize_pcm_async(self._tts, sentence) or b""
            except Exception as err:
                self.last_error = f"tts: {err}"
                await self._close_stream(generation, reply)
                return
            if self.session_generation != generation or self._client is None:
                return
            for payload in _iter_payloads(pcm):
                if self.session_generation != generation or self._client is None:
                    return
                self._client.send_voice_assistant_audio(payload)
                # Model the exact satellite-bound payload as the lane far-end
                # (pre-output PCM). 3 ms link + 12 ms speaker model, both on
                # the same monotonic clock as the microphone packets.
                if self._ingress is not None:
                    try:
                        self._ingress.push_tts_reference(
                            self._stream(),
                            pcm16_to_float32(payload),
                            16000,
                            time.monotonic_ns() + 3_000_000 + 12_000_000,
                        )
                    except Exception:
                        pass
                if start_time is None:
                    start_time = loop.time()
                    print(
                        f"  🔊 Voice PE first PCM chunk on wire "
                        f"(ring holds 0.384 s; paced streaming active)",
                        flush=True,
                    )
                audio_duration_sent += seconds_in_chunk
                wait_s = (audio_duration_sent - 0.384) - (
                    loop.time() - (start_time or loop.time())
                )
                if wait_s > 0:
                    await asyncio.sleep(wait_s)

        if self.session_generation != generation or self._client is None:
            return
        await self._event("TTS_STREAM_END", {})
        print(
            f"  🔊 Voice PE PCM stream closed: {audio_duration_sent:.2f} s of "
            f"audio in {len(sentences)} sentence chunk(s) over the Native API",
            flush=True,
        )
        await self._end_run(reply, generation)

    async def _close_stream(
        self, generation: int, reply: str = "", *, stream: bool = True
    ) -> None:
        if self.session_generation != generation or self._client is None:
            return
        if stream:
            await self._event("TTS_STREAM_END", {})
        await self._end_run(reply, generation)

    async def _deliver_tts_by_url(
        self, reply: str, generation: Optional[int] = None
    ) -> None:
        """Non-speaker egress: WAV served over LAN HTTP, referenced by ``TTS_END``."""
        generation = (
            self.session_generation if generation is None else int(generation)
        )
        try:
            pcm = await synthesize_pcm_async(self._tts, reply) or b""
        except Exception as err:
            self.last_error = f"tts: {err}"
            await self._end_run("", generation, stream=False)
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
            wav_n = int(self.metrics.get(f"wav_bytes_{self._tts_media_id}", 0) or 0)
            print(
                f"  📦 Voice PE WAV ready: {wav_n} B published, satellite fetches "
                f"and plays: {url}",
                flush=True,
            )
            self._bump_metric("tts_url_deliveries")
            # The satellite fetches and plays the WAV itself; this is the entry
            # the next ``AnnounceFinished`` of the same generation closes.
            self._pending_playback.append(PendingPlayback(
                generation=int(self.connection_generation),
                session_generation=generation,
                media_id=self._tts_media_id,
                egress="url",
            ))
        await self._end_run(reply, generation, stream=False)

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
        # The payload itself is counted per generation, so a checkpoint can tell
        # "a URL was made" from "a non-empty WAV was published".
        self.metrics[f"wav_bytes_{key}"] = int(self._http.payload_bytes(key))
        return f"http://{self._lan_ip or self._host}:{self._http.port}{path}"

    def media_delivery(self) -> dict:
        """Delivery trail of the current media key, straight from the server."""
        key = str(self._tts_media_id or "")
        info = {"key": key, "stored_bytes": 0, "hits": 0, "status": None,
                "served_bytes": 0, "content_type": "", "port": 0}
        if self._http is not None and key:
            info.update(self._http.delivery_for(key))
        info["wav_bytes_metric"] = int(self.metrics.get(f"wav_bytes_{key}", 0) or 0)
        info["deliveries"] = int(self.metrics.get("tts_url_deliveries", 0) or 0)
        info["announcements_finished"] = int(
            self.metrics.get("announcements_finished", 0) or 0
        )
        info["last_announce_success"] = self.last_announce_success
        info["last_finished_generation"] = int(self.last_finished_generation)
        return info

    async def tts_media_url(self, text: str) -> str:
        """Public synthesis helper: LAN URL of ``text``, ``""`` when unavailable."""
        try:
            pcm = await synthesize_pcm_async(self._tts, text) or b""
        except Exception as err:
            self.last_error = f"tts: {err}"
            return ""
        if not pcm:
            return ""
        return await self._serve_wav(pcm)

    async def _end_run(
        self, reply: str, generation: Optional[int] = None, *, stream: bool = True
    ) -> None:
        """Close the run; the firmware carries an open follow-up itself."""
        if self._client is None:
            return
        generation = (
            self.session_generation if generation is None else int(generation)
        )
        # ``RUN_END`` is always the last event of a run, so the ring leaves
        # ``replying`` in both cases. With ``continue_conversation`` set in
        # ``INTENT_END`` the stock firmware reopens the microphone itself and
        # calls the start callback again - no extra RPC is needed and sending
        # one would only restart the already playing reply.
        await self._event("RUN_END", {})
        self._bump_metric("run_end")
        # The playback latch is what keeps the avatar steady between ``RUN_END``
        # and the device's own report: a media push to ``idle`` after an active
        # state, or the ``AnnounceFinished`` that closes this very delivery.
        if self._pending_playback:
            self._playback_latch = int(self._pending_playback[-1].session_generation)
        elif reply:
            self._playback_latch = generation
        if self._should_continue(reply):
            self.session_state = SessionState.CONTINUE_PENDING
            self._mark_event("continue_pending")
            self._sync_face_state()
            self._lease_after_run()
            return
        self.session = None
        self.session_state = SessionState.IDLE
        self._mark_event("run_end")
        self._sync_face_state()
        self._lease_after_run()

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
        """Stock dialog rule: only an open question keeps the run waiting.

        The firmware's ``continue_conversation`` flag is taken from the
        reply's shape: ``?`` ends the statement in an open question, every
        other terminator closes the run. Muted devices and a disabled config
        never keep the dialog.
        """
        if not self.config.continued_conversation or not reply:
            return False
        if self.media.muted:
            return False
        text = str(reply).strip()
        return text.endswith("?") or text.endswith("？")

    async def _start_udp_server(self) -> Optional[int]:
        """UDP microphone for firmware without the ``API_AUDIO`` flag."""
        if self._ingress is None:
            return None
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setblocking(False)
        sock.bind(("", 0))
        _transport, protocol = await asyncio.get_running_loop().create_datagram_endpoint(
            lambda: UdpAudioServer(self._ingress),
            sock=sock,
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
        source_status = (
            self._ingress.source_status(self._stream())
            if self._ingress is not None
            else {"source_id": self.device_id, "aec_state": "unknown"}
        )
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
            "lease": self.lease.snapshot(),
            "source_status": source_status,
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
