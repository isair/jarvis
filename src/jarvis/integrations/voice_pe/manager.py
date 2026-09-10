"""Multi-device Voice PE manager: one asyncio loop thread, N satellites.

The Jarvis runtime is thread-based, so the manager owns a single background
event loop and hands work to it. Each physical device gets exactly one
:class:`VoicePEDevice` (and therefore exactly one Voice Assistant subscriber).

Recovery: ``ReconnectLogic`` backs off with jitter and wakes on the mDNS
announcement, and every reconnect re-reads entities and feature flags because
entity keys are not application constants. Audio and replies of an old
generation are never replayed.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Optional

from . import config as pe_config
from .device import VoicePEDevice
from .discovery import discover
from .models import DeviceState, VoicePEConfig

try:  # pragma: no cover - trivial import shim
    from jarvis.debug import debug_log
except ImportError:  # pragma: no cover
    def debug_log(message: str, category: str = "debug") -> None:  # type: ignore[misc]
        pass


class SinkFanout:
    """Fan the listener's pipeline milestones out to every attached device.

    Only the device holding the open session has ``session`` set, so the
    others drop each milestone without sending events.
    """

    def __init__(self, devices: list[VoicePEDevice]) -> None:
        self._devices = list(devices)

    def on_vad_start(self) -> None:
        for device in self._devices:
            device.on_vad_start()

    def on_vad_end(self) -> None:
        for device in self._devices:
            device.on_vad_end()

    def on_transcript(self, text: str) -> None:
        for device in self._devices:
            device.on_transcript(text)

    def on_reply(self, reply: str) -> None:
        for device in self._devices:
            device.on_reply(reply)

    def on_error(self, code: str, message: str) -> None:
        for device in self._devices:
            device.on_error(code, message)


class VoicePEManager:
    """Owns the event loop thread and the per-device connections."""

    def __init__(self, settings: Any, listener: Any, tts_engine: Any) -> None:
        self.settings = settings
        self.config: VoicePEConfig = pe_config.from_settings(settings)
        self._listener = listener
        self._tts = tts_engine
        self._devices: list[VoicePEDevice] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._metrics: dict = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return bool(self.config.enabled)

    def start(self) -> bool:
        """Start the loop thread. Returns False when disabled."""
        if not self.enabled:
            return False
        if self._thread is not None:
            return True
        self._thread = threading.Thread(
            target=self._run_loop, name="voice_pe", daemon=True
        )
        self._thread.start()
        return True

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._astart())
            loop.run_forever()
        finally:
            try:
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()
            except Exception:
                pass
            loop.close()

    async def _astart(self) -> None:
        hosts = self._candidate_hosts()
        if not hosts and self.config.discovery_enabled:
            # No manual host and nothing stored yet: one mDNS pass seeds the
            # device list, then the same nodes become the reconnect sources.
            from .discovery import discover

            try:
                found = await discover(self.config)
            except Exception as err:
                found = []
                debug_log(f"component=voice_pe event=discovery_failed error={err}", "voice")
            for entry in found:
                mac = str(entry.get("mac_address") or "")
                if mac:
                    pe_config.save_device_metadata(
                        mac,
                        {
                            "mac_address": mac,
                            "node_name": entry.get("node_name", ""),
                            "project_name": entry.get("project_name", ""),
                            "project_version": entry.get("project_version", ""),
                            "voice_feature_flags": entry.get("voice_feature_flags", 0),
                            "addresses": [entry["host"]],
                            "port": int(entry.get("port") or self.config.port),
                        },
                    )
                hosts.append(
                    (
                        str(entry["host"]),
                        int(entry.get("port") or self.config.port),
                        str(entry.get("node_name") or "") or None,
                        mac or None,
                    )
                )

        for host, port, name, mac in hosts:
            psk = self._psk_for(mac, name)
            device = VoicePEDevice(
                self.config,
                listener=self._listener,
                tts_engine=self._tts,
                host=host,
                port=port,
                psk=psk,
                device_name=name,
                expected_mac=(mac or "").replace(":", "").lower() or None,
                metrics=self._metrics,
            )
            self._register_builtin_actions(device)
            self._devices.append(device)
            try:
                await device.start()
            except Exception as err:
                device.handle_auth_error(err)
                debug_log(
                    f"component=voice_pe host={host} error={err}",
                    "voice",
                )
        # One shared VAD/STT path, N satellites: the fan-out hands each
        # milestone to every device and the one holding the open session
        # answers, the others ignore it.
        if self._listener is not None and self._devices:
            try:
                self._listener._voice_pe_sink = SinkFanout(self._devices)
            except Exception:
                pass
        debug_log(
            f"component=voice_pe event=manager_start devices={len(self._devices)}",
            "voice",
        )

    async def _astop(self) -> None:
        if self._listener is not None:
            try:
                self._listener._voice_pe_sink = None
            except Exception:
                pass
        for device in self._devices:
            try:
                await device.stop()
            except Exception:
                pass
        self._devices = []

    def stop(self) -> None:
        """Stop devices and the loop thread."""
        loop = self._loop
        if loop is not None:
            try:
                future = asyncio.run_coroutine_threadsafe(self._astop(), loop)
                future.result(timeout=3.0)
            except Exception:
                pass
            try:
                loop.call_soon_threadsafe(loop.stop)
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        self._loop = None

    def restart(self) -> None:
        """Re-read config and rebuild the device list (UI re-save helper)."""
        self.config = pe_config.from_settings(self.settings)
        self.stop()
        self.start()

    # ------------------------------------------------------------------
    # Device discovery of hosts
    # ------------------------------------------------------------------

    def _candidate_hosts(self) -> list[tuple[str, int, Optional[str], Optional[str]]]:
        cfg = self.config
        hosts: list[tuple[str, int, Optional[str], Optional[str]]] = []

        if cfg.host:
            hosts.append((cfg.host, cfg.port, cfg.device_name, cfg.mac_address))
            return hosts

        for mac, meta in cfg.devices.items():
            addresses = meta.get("addresses") or []
            if not addresses:
                continue
            hosts.append(
                (
                    str(addresses[0]),
                    int(meta.get("port") or cfg.port),
                    str(meta.get("node_name") or "") or None,
                    str(mac),
                )
            )
        return hosts

    def _psk_for(self, mac: Optional[str], name: Optional[str]) -> Optional[str]:
        if mac:
            psk = pe_config.get_psk(self.config, mac)
            if psk:
                return psk
        for meta in self.config.devices.values():
            if name and meta.get("node_name") == name and meta.get("noise_psk"):
                return str(meta["noise_psk"])
        return None

    def _register_builtin_actions(self, device: VoicePEDevice) -> None:
        device.actions.register("cancel_current_agent_run", _cancel_agent_run)
        device.actions.register("toaster_easter_egg", _easter_egg)

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    @property
    def devices(self) -> list[VoicePEDevice]:
        return list(self._devices)

    def device(self, key: str) -> Optional[VoicePEDevice]:
        """Find one device by node name, friendly name or MAC address."""
        needle = str(key or "").strip().lower()
        if not needle:
            return self._devices[0] if self._devices else None
        for device in self._devices:
            identity = device.identity
            candidates = {
                str(identity.get("node_name", "")).lower(),
                str(identity.get("friendly_name", "")).lower(),
                str(identity.get("mac_address", "")).lower(),
                str(identity.get("mac_address", "")).replace(":", "").lower(),
                str(device._host).lower(),
            }
            if needle in candidates:
                return device
        return None

    def register_action(self, name: str, handler) -> None:
        for device in self._devices:
            device.actions.register(name, handler)

    def metrics(self) -> dict:
        """Metric snapshot with the per-device counters folded in."""
        merged = dict(self._metrics)
        merged["active_devices"] = sum(
            1
            for device in self._devices
            if device.state in (DeviceState.READY, DeviceState.VOICE_ACTIVE)
        )
        merged["device_count"] = len(self._devices)
        for device in self._devices:
            name = device.identity.get("node_name") or device._host
            merged[f"{name}.audio_queue_ms"] = device.health_snapshot()["audio_queue_ms"]
        return merged

    def health(self) -> dict:
        """Health snapshot list for the diagnostics panel and the CLI."""
        return {
            "enabled": self.enabled,
            "devices": [device.health_snapshot() for device in self._devices],
            "metrics": self.metrics(),
        }

    # ------------------------------------------------------------------
    # Direct commands used by the CLI
    # ------------------------------------------------------------------

    async def aasync_discover(self) -> list[dict]:
        return await discover(self.config)

    async def announce(self, key: str, text: str, *, start_conversation: bool = True) -> bool:
        """Announce over the Voice Assistant RPC with a playable ``media_id``.

        The satellite fetches ``media_id`` as a URL, so plain text is first
        synthesized into a WAV and published on the LAN HTTP server; a text
        that already is a URL is passed through unchanged.
        """
        device = self.device(key)
        if device is None or device.media is None:
            return False
        text = text or ""
        media_id = text if "://" in text else await device.tts_media_url(text)
        return await device.media.announce(
            media_id or "",
            text=text,
            start_conversation=start_conversation
            and device.capabilities.start_conversation,
        )

    async def play(self, key: str, url: str) -> bool:
        device = self.device(key)
        if device is None or not url:
            return False
        device.media.play_url(url)
        return True

    async def stop_media(self, key: str) -> bool:
        device = self.device(key)
        if device is None:
            return False
        device.media.stop()
        return True

    async def set_led(self, key: str, rgb, brightness) -> bool:
        from .led import apply_led

        device = self.device(key)
        if device is None:
            return False
        key_id = device.entities.led_key()
        if key_id is None:
            return False
        apply_led(device._client, key_id, device.config, rgb=rgb, brightness=brightness)
        return True


# ----------------------------------------------------------------------
# Built-in button actions
# ----------------------------------------------------------------------

def _cancel_agent_run() -> None:
    """Cancel the in-flight reply on both the voice and the text path."""
    try:
        from jarvis.daemon import cancel_active_chat_query

        cancel_active_chat_query()
    except Exception:
        pass


def _easter_egg() -> None:
    """Mirror the device's own easter-egg pulse onto the toaster overlay."""
    try:
        from desktop_app.face_widget import JarvisState, get_jarvis_state

        get_jarvis_state().set_state(JarvisState.SUCCESS, label="voice_pe")
    except Exception:
        pass
