"""Discovery for ESPHome Voice PE nodes.

Source order: mDNS, the last known IP from persisted metadata, then the
manually configured host/IP. Identity is taken from ``DeviceInfoResponse``
(MAC, node name, project metadata), never from the hostname alone: the IP is
a mutable attribute while the MAC is the stable identity.

A foreign ESPHome node is not adopted automatically. It counts as a Voice PE
when its project/model metadata matches the official firmware, or when the
user confirmed it explicitly.
"""

from __future__ import annotations

import asyncio
from typing import Any, Optional

from .models import DEFAULT_API_PORT, VoicePEConfig, make_client
from .config import get_psk

#: Official Voice PE metadata markers (project name / node name prefixes).
#: The retail unit advertises the short ``ha-voice-pe-XXXXXX`` node name.
PROJECT_MARKERS = ("home-assistant-voice", "home-assistant-voice-pe", "ha-voice-pe")
NODE_NAME_PREFIXES = ("home-assistant-voice", "ha-voice-pe")

MDNS_TIMEOUT_S = 2.5


def match_voice_pe(device_info: Any) -> bool:
    """True when ``DeviceInfo`` metadata identifies the official Voice PE."""
    if device_info is None:
        return False
    project = str(getattr(device_info, "project_name", "") or "").lower()
    for marker in PROJECT_MARKERS:
        if marker in project:
            return True
    node_name = str(getattr(device_info, "name", "") or "").lower()
    if node_name.startswith(NODE_NAME_PREFIXES):
        return True
    model = str(getattr(device_info, "model", "") or "").lower()
    return "home-assistant-voice" in model


async def _browse_esphome(azc: Any, timeout_s: float) -> list[dict]:
    """One mDNS browse over ``_esphomelib._tcp.local.`` on the owner loop.

    Every object here is bound to the loop of ``azc`` (the ``AsyncZeroconf``
    instance), so no cross-loop handoff is needed.
    """
    from zeroconf.asyncio import AsyncServiceBrowser

    from .models import ESPHOME_MDNS_TYPE

    names: list[str] = []

    def _handler(_zc: Any, _type_: Any, _state: Any, new_names: Any) -> None:
        for name in new_names or []:
            if name not in names:
                names.append(name)

    browser = AsyncServiceBrowser(azc.zeroconf, [ESPHOME_MDNS_TYPE], [_handler])
    found: list[dict] = []
    try:
        await asyncio.sleep(timeout_s)
        for name in list(names):
            info = await azc.async_get_service_info(
                ESPHOME_MDNS_TYPE, name, timeout=int(timeout_s * 1000)
            )
            if info is None or not info.addresses:
                continue
            properties = {
                str(key): (value.decode() if isinstance(value, bytes) else str(value))
                for key, value in (info.decoded_properties or {}).items()
            }
            found.append(
                {
                    "name": str(info.server or info.name or name),
                    # The instance label is the stable ESPHome node name.
                    "node_name": str(name).split("._esphomelib")[0],
                    "addresses": [str(a) for a in info.addresses],
                    "port": int(info.port or DEFAULT_API_PORT),
                    "properties": properties,
                    "source": "mdns",
                }
            )
    finally:
        try:
            await browser.async_cancel()
        except Exception:
            pass

    return found


async def probe(host: str, port: int, config: VoicePEConfig, psk: Optional[str],
                expected_name: Optional[str] = None) -> Optional[Any]:
    """Open a short Native API connection and return its ``DeviceInfo``.

    Every request is retried once: the single-client limit of the Native API
    makes the first attempt after a lease change end with a closed socket.
    """
    client = make_client(host, port, psk, device_name=expected_name)
    info: Any = None
    try:
        for _ in range(2):
            try:
                await client.connect()
                info = await client.device_info()
                break
            except Exception:
                try:
                    await client.disconnect(True)
                except Exception:
                    pass
                await asyncio.sleep(0.3)
        return info
    finally:
        try:
            await client.disconnect(True)
        except Exception:
            pass


async def discover(config: VoicePEConfig, *, confirmed: Optional[set[str]] = None) -> list[dict]:
    """Return candidate dicts in preference order, Voice PE matches first."""
    candidates: list[dict] = []

    # 1. mDNS / ZeroConf. One ``AsyncZeroconf`` owns the loop for the whole
    # browse, so every zeroconf object stays on that same loop.
    if config.discovery_enabled:
        azc: Any = None
        try:
            from zeroconf.asyncio import AsyncZeroconf

            azc = AsyncZeroconf()
        except Exception:
            azc = None
        if azc is not None:
            try:
                found = await _browse_esphome(azc, MDNS_TIMEOUT_S)
            except Exception:
                found = []
            for entry in found:
                for address in entry["addresses"]:
                    candidates.append({**entry, "host": address})
            try:
                await azc.async_close()
            except Exception:
                pass

    # 2. Last known addresses from persisted metadata
    for meta in config.devices.values():
        for address in (meta.get("addresses") or []):
            if address and not any(c["host"] == address for c in candidates):
                candidates.append(
                    {
                        "host": str(address),
                        "port": int(meta.get("port") or config.port),
                        "name": str(meta.get("node_name") or ""),
                        "source": "last_known",
                    }
                )
    if config.host and not any(c["host"] == config.host for c in candidates):
        # 3. Manual host/IP
        candidates.append(
            {
                "host": config.host,
                "port": config.port,
                "name": config.device_name or "",
                "source": "manual",
            }
        )

    # Confirm identity through the API and keep only Voice PE matches. Each
    # stored PSK is tried in turn; a node without encryption answers with None
    # for the key anyway.
    confirmed = confirmed or set()
    stored_psks: list[str] = []
    for meta in config.devices.values():
        if not isinstance(meta, dict):
            continue
        value = str(meta.get("noise_psk") or "").strip()
        if value and value not in stored_psks:
            stored_psks.append(value)
    env_psk = get_psk(config, "") or None
    if env_psk and env_psk not in stored_psks:
        stored_psks.insert(0, env_psk)

    results: list[dict] = []
    for entry in candidates:
        node_name = str(entry.get("node_name") or "").strip() or None
        info: Any = None
        for psk in [None, *stored_psks]:
            info = await probe(
                entry["host"],
                int(entry.get("port") or config.port),
                config,
                psk,
                expected_name=node_name,
            )
            if info is not None:
                break
        if info is None:
            continue
        mac = str(getattr(info, "mac_address", "") or "")
        is_pe = match_voice_pe(info) or mac in confirmed
        if not is_pe:
            continue
        results.append(
            {
                "host": entry["host"],
                "port": int(entry.get("port") or config.port),
                "source": entry.get("source", "mdns"),
                "node_name": str(getattr(info, "name", "") or ""),
                "friendly_name": str(getattr(info, "friendly_name", "") or ""),
                "mac_address": mac,
                "project_name": str(getattr(info, "project_name", "") or ""),
                "project_version": str(getattr(info, "project_version", "") or ""),
                "model": str(getattr(info, "model", "") or ""),
                "esphome_version": str(getattr(info, "esphome_version", "") or ""),
                "voice_feature_flags": int(
                    getattr(info, "voice_assistant_feature_flags", 0) or 0
                ),
                "api_encryption_supported": bool(
                    getattr(info, "api_encryption_supported", False)
                ),
                "api_encryption_provisionable": bool(
                    getattr(info, "api_encryption_provisionable", False)
                ),
            }
        )
    return results
