"""Config mapping and persisted metadata for the Voice PE integration.

The Jarvis config framework is a flat frozen ``Settings`` dataclass loaded
from one JSON file, so every Voice PE key is a flat ``voice_pe_*`` key plus
two structured keys (``voice_pe_devices``, ``voice_pe_button_actions``).

``noise_psk`` and the Wi-Fi passphrase are only ever written through
``jarvis.config._save_json``, which chmods the file to ``0o600``. No secret is
printed by ``debug_log``; the length is logged instead.
"""

from __future__ import annotations

import base64
import os
from typing import Any, Dict, Optional, Tuple

from .models import DEFAULT_API_PORT, VoicePEConfig

#: Structured keys in config.json.
DEVICES_KEY = "voice_pe_devices"
BUTTON_ACTIONS_KEY = "voice_pe_button_actions"

#: Safe-by-default centre-button event mapping (single click stays local).
DEFAULT_BUTTON_ACTIONS: Dict[str, str] = {
    "double_press": "toggle_overlay",
    "triple_press": "open_command_palette",
    "long_press": "cancel_current_agent_run",
    "easter_egg_press": "toaster_easter_egg",
}


def _opt_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_rgb(value: Any, default: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Parse a colour into three 0..1 floats.

    Accepted forms: a 3-number list, ``"0.55,0,1"``, ``"8c00ff"`` /
    ``"#8c00ff"``. Anything incomplete falls back to ``default``.
    """
    if value is None:
        return default
    if isinstance(value, (tuple, list)) and len(value) == 3:
        return tuple(round(_as_float(part, 0.0), 4) for part in value)  # type: ignore[return-value]
    text = str(value).strip().lstrip("#")
    if not text:
        return default
    if "," in text:
        parts = [part.strip() for part in text.split(",")]
        decimal_only = all("." in part or part.isdigit() for part in parts)
        channels: list[float] = []
        for part in parts:
            if not part:
                return default
            if decimal_only:
                try:
                    channels.append(float(part))
                except ValueError:
                    return default
            else:
                try:
                    channels.append(int(part, 16) / 255.0)
                except ValueError:
                    return default
        if len(channels) != 3:
            return default
        return tuple(round(channel, 4) for channel in channels)  # type: ignore[return-value]
    if len(text) == 6:
        channels = []
        for index in (0, 2, 4):
            try:
                channels.append(int(text[index: index + 2], 16) / 255.0)
            except ValueError:
                return default
        return tuple(round(channel, 4) for channel in channels)  # type: ignore[return-value]
    return default


def from_settings(settings: Any) -> VoicePEConfig:
    """Build a :class:`VoicePEConfig` off the flat ``Settings`` fields."""
    raw_devices = getattr(settings, "voice_pe_devices", None)
    devices: Dict[str, Dict[str, Any]] = {}
    if isinstance(raw_devices, dict):
        for mac, meta in raw_devices.items():
            if isinstance(meta, dict):
                devices[str(mac)] = dict(meta)

    raw_actions: Any = getattr(settings, "voice_pe_button_actions", None)
    actions = dict(DEFAULT_BUTTON_ACTIONS)
    if isinstance(raw_actions, dict):
        for key, value in raw_actions.items():
            if value:
                actions[str(key)] = str(value)

    psk_id = _opt_str(getattr(settings, "voice_pe_noise_psk_secret_id", None))
    if not psk_id and devices:
        # First paired MAC acts as the implicit secret id.
        psk_id = next(iter(sorted(devices)))

    return VoicePEConfig(
        enabled=bool(getattr(settings, "voice_pe_enabled", False)),
        discovery_enabled=bool(getattr(settings, "voice_pe_discovery_enabled", True)),
        host=_opt_str(getattr(settings, "voice_pe_host", None)),
        port=_as_int(getattr(settings, "voice_pe_port", DEFAULT_API_PORT), DEFAULT_API_PORT),
        device_name=_opt_str(getattr(settings, "voice_pe_device_name", None)),
        mac_address=_opt_str(getattr(settings, "voice_pe_mac_address", None)),
        noise_psk_secret_id=psk_id,
        room=_opt_str(getattr(settings, "voice_pe_room", None)),
        disable_wake_words=bool(getattr(settings, "voice_pe_disable_wake_words", True)),
        prefer_api_audio=bool(getattr(settings, "voice_pe_prefer_api_audio", True)),
        preferred_input_channel=_as_int(
            getattr(settings, "voice_pe_preferred_input_channel", 0), 0
        ),
        continued_conversation=bool(
            getattr(settings, "voice_pe_continued_conversation", True)
        ),
        conversation_timeout_s=_as_float(
            getattr(settings, "voice_pe_conversation_timeout_s", 300.0), 300.0
        ),
        reconnect_min_s=_as_float(getattr(settings, "voice_pe_reconnect_min_s", 1.0), 1.0),
        reconnect_max_s=_as_float(getattr(settings, "voice_pe_reconnect_max_s", 30.0), 30.0),
        audio_queue_ms=_as_int(getattr(settings, "voice_pe_audio_queue_ms", 300), 300),
        led_brightness=_as_float(getattr(settings, "voice_pe_led_brightness", 0.66), 0.66),
        led_rgb=_parse_rgb(getattr(settings, "voice_pe_led_rgb", None), (0.55, 0.0, 1.0)),
        button_actions=actions,
        devices=devices,
    )


# ---------------------------------------------------------------------------
# Secrets + persisted metadata (one JSON file, 0o600, existing loader helpers)
# ---------------------------------------------------------------------------

def _config_io():
    from jarvis.config import default_config_path, _load_json, _save_json

    return default_config_path, _load_json, _save_json


def get_psk(config: VoicePEConfig, mac: str = "") -> Optional[str]:
    """PSK for one device: env override first, then the stored metadata."""
    env = os.environ.get("JARVIS_VOICE_PE_PSK", "").strip()
    if env:
        return env
    meta = config.devices.get(mac) or {}
    psk = str(meta.get("noise_psk", "") or "").strip()
    return psk or None


def new_noise_key() -> bytes:
    """32 cryptographically random bytes for the Noise PSK."""
    return os.urandom(32)


def encode_psk(key: bytes) -> str:
    return base64.b64encode(bytes(key)).decode("ascii")


def save_device_metadata(mac: str, meta: Dict[str, Any]) -> bool:
    """Merge per-device metadata into ``voice_pe_devices``."""
    if not mac:
        return False
    default_path, load_json, save_json = _config_io()
    path = default_path()
    data = load_json(path)
    devices = data.get(DEVICES_KEY)
    if not isinstance(devices, dict):
        devices = {}
    merged = dict(devices.get(mac) or {})
    merged.update(meta)
    devices[mac] = merged
    data[DEVICES_KEY] = devices
    return bool(save_json(path, data))


def enable_integration(value: bool = True) -> bool:
    """Write ``voice_pe_enabled`` so a paired unit is actually started."""
    default_path, load_json, save_json = _config_io()
    path = default_path()
    data = load_json(path)
    data["voice_pe_enabled"] = bool(value)
    return bool(save_json(path, data))


def forget_device(mac: str) -> bool:
    """Drop Jarvis-side metadata for one MAC. No factory reset is attempted."""
    if not mac:
        return False
    default_path, load_json, save_json = _config_io()
    path = default_path()
    data = load_json(path)
    devices = data.get(DEVICES_KEY)
    if not isinstance(devices, dict) or mac not in devices:
        return True
    devices.pop(mac, None)
    data[DEVICES_KEY] = devices
    return bool(save_json(path, data))


def fold_button_actions(value: Any) -> Dict[str, str]:
    """Normalise the button mapping into a flat ``{event: action}`` dict.

    Accepts the dict form and the list form the settings UI writes, i.e.
    ``["double_press=toggle_overlay", "long_press -> cancel_current_agent_run"]``.
    """
    folded: Dict[str, str] = {}
    if isinstance(value, dict):
        for key, item in value.items():
            if item:
                folded[str(key).strip()] = str(item).strip()
        return folded
    if isinstance(value, (list, tuple)):
        for item in value:
            text = str(item or "").strip()
            if not text:
                continue
            for separator in ("=", "->"):
                if separator in text:
                    key, _, action = text.partition(separator)
                    if key.strip() and action.strip():
                        folded[key.strip()] = action.strip()
                    break
    return folded


def wizard_status(settings: Any) -> tuple[bool, str]:
    """One-line status for the setup wizard's system-status card.

    Reads only the stored metadata so the wizard stays non-blocking; the live
    mDNS scan is what ``jarvis voice-pe discover`` performs.
    """
    config = from_settings(settings)
    if not config.enabled:
        return False, "Disabled"
    paired = [
        str(meta.get("node_name") or mac)
        for mac, meta in config.devices.items()
        if isinstance(meta, dict)
    ]
    if paired:
        return True, f"Paired: {', '.join(sorted(paired))}"
    if config.host:
        return True, f"Host {config.host}:{config.port}"
    return False, "No device stored - run: jarvis voice-pe pair"

