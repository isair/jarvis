"""LED ring control for the Voice PE integration.

The user-facing light entity is ``led_ring``: on/off, RGB colour, brightness
and (when declared) transition. The internal ``voice_assistant_leds`` light is
driven by the firmware itself from the standard Voice Assistant events, so it
is not addressed here.
"""

from __future__ import annotations

from typing import Optional, Tuple

from .models import VoicePEConfig

#: Stock initial state of ``led_ring`` from the official firmware.
STOCK_LED_RGB: Tuple[float, float, float] = (0.094, 0.733, 0.949)
STOCK_LED_BRIGHTNESS = 0.66


def apply_led(
    client,
    key: int,
    config: VoicePEConfig,
    *,
    on: bool = True,
    rgb: Optional[Tuple[float, float, float]] = None,
    brightness: Optional[float] = None,
    transition_s: Optional[float] = None,
) -> None:
    """Send one ``light_command``; a missing light entity is a no-op."""
    if key is None:
        return
    colour = rgb if rgb is not None else config.led_rgb
    level = brightness if brightness is not None else config.led_brightness
    kwargs: dict = {"state": bool(on), "brightness": float(level)}
    if colour is not None and len(colour) == 3:
        kwargs["rgb"] = (float(colour[0]), float(colour[1]), float(colour[2]))
    if transition_s is not None:
        kwargs["transition_length"] = float(transition_s)
    client.light_command(key=key, **kwargs)


def parse_hex_rgb(text: str) -> Optional[Tuple[float, float, float]]:
    """``8c00ff`` / ``#8c00ff`` / ``0.55,0,1`` to 0..1 channel floats."""
    from .config import _parse_rgb

    parsed = _parse_rgb(text, (0.0, 0.0, 0.0))
    if parsed == (0.0, 0.0, 0.0) and not (text or "").strip():
        return None
    return parsed


def sync_defaults(client, index, config: VoicePEConfig) -> Optional[dict]:
    """Push the configured accent colour once the session is READY."""
    key = index.led_key() if index is not None else None
    if key is None:
        return None
    apply_led(client, key, config)
    return {"key": key, "rgb": config.led_rgb, "brightness": config.led_brightness}
