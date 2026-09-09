"""Entity index for one Voice PE connection generation.

Entity ``key`` values are per-connection numbers, not application constants, so
every lookup resolves through ``object_id`` (then ``name``) and is refreshed
after each reconnect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

#: Public light of the retail firmware (``led_ring``); ``voice_assistant_leds``
#: is ``internal: true`` and is never looked up by a hardcoded key.
LED_RING_OBJECT_ID = "led_ring"
BUTTON_EVENT_OBJECT_ID = "button_press_event"
#: Soft mute switch object ids published by the stock firmware.
MUTE_SWITCH_OBJECT_ID = "master_mute_switch"


def _object_id(info: Any) -> str:
    return str(getattr(info, "object_id", "") or "")


def _name(info: Any) -> str:
    return str(getattr(info, "name", "") or "").strip().lower()


#: Wire model names per entity kind, with the distinguishing attributes. The
#: probes come first so a renamed model class still resolves.
_KIND_BY_ATTR = (
    ("supported_color_modes", "light"),
    ("supported_formats", "media_player"),
    ("event_types", "event"),
    # ``multiple`` belongs to binary sensors and is checked before
    # ``assumed_state``, which both binary sensors and switches declare.
    ("multiple", "binary_sensor"),
    ("assumed_state", "switch"),
)
_KIND_BY_CLASS = {
    "LightInfo": "light",
    "MediaPlayerInfo": "media_player",
    "EventInfo": "event",
    "SwitchInfo": "switch",
    "BinarySensorInfo": "binary_sensor",
    "SensorInfo": "sensor",
    "NumberInfo": "number",
    "SelectInfo": "select",
    "TextInfo": "text",
}


def kind_of(info: Any) -> str:
    """Entity kind of one ``EntityInfo`` entry (empty string when unknown)."""
    for attribute, kind in _KIND_BY_ATTR:
        if hasattr(info, attribute):
            return kind
    return _KIND_BY_CLASS.get(type(info).__name__, "")


@dataclass
class EntityIndex:
    """Lookup table rebuilt after ``list_entities`` on every generation."""

    infos: list[Any] = field(default_factory=list)
    by_key: dict[int, Any] = field(default_factory=dict)
    led: Optional[Any] = None
    button_event: Optional[Any] = None
    media_player: Optional[Any] = None
    mute_switch: Optional[Any] = None
    hardware_mute: Optional[Any] = None
    timers: list[Any] = field(default_factory=list)

    def build(self, entities: list[Any]) -> "EntityIndex":
        """Index one ``list_entities_services()`` result."""
        self.infos = list(entities)
        self.by_key = {}
        self.led = None
        self.button_event = None
        self.media_player = None
        self.mute_switch = None
        self.hardware_mute = None
        self.timers = []

        for info in entities:
            key = int(getattr(info, "key", 0) or 0)
            if key:
                self.by_key[key] = info
            kind = kind_of(info)
            object_id = _object_id(info)
            name = _name(info)

            if kind == "light":
                if object_id == LED_RING_OBJECT_ID:
                    self.led = info
                elif self.led is None:
                    self.led = info
            elif kind == "event":
                event_types = [str(t) for t in (getattr(info, "event_types", []) or [])]
                if object_id == BUTTON_EVENT_OBJECT_ID or "double_press" in event_types:
                    self.button_event = info
            elif kind == "media_player":
                # The internal mixer stays unexposed; the first published
                # media player is the primary one.
                if self.media_player is None:
                    self.media_player = info
            elif kind == "switch":
                if object_id in {"master_mute_switch", "mute_switch"} or name.startswith("mute"):
                    self.mute_switch = info
            elif kind == "binary_sensor" and name.startswith("mute"):
                self.hardware_mute = info

        return self

    # -- helpers ---------------------------------------------------------

    def info_for(self, key: int) -> Optional[Any]:
        return self.by_key.get(int(key))

    def led_key(self) -> Optional[int]:
        if self.led is None:
            return None
        return int(getattr(self.led, "key", 0) or 0) or None

    def media_key(self) -> Optional[int]:
        if self.media_player is None:
            return None
        return int(getattr(self.media_player, "key", 0) or 0) or None

    def mute_key(self) -> Optional[int]:
        if self.mute_switch is None:
            return None
        return int(getattr(self.mute_switch, "key", 0) or 0) or None

    def event_types(self) -> list[str]:
        if self.button_event is None:
            return []
        return [str(t) for t in (getattr(self.button_event, "event_types", []) or [])]

    def supports_transition(self) -> bool:
        """Colour-mode list of the ring light declares brightness support."""
        if self.led is None:
            return False
        modes = getattr(self.led, "supported_color_modes", []) or []
        return len(modes) > 0

    def led_color_modes(self) -> list[str]:
        if self.led is None:
            return []
        return [str(m) for m in (getattr(self.led, "supported_color_modes", []) or [])]

    def effects(self) -> tuple[str, ...]:
        if self.led is None:
            return ()
        return tuple(str(e) for e in (getattr(self.led, "effects", []) or []))


def describe(index: EntityIndex) -> dict[str, Any]:
    """Small plain-dict view of the index for the UI card and the CLI."""
    return {
        "led_ring": _object_id(index.led) if index.led is not None else None,
        "led_color_modes": index.led_color_modes(),
        "led_effects": list(index.effects()),
        "button_event": _object_id(index.button_event) if index.button_event is not None else None,
        "button_event_types": index.event_types(),
        "media_player": _object_id(index.media_player) if index.media_player is not None else None,
        "mute_switch": _object_id(index.mute_switch) if index.mute_switch is not None else None,
        "hardware_mute": _object_id(index.hardware_mute) if index.hardware_mute is not None else None,
    }
