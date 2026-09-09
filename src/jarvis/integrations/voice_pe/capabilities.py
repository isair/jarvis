"""Capability decoding for the Voice PE integration.

Decisions come from ``voice_assistant_feature_flags`` (or the API 1.15
``DeviceCapabilitiesResponse``) and from the entities the node actually
publishes, never from a firmware version string. Every capability degrades on
its own so one missing piece does not fail the whole integration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .entities import (
    BUTTON_EVENT_OBJECT_ID,
    LED_RING_OBJECT_ID,
    MUTE_SWITCH_OBJECT_ID,
    kind_of,
)
from .models import (
    FEATURE_ANNOUNCE,
    FEATURE_API_AUDIO,
    FEATURE_MULTI_CHANNEL_AUDIO,
    FEATURE_SPEAKER,
    FEATURE_START_CONVERSATION,
    FEATURE_TIMERS,
    FEATURE_VOICE_ASSISTANT,
    feature_list,
)


@dataclass(frozen=True)
class CapabilitySnapshot:
    """Immutable view of one connection generation's capabilities."""

    voice_assistant: bool = False
    api_audio: bool = False
    speaker: bool = False
    multi_channel_audio: bool = False
    announce: bool = False
    start_conversation: bool = False
    timers: bool = False
    has_media_player: bool = False
    has_led_ring: bool = False
    has_button_event: bool = False
    has_mute_switch: bool = False
    led_effects: tuple[str, ...] = ()
    media_formats: tuple[str, ...] = ()
    feature_flags: int = 0
    entity_count: int = 0
    service_count: int = 0

    def names(self) -> list[str]:
        """Feature-flag names, for the health snapshot and the UI card."""
        return feature_list(self.feature_flags)

    @property
    def uses_api_audio(self) -> bool:
        """True when microphone + TTS ride the Native API, not UDP."""
        return bool(self.api_audio and self.speaker)


def _is_voice_assistant_entity(info: Any) -> bool:
    return str(getattr(info, "object_id", "") or "") in {
        "voice_assistant",
        "va",
    }


def build_snapshot(
    device_info: Any,
    entities: list[Any],
    services: Optional[list[Any]] = None,
    capabilities: Any = None,
) -> CapabilitySnapshot:
    """Decode feature flags and published entities into one snapshot.

    ``capabilities`` is the optional ``DeviceCapabilities`` model; from API
    1.15 its ``voice_assistant.feature_flags`` wins over the flat
    ``DeviceInfo`` field.
    """
    flags = int(getattr(device_info, "voice_assistant_feature_flags", 0) or 0)
    if capabilities is not None:
        va_caps = getattr(capabilities, "voice_assistant", None)
        caps_flags = int(getattr(va_caps, "feature_flags", 0) or 0)
        if caps_flags:
            flags = caps_flags

    has_media = False
    has_led = False
    has_event = False
    has_switch_mute = False
    led_effects: tuple[str, ...] = ()
    media_formats: tuple[str, ...] = ()

    for info in entities:
        object_id = str(getattr(info, "object_id", "") or "")
        kind = kind_of(info)
        if kind == "media_player":
            has_media = True
            formats = getattr(info, "supported_formats", []) or []
            media_formats = tuple(
                str(getattr(fmt, "format", "") or "") for fmt in formats
            )
        elif kind == "light":
            if object_id == LED_RING_OBJECT_ID or not has_led:
                has_led = True
                if object_id == LED_RING_OBJECT_ID:
                    led_effects = tuple(str(e) for e in (getattr(info, "effects", []) or []))
        elif kind == "event":
            event_types = [str(t) for t in (getattr(info, "event_types", []) or [])]
            if object_id == BUTTON_EVENT_OBJECT_ID or "double_press" in event_types:
                has_event = True
        elif kind == "switch":
            name = str(getattr(info, "name", "") or "").strip().lower()
            if object_id in {MUTE_SWITCH_OBJECT_ID, "mute_switch"} or name in {"mute", "mute switch"}:
                has_switch_mute = True

    if not flags and any(_is_voice_assistant_entity(e) for e in entities):
        # Very old firmware: only a legacy version marker, assume the base
        # assistant without any of the optional extras.
        flags = FEATURE_VOICE_ASSISTANT

    return CapabilitySnapshot(
        voice_assistant=bool(flags & FEATURE_VOICE_ASSISTANT),
        api_audio=bool(flags & FEATURE_API_AUDIO),
        speaker=bool(flags & FEATURE_SPEAKER),
        multi_channel_audio=bool(flags & FEATURE_MULTI_CHANNEL_AUDIO),
        announce=bool(flags & FEATURE_ANNOUNCE),
        start_conversation=bool(flags & FEATURE_START_CONVERSATION),
        timers=bool(flags & FEATURE_TIMERS),
        has_media_player=has_media,
        has_led_ring=has_led,
        has_button_event=has_event,
        has_mute_switch=has_switch_mute,
        led_effects=led_effects,
        media_formats=media_formats,
        feature_flags=flags,
        entity_count=len(entities),
        service_count=len(services or []),
    )
