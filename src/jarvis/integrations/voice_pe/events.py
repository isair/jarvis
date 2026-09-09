"""Voice Assistant events and centre-button event mapping.

The pipeline event sequence is also the contract of the stock LED state
machine, so the phases come from these events rather than from manual per-
phase light commands:

``RUN_START, STT_START, STT_VAD_START, STT_VAD_END, STT_END, INTENT_START,
INTENT_END, TTS_START, TTS_STREAM_START, TTS_STREAM_END, RUN_END``.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

try:  # pragma: no cover - trivial import shim
    from aioesphomeapi import VoiceAssistantEventType as VA_EVENT
except ImportError:  # pragma: no cover
    VA_EVENT = None  # type: ignore[assignment]

#: Numeric fallbacks matching ``VoiceAssistantEventType``.
EVENT_IDS = {
    "ERROR": 0,
    "RUN_START": 1,
    "RUN_END": 2,
    "STT_START": 3,
    "STT_END": 4,
    "INTENT_START": 5,
    "INTENT_END": 6,
    "TTS_START": 7,
    "TTS_END": 8,
    "WAKE_WORD_START": 9,
    "WAKE_WORD_END": 10,
    "STT_VAD_START": 11,
    "STT_VAD_END": 12,
    "TTS_STREAM_START": 98,
    "TTS_STREAM_END": 99,
    "INTENT_PROGRESS": 100,
}

#: LED phase each event leaves the device in. Matches the ``on_*`` triggers of
#: the official firmware: ``on_listening`` is the waiting phase, the VAD start
#: is the listening phase, the VAD end is thinking, TTS is replying.
EVENT_LED_PHASE = {
    "RUN_START": "waiting_for_command",
    "STT_START": "waiting_for_command",
    "STT_VAD_START": "listening_for_command",
    "STT_VAD_END": "thinking",
    "STT_END": "thinking",
    "INTENT_START": "thinking",
    "INTENT_END": "thinking",
    "INTENT_PROGRESS": "replying",
    "TTS_START": "replying",
    "TTS_STREAM_START": "replying",
    "TTS_STREAM_END": "replying",
    "RUN_END": "idle",
    "ERROR": "error",
}

#: Event-entity types a stock device publishes on ``button_press_event``.
STOCK_BUTTON_EVENTS = ("double_press", "triple_press", "long_press", "easter_egg_press")

#: The Jarvis actions a mapping may name. The single click stays on-device.
KNOWN_ACTIONS = (
    "toggle_overlay",
    "open_command_palette",
    "cancel_current_agent_run",
    "toaster_easter_egg",
    "ignore",
)


def event_type(name: str):
    """Enum member for an event name, with the numeric id as fallback."""
    if VA_EVENT is not None:
        member = getattr(VA_EVENT, f"VOICE_ASSISTANT_{name}", None)
        if member is not None:
            return member
    return EVENT_IDS[name]


def send_event(client, name: str, data: Optional[dict[str, str]] = None) -> None:
    """Send one Voice Assistant event with ``str``-only payloads."""
    payload: dict[str, str] = {}
    for key, value in (data or {}).items():
        if value is None:
            continue
        payload[str(key)] = str(value)
    client.send_voice_assistant_event(event_type(name), payload)


def resolve_action(event_value: str, mapping: dict[str, str]) -> str:
    """Map an event-entity value to a Jarvis action name (``ignore`` else)."""
    name = str(event_value or "").strip()
    action = mapping.get(name)
    if action:
        return str(action)
    if name in KNOWN_ACTIONS:
        return name
    return "ignore"


class ActionRunner:
    """Runs mapped button actions against callbacks the desktop app registers."""

    def __init__(self) -> None:
        self._handlers: dict[str, Callable[[], Any]] = {}

    def register(self, name: str, handler: Callable[[], Any]) -> None:
        self._handlers[str(name)] = handler

    def run(self, action: str) -> str:
        handler = self._handlers.get(action)
        if handler is None:
            return f"unhandled:{action}"
        try:
            handler()
        except Exception as err:  # keep the transport alive
            return f"error:{action}:{err}"
        return f"ok:{action}"
