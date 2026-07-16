"""Instant command registry - regex fast-path that skips the LLM.

Registered commands match against the user's query BEFORE the reply engine
runs. A matching handler returns a reply string instantly without invoking
the LLM. Useful for time, date, greetings, thanks, and other trivial
patterns that don't need model inference.

Usage:
    from .instant_commands import register_command, match_instant_command

    register_command("time", r"what(?:'s| is) the time|what time is it", handler_fn)

Or with the decorator:
    @register_command("time", r"what(?:'s| is) the time|what time is it")
    def handle_time(text, cfg, db, context):
        return f"The current time is {datetime.now():%I:%M %p}."
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Callable, Optional

# Context dict keys passed to handlers
CONTEXT_LAST_TTS = "last_tts_text"

# Registry: (name, compiled_pattern, handler)
_registry: list[tuple[str, "re.Pattern[str]", Callable]] = []


def register_command(name: str, pattern: str):
    """Decorator to register an instant command handler.

    Args:
        name: Short label for logging (e.g. "time", "greeting").
        pattern: Regex searched against the lower-cased query text.
                 Uses ``re.search`` so it matches anywhere in the text.
    """
    compiled = re.compile(pattern, re.IGNORECASE)

    def decorator(fn: Callable) -> Callable:
        _registry.append((name, compiled, fn))
        return fn

    return decorator


def try_handle_instant_command(text: str, listener: Any = None) -> tuple[bool, Optional[str]]:
    """Top-level entry used by the listener.

    Matches *text* against the registry and returns ``(handled, reply)``.
    If no instant command matches, returns ``(False, None)`` so the caller
    falls through to the normal LLM pipeline.

    The listener is passed so the repeat command can read the last spoken
    text from the echo detector without a hard dependency on its internals.
    """
    cfg = getattr(listener, "cfg", None)
    db = getattr(listener, "db", None)
    context: dict[str, Any] = {}
    if listener is not None:
        last_tts = getattr(getattr(listener, "echo_detector", None), "_last_tts_text", None)
        if last_tts:
            context[CONTEXT_LAST_TTS] = last_tts
    reply = match_instant_command(text, cfg, db, context)
    if reply is not None:
        return (True, reply)
    return (False, None)


def match_instant_command(
    text: str,
    cfg,
    db,
    context: Optional[dict[str, Any]] = None,
) -> Optional[str]:
    """Try to handle *text* as an instant command.

    Iterates the registry in registration order. The first handler whose
    pattern matches and returns a non-None reply wins.

    Args:
        text: The user's query (used as-is, handler receives lower-cased).
        cfg: Application config (for locale, format preferences).
        db: Database instance (for data-backed handlers).
        context: Optional extras the listener injects, e.g.
                 ``{CONTEXT_LAST_TTS: "..."}`` for repeat commands.

    Returns:
        A reply string if matched, or ``None`` to fall through to the LLM.
    """
    text_lower = text.strip().lower()
    if not text_lower:
        return None

    for name, pattern, handler in _registry:
        if pattern.search(text_lower):
            try:
                reply = handler(text_lower, cfg, db, context or {})
                if reply is not None:
                    return reply
            except Exception:
                continue

    return None


# ---------------------------------------------------------------------------
# Built-in handlers
# ---------------------------------------------------------------------------

@register_command("time", r"what(?:'s| is) the time|what time is it|tell me the time")
def _handle_time(text: str, cfg, db, context: dict) -> Optional[str]:
    now = datetime.now()
    return f"The current time is {now:%I:%M %p}."


@register_command("date", r"what(?:'s| is) the date|what day is it(?: today)?|what(?:'s| is) today(?:'s date)?|tell me the date")
def _handle_date(text: str, cfg, db, context: dict) -> Optional[str]:
    now = datetime.now()
    return f"Today is {now:%A, %B %d, %Y}."


@register_command("greeting", r"^(?:hello|hi|hey)\b")
def _handle_greeting(text: str, cfg, db, context: dict) -> Optional[str]:
    hour = datetime.now().hour
    if hour < 12:
        period = "morning"
    elif hour < 17:
        period = "afternoon"
    else:
        period = "evening"
    return f"Good {period}! How can I help you?"


@register_command("thanks", r"thanks|thank you")
def _handle_thanks(text: str, cfg, db, context: dict) -> Optional[str]:
    return "You are welcome!"


@register_command("farewell", r"^(?:good)?bye\b|see you later|see ya|talk to you later")
def _handle_farewell(text: str, cfg, db, context: dict) -> Optional[str]:
    return "Goodbye! Talk to you later."


@register_command("who_are_you", r"who are you|what are you|introduce yourself|tell me about yourself")
def _handle_who(text: str, cfg, db, context: dict) -> Optional[str]:
    return (
        "I am Jarvis, your personal AI assistant. "
        "I run completely locally on your machine, so your data stays private. "
        "I can answer questions, search the web, manage tools, and help with tasks."
    )


@register_command("capabilities", r"what can you do|what are your capabilities|what features|help(?: me)?$")
def _handle_capabilities(text: str, cfg, db, context: dict) -> Optional[str]:
    return (
        "I can answer questions using my local knowledge, search the web for up-to-date "
        "information, manage your calendar and tasks, control applications through tools, "
        "and remember things you tell me. Just ask me anything."
    )


@register_command("how_are_you", r"how are you|how(?:'s| is) it going|how are you doing")
def _handle_how_are_you(text: str, cfg, db, context: dict) -> Optional[str]:
    return "I am doing well, thank you for asking. How can I help you?"


@register_command("repeat", r"repeat (?:that|yourself)|say that again|what did you say|can you repeat|say it again")
def _handle_repeat(text: str, cfg, db, context: dict) -> Optional[str]:
    last_tts = context.get(CONTEXT_LAST_TTS)
    if last_tts and last_tts.strip():
        return last_tts
    return "I do not have anything to repeat."


@register_command("praise", r"good(?: job| work| one)|well done|you(?:'?:?re| are) (?:great|awesome|amazing|fantastic|the best)|nice work|great job")
def _handle_praise(text: str, cfg, db, context: dict) -> Optional[str]:
    return "Thank you. I am glad I could help."
