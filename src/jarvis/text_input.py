"""Cross-process text queries to Jarvis (typed input, no wake word).

When the voice listener runs in-process, queries go to its queue immediately.
Otherwise lines are appended to ``~/.config/jarvis/text_inbox.jsonl`` for the
daemon to drain.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Optional

from .debug import debug_log

if TYPE_CHECKING:
    from .listening.listener import VoiceListener

TextDeliveryFn = Callable[[str, Optional[list[str]]], bool]

_inbox_lock = threading.Lock()
_listener: Optional["VoiceListener"] = None
_external_delivery: Optional[TextDeliveryFn] = None


def inbox_path() -> Path:
    return Path.home() / ".config" / "jarvis" / "text_inbox.jsonl"


def register_voice_listener(listener: "VoiceListener") -> None:
    """Called when the voice listener thread is ready."""
    global _listener
    _listener = listener
    _drain_inbox_file_into_listener(listener)


def unregister_voice_listener(listener: "VoiceListener") -> None:
    global _listener
    if _listener is listener:
        _listener = None


def register_text_delivery(fn: TextDeliveryFn) -> None:
    """Desktop app: deliver via daemon stdin before inbox fallback."""
    global _external_delivery
    _external_delivery = fn


def unregister_text_delivery(fn: TextDeliveryFn) -> None:
    global _external_delivery
    if _external_delivery is fn:
        _external_delivery = None


def deliver_text_query(text: str, image_paths: Optional[list[str]] = None) -> str:
    """Queue a typed message. Returns delivery channel: stdin, listener, inbox, or empty."""
    cleaned = (text or "").strip()
    images = [str(p) for p in (image_paths or []) if p and str(p).strip()]
    if not cleaned and not images:
        return ""

    deliver = _external_delivery
    if deliver is not None:
        try:
            if deliver(cleaned, images or None):
                debug_log(
                    f"text query delivered via desktop hook: {cleaned[:80]!r}",
                    "text_input",
                )
                return "stdin"
        except Exception as exc:
            debug_log(f"external text delivery failed: {exc}", "text_input")

    listener = _listener
    if listener is not None:
        listener.enqueue_text_query(cleaned, image_paths=images)
        debug_log(
            f"text query queued in-process: {cleaned[:80]!r} images={len(images)}",
            "text_input",
        )
        return "listener"

    _append_inbox_line(cleaned, images)
    debug_log(
        f"text query written to inbox: {cleaned[:80]!r} images={len(images)}",
        "text_input",
    )
    return "inbox"


def submit_text_query(text: str, image_paths: Optional[list[str]] = None) -> bool:
    """Queue a typed user message. Returns False if text and images are both empty."""
    return bool(deliver_text_query(text, image_paths=image_paths))


def process_inbox_file() -> int:
    """Drain pending inbox lines into the registered listener. Returns count."""
    listener = _listener
    if listener is None:
        return 0
    return _drain_inbox_file_into_listener(listener)


def _append_inbox_line(text: str, image_paths: Optional[list[str]] = None) -> None:
    path = inbox_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict = {"text": text}
    if image_paths:
        payload["images"] = image_paths
    line = json.dumps(payload, ensure_ascii=False)
    with _inbox_lock:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def _drain_inbox_file_into_listener(listener: "VoiceListener") -> int:
    path = inbox_path()
    if not path.is_file():
        return 0

    with _inbox_lock:
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError:
            return 0
        try:
            path.unlink()
        except OSError:
            pass

    count = 0
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            payload = {"text": line}
        text = str(payload.get("text") or "").strip()
        images = payload.get("images") or []
        if not isinstance(images, list):
            images = []
        image_paths = [str(p) for p in images if p]
        if text or image_paths:
            listener.enqueue_text_query(text, image_paths=image_paths)
            count += 1
    if count:
        debug_log(f"drained {count} text query(s) from inbox file", "text_input")
    return count
