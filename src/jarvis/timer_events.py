"""Typed local timer-alarm events shared by the daemon and desktop."""
import json
import hmac
import os
import secrets

TIMER_ALARM_IPC_PREFIX = "__TIMER_ALARM__:"
TIMER_ALARM_MAX_SECONDS = 30
_ALARM_TOKEN = os.environ.get("JARVIS_TIMER_ALARM_TOKEN") or secrets.token_hex(32)


def alarm_event_token() -> str:
    """Share this desktop process capability only with its owned daemon."""
    return _ALARM_TOKEN


def encode_alarm_event(event_type: str, data: dict) -> str:
    """Frame a genuine event independently of conversational log text."""
    return TIMER_ALARM_IPC_PREFIX + json.dumps({"type": event_type, "data": data, "token": _ALARM_TOKEN})


def parse_alarm_event(line: str) -> dict | None:
    """Accept known alarm events whose fields are safe for desktop dispatch."""
    line = line.rstrip("\r\n")
    if not line.startswith(TIMER_ALARM_IPC_PREFIX):
        return None
    try:
        event = json.loads(line[len(TIMER_ALARM_IPC_PREFIX):])
    except (ValueError, RecursionError):
        return None
    if not isinstance(event, dict) or event.get("type") not in ("start", "stop"):
        return None
    token = event.get("token")
    if not isinstance(token, str) or not token.isascii() or not hmac.compare_digest(token, _ALARM_TOKEN):
        return None
    data = event.get("data")
    if not isinstance(data, dict):
        return None
    identity = data.get("id")
    if identity is not None and (not isinstance(identity, str) or not identity):
        return None
    if event["type"] == "start":
        if not identity:
            return None
        if any(data.get(key) is not None and not isinstance(data[key], str)
               for key in ("label", "duration_human")):
            return None
        cap = data.get("auto_stop_sec")
        if cap is not None and (type(cap) is not int or not 1 <= cap <= TIMER_ALARM_MAX_SECONDS):
            return None
    else:
        if "all" in data and type(data["all"]) is not bool:
            return None
        if not identity and not data.get("all", False):
            return None
    return {"type": event["type"], "data": data}
