"""Typed local timer-alarm events shared by the daemon and desktop."""
import json

TIMER_ALARM_IPC_PREFIX = "__TIMER_ALARM__:"
TIMER_ALARM_MAX_SECONDS = 30


def parse_alarm_event(line: str) -> dict | None:
    """Accept known alarm events whose fields are safe for desktop dispatch."""
    line = line.strip()
    if not line.startswith(TIMER_ALARM_IPC_PREFIX):
        return None
    try:
        event = json.loads(line[len(TIMER_ALARM_IPC_PREFIX):])
    except (ValueError, RecursionError):
        return None
    if not isinstance(event, dict) or event.get("type") not in ("start", "stop"):
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
    return event
