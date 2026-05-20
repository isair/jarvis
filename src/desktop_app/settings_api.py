"""Settings metadata and save helpers for the Jarvis shell (non-PyQt)."""

from __future__ import annotations

import json
from typing import Any

from jarvis.config import (
    default_config_path,
    get_default_config,
    load_config,
    _save_json,
)
from jarvis.debug import debug_log

from desktop_app.settings_window import CATEGORIES, FIELD_METADATA

# Categories without custom PyQt-only pages (MCP uses separate UI).
_SHELL_CATEGORIES = [c for c, _ in CATEGORIES if c != "mcps"]


def _audio_device_choices() -> list[dict[str, str]]:
    try:
        import sounddevice as sd

        choices = [{"value": "", "label": "System default"}]
        for idx, dev in enumerate(sd.query_devices()):
            try:
                if int(dev.get("max_input_channels", 0)) > 0:
                    name = str(dev.get("name", f"Device {idx}"))
                    choices.append({"value": str(idx), "label": f"[{idx}] {name}"})
            except Exception:
                continue
        return choices
    except Exception as exc:
        debug_log(f"settings device list skipped: {exc}", "desktop")
        return [{"value": "", "label": "System default"}]


def _field_to_dict(fm: Any) -> dict[str, Any]:
    from desktop_app.settings_window import FieldMeta

    assert isinstance(fm, FieldMeta)
    out: dict[str, Any] = {
        "key": fm.key,
        "label": fm.label,
        "description": fm.description,
        "category": fm.category,
        "fieldType": fm.field_type,
        "nullable": fm.nullable,
    }
    if fm.min_val is not None:
        out["min"] = fm.min_val
    if fm.max_val is not None:
        out["max"] = fm.max_val
    if fm.step is not None:
        out["step"] = fm.step
    if fm.suffix:
        out["suffix"] = fm.suffix
    if fm.field_type == "device":
        out["choices"] = _audio_device_choices()
    elif fm.choices:
        out["choices"] = [
            {"value": str(v), "label": str(lbl)} for v, lbl in fm.choices
        ]
    return out


def export_settings_bundle() -> dict[str, Any]:
    fields = [
        _field_to_dict(fm)
        for fm in FIELD_METADATA
        if fm.category in _SHELL_CATEGORIES
    ]
    categories = [
        {"id": cat_id, "label": label}
        for cat_id, label in CATEGORIES
        if cat_id in _SHELL_CATEGORIES
    ]
    return {
        "ok": True,
        "categories": categories,
        "fields": fields,
        "configPath": str(default_config_path()),
    }


def build_default_values() -> dict[str, Any]:
    """Default config values for shell reset (same keys as the form)."""
    defaults = get_default_config()
    values: dict[str, Any] = {}
    for fm in FIELD_METADATA:
        if fm.category not in _SHELL_CATEGORIES:
            continue
        values[fm.key] = defaults.get(fm.key)
    return values


def build_merged_values() -> dict[str, Any]:
    """Effective config: defaults overlaid with config.json."""
    defaults = get_default_config()
    merged = dict(defaults)
    merged.update(load_config())
    values: dict[str, Any] = {}
    for fm in FIELD_METADATA:
        if fm.category not in _SHELL_CATEGORIES:
            continue
        values[fm.key] = merged.get(fm.key, defaults.get(fm.key))
    return values


def _coerce_submitted(fm: Any, raw: Any) -> Any:
    from desktop_app.settings_window import FieldMeta

    assert isinstance(fm, FieldMeta)
    if fm.field_type == "bool":
        if isinstance(raw, bool):
            return raw
        return str(raw).lower() in ("1", "true", "yes", "on")
    if fm.field_type == "int":
        if raw is None or raw == "":
            return None if fm.nullable else 0
        return int(raw)
    if fm.field_type == "float":
        if raw is None or raw == "":
            return None if fm.nullable else 0.0
        return float(raw)
    if fm.field_type == "list":
        if isinstance(raw, list):
            return raw
        if isinstance(raw, str):
            text = raw.strip()
            if not text:
                return []
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass
            return [line.strip() for line in text.splitlines() if line.strip()]
        return []
    if fm.field_type in ("choice", "device"):
        if raw is None or raw == "":
            return None if fm.nullable else ""
        if fm.key == "sample_rate":
            try:
                return int(raw)
            except (TypeError, ValueError):
                return 16000
        return str(raw)
    text = "" if raw is None else str(raw).strip()
    if fm.nullable and text == "":
        return None
    return text


def save_settings_from_values(values: dict[str, Any]) -> tuple[bool, str]:
    """Persist settings using the same rules as the PyQt settings window."""
    defaults = get_default_config()
    config = dict(load_config())

    for fm in FIELD_METADATA:
        if fm.category not in _SHELL_CATEGORIES:
            continue
        if fm.key not in values:
            continue
        val = _coerce_submitted(fm, values[fm.key])
        default_val = defaults.get(fm.key)
        if val == default_val or (val is None and default_val is None):
            config.pop(fm.key, None)
        else:
            config[fm.key] = val

    path = default_config_path()
    if _save_json(path, config):
        debug_log("settings saved via shell API", "desktop")
        return True, f"Saved to {path}. Restart listening for voice changes."
    return False, f"Could not write {path}"
