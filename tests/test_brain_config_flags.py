"""Phase 4 — Cora Brain Foundation config flags.

Every new flag must (a) exist in get_default_config with the documented
default, (b) round-trip through load_settings onto the frozen Settings
dataclass, and (c) be fail-safe/clamped on bad input. All defaults must keep
runtime behaviour unchanged (every gate OFF / inactive).

Isolation: JARVIS_CONFIG_PATH points load_settings at a temp file, so the
live ~/.config/jarvis/config.json is never read or written.
"""

from __future__ import annotations

import json
import dataclasses

import pytest

from src.jarvis.config import get_default_config, load_settings, Settings


BRAIN_BOOL_DEFAULTS = {
    "legacy_knowledge_auto_write_enabled": False,
    "owner_profile_enabled": False,
    "identity_registry_enabled": False,
    "state_memory_enabled": False,
    "memory_require_confirmation": True,   # safety default: require confirm
    "internet_learning_enabled": False,
    "self_eval_enabled": False,
    "owner_triggered_development_enabled": False,
    "audit_panel_enabled": False,
}
BRAIN_OTHER_DEFAULTS = {
    "owner_profile_max_chars": 600,
    "development_agent_provider": "disabled",
}


def _load_with(tmp_path, monkeypatch, cfg: dict | None):
    p = tmp_path / "config.json"
    if cfg is not None:
        p.write_text(json.dumps(cfg), encoding="utf-8")
    # else: leave nonexistent → pure defaults
    monkeypatch.setenv("JARVIS_CONFIG_PATH", str(p))
    return load_settings()


def test_defaults_present_in_default_config():
    d = get_default_config()
    for k, v in {**BRAIN_BOOL_DEFAULTS, **BRAIN_OTHER_DEFAULTS}.items():
        assert k in d, f"{k} missing from get_default_config()"
        assert d[k] == v, f"{k} default {d[k]!r} != expected {v!r}"


def test_dataclass_has_all_brain_fields():
    fields = {f.name for f in dataclasses.fields(Settings)}
    for k in {**BRAIN_BOOL_DEFAULTS, **BRAIN_OTHER_DEFAULTS}:
        assert k in fields, f"Settings dataclass missing field {k}"


def test_load_settings_pure_defaults_all_off(tmp_path, monkeypatch):
    s = _load_with(tmp_path, monkeypatch, None)
    for k, v in BRAIN_BOOL_DEFAULTS.items():
        assert getattr(s, k) == v, f"{k} loaded {getattr(s, k)!r} != {v!r}"
    assert s.owner_profile_max_chars == 600
    assert s.development_agent_provider == "disabled"


def test_flags_round_trip_when_enabled(tmp_path, monkeypatch):
    cfg = {
        "legacy_knowledge_auto_write_enabled": True,
        "owner_profile_enabled": True,
        "identity_registry_enabled": True,
        "state_memory_enabled": True,
        "internet_learning_enabled": True,
        "self_eval_enabled": True,
        "owner_triggered_development_enabled": True,
        "audit_panel_enabled": True,
        "owner_profile_max_chars": 800,
        "development_agent_provider": "claude_cli",
    }
    s = _load_with(tmp_path, monkeypatch, cfg)
    for k in cfg:
        if isinstance(cfg[k], bool):
            assert getattr(s, k) is True
    assert s.owner_profile_max_chars == 800
    assert s.development_agent_provider == "claude_cli"


def test_development_provider_invalid_fails_safe(tmp_path, monkeypatch):
    s = _load_with(tmp_path, monkeypatch, {"development_agent_provider": "rm -rf /; evil"})
    assert s.development_agent_provider == "disabled"


def test_owner_profile_max_chars_clamped(tmp_path, monkeypatch):
    lo = _load_with(tmp_path, monkeypatch, {"owner_profile_max_chars": 1})
    assert lo.owner_profile_max_chars == 120
    hi = _load_with(tmp_path, monkeypatch, {"owner_profile_max_chars": 999999})
    assert hi.owner_profile_max_chars == 4000


def test_memory_require_confirmation_defaults_true_off_is_explicit(tmp_path, monkeypatch):
    # Safety-critical: confirmation is ON unless the owner explicitly disables it.
    s_default = _load_with(tmp_path, monkeypatch, None)
    assert s_default.memory_require_confirmation is True
    s_off = _load_with(tmp_path, monkeypatch, {"memory_require_confirmation": False})
    assert s_off.memory_require_confirmation is False
