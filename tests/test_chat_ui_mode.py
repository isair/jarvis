"""Phase 3A: chat_ui_mode config (fail-closed) + app.py mode-selector wiring."""

from __future__ import annotations

import json
import os
import re

from jarvis.config import load_settings

APP_SRC = os.path.join(
    os.path.dirname(__file__), "..", "src", "desktop_app", "app.py"
)


def _settings_with(tmp_path, cfg: dict):
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    old = os.environ.get("JARVIS_CONFIG_PATH")
    os.environ["JARVIS_CONFIG_PATH"] = str(p)
    try:
        return load_settings()
    finally:
        if old is None:
            os.environ.pop("JARVIS_CONFIG_PATH", None)
        else:
            os.environ["JARVIS_CONFIG_PATH"] = old


# --- config fail-closed -----------------------------------------------------

def test_default_mode_is_classic(tmp_path):
    assert _settings_with(tmp_path, {}).chat_ui_mode == "classic"


def test_modern_mode_parsed(tmp_path):
    assert _settings_with(tmp_path, {"chat_ui_mode": "modern"}).chat_ui_mode == "modern"


def test_case_and_whitespace_normalised(tmp_path):
    assert _settings_with(tmp_path, {"chat_ui_mode": "  MODERN "}).chat_ui_mode == "modern"


def test_unknown_mode_fails_closed_to_classic(tmp_path):
    for bad in ["bogus", "web", "electron", "", "123", "moderns"]:
        assert _settings_with(tmp_path, {"chat_ui_mode": bad}).chat_ui_mode == "classic", bad


def test_null_mode_fails_closed(tmp_path):
    assert _settings_with(tmp_path, {"chat_ui_mode": None}).chat_ui_mode == "classic"


# --- app.py structural wiring ----------------------------------------------

def _src():
    with open(APP_SRC, encoding="utf-8") as f:
        return f.read()


def test_show_chat_reads_mode_fail_closed():
    s = _src()
    body = s.split("def show_chat", 1)[1].split("\n    def ", 1)[0]
    assert 'load_config().get("chat_ui_mode", "classic")' in body
    assert 'if mode not in ("classic", "modern")' in body
    assert 'mode = "classic"' in body  # fail-closed default


def test_show_chat_has_both_windows():
    s = _src()
    body = s.split("def show_chat", 1)[1].split("\n    def ", 1)[0]
    assert "ModernChatWindow" in body and "ChatWindow" in body
    # modern only under the modern branch; classic is the else fallback
    assert 'if mode == "modern"' in body and "else:" in body


def test_modern_window_attr_initialised():
    s = _src()
    assert "self.modern_chat_window = None" in s


def test_chat_action_still_gated_on_chat_ui_enabled():
    # Phase 3A must not weaken the Phase 2 feature gate.
    s = _src()
    assert 'load_config().get("chat_ui_enabled", False)' in s
