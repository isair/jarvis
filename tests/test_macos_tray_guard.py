"""Tray callback safety does not prevent ordinary mouse activation."""
from types import SimpleNamespace

import pytest

from desktop_app import macos_tray

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("event", [None, SimpleNamespace(type=lambda: "application"), SimpleNamespace(type=lambda: "keyboard")])
def test_non_mouse_callback_does_not_reach_click_count(event):
    def native_callback(*args):
        pytest.fail("Native tray callback must not read clickCount for this event")
    callback = macos_tray._guard_tray_callback(native_callback, lambda: event, {"mouse"})
    callback(object(), object())


@pytest.mark.parametrize("notification", [(), (object(),)])
def test_mouse_callback_preserves_activation_and_arguments(notification):
    sender = object()
    activations = []
    def native_callback(actual_sender, *arguments):
        activations.append((actual_sender, arguments))
    callback = macos_tray._guard_tray_callback(native_callback, lambda: SimpleNamespace(type=lambda: "mouse"), {"mouse"})
    callback(sender, *notification)
    assert activations == [(sender, notification)]


def test_other_platforms_do_not_install_native_guard(monkeypatch):
    monkeypatch.setattr(macos_tray.sys, "platform", "linux")
    assert macos_tray.install_macos_tray_event_guard() is False


def test_missing_native_bridge_records_a_diagnostic(monkeypatch):
    diagnostics = []
    monkeypatch.setattr(macos_tray.sys, "platform", "darwin")
    monkeypatch.setitem(macos_tray.sys.modules, "objc", None)
    monkeypatch.setattr(macos_tray, "debug_log", lambda message, category: diagnostics.append(message))
    assert macos_tray.install_macos_tray_event_guard() is False
    assert any("tray event guard unavailable" in message for message in diagnostics)
