"""Guard Qt Cocoa tray activation against non-mouse AppKit events."""
from __future__ import annotations

import sys

from jarvis.debug import debug_log

_installed = False


def _guard_tray_callback(original, current_event, mouse_types):
    def guarded(sender, *arguments):
        event = current_event()
        if event is not None and event.type() in mouse_types:
            original(sender, *arguments)
        else:
            debug_log("ignored non-mouse Qt tray activation", "desktop")
    return guarded


def install_macos_tray_event_guard() -> bool:
    """Install before showing tray icons, after QApplication initialisation."""
    global _installed
    if sys.platform != "darwin":
        return False
    if _installed:
        return True

    try:
        import objc
        import AppKit

        delegate = objc.lookUpClass("QStatusItemDelegate")
        mouse_types = frozenset(getattr(AppKit, name) for name in (
            "NSEventTypeLeftMouseDown", "NSEventTypeLeftMouseUp",
            "NSEventTypeRightMouseDown", "NSEventTypeRightMouseUp",
            "NSEventTypeOtherMouseDown", "NSEventTypeOtherMouseUp",
            "NSEventTypeLeftMouseDragged", "NSEventTypeRightMouseDragged",
            "NSEventTypeOtherMouseDragged", "NSEventTypeMouseMoved",
        ))
        current_event = AppKit.NSApplication.sharedApplication().currentEvent
        methods = []
        for selector, signature in (
            (b"statusItemMenuBeganTracking:", b"v@:@"),
            (b"statusItemClicked", b"v@:"),
        ):
            # Capture the native implementation, not a selector that resolves
            # through the replacement and recursively calls itself.
            original = delegate.instanceMethodForSelector_(selector)
            methods.append(objc.selector(
                _guard_tray_callback(original, current_event, mouse_types),
                selector=selector, signature=signature,
            ))
        objc.classAddMethods(delegate, methods)
    except Exception as error:
        debug_log(f"macOS tray event guard unavailable: {error!r}", "desktop")
        return False

    _installed = True
    debug_log("installed macOS tray non-mouse event guard", "desktop")
    return True
