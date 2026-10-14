"""Shared native-audio safety for unit tests and eval qualification."""

import importlib
import os
import sys

import pytest


_audio_attempts: set[str] = set()


def _reject_audio_operation(operation):
    def blocked(*args, **kwargs):
        context = os.environ.get("PYTEST_CURRENT_TEST", "test collection or shutdown")
        _audio_attempts.add(f"{operation} ({context})")
        raise AssertionError(f"🔇 Physical audio access is forbidden in tests: {operation}. Use an owned fake.")
    return blocked


def pytest_sessionstart(session):
    """Keep native audio entry points blocked for the test process lifetime."""
    _audio_attempts.clear()
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    operations = {
        "sounddevice": ("play", "rec", "playrec", "InputStream", "OutputStream",
                        "RawInputStream", "RawOutputStream", "Stream", "RawStream"),
        "winsound": ("PlaySound", "Beep", "MessageBeep"),
        "pygame": ("init", "mixer.init", "mixer.music.play"),
    }
    for backend, names in operations.items():
        try:
            module = importlib.import_module(backend)
        except (ImportError, OSError):
            continue
        for name in names:
            target = module
            parts = name.split(".")
            for part in parts[:-1]:
                target = getattr(target, part, None) if target else None
            if target and hasattr(target, parts[-1]):
                setattr(target, parts[-1], _reject_audio_operation(f"{backend}.{name}"))


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_sessionfinish(session, exitstatus):
    """Caught audio errors in all shutdown hooks still make qualification fail."""
    yield
    if not _audio_attempts:
        return
    message = "🔇 Physical audio access rejected during tests:\n" + "\n".join(
        f"  🔇 {attempt}" for attempt in sorted(_audio_attempts)
    )
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_line(message)
    else:
        sys.stderr.write(message + "\n")
    if session.exitstatus == pytest.ExitCode.OK:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


