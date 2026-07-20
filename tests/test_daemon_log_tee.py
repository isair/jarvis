"""Diagnostic tee for the daemon's log stream.

The daemon writes into a pipe owned by desktop_app (consumed for the Log
Viewer), so no external redirection can capture it — that is why calibration
had nothing to measure. The tee mirrors each line to disk *after* the Log
Viewer receives it, and only when JARVIS_VOICE_DEBUG=1.
"""

import io
import os
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.desktop_app.app import JarvisSystemTray


class _FakeStdout:
    def __init__(self, lines):
        self._lines = list(lines)

    def readline(self):
        return self._lines.pop(0) if self._lines else ""


def _tray(lines, tmp_path):
    tray = JarvisSystemTray.__new__(JarvisSystemTray)
    tray.daemon_process = Mock()
    tray.daemon_process.stdout = _FakeStdout(lines)
    tray.log_signals = Mock()
    tray.log_signals.new_log = Mock()
    tray._CALIB_LOG_NAME = "_calib-daemon.log"
    tray._CALIB_LOG_MAX_BYTES = 20 * 1024 * 1024
    return tray


def _tee_to(tmp_path, monkeypatch, tray):
    """Point the tee at tmp_path instead of the repo root."""
    target = tmp_path / "_calib-daemon.log"
    real_open = JarvisSystemTray._open_calib_tee

    def patched(self):
        if os.environ.get("JARVIS_VOICE_DEBUG") != "1":
            return None
        return open(target, "a", encoding="utf-8", buffering=1, errors="replace")

    monkeypatch.setattr(JarvisSystemTray, "_open_calib_tee", patched)
    return target


LINES = [
    "🎙️  Listening!\n",
    '📝 Heard: "Jarvis, ce zi este astăzi"\n',
    "[  voice   ] UTT_SUMMARY id=u0001 rms_p50=0.000086 voiced_ms=420 verdict=ACCEPTED\n",
]


# ------------------------------------------------------------------ 1. no flag

@pytest.mark.unit
def test_no_file_and_identical_behaviour_without_flag(tmp_path, monkeypatch):
    monkeypatch.delenv("JARVIS_VOICE_DEBUG", raising=False)
    target = _tee_to(tmp_path, monkeypatch, None)
    tray = _tray(LINES, tmp_path)

    tray._read_daemon_logs()

    assert not target.exists(), "no file may be created without the flag"
    assert tray.log_signals.new_log.emit.call_count == len(LINES)


# ------------------------------------------------- 2. captures stdout + stderr

@pytest.mark.unit
def test_captures_every_line_with_flag(tmp_path, monkeypatch):
    """Popen merges stderr into stdout, so one stream carries both."""
    monkeypatch.setenv("JARVIS_VOICE_DEBUG", "1")
    target = _tee_to(tmp_path, monkeypatch, None)
    tray = _tray(LINES, tmp_path)

    tray._read_daemon_logs()

    written = target.read_text(encoding="utf-8")
    assert "Listening!" in written                    # stdout-side line
    assert "UTT_SUMMARY" in written                   # stderr-side (debug_log)
    assert "📝 Heard" in written


# --------------------------------------------- 3. Log Viewer still gets lines

@pytest.mark.unit
def test_log_viewer_receives_unchanged_lines(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_VOICE_DEBUG", "1")
    _tee_to(tmp_path, monkeypatch, None)
    tray = _tray(LINES, tmp_path)

    tray._read_daemon_logs()

    emitted = [c.args[0] for c in tray.log_signals.new_log.emit.call_args_list]
    assert emitted == LINES, "UI must receive the original lines, unmodified"


# ---------------------------------------------------- 4. no loss, no deadlock

@pytest.mark.unit
def test_no_lines_lost_under_volume(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_VOICE_DEBUG", "1")
    target = _tee_to(tmp_path, monkeypatch, None)
    many = [f"line {i}\n" for i in range(500)]
    tray = _tray(many, tmp_path)

    tray._read_daemon_logs()

    body = target.read_text(encoding="utf-8")
    assert tray.log_signals.new_log.emit.call_count == 500
    for i in (0, 250, 499):
        assert f"line {i}\n" in body, f"line {i} lost"


@pytest.mark.unit
def test_tee_failure_does_not_break_log_reading(tmp_path, monkeypatch):
    """A broken tee must degrade to Log-Viewer-only, never kill the reader."""
    monkeypatch.setenv("JARVIS_VOICE_DEBUG", "1")

    class _Exploding(io.StringIO):
        def write(self, _s):
            raise OSError("disk full")

    monkeypatch.setattr(JarvisSystemTray, "_open_calib_tee", lambda self: _Exploding())
    tray = _tray(LINES, tmp_path)

    tray._read_daemon_logs()  # must not raise

    assert tray.log_signals.new_log.emit.call_count == len(LINES)


# ------------------------------------------------------------- 5. redaction

@pytest.mark.unit
def test_secrets_are_redacted_before_disk(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_VOICE_DEBUG", "1")
    target = _tee_to(tmp_path, monkeypatch, None)
    secret = "sk-ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    token = "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    tray = _tray([f"key={secret}\n", f"tok={token}\n", "Authorization: Bearer abc.def\n"], tmp_path)

    tray._read_daemon_logs()

    body = target.read_text(encoding="utf-8")
    assert secret not in body, "OpenAI-style key reached disk"
    assert token not in body, "GitHub token reached disk"
    assert "REDACTED" in body

    # ...while the UI still shows the untouched line.
    emitted = "".join(c.args[0] for c in tray.log_signals.new_log.emit.call_args_list)
    assert secret in emitted


# ------------------------------------------------------ 6. handle is closed

@pytest.mark.unit
def test_handle_closed_at_shutdown(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_VOICE_DEBUG", "1")
    opened = {}

    def patched(self):
        h = open(tmp_path / "_calib-daemon.log", "a", encoding="utf-8", buffering=1)
        opened["h"] = h
        return h

    monkeypatch.setattr(JarvisSystemTray, "_open_calib_tee", patched)
    tray = _tray(LINES, tmp_path)

    tray._read_daemon_logs()

    assert opened["h"].closed, "tee handle must be closed when the child ends"


@pytest.mark.unit
def test_handle_closed_even_when_reader_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_VOICE_DEBUG", "1")
    opened = {}

    def patched(self):
        h = open(tmp_path / "_calib-daemon.log", "a", encoding="utf-8", buffering=1)
        opened["h"] = h
        return h

    monkeypatch.setattr(JarvisSystemTray, "_open_calib_tee", patched)

    class _Boom:
        def readline(self):
            raise RuntimeError("pipe died")

    tray = _tray([], tmp_path)
    tray.daemon_process.stdout = _Boom()

    tray._read_daemon_logs()  # handled internally

    assert opened["h"].closed


# ------------------------------------------------------------ size limiting

@pytest.mark.unit
def test_write_cap_is_enforced(tmp_path, monkeypatch):
    monkeypatch.setenv("JARVIS_VOICE_DEBUG", "1")
    target = _tee_to(tmp_path, monkeypatch, None)
    tray = _tray([f"x{i}\n" for i in range(200)], tmp_path)
    tray._CALIB_LOG_MAX_BYTES = 50  # tiny cap for the test

    tray._read_daemon_logs()

    assert target.stat().st_size < 5000, "cap must stop unbounded growth"
    # The UI is never capped.
    assert tray.log_signals.new_log.emit.call_count == 200


# ---------------------------------------------------------- IP redaction

@pytest.mark.unit
def test_public_ip_is_redacted_but_loopback_kept(tmp_path, monkeypatch):
    """redact() carries no IP rule, so the tee adds one — the daemon logs the
    machine's public address on every start and these files get shared.

    Loopback and RFC1918 ranges are deliberately kept: they are diagnostic
    signal (which Ollama port, which LAN peer), not personal data.
    """
    monkeypatch.setenv("JARVIS_VOICE_DEBUG", "1")
    target = _tee_to(tmp_path, monkeypatch, None)
    tray = _tray([
        "[ location ] Public IP resolved via OpenDNS: 203.0.113.45\n",
        "ollama at http://127.0.0.1:11434\n",
        "lan peer 192.168.1.10\n",
    ], tmp_path)

    tray._read_daemon_logs()

    body = target.read_text(encoding="utf-8")
    assert "203.0.113.45" not in body, "public IP reached disk"
    assert "[REDACTED_IP]" in body
    assert "127.0.0.1" in body, "loopback must stay — diagnostic signal"
    assert "192.168.1.10" in body, "private range must stay"

    # The Log Viewer is unaffected; redaction applies to the disk copy only.
    emitted = "".join(c.args[0] for c in tray.log_signals.new_log.emit.call_args_list)
    assert "203.0.113.45" in emitted
