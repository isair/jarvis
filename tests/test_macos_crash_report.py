"""
Tests for collecting the macOS native crash report (`.ips`) so that
"Fatal Python error: Aborted" crashes (which faulthandler cannot explain —
it only dumps Python frames) become diagnosable from the app's own crash
report. See #584/#575/#576: the SIGABRT source is a C-level abort in the
main Qt thread whose native stack is only recorded in
`~/Library/Logs/DiagnosticReports/Jarvis-*.ips`.
"""

import json
from pathlib import Path

from unittest.mock import patch


def _write_ips(path: Path, mtime: float, *, app_name: str = "Jarvis",
               exc_type: str = "SIGABRT", indicator: str = "DirtyVM_FLUSH",
               frames: list) -> None:
    """Write a minimal but structurally valid Apple .ips crash report."""
    body = {
        "app_name": app_name,
        "timestamp": "2026-08-08 13:35:39.539733",
        "exception": {"type": exc_type, "signal": 6},
        "termination": {"indicator": indicator},
        "threads": [
            {"id": 1, "frames": [{"imageOffset": 0, "symbol": f} for f in frames]},
            {"triggered": True, "id": 2,
             "frames": [{"imageOffset": i * 16, "symbol": s}
                        for i, s in enumerate(frames)]},
        ],
    }
    # .ips files are two-line JSON-ish: a metadata line then the payload.
    path.write_text(
        json.dumps({"app_name": app_name}) + "\n" + json.dumps(body),
        encoding="utf-8",
    )
    import os
    os.utime(path, (mtime, mtime))


def _fresh_mtime() -> float:
    """An mtime safely newer than any crash log written during the test."""
    import time
    return time.time() + 60


class TestCollectMacosCrashReport:
    def test_returns_none_when_diagnostics_dir_missing(self, tmp_path):
        from desktop_app.app import collect_macos_crash_report
        crash_log = tmp_path / "jarvis_desktop_crash.log"
        crash_log.write_text("x")
        missing = tmp_path / "nope"
        assert collect_macos_crash_report(crash_log, diagnostics_dir=missing) is None

    def test_returns_none_when_dir_empty(self, tmp_path):
        from desktop_app.app import collect_macos_crash_report
        crash_log = tmp_path / "jarvis_desktop_crash.log"
        crash_log.write_text("x")
        assert collect_macos_crash_report(crash_log, diagnostics_dir=tmp_path) is None

    def test_returns_none_when_no_matching_app(self, tmp_path):
        from desktop_app.app import collect_macos_crash_report
        crash_log = tmp_path / "jarvis_desktop_crash.log"
        crash_log.write_text("x")
        other = tmp_path / "OtherApp-2026-08-08-133539.ips"
        _write_ips(other, mtime=100.0, app_name="OtherApp", frames=["main"])
        assert collect_macos_crash_report(crash_log, diagnostics_dir=tmp_path) is None

    def test_returns_none_when_report_older_than_crash_log(self, tmp_path):
        from desktop_app.app import collect_macos_crash_report
        crash_log = tmp_path / "jarvis_desktop_crash.log"
        crash_log.write_text("x")
        # Crash report written BEFORE this session's log: stale, ignore it.
        stale = tmp_path / "Jarvis-2026-07-01-000000.ips"
        _write_ips(stale, mtime=50.0, frames=["main"])
        assert collect_macos_crash_report(crash_log, diagnostics_dir=tmp_path) is None

    def test_extracts_native_stack_from_fresh_report(self, tmp_path):
        from desktop_app.app import collect_macos_crash_report
        crash_log = tmp_path / "jarvis_desktop_crash.log"
        crash_log.write_text("x")
        ips = tmp_path / "Jarvis-2026-08-08-133539.ips"
        frames = ["__pthread_kill", "abort", "qt_message_fatal",
                  "QThread::start", "main"]
        _write_ips(ips, mtime=_fresh_mtime(), frames=frames)
        result = collect_macos_crash_report(crash_log, diagnostics_dir=tmp_path)
        assert result is not None
        assert "SIGABRT" in result
        assert "DirtyVM_FLUSH" in result
        assert "__pthread_kill" in result and "abort" in result
        assert "qt_message_fatal" in result
        assert ips.name in result

    def test_handles_pretty_printed_multiline_payload(self, tmp_path):
        """Real .ips payloads can be pretty-printed across many lines."""
        from desktop_app.app import collect_macos_crash_report
        crash_log = tmp_path / "jarvis_desktop_crash.log"
        crash_log.write_text("x")
        ips = tmp_path / "Jarvis-2026-08-08-133539.ips"
        payload = (
            '{\n  "app_name": "Jarvis",\n'
            '  "exception": {"type": "SIGABRT"},\n'
            '  "threads": [{"triggered": true, "frames": ['
            '{"symbol": "abort"}, {"symbol": "main"}]}]\n}'
        )
        ips.write_text('{"app_name": "Jarvis"}\n' + payload)
        import os as _os
        _os.utime(ips, (_fresh_mtime(), _fresh_mtime()))
        result = collect_macos_crash_report(crash_log, diagnostics_dir=tmp_path)
        assert result is not None
        assert "SIGABRT" in result and "abort" in result

    def test_ignores_non_string_symbols(self, tmp_path):
        """Frames without a demangled symbol render as '?' — never fall back
        to the integer ``symbolLocation`` byte offset."""
        from desktop_app.app import collect_macos_crash_report
        crash_log = tmp_path / "jarvis_desktop_crash.log"
        crash_log.write_text("x")
        ips = tmp_path / "Jarvis-2026-08-08-133539.ips"
        _write_ips(ips, mtime=_fresh_mtime(),
                   frames=["abort", "__pthread_kill", "main"])
        # Replace the payload with one whose frames carry only symbolLocation.
        import os as _os
        body = {
            "app_name": "Jarvis",
            "exception": {"type": "SIGABRT"},
            "threads": [{"triggered": True, "id": 2,
                         "frames": [{"imageOffset": 4096}]}],
        }
        ips.write_text('{"app_name": "Jarvis"}\n' + json.dumps(body))
        _os.utime(ips, (_fresh_mtime(), _fresh_mtime()))
        result = collect_macos_crash_report(crash_log, diagnostics_dir=tmp_path)
        assert result is not None and "?" in result
        assert "4096" not in result

    def test_picks_newest_report(self, tmp_path):
        from desktop_app.app import collect_macos_crash_report
        crash_log = tmp_path / "jarvis_desktop_crash.log"
        crash_log.write_text("x")
        older = tmp_path / "Jarvis-2026-08-08-100000.ips"
        newer = tmp_path / "Jarvis-2026-08-08-133539.ips"
        _write_ips(older, mtime=_fresh_mtime() - 10, frames=["old_main"])
        _write_ips(newer, mtime=_fresh_mtime(), frames=["new_main"])
        result = collect_macos_crash_report(crash_log, diagnostics_dir=tmp_path)
        assert result is not None and "new_main" in result and "old_main" not in result

    def test_malformed_report_returns_none(self, tmp_path):
        from desktop_app.app import collect_macos_crash_report
        crash_log = tmp_path / "jarvis_desktop_crash.log"
        crash_log.write_text("x")
        bad = tmp_path / "Jarvis-2026-08-08-133539.ips"
        bad.write_text("not json at all")
        import os as _os
        _os.utime(bad, (_fresh_mtime(), _fresh_mtime()))
        assert collect_macos_crash_report(crash_log, diagnostics_dir=tmp_path) is None

    def test_skipped_on_non_macos(self, tmp_path):
        from desktop_app.app import collect_macos_crash_report
        crash_log = tmp_path / "jarvis_desktop_crash.log"
        crash_log.write_text("x")
        ips = tmp_path / "Jarvis-2026-08-08-133539.ips"
        _write_ips(ips, mtime=_fresh_mtime(), frames=["main"])
        with patch.object(Path, "__str__", return_value="x"):
            pass
        with patch("desktop_app.app.sys.platform", "win32"):
            assert collect_macos_crash_report(
                crash_log, diagnostics_dir=tmp_path) is None


class TestPreviousCrashIncludesNativeStack:
    def test_previous_crash_content_includes_native_stack(self, tmp_path, monkeypatch):
        """check_previous_crash() must surface the macOS native stack so the
        crash dialog / report-issue body carries the C-level abort source."""
        from desktop_app import app as desktop_app
        crash_log, crash_marker, _ = desktop_app.get_crash_paths()

        # Redirect paths to a temp dir for this test.
        monkeypatch.setattr(desktop_app, "get_crash_paths",
                            lambda: (tmp_path / "jarvis_desktop_crash.log",
                                     tmp_path / ".crash_marker",
                                     tmp_path / "previous_crash.log"))
        crash_log, crash_marker, _ = desktop_app.get_crash_paths()

        crash_log.write_text("=== Jarvis Desktop App Crash Log ===\nFatal Python error: Aborted\n")
        crash_marker.touch()

        ips = tmp_path / "Jarvis-2026-08-08-133539.ips"
        _write_ips(ips, mtime=_fresh_mtime(), frames=["__pthread_kill", "abort", "qt_message_fatal"])
        # Point the wiring's collector at the temp diagnostics dir so the
        # end-to-end path (check_previous_crash -> collector -> content) is
        # exercised without touching the real home directory.
        _real_collect = desktop_app.collect_macos_crash_report
        monkeypatch.setattr(
            desktop_app, "collect_macos_crash_report",
            lambda crash_log: _real_collect(crash_log, diagnostics_dir=tmp_path),
        )
        with patch.object(desktop_app.sys, "platform", "darwin"):
            content = desktop_app.check_previous_crash()
        assert content is not None
        assert "Fatal Python error: Aborted" in content
        assert "Native crash report" in content
        assert "__pthread_kill" in content
