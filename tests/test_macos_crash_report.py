"""Synthetic native crash reports identify the applicable report and crashed thread."""

import json
import os
from types import SimpleNamespace

import pytest

from desktop_app import app as desktop

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("desktop_log_home", "no_network")]


@pytest.fixture(autouse=True)
def macos_platform(monkeypatch):
    owned_sys = SimpleNamespace(**vars(desktop.sys))
    owned_sys.platform = "darwin"
    monkeypatch.setattr(desktop, "sys", owned_sys)


@pytest.fixture
def crash_log(tmp_path):
    path = tmp_path / "jarvis_desktop_crash.log"
    path.write_text("Fatal Python error: Aborted\n")
    return path


def _write_ips(path, mtime, frames, *, indent=None):
    body = {
        "app_name": "Jarvis", "exception": {"type": "SIGABRT"},
        "termination": {"indicator": "Synthetic termination"},
        "threads": [{"id": 1, "frames": [{"symbol": "idle_worker"}]},
                    {"id": 2, "triggered": True, "frames": frames}],
    }
    path.write_text(json.dumps({"app_name": "Jarvis"}) + "\n" + json.dumps(body, indent=indent))
    os.utime(path, (mtime, mtime))


@pytest.mark.parametrize("condition", ["missing", "empty", "other_app", "stale", "malformed"])
def test_unavailable_or_inapplicable_report_returns_none(crash_log, tmp_path, condition):
    directory = tmp_path / "missing" if condition == "missing" else tmp_path
    fresh = crash_log.stat().st_mtime + 10
    if condition in {"other_app", "stale"}:
        name = "OtherApp-fixture.ips" if condition == "other_app" else "Jarvis-fixture.ips"
        _write_ips(tmp_path / name, fresh if condition == "other_app" else fresh - 20,
                   [{"symbol": "abort"}])
    elif condition == "malformed":
        report = tmp_path / "Jarvis-fixture.ips"
        report.write_text("not json")
        os.utime(report, (fresh, fresh))
    assert desktop.collect_macos_crash_report(crash_log, diagnostics_dir=directory) is None


@pytest.mark.parametrize("indent", [None, 2])
def test_fresh_report_includes_crashed_thread_and_exception(crash_log, tmp_path, indent):
    report = tmp_path / "Jarvis-fixture.ips"
    frames = [{"symbol": symbol} for symbol in ("__pthread_kill", "abort", "qt_message_fatal")]
    _write_ips(report, crash_log.stat().st_mtime + 10, frames, indent=indent)
    result = desktop.collect_macos_crash_report(crash_log, diagnostics_dir=tmp_path)
    assert result is not None
    assert report.name in result and "SIGABRT" in result and "Synthetic termination" in result
    for frame in frames:
        assert frame["symbol"] in result
    assert "idle_worker" not in result


@pytest.mark.parametrize("symbol", [None, "", 123, ["not a symbol"]])
def test_invalid_symbols_are_unknown_without_exposing_numeric_offsets(crash_log, tmp_path, symbol):
    _write_ips(tmp_path / "Jarvis-fixture.ips", crash_log.stat().st_mtime + 10,
               [{"symbol": symbol, "symbolLocation": 4096, "imageOffset": 8192}])
    result = desktop.collect_macos_crash_report(crash_log, diagnostics_dir=tmp_path)
    assert result is not None and "?" in result
    assert "4096" not in result and "8192" not in result and "idle_worker" not in result


def test_newest_applicable_report_wins(crash_log, tmp_path):
    fresh = crash_log.stat().st_mtime + 10
    _write_ips(tmp_path / "Jarvis-older.ips", fresh, [{"symbol": "old_main"}])
    _write_ips(tmp_path / "Jarvis-newer.ips", fresh + 10, [{"symbol": "new_main"}])
    result = desktop.collect_macos_crash_report(crash_log, diagnostics_dir=tmp_path)
    assert result is not None and "new_main" in result and "old_main" not in result


def test_only_the_requested_number_of_crashed_frames_is_reported(crash_log, tmp_path):
    frames = [{"symbol": f"crash_frame_{i}"} for i in range(6)]
    limit = len(frames) // 2
    _write_ips(tmp_path / "Jarvis-fixture.ips", crash_log.stat().st_mtime + 10, frames)
    result = desktop.collect_macos_crash_report(crash_log, diagnostics_dir=tmp_path, max_frames=limit)
    assert result is not None and "idle_worker" not in result
    for index, frame in enumerate(frames):
        assert (frame["symbol"] in result) == (index < limit)


def test_native_reports_are_ignored_on_other_platforms(crash_log, tmp_path, monkeypatch):
    _write_ips(tmp_path / "Jarvis-fixture.ips", crash_log.stat().st_mtime + 10, [{"symbol": "abort"}])
    monkeypatch.setattr(desktop.sys, "platform", "win32")
    assert desktop.collect_macos_crash_report(crash_log, diagnostics_dir=tmp_path) is None


def test_previous_crash_includes_the_native_crashed_thread(crash_log, tmp_path, monkeypatch):
    marker = tmp_path / ".crash_marker"
    marker.touch()
    monkeypatch.setattr(desktop, "get_crash_paths", lambda: (crash_log, marker, tmp_path / "previous_crash.log"))
    _write_ips(tmp_path / "Jarvis-fixture.ips", crash_log.stat().st_mtime + 10,
               [{"symbol": "__pthread_kill"}, {"symbol": "abort"}])
    collector = desktop.collect_macos_crash_report
    monkeypatch.setattr(desktop, "collect_macos_crash_report",
                        lambda log: collector(log, diagnostics_dir=tmp_path))
    content = desktop.check_previous_crash()
    assert content is not None
    assert "Fatal Python error: Aborted" in content and "Native crash report" in content
    assert "__pthread_kill" in content and "idle_worker" not in content
