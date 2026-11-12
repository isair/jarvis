"""The real daemon reads desktop stdin only in supported launch modes."""

import io
import json
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from jarvis import daemon
from jarvis.config import load_settings

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("no_network")]


@pytest.mark.parametrize("bundled", [False, True])
@pytest.mark.parametrize("platform,ipc,development_reads", [
    ("win32", None, True), ("win32", "0", True), ("win32", "1", True),
    ("darwin", None, False), ("darwin", "0", False), ("darwin", "1", True),
    ("linux", None, False), ("linux", "0", False), ("linux", "1", True),
])
def test_daemon_launch_mode_controls_synthetic_stdin(monkeypatch, tmp_path, platform, ipc, development_reads, bundled):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "_config_version": 3, "db_path": str(tmp_path / "synthetic.sqlite"),
        "sqlite_vss_path": None, "tts_enabled": False, "dictation_enabled": False,
        "location_enabled": False, "mcps": {}, "embedding_model": "",
        "ollama_embed_model": "", "use_stdin": True,
    }), encoding="utf-8")
    monkeypatch.setenv("JARVIS_CONFIG_PATH", str(config))
    cfg = load_settings()
    monkeypatch.setattr(daemon, "load_settings", lambda: cfg)
    monkeypatch.setattr(daemon, "_install_signal_handlers", lambda: None)
    monkeypatch.setattr(daemon, "_check_and_update_diary", lambda *a, **k: None)
    monkeypatch.setattr("jarvis.tools.external.mcp_runtime.shutdown_runtime", lambda: None)
    monkeypatch.setattr(daemon, "VoiceListener", lambda *a, **k: SimpleNamespace(
        start=lambda: None, stop=lambda: None, join=lambda **k: None,
        is_alive=lambda: False,
    ))
    monkeypatch.setattr(daemon, "create_tts_engine", lambda **k: SimpleNamespace(
        enabled=False, stop=lambda: None,
    ))
    for name in ("_global_cfg", "_global_db", "_global_voice_listener",
                 "_global_tts_engine", "_global_dictation_engine",
                 "_global_dialogue_memory", "_warm_profile_graph_listener"):
        monkeypatch.setattr(daemon, name, None)
    monkeypatch.setattr(daemon, "_global_stop_requested", False)

    control_line = "SHUTDOWN\n"
    stream = io.StringIO(control_line)
    owned_sys = SimpleNamespace(**vars(sys))
    owned_sys.platform, owned_sys.frozen, owned_sys.stdin = platform, bundled, stream
    monkeypatch.setattr(daemon, "sys", owned_sys)
    if ipc is None:
        monkeypatch.delenv("JARVIS_STDIN_IPC", raising=False)
    else:
        monkeypatch.setenv("JARVIS_STDIN_IPC", ipc)

    class ControlledThread:
        def __init__(self, *, target, daemon):
            self.target = target
        def start(self):
            self.target()

    owned_threads = SimpleNamespace(**vars(threading))
    owned_threads.Thread = ControlledThread
    monkeypatch.setattr(daemon, "threading", owned_threads)
    owned_clock = SimpleNamespace(**vars(time))
    owned_clock.sleep = lambda seconds: daemon.request_stop()
    monkeypatch.setattr(daemon, "time", owned_clock)

    daemon._run_daemon()
    expected_consumed = len(control_line) if development_reads and not bundled else 0
    assert stream.tell() == expected_consumed
