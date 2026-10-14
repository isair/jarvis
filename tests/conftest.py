import importlib
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

# Robustly locate repository root (directory containing src/jarvis)
_this_file = Path(__file__).resolve()
ROOT = None
for parent in _this_file.parents:
    if (parent / "src" / "jarvis").exists():
        ROOT = parent
        break
if ROOT is None:
    # Fallback to two levels up
    ROOT = _this_file.parent.parent

SRC = ROOT / "src"
# Repository imports expose test helpers; SRC matches the installed package layout.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Import Jarvis through its production package name.
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from jarvis.config import get_default_config


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
    operations = {
        "sounddevice": ("play", "rec", "playrec", "InputStream", "OutputStream",
                        "RawInputStream", "RawOutputStream", "Stream", "RawStream"),
        "winsound": ("PlaySound", "Beep", "MessageBeep"),
    }
    for backend, names in operations.items():
        try:
            module = importlib.import_module(backend)
        except (ImportError, OSError):
            continue
        for name in names:
            if hasattr(module, name):
                setattr(module, name, _reject_audio_operation(f"{backend}.{name}"))


def pytest_sessionfinish(session, exitstatus):
    """Caught audio errors still make qualification fail."""
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
    if exitstatus == pytest.ExitCode.OK:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED


@dataclass
class MockConfig:
    """Minimal config object for unit tests that need a config."""
    # Provider-aware fields. Default to Ollama at localhost so tests
    # that don't care about providers keep the historical behaviour.
    llm_provider: str = "ollama"
    llm_base_url: str = "http://localhost:11434"
    llm_api_key: str = ""
    # ``llm_chat_model`` defaults to empty so tests that pin
    # ``ollama_chat_model = "gpt-oss:20b"`` to exercise the LARGE-model
    # branch get the legacy alias promoted into ``llm_chat_model`` by
    # ``__post_init__`` — same shape ``load_settings()`` produces.
    llm_chat_model: str = ""
    wake_word: str = "jarvis"
    wake_aliases: list = field(default_factory=list)
    whisper_model: str = "small"
    embedding_provider: str = ""
    embedding_base_url: str = ""
    embedding_api_key: str = ""
    embedding_model: str = "nomic-embed-text"
    ollama_base_url: str = "http://localhost:11434"
    ollama_chat_model: str = "gemma4:e2b"
    ollama_embed_model: str = "nomic-embed-text"
    db_path: str = ":memory:"
    sqlite_vss_path: Optional[str] = None
    voice_debug: bool = True
    tts_enabled: bool = False
    tts_engine: str = "piper"
    tts_voice: Optional[str] = None
    tts_rate: int = 200
    tts_piper_model_path: Optional[str] = None
    tts_piper_speaker: Optional[int] = None
    tts_piper_length_scale: float = 1.0
    tts_piper_noise_scale: float = 0.667
    tts_piper_noise_w: float = 0.8
    tts_piper_sentence_silence: float = 0.2
    tts_chatterbox_device: str = "cpu"
    tts_chatterbox_audio_prompt: Optional[str] = None
    tts_chatterbox_exaggeration: float = 0.5
    tts_chatterbox_cfg_weight: float = 0.5
    web_search_enabled: bool = True
    brave_search_api_key: str = ""
    wikipedia_fallback_enabled: bool = True
    llm_tools_timeout_sec: float = 8.0
    llm_embedding_timeout_sec: float = 10.0
    llm_chat_timeout_sec: float = 45.0
    agentic_max_turns: int = 8
    tool_selection_strategy: str = "embedding"
    fast_model: str = ""
    memory_enrichment_max_results: int = 5
    memory_enrichment_source: str = "diary"
    location_enabled: bool = False
    location_ip_address: Optional[str] = None
    location_auto_detect: bool = False
    location_cgnat_resolve_public_ip: bool = False
    location_cache_minutes: int = 60
    dialogue_memory_timeout: int = 300
    llm_thinking_enabled: bool = False
    intent_judge_thinking_enabled: bool = False
    dictation_thinking_enabled: bool = False
    stop_commands: list = field(default_factory=lambda: get_default_config()["stop_commands"])
    stop_command_fuzzy_ratio: float = field(default_factory=lambda: get_default_config()["stop_command_fuzzy_ratio"])
    dictation_hotkey: str = "ctrl+alt+space"
    mcps: Dict[str, Any] = field(default_factory=dict)
    use_stdin: bool = True

    def __post_init__(self) -> None:
        # Mirror ``load_settings``: when the provider-aware fields are
        # left empty, promote the legacy ``ollama_*`` aliases. Tests can
        # set either pair and end up with consistent reads on either side.
        if not self.llm_chat_model:
            self.llm_chat_model = self.ollama_chat_model
        if not self.llm_base_url:
            self.llm_base_url = self.ollama_base_url
        if not self.embedding_model:
            self.embedding_model = self.ollama_embed_model


@pytest.fixture(autouse=True)
def _isolate_user_config_path(tmp_path_factory, monkeypatch):
    """Select a temporary configuration for every runtime and desktop reader."""
    sandbox = tmp_path_factory.mktemp("jarvis_config_sandbox")
    monkeypatch.setenv("JARVIS_CONFIG_PATH", str(sandbox / "config.json"))


@pytest.fixture
def mock_config():
    """Provide a mock configuration for unit tests."""
    return MockConfig()


@pytest.fixture(autouse=True)
def _isolate_dictation_history_path(tmp_path, monkeypatch):
    """Default-history consumers share this test's private storage boundary."""
    from jarvis.dictation import paths

    monkeypatch.setattr(paths, "default_history_path", lambda: tmp_path / "default-history.json")


@pytest.fixture(scope="session")
def runtime_temp_root(tmp_path_factory):
    return tmp_path_factory.mktemp("jarvis_runtime_sandbox")


@pytest.fixture(autouse=True)
def _isolate_runtime_tempfiles(runtime_temp_root, monkeypatch):
    """Keep temporary IPC state separate from a running user's Jarvis instance."""
    import tempfile
    monkeypatch.setattr(tempfile, "tempdir", str(runtime_temp_root))
    for variable in ("TMPDIR", "TEMP", "TMP"):
        monkeypatch.setenv(variable, str(runtime_temp_root))


@pytest.fixture
def desktop_log_home(tmp_path, monkeypatch):
    """Resolve desktop diagnostics inside this test's temporary home."""
    from unittest.mock import MagicMock
    from desktop_app import paths

    owned_path = MagicMock(wraps=Path)
    owned_path.home.return_value = tmp_path
    monkeypatch.setattr(paths, "Path", owned_path)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "appdata"))


@pytest.fixture
def db():
    """Provide an in-memory database for unit tests."""
    from jarvis.memory.db import Database
    database = Database(":memory:", sqlite_vss_path=None)
    yield database
    database.close()


@pytest.fixture
def dialogue_memory():
    """Provide a dialogue memory instance for unit tests."""
    from jarvis.memory.conversation import DialogueMemory
    return DialogueMemory(inactivity_timeout=300, max_interactions=20)


@pytest.fixture(scope="session")
def _qapplication():
    """Keep one QApplication alive for the entire test process.

    Qt requires exactly one QApplication per process.  Re-uses an existing
    instance when present so repeated test runs inside a single session
    don't error.  The offscreen platform is set as a default so headless CI
    (no DISPLAY, no xvfb) can construct a real QApplication instead of
    aborting; machines with a real display are unaffected because the
    platform is only set when it is not already configured.
    """
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    app.setQuitOnLastWindowClosed(False)
    yield app


@pytest.fixture
def qapp(_qapplication):
    """Dispose each test's widgets while the shared application is still alive."""
    from PyQt6 import sip
    from PyQt6.QtCore import QCoreApplication, QEvent

    existing = set(_qapplication.topLevelWidgets())
    yield _qapplication
    for widget in _qapplication.topLevelWidgets():
        if widget not in existing and not sip.isdeleted(widget):
            widget.deleteLater()
    # Flush deferred destruction, not unrelated timers or application callbacks.
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


@pytest.fixture(autouse=True)
def _isolate_daemon_lock(tmp_path, monkeypatch):
    """Daemon entry-point tests must not contend with the user's runtime."""
    monkeypatch.setenv("JARVIS_DAEMON_LOCK", str(tmp_path / "jarvis_daemon.lock"))


@pytest.fixture
def no_network(monkeypatch):
    """Opt-in guard for tests that require every dependency to stay in-process."""
    import socket

    attempts = []

    def blocked(sock, address):
        attempts.append(address)
        raise AssertionError("Offline test attempted a network connection")

    from importlib.util import find_spec

    def blocked_native(*args, **kwargs):
        attempts.append("native HTTP request")
        raise AssertionError("Offline test attempted a native network connection")

    # Minimal dependency jobs cannot load the optional native client.
    if find_spec("curl_cffi") is not None:
        from curl_cffi.requests import Session
        monkeypatch.setattr(Session, "request", blocked_native)
    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    yield
    assert not attempts, f"Unmocked network attempts: {attempts}"


@pytest.fixture
def owned_loopback_network(monkeypatch):
    """Permit HTTP and socket connections only to endpoints owned by this test."""
    import ipaddress
    import socket
    from urllib.parse import urlsplit
    import requests

    bound_sockets = []
    denied = []
    real_bind = socket.socket.bind
    real_connect = socket.socket.connect
    real_connect_ex = socket.socket.connect_ex
    real_send = requests.Session.send

    def address(host):
        try:
            return ipaddress.ip_address(host)
        except ValueError:
            try:
                return ipaddress.ip_address(socket.getaddrinfo(
                    host, None, flags=socket.AI_NUMERICHOST,
                )[0][4][0])
            except (OSError, TypeError):
                return None

    def is_loopback(host):
        selected = address(host)
        return selected is not None and selected.is_loopback

    def reject():
        denied.append("unowned network operation")
        raise AssertionError("Local provider tests require an owned loopback endpoint")

    def owns(host, port):
        return is_loopback(host) and any(
            stream.fileno() >= 0 and stream.getsockname()[1] == port
            and address(stream.getsockname()[0]) == address(host)
            for stream in bound_sockets
        )

    def bind(stream, address):
        if not isinstance(address, tuple) or not is_loopback(address[0]):
            reject()
        result = real_bind(stream, address)
        bound_sockets.append(stream)
        return result

    def connect(stream, address, *, operation):
        if not isinstance(address, tuple) or not owns(address[0], address[1]):
            reject()
        return operation(stream, address)

    def send(session, request, **kwargs):
        target = urlsplit(request.url)
        if not owns(target.hostname, target.port):
            reject()
        return real_send(session, request, **kwargs)

    monkeypatch.setattr(socket.socket, "bind", bind)
    monkeypatch.setattr(socket.socket, "connect", lambda stream, address: connect(stream, address, operation=real_connect))
    monkeypatch.setattr(socket.socket, "connect_ex", lambda stream, address: connect(stream, address, operation=real_connect_ex))
    monkeypatch.setattr(socket, "getfqdn", lambda name="": "localhost")
    monkeypatch.setattr(requests.Session, "send", send)
    yield
    for stream in bound_sockets:
        stream.close()
    assert not denied, "A local provider test attempted an unowned network operation"


@pytest.fixture
def owned_loopback_http(owned_loopback_network, monkeypatch):
    """Keep owned HTTP tests independent of ambient proxies and credentials."""
    import requests

    real_init = requests.Session.__init__

    def initialise(session, *args, **kwargs):
        real_init(session, *args, **kwargs)
        session.trust_env = False

    monkeypatch.setattr(requests.Session, "__init__", initialise)


@pytest.fixture
def settings_config_path(qapp, tmp_path, monkeypatch):
    """Isolate the settings dialogue from user configuration and audio devices."""
    from desktop_app import settings_window as sw

    path = tmp_path / "config.json"
    path.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("JARVIS_CONFIG_PATH", str(path))
    monkeypatch.setattr(sw, "get_input_devices", lambda: [("", "Default"), ("7", "Microphone")])
    monkeypatch.setattr(sw.QMessageBox, "information", lambda *args: None)
    return path


@pytest.fixture
def silent_thinking_tunes(monkeypatch):
    """Exercise real cue workers against owned output streams and drain them."""
    import sounddevice
    from jarvis.output import tune_player

    players = []
    streams = []
    real_player = tune_player.TunePlayer

    class SilentStream:
        def __init__(self, **kwargs):
            self.closed = False
            streams.append(self)

        def start(self):
            pass

        def close(self):
            self.closed = True

    class OwnedPlayer(real_player):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            players.append(self)

    monkeypatch.setattr(sounddevice, 'OutputStream', SilentStream)
    monkeypatch.setattr(tune_player, 'TunePlayer', OwnedPlayer)
    try:
        yield
    finally:
        workers = [player._thread for player in players if player._thread is not None]
        for player in players:
            player.stop_tune()
        for worker in workers:
            worker.join(timeout=5)
        assert all(not worker.is_alive() for worker in workers), 'Test left a thinking-cue worker running'
        assert all(not player.is_playing() for player in players), 'Test left a thinking cue active'
        assert all(stream.closed for stream in streams), 'Test left an output stream open'
