"""Private prompt diagnostics retain complete data without interrupting replies."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from jarvis.reply import prompt_dump

pytestmark = pytest.mark.unit


@pytest.fixture
def tmp_home(tmp_path, monkeypatch):
    """Redirect Path.home() so dumps land in a sandbox."""
    from unittest.mock import MagicMock
    owned_path = MagicMock(wraps=Path)
    owned_path.home.return_value = tmp_path
    monkeypatch.setattr(prompt_dump, "Path", owned_path)
    return tmp_path


class TestGating:
    def test_disabled_by_default(self, tmp_home, monkeypatch):
        monkeypatch.delenv("JARVIS_DUMP_PROMPTS", raising=False)
        result = prompt_dump.dump_reply_turn(
            session_id="abc12345",
            turn=1,
            query="hi",
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            tools_schema=None,
            use_text_tools=False,
        )
        assert result is None
        # No files created.
        assert not (tmp_home / ".local" / "share" / "jarvis" / "prompts").exists()

    @pytest.mark.parametrize("value", ["1", "true", "YES", "on"])
    def test_enabled_by_truthy_env_values(self, tmp_home, monkeypatch, value):
        monkeypatch.setenv("JARVIS_DUMP_PROMPTS", value)
        assert prompt_dump.is_enabled()

    @pytest.mark.parametrize("value", ["", "0", "false", "no"])
    def test_disabled_by_falsy_env_values(self, tmp_home, monkeypatch, value):
        monkeypatch.setenv("JARVIS_DUMP_PROMPTS", value)
        assert not prompt_dump.is_enabled()


class TestDumpContents:
    def test_writes_full_payload(self, tmp_home, monkeypatch, capsys):
        monkeypatch.setenv("JARVIS_DUMP_PROMPTS", "1")
        messages = [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "Tell me about Possessor"},
        ]
        tools = [{"type": "function", "function": {"name": "webSearch"}}]
        path = prompt_dump.dump_reply_turn(
            session_id="deadbeef",
            turn=2,
            query="Tell me about Possessor",
            model="gemma4:e2b",
            messages=messages,
            tools_schema=tools,
            use_text_tools=False,
            response={"message": {"content": "Under the Skin"}},
        )
        assert path is not None
        assert path.exists()
        assert path.parent == tmp_home / ".local" / "share" / "jarvis" / "prompts"
        assert "deadbeef" in path.name
        assert "t02" in path.name

        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["session_id"] == "deadbeef"
        assert payload["turn"] == 2
        assert payload["query"] == "Tell me about Possessor"
        assert payload["model"] == "gemma4:e2b"
        assert payload["messages"] == messages
        assert payload["tools_schema"] == tools
        assert payload["use_text_tools"] is False
        assert payload["response"]["message"]["content"] == "Under the Skin"

        # User-visible line should mention the path so they know where to grab it.
        out = capsys.readouterr().out
        assert str(path) in out

    def test_session_ids_are_unique_per_call(self):
        ids = {prompt_dump.new_session_id() for _ in range(50)}
        assert len(ids) == 50

    def test_dump_failure_is_swallowed(self, tmp_home, monkeypatch):
        """A broken serialiser must not propagate — prompts dump is a
        diagnostic aid, never a hard dependency of reply generation."""
        monkeypatch.setenv("JARVIS_DUMP_PROMPTS", "1")

        class Unserialisable:
            def __repr__(self):
                raise RuntimeError("nope")

        result = prompt_dump.dump_reply_turn(
            session_id="abc", turn=1, query="q", model="m", messages=[],
            tools_schema=None, use_text_tools=False, response=Unserialisable(),
        )
        assert result is None  # swallowed, not raised


@pytest.mark.skipif(os.name != "posix", reason="POSIX permissions")
@pytest.mark.parametrize("existing", [False, True])
def test_diagnostics_are_private_under_a_permissive_umask(tmp_path, existing):
    import subprocess
    import sys

    script = r'''
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
import json, os, stat, sys, time
root = Path(sys.argv[1])
config = root / 'private-config.json'
config.write_text(json.dumps({'_config_version': 3, 'db_path': str(root/'synthetic.sqlite'), 'voice_debug': False}))
os.environ['JARVIS_CONFIG_PATH'] = str(config)
os.environ['JARVIS_DUMP_PROMPTS'] = '1'
from jarvis.reply import prompt_dump
owned_path = MagicMock(wraps=Path)
owned_path.home.return_value = root
prompt_dump.Path = owned_path
prompt_dump.time = SimpleNamespace(time=time.time, strftime=lambda fmt: 'fixed')
os.umask(0o022)
target = root/'.local/share/jarvis/prompts/turn-fixed-fixture-t01.json'
if sys.argv[2] == 'existing':
    target.parent.mkdir(parents=True)
    target.write_text('Retained old diagnostic')
    target.chmod(0o644)
path = prompt_dump.dump_reply_turn(session_id='fixture', turn=1, query='Synthetic private query', model='fixture', messages=[], tools_schema=None, use_text_tools=False)
print('DIAG_RESULT:' + json.dumps({'mode': stat.S_IMODE(path.stat().st_mode), 'directory_mode': stat.S_IMODE(path.parent.stat().st_mode), 'query': json.loads(path.read_text())['query']}))
'''
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path), "existing" if existing else "new"],
                            env=env, text=True, capture_output=True, check=True)
    report = json.loads(next(line.removeprefix("DIAG_RESULT:") for line in result.stdout.splitlines()
                             if line.startswith("DIAG_RESULT:")))
    assert report["mode"] == 0o600
    if not existing:
        assert report["directory_mode"] == 0o700
    assert report["query"] == "Synthetic private query"


def test_failed_serialisation_preserves_saved_diagnostics_and_private_errors(tmp_home, monkeypatch):
    monkeypatch.setenv("JARVIS_DUMP_PROMPTS", "1")
    from types import SimpleNamespace
    import time
    monkeypatch.setattr(prompt_dump, "time", SimpleNamespace(time=time.time, strftime=lambda fmt: "fixed"))
    options = dict(session_id="fixture", turn=1, query="Retained query", model="fixture",
                   messages=[], tools_schema=None, use_text_tools=False)
    saved = prompt_dump.dump_reply_turn(**options)
    original = saved.read_bytes()
    before = set(saved.parent.iterdir())
    secret = "private-fixture-value"
    logs = []
    monkeypatch.setattr(prompt_dump, "debug_log", lambda message, category: logs.append(message))

    class FailedValue:
        def __str__(self):
            raise ValueError(secret)

    assert prompt_dump.dump_reply_turn(**options, response=FailedValue()) is None
    assert saved.read_bytes() == original
    assert set(saved.parent.iterdir()) == before
    assert all(secret not in message for message in logs)


def test_failed_publication_preserves_saved_diagnostics_and_cleans_temporary_files(tmp_home, monkeypatch):
    monkeypatch.setenv("JARVIS_DUMP_PROMPTS", "1")
    from types import SimpleNamespace
    import time
    monkeypatch.setattr(prompt_dump, "time", SimpleNamespace(time=time.time, strftime=lambda fmt: "fixed"))
    options = dict(session_id="fixture", turn=1, query="Retained query", model="fixture",
                   messages=[], tools_schema=None, use_text_tools=False)
    saved = prompt_dump.dump_reply_turn(**options)
    original = saved.read_bytes()
    before = set(saved.parent.iterdir())
    owned_os = SimpleNamespace(**vars(os))

    def blocked_replace(*args):
        raise PermissionError("Synthetic publication failure")

    owned_os.replace = blocked_replace
    monkeypatch.setattr(prompt_dump, "os", owned_os)
    assert prompt_dump.dump_reply_turn(**options, response="New fixture response") is None
    assert saved.read_bytes() == original
    assert set(saved.parent.iterdir()) == before
