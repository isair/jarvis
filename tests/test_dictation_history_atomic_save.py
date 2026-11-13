"""Saved dictations remain complete when a history write cannot finish."""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from jarvis.dictation import history as history_module

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("operation", ["add", "delete", "clear"])
def test_partial_serialisation_preserves_saved_history(tmp_path, monkeypatch, operation):
    path = tmp_path / "history.json"
    history = history_module.DictationHistory(path=path)
    entry = history.add("Already saved dictation")
    before = path.read_bytes()

    def interrupted_dump(entries, stream, **kwargs):
        stream.write('[{"id":')
        stream.flush()
        raise OSError("Synthetic disk-write failure")

    monkeypatch.setattr(history_module, "json", MagicMock(wraps=json))
    monkeypatch.setattr(history_module.json, "dump", interrupted_dump)
    with pytest.raises(history_module.HistoryUpdateError):
        if operation == "add":
            history.add("New unsaved dictation")
        elif operation == "delete":
            history.delete(entry["id"])
        else:
            history.clear()

    assert path.read_bytes() == before
    assert history.get_all() == [entry]
    assert history_module.DictationHistory(path=path).get_all() == [entry]
    assert set(tmp_path.iterdir()) == {path, Path(str(path) + ".lock")}


def test_successful_save_publishes_complete_unicode_entries(tmp_path):
    path = tmp_path / "nested" / "history.json"
    history = history_module.DictationHistory(path=path)
    first = history.add("İstanbul, café, 東京", duration=1.25)
    second = history.add("Another complete entry")
    assert history_module.DictationHistory(path=path).get_all() == [second, first]
    assert "東京" in path.read_text(encoding="utf-8")
    assert set(path.parent.iterdir()) == {path, Path(str(path) + ".lock")}


@pytest.mark.parametrize("existing", [False, True])
def test_failed_replacement_preserves_the_destination(tmp_path, monkeypatch, existing):
    path = tmp_path / "history.json"
    history = history_module.DictationHistory(path=path)
    saved = [history.add("Already saved")] if existing else []
    before = path.read_bytes() if existing else None
    module_os = MagicMock(wraps=os)
    monkeypatch.setattr(history_module, "os", module_os, raising=False)
    module_os.replace.side_effect = PermissionError("Synthetic replacement denied")

    with pytest.raises(history_module.HistoryUpdateError):
        history.add("Not published")

    assert path.exists() is existing
    if existing:
        assert path.read_bytes() == before
    assert history.get_all() == saved
    assert history_module.DictationHistory(path=path).get_all() == saved
    assert set(tmp_path.iterdir()) == ({path, Path(str(path) + ".lock")} if existing else {Path(str(path) + ".lock")})


@pytest.mark.skipif(os.name == "nt", reason="POSIX file-mode contract")
def test_saved_history_is_private(tmp_path):
    path = tmp_path / "history.json"
    history_module.DictationHistory(path=path).add("Private dictation")
    assert path.stat().st_mode & 0o777 == 0o600
