"""Unreadable saved history must not be mistaken for an empty store."""

import json
from unittest.mock import MagicMock

import pytest

from jarvis.dictation import history as history_module

pytestmark = pytest.mark.unit


@pytest.fixture(params=["read_denied", "read_missing", "bad_json", "wrong_root"])
def unavailable_history(tmp_path, monkeypatch, request):
    path = tmp_path / "history.json"
    history = history_module.DictationHistory(path=path)
    saved = history.add("Previously saved dictation", duration=1.25)
    if request.param in {"read_denied", "read_missing"}:
        module_json = MagicMock(wraps=json)
        failure = PermissionError if request.param == "read_denied" else FileNotFoundError
        module_json.load.side_effect = failure("Synthetic failure reading an opened history file")
        monkeypatch.setattr(history_module, "json", module_json)
    elif request.param == "bad_json":
        path.write_text('[{"text": "Incomplete external edit"', encoding="utf-8")
    else:
        path.write_text('{"unknown": "Keep these external bytes"}', encoding="utf-8")
    return history, path, saved


@pytest.mark.parametrize("operation", ["add", "delete", "clear"])
def test_mutations_preserve_unavailable_history(unavailable_history, operation):
    history, path, saved = unavailable_history
    before = path.read_bytes()
    with pytest.raises(history_module.HistoryUpdateError):
        if operation == "add":
            history.add("A new dictation")
        elif operation == "delete":
            history.delete(saved["id"])
        else:
            history.clear()
    assert path.read_bytes() == before
    assert history.get_all() == [saved]


def test_failed_refresh_retains_the_last_valid_snapshot(unavailable_history):
    history, path, saved = unavailable_history
    before = path.read_bytes()
    history.reload_from_disk()
    assert history.get_all() == [saved]
    assert path.read_bytes() == before


@pytest.mark.parametrize("initial", [None, "[]"])
def test_a_new_or_valid_empty_store_can_save_entries(tmp_path, initial):
    path = tmp_path / "history.json"
    if initial is not None:
        path.write_text(initial)
    history = history_module.DictationHistory(path=path)
    assert history.get_all() == []
    saved = history.add("İstanbul, café, 東京")
    assert history_module.DictationHistory(path=path).get_all() == [saved]


@pytest.mark.parametrize("content", ["", "Invalid JSON", '{"unknown": true}'])
def test_initial_unavailable_history_is_gracefully_empty_without_rewriting(tmp_path, content):
    path = tmp_path / "history.json"
    path.write_text(content)
    assert history_module.DictationHistory(path=path).get_all() == []
    assert path.read_text() == content


def test_failed_write_retains_its_snapshot_when_further_reads_fail(tmp_path, monkeypatch):
    path = tmp_path / "history.json"
    history = history_module.DictationHistory(path=path)
    saved = history.add("Keep the committed entry")
    before = path.read_bytes()
    module_json = MagicMock(wraps=json)
    def interrupted_save(entries, stream, **kwargs):
        stream.write("[")
        stream.flush()
        module_json.load.side_effect = PermissionError("Synthetic subsequent read denial")
        raise OSError("Synthetic write failure")
    module_json.dump.side_effect = interrupted_save
    monkeypatch.setattr(history_module, "json", module_json)
    with pytest.raises(history_module.HistoryUpdateError):
        history.add("Not saved")
    assert path.read_bytes() == before
    assert history.get_all() == [saved]
