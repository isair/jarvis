"""Saved dictation records remain safe for their existing consumers."""

import json
import time

import pytest

from jarvis.dictation.history import DictationHistory, HistoryUpdateError

pytestmark = pytest.mark.unit

INVALID_FIELDS = [
    ("id", None), ("id", ""), ("id", 42), ("id", "\ud800"),
    ("text", None), ("text", 42), ("text", "\ud800"),
    ("timestamp", None), ("timestamp", "invalid"), ("timestamp", True),
    ("timestamp", float("nan")), ("timestamp", float("inf")), ("timestamp", 10 ** 50),
    ("duration", None), ("duration", "invalid"), ("duration", True),
    ("duration", float("nan")), ("duration", float("inf")),
]


def record():
    return {"id": "fixture-entry", "text": "İstanbul, café, 東京", "timestamp": time.time(), "duration": 1.0}


@pytest.mark.parametrize("field,value", INVALID_FIELDS)
def test_invalid_fields_are_not_loaded_or_rewritten(tmp_path, field, value):
    path = tmp_path / "history.json"
    invalid = {**record(), field: value}
    path.write_text(json.dumps([invalid]), encoding="utf-8")
    before = path.read_bytes()
    assert DictationHistory(path=path).get_all() == []
    assert path.read_bytes() == before


@pytest.mark.parametrize("invalid", [None, "invalid", {}, {"id": "missing-text"}, {"text": "missing-id"}])
def test_unknown_record_shapes_are_not_loaded_or_rewritten(tmp_path, invalid):
    path = tmp_path / "history.json"
    path.write_text(json.dumps([invalid]), encoding="utf-8")
    before = path.read_bytes()
    assert DictationHistory(path=path).get_all() == []
    assert path.read_bytes() == before


@pytest.fixture(params=["null", "timestamp", "duration"])
def malformed_history(tmp_path, request):
    path = tmp_path / "history.json"
    history = DictationHistory(path=path)
    saved = history.add("Keep the valid cached dictation")
    invalid = None if request.param == "null" else {**record(), request.param: "invalid"}
    path.write_text(json.dumps([invalid]), encoding="utf-8")
    return history, path, saved


@pytest.mark.parametrize("operation", ["add", "delete", "clear"])
def test_malformed_records_cannot_be_mutated(malformed_history, operation):
    history, path, saved = malformed_history
    before = path.read_bytes()
    with pytest.raises(HistoryUpdateError):
        if operation == "add":
            history.add("New entry")
        elif operation == "delete":
            history.delete(saved["id"])
        else:
            history.clear()
    assert path.read_bytes() == before
    assert history.get_all() == [saved]


def test_refresh_retains_valid_entries_when_records_become_malformed(malformed_history):
    history, path, saved = malformed_history
    before = path.read_bytes()
    history.reload_from_disk()
    assert history.get_all() == [saved]
    assert path.read_bytes() == before


@pytest.mark.parametrize("metadata", [True, False])
def test_valid_legacy_records_and_extra_fields_remain_literal(tmp_path, metadata):
    path = tmp_path / "history.json"
    legacy = {"id": "legacy-entry", "text": "İstanbul, café, 東京", "source": {"name": "Local fixture"}}
    if metadata:
        legacy.update(timestamp=time.time(), duration=1.0)
    path.write_text(json.dumps([legacy], ensure_ascii=False), encoding="utf-8")
    history = DictationHistory(path=path)
    assert history.get_all() == [legacy]
    added = history.add("New valid entry")
    assert DictationHistory(path=path).get_all() == [added, legacy]


@pytest.mark.parametrize("text,duration", [(None, 1.0), ("New entry", float("inf")), ("New entry", float("nan"))])
def test_invalid_new_entries_are_not_published(tmp_path, text, duration):
    path = tmp_path / "history.json"
    history = DictationHistory(path=path)
    saved = history.add("Previously saved")
    before = path.read_bytes()
    with pytest.raises(HistoryUpdateError):
        history.add(text, duration=duration)
    assert path.read_bytes() == before
    assert history.get_all() == [saved]
