"""
Regression tests: ``_save_json`` must never leave ``config.json`` truncated
or partially written if the process dies mid-save.

The config file can hold ``llm_api_key`` and every other user setting, so a
crash between opening the file for writing and finishing the write used to
leave the user with a corrupted or empty config. ``_save_json`` now writes
to a temp file in the same directory and ``os.replace()``s it over the
target, so the target either has the old contents or the new contents,
never a partial write.
"""

import json
import os

import pytest

from jarvis.config import _save_json


def test_save_json_writes_new_contents_via_replace(tmp_path, monkeypatch):
    """The temp file holds the old contents at the moment of the atomic swap."""
    cfg_path = tmp_path / "config.json"
    original = {"llm_api_key": "original-secret"}
    cfg_path.write_text(json.dumps(original))

    replace_calls = []
    real_replace = os.replace

    def spy_replace(src, dst):
        # Destination must still hold the original bytes right up until the swap.
        assert cfg_path.read_text() == json.dumps(original)
        replace_calls.append((src, dst))
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", spy_replace)

    ok = _save_json(cfg_path, {"llm_api_key": "new-secret"})

    assert ok is True
    assert replace_calls
    assert json.loads(cfg_path.read_text()) == {"llm_api_key": "new-secret"}


def test_save_json_leaves_original_intact_on_failure_between_write_and_rename(
    tmp_path, monkeypatch
):
    """A crash between the temp-file write and the rename must not corrupt the config."""
    cfg_path = tmp_path / "config.json"
    original = {"llm_api_key": "original-secret", "some_setting": True}
    cfg_path.write_text(json.dumps(original))

    def failing_replace(src, dst):
        raise OSError("simulated crash between write and rename")

    monkeypatch.setattr(os, "replace", failing_replace)

    ok = _save_json(cfg_path, {"llm_api_key": "corrupted-write"})

    assert ok is False
    assert json.loads(cfg_path.read_text()) == original


def test_save_json_leaves_no_leftover_temp_file_on_failure(tmp_path, monkeypatch):
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({"llm_api_key": "original-secret"}))

    def failing_replace(src, dst):
        raise OSError("simulated crash between write and rename")

    monkeypatch.setattr(os, "replace", failing_replace)

    _save_json(cfg_path, {"llm_api_key": "corrupted-write"})

    leftover = [p for p in tmp_path.iterdir() if p != cfg_path]
    assert leftover == []


def test_save_json_no_leftover_temp_files_on_success(tmp_path):
    cfg_path = tmp_path / "config.json"

    ok = _save_json(cfg_path, {"a": 1})

    assert ok is True
    leftover = [p for p in tmp_path.iterdir() if p != cfg_path]
    assert leftover == []


@pytest.mark.skipif(os.name == "nt", reason="POSIX-only permission check")
def test_save_json_sets_permissions_to_0600_on_posix(tmp_path):
    cfg_path = tmp_path / "config.json"

    _save_json(cfg_path, {"llm_api_key": "secret"})

    assert (cfg_path.stat().st_mode & 0o777) == 0o600
