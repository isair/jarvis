"""Independent history writers retain each other's completed mutations."""

import multiprocessing
import os
from pathlib import Path

import pytest

from jarvis.dictation.history import DictationHistory

pytestmark = pytest.mark.unit


def _write(path, operation, target, ready, release, done, pause):
    history = DictationHistory(path=Path(path))
    if pause:
        save = history._save
        def paused_save():
            ready.set()
            if not release.wait(timeout=10):
                raise TimeoutError("Synthetic writer was not released")
            return save()
        history._save = paused_save
    if operation == "add":
        history.add(target)
    elif operation == "delete":
        history.delete(target)
    else:
        history.clear()
    done.set()


@pytest.mark.parametrize("operation", ["add", "delete", "clear"])
def test_subprocess_mutations_use_the_previous_committed_history(tmp_path, operation):
    path = tmp_path / "history.json"
    initial = DictationHistory(path=path).add("Initial entry")
    context = multiprocessing.get_context("spawn")
    ready, release, first_done, second_done = [context.Event() for _ in range(4)]
    first = context.Process(target=_write, args=(str(path), "add", "First writer", ready, release, first_done, True))
    target = "Second writer" if operation == "add" else initial["id"]
    second = context.Process(target=_write, args=(str(path), operation, target, ready, release, second_done, False))
    first.start()
    try:
        assert ready.wait(timeout=5)
        second.start()
        # Allow the competing process to attempt its write while the first
        # mutation has read its input but cannot yet publish its output.
        second_done.wait(timeout=0.5)
        release.set()
        first.join(timeout=5)
        second.join(timeout=5)
        assert first.exitcode == 0 and second.exitcode == 0
        texts = {entry["text"] for entry in DictationHistory(path=path).get_all()}
        expected = {"Initial entry", "First writer", "Second writer"} if operation == "add" else (
            {"First writer"} if operation == "delete" else set())
        assert texts == expected
    finally:
        release.set()
        for process in (first, second):
            if process.pid is not None:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=5)
                process.close()


@pytest.mark.parametrize("operation", ["add", "delete", "clear"])
def test_busy_history_does_not_report_a_successful_mutation(tmp_path, operation):
    from filelock import FileLock
    from jarvis.dictation.history import HistoryUpdateError

    path = tmp_path / "history.json"
    history = DictationHistory(path=path)
    entry = history.add("Keep this saved dictation")
    before = path.read_bytes()
    with FileLock(str(path) + ".lock", mode=0o600):
        with pytest.raises(HistoryUpdateError):
            if operation == "add":
                history.add("Not saved")
            elif operation == "delete":
                history.delete(entry["id"])
            else:
                history.clear()
    assert path.read_bytes() == before
    assert history.get_all() == [entry]


def test_other_history_files_remain_writable(tmp_path):
    context = multiprocessing.get_context("spawn")
    ready, release, done = [context.Event() for _ in range(3)]
    first_path = tmp_path / "first.json"
    process = context.Process(target=_write, args=(str(first_path), "add", "Held writer", ready, release, done, True))
    process.start()
    try:
        assert ready.wait(timeout=5)
        entry = DictationHistory(path=tmp_path / "second.json").add("Independent entry")
        assert entry["text"] == "Independent entry"
        release.set()
        process.join(timeout=5)
        assert process.exitcode == 0
    finally:
        release.set()
        if process.is_alive():
            process.terminate()
        process.join(timeout=5)
        process.close()


def test_terminated_writer_releases_history_ownership(tmp_path):
    path = tmp_path / "history.json"
    original = DictationHistory(path=path).add("Saved before termination")
    context = multiprocessing.get_context("spawn")
    ready, release, done = [context.Event() for _ in range(3)]
    process = context.Process(target=_write, args=(str(path), "add", "Unfinished entry", ready, release, done, True))
    process.start()
    try:
        assert ready.wait(timeout=5)
        process.terminate()
        process.join(timeout=5)
        assert not process.is_alive()
        other = DictationHistory(path=path)
        added = other.add("Saved after termination")
        assert other.get_all() == [added, original]
    finally:
        if process.is_alive():
            process.terminate()
        process.join(timeout=5)
        process.close()


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink contract")
def test_lock_alias_does_not_modify_an_unrelated_file(tmp_path):
    from jarvis.dictation.history import HistoryUpdateError

    path = tmp_path / "history.json"
    unrelated = tmp_path / "unrelated.txt"
    unrelated.write_text("Keep this unrelated private fixture")
    Path(str(path) + ".lock").symlink_to(unrelated)
    denied = False
    try:
        DictationHistory(path=path).add("Not published")
    except HistoryUpdateError:
        denied = True
    assert unrelated.read_text() == "Keep this unrelated private fixture"
    assert denied and not path.exists()
