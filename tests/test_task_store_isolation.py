"""Task discovery, resumption and evidence stay within one database."""
import json

import pytest

from jarvis.reply.task_state import TaskStore

pytestmark = pytest.mark.unit


def test_neighbouring_databases_do_not_discover_each_others_tasks(tmp_path):
    private = TaskStore(tmp_path / "private.db")
    private.begin("Private unfinished request", ["Pending private action"])
    fresh = TaskStore(tmp_path / "fresh.db")
    assert fresh.latest_incomplete() is None


@pytest.mark.parametrize("access", ["resume", "compact_context", "prior_step_results", "read_result"])
def test_another_database_cannot_resume_or_read_a_known_task(tmp_path, access):
    private = TaskStore(tmp_path / "private.db")
    task = private.begin("Private request", ["Read private data"])
    result_id = private.record_result(task.task_id, step_index=0, tool_name="readFile",
                                     success=True, full_text="private evidence",
                                     signature="private read", mutating=False)
    fresh = TaskStore(tmp_path / "fresh.db")
    with pytest.raises((FileNotFoundError, ValueError)):
        if access == "read_result":
            fresh.read_result(result_id, task_id=task.task_id)
        elif access == "prior_step_results":
            fresh.prior_step_results(task.task_id, max_chars=500)
        else:
            getattr(fresh, access)(task.task_id)


def test_equivalent_database_paths_reopen_the_same_tasks_and_evidence(tmp_path):
    original = TaskStore(tmp_path / "one.db")
    task = original.begin("Keep this task", ["Fetch evidence"])
    result_id = original.record_result(task.task_id, step_index=0, tool_name="fetch",
                                      success=True, full_text="owned evidence",
                                      signature="read", mutating=False)
    reopened = TaskStore(tmp_path / "subdirectory" / ".." / "one.db")
    assert reopened.resume(task.task_id).task_id == task.task_id
    assert reopened.read_result(result_id, task_id=task.task_id)["text"] == "owned evidence"


@pytest.mark.parametrize("identity", ["foreign", "missing"])
def test_unattributed_or_foreign_copied_records_cannot_be_discovered_or_resumed(tmp_path, identity):
    private = TaskStore(tmp_path / "private.db")
    task = private.begin("Private request", ["Private step"])
    fresh = TaskStore(tmp_path / "fresh.db")
    source = private.root / (task.task_id + ".json")
    record = json.loads(source.read_text(encoding="utf-8"))
    if identity == "missing":
        record.pop("database_id", None)
    destination = fresh.root / source.name
    destination.write_text(json.dumps(record), encoding="utf-8")
    assert fresh.latest_incomplete() is None
    with pytest.raises(ValueError):
        fresh.resume(task.task_id)


def test_unattributed_sibling_journal_is_preserved_without_automatic_discovery(tmp_path):
    journal = tmp_path / ".jarvis_tasks"
    journal.mkdir()
    legacy = journal / "unattributed.json"
    original = '{"objective":"private legacy words","status":"active"}'
    legacy.write_text(original, encoding="utf-8")
    fresh = TaskStore(tmp_path / "fresh.db")
    assert fresh.latest_incomplete() is None
    assert legacy.read_text(encoding="utf-8") == original


def test_copied_task_and_result_files_do_not_grant_another_database_evidence_access(tmp_path):
    private = TaskStore(tmp_path / "private.db")
    task = private.begin("Private request", ["Private read"])
    result_id = private.record_result(task.task_id, step_index=0, tool_name="readFile",
                                     success=True, full_text="private evidence",
                                     signature="private read", mutating=False)
    fresh = TaskStore(tmp_path / "fresh.db")
    for source in private.root.glob("*.json"):
        (fresh.root / source.name).write_bytes(source.read_bytes())
    for source in private.results_dir.iterdir():
        (fresh.results_dir / source.name).write_bytes(source.read_bytes())
    with pytest.raises(ValueError):
        fresh.read_result(result_id, task_id=task.task_id)
    assert private.read_result(result_id, task_id=task.task_id)["text"] == "private evidence"


def test_task_filename_cannot_claim_another_record_identity(tmp_path):
    import uuid
    store = TaskStore(tmp_path / "one.db")
    task = store.begin("One task", ["One step"])
    path = store.root / (task.task_id + ".json")
    record = json.loads(path.read_text(encoding="utf-8"))
    record["task_id"] = uuid.uuid4().hex
    path.write_text(json.dumps(record), encoding="utf-8")
    assert store.latest_incomplete() is None
    with pytest.raises(ValueError):
        store.resume(task.task_id)


def test_memory_only_database_does_not_create_a_shared_persistent_journal(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError):
        TaskStore(":memory:")
    assert list(tmp_path.iterdir()) == []
