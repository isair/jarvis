"""Behavioural coverage for inspecting and correcting local facts."""
import pytest

pytest.importorskip("flask")
from jarvis.memory.facts import FactStore
from desktop_app import memory_viewer

pytestmark = pytest.mark.unit


@pytest.fixture
def facts_client(tmp_path, monkeypatch):
    path = str(tmp_path / "facts.db")
    store = FactStore(path)
    fact = store.add_fact(
        "I live in Paris", kind="user", owner="user", subject="user",
        source_ref="dialogue:test", source_type="dialogue", source_role="user",
        source_channel="text", source_text="I live in Paris", evidence="I live in Paris",
        observed_at="2026-01-01T00:00:00+00:00",
    )
    monkeypatch.setattr(memory_viewer, "_get_db_path", lambda: path)
    with memory_viewer.app.test_client() as client:
        yield client, fact
    store.close()


def test_facts_show_source_and_evidence(facts_client):
    client, fact = facts_client
    response = client.get("/api/facts")
    assert response.status_code == 200
    assert response.json["facts"][0]["source"]["evidence"] == fact["text"]
    detail = client.get(f"/api/facts/{fact['id']}")
    assert detail.json["fact"]["source"]["source_type"] == "dialogue"
    assert len(detail.json["history"]) == 1


def test_correction_supersedes_without_erasing_evidence(facts_client):
    client, fact = facts_client
    response = client.put(f"/api/facts/{fact['id']}", json={"text": "I live in London"})
    assert response.status_code == 200
    corrected = response.json["fact"]
    assert corrected["text"] == "I live in London"
    assert corrected["source"]["source_type"] == "manual"
    assert corrected["supersedes_id"] == fact["id"]
    assert client.get("/api/facts").json["facts"] == [corrected]
    history = client.get(f"/api/facts/{corrected['id']}").json["history"]
    assert [item["status"] for item in history] == ["superseded", "active"]


def test_retraction_excludes_fact_but_preserves_visible_history(facts_client):
    client, fact = facts_client
    response = client.post(f"/api/facts/{fact['id']}/retract", json={})
    assert response.status_code == 200
    assert client.get("/api/facts").json["facts"] == []
    assert client.get("/api/facts?status=retracted").json["facts"][0]["id"] == fact["id"]
    assert client.get(f"/api/facts/{fact['id']}").json["history"][0]["text"] == fact["text"]
    assert client.put(f"/api/facts/{fact['id']}", json={"text": "revived"}).status_code == 409


@pytest.mark.parametrize("body", [None, [], {"text": ""}, {"text": 3}, {"text": "x" * 4001}])
def test_invalid_corrections_leave_memory_unchanged(facts_client, body):
    client, fact = facts_client
    assert client.put(f"/api/facts/{fact['id']}", json=body).status_code == 400
    assert client.get(f"/api/facts/{fact['id']}").json["fact"]["text"] == fact["text"]


def test_missing_facts_and_invalid_filters(facts_client):
    client, _ = facts_client
    assert client.get("/api/facts/99999").status_code == 404
    assert client.put("/api/facts/99999", json={"text": "hello"}).status_code == 404
    for query in ("status=invalid", "limit=bad", "offset=-1", "limit=0"):
        assert client.get("/api/facts?" + query).status_code == 400
