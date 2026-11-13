"""Malformed merge proposals preserve the whole current fact collection."""
import json

import pytest

from jarvis.memory import graph_ops
from jarvis.memory.graph import GraphMemoryStore

pytestmark = pytest.mark.unit


@pytest.fixture
def store(tmp_path):
    graph = GraphMemoryStore(str(tmp_path / "graph.db"))
    yield graph
    graph.close()


@pytest.mark.parametrize("invalid", [None, True, 42, {}, []])
def test_invalid_fact_entry_rejects_the_entire_proposal(store, monkeypatch, invalid):
    current = "First independent fact.\nSecond independent fact."
    monkeypatch.setattr(graph_ops, "call_llm_direct", lambda **kwargs: json.dumps({"facts": ["First independent fact.", invalid]}))
    node = store.create_node("Profile", "Personal facts", data=current, parent_id="user")
    result = graph_ops.merge_node_data(store, node.id, ["New fact."], None, "synthetic")
    assert result.success is False
    assert store.get_node(node.id).data == current


def test_invalid_merge_keeps_existing_facts_and_appends_new_fact(store, monkeypatch):
    current = ["First independent fact.", "Second independent fact."]
    new_fact = "The user enjoys sailing."
    responses = iter([
        json.dumps([{"branch": "USER", "fact": new_fact}]),
        json.dumps({"facts": [new_fact, {"unexpected": "malformed fact"}]}),
    ])
    monkeypatch.setattr(graph_ops, "call_llm_direct", lambda **kwargs: next(responses))
    monkeypatch.setattr(graph_ops, "_review_graph_facts", lambda facts, *args: facts)
    store.update_node("user", data="\n".join(current))
    result = graph_ops.update_graph_from_dialogue(store, "User enjoys sailing.", None, "synthetic")
    assert store.get_node("user").data.splitlines() == current + [new_fact]
    assert [fact for fact, _ in result.stored] == [new_fact]


def test_valid_fact_strings_still_allow_blank_items(store, monkeypatch):
    current = ["First independent fact.", "Second independent fact."]
    new_fact = "The user enjoys sailing."
    monkeypatch.setattr(graph_ops, "call_llm_direct", lambda **kwargs: json.dumps({"facts": current + ["", "  ", new_fact]}))
    node = store.create_node("Profile", "Personal facts", data="\n".join(current), parent_id="user")
    result = graph_ops.merge_node_data(store, node.id, [new_fact], None, "synthetic")
    assert result.success is True
    assert store.get_node(node.id).data.splitlines() == current + [new_fact]
    assert result.incorporated_indices == [0]
