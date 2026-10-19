"""Graph rewrites cannot bypass the stored-line growth bound."""
import json

import pytest

from jarvis.memory import graph_ops
from jarvis.memory.graph import GraphMemoryStore

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("over_limit", [False, True])
@pytest.mark.parametrize("separator", ["\n", "\n\n"])
def test_merge_growth_limit_counts_every_stored_line(tmp_path, monkeypatch, over_limit, separator):
    store = GraphMemoryStore(str(tmp_path / "graph.db"))
    current = ["Original independent fact.", "Another original fact."]
    new_facts = ["New fact."]
    limit = len(current) + len(new_facts) + graph_ops._MERGE_GROWTH_SLACK
    lines = [f"Proposed fact {index}." for index in range(limit + int(over_limit))]
    monkeypatch.setattr(graph_ops, "call_llm_direct", lambda **kwargs: json.dumps({"facts": [separator.join(lines)]}))
    try:
        node = store.create_node("Profile", "Personal facts", data="\n".join(current), parent_id="user")
        result = graph_ops.merge_node_data(store, node.id, new_facts, None, "synthetic")
        assert result.success is not over_limit
        assert store.get_node(node.id).data == (node.data if over_limit else separator.join(lines))
    finally:
        store.close()


def test_oversized_merge_falls_back_without_losing_existing_or_new_facts(tmp_path, monkeypatch):
    store = GraphMemoryStore(str(tmp_path / "graph.db"))
    current = ["Original independent fact.", "Another original fact."]
    new_fact = "The user enjoys sailing."
    limit = len(current) + 1 + graph_ops._MERGE_GROWTH_SLACK
    responses = iter([
        json.dumps([{"branch": "USER", "fact": new_fact}]),
        json.dumps({"facts": ["\n".join(f"Invented fact {i}." for i in range(limit + 1))]}),
    ])
    monkeypatch.setattr(graph_ops, "call_llm_direct", lambda **kwargs: next(responses))
    monkeypatch.setattr(graph_ops, "_review_graph_facts", lambda facts, *args: facts)
    try:
        store.update_node("user", data="\n".join(current))
        result = graph_ops.update_graph_from_dialogue(store, "User enjoys sailing.", None, "synthetic")
        assert store.get_node("user").data.splitlines() == current + [new_fact]
        assert [fact for fact, _ in result.stored] == [new_fact]
    finally:
        store.close()
