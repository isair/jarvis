"""Tests for the graph mutation listener registry and the warm-profile
invalidation hook it powers.

The registry lets consumers (notably ``DialogueMemory``'s warm-profile
cache) react to writes against the User / Directives branches mid-
conversation. World-branch writes must NOT invalidate the warm profile,
since the warm profile does not include world facts.
"""

from __future__ import annotations

import os
import tempfile

import pytest

from src.jarvis.memory.conversation import DialogueMemory
from src.jarvis.memory.graph import (
    BRANCH_DIRECTIVES,
    BRANCH_USER,
    BRANCH_WORLD,
    GraphMemoryStore,
    register_graph_mutation_listener,
    unregister_graph_mutation_listener,
)


@pytest.fixture
def graph_store():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    store = GraphMemoryStore(path)
    yield store
    try:
        os.unlink(path)
    except OSError:
        pass


@pytest.mark.unit
class TestMutationListenerRegistry:
    def test_create_under_user_notifies_with_user_branch(self, graph_store):
        events: list[dict] = []

        def cb(*, action, node_id, branch):
            events.append({"action": action, "node_id": node_id, "branch": branch})

        register_graph_mutation_listener(cb)
        try:
            graph_store.create_node("Alice", "user fact", parent_id=BRANCH_USER)
        finally:
            unregister_graph_mutation_listener(cb)

        actions = [e["action"] for e in events]
        branches = [e["branch"] for e in events]
        assert "create" in actions
        assert BRANCH_USER in branches

    def test_update_under_directives_notifies_with_directives_branch(self, graph_store):
        node = graph_store.create_node(
            "be brief", "rule", parent_id=BRANCH_DIRECTIVES,
        )
        events: list[dict] = []

        def cb(*, action, node_id, branch):
            events.append({"action": action, "node_id": node_id, "branch": branch})

        register_graph_mutation_listener(cb)
        try:
            graph_store.update_node(node.id, data="updated")
        finally:
            unregister_graph_mutation_listener(cb)

        update_events = [e for e in events if e["action"] == "update"]
        assert update_events
        assert update_events[-1]["branch"] == BRANCH_DIRECTIVES

    def test_delete_under_world_notifies_with_world_branch(self, graph_store):
        node = graph_store.create_node(
            "Paris", "city", parent_id=BRANCH_WORLD,
        )
        events: list[dict] = []

        def cb(*, action, node_id, branch):
            events.append({"action": action, "node_id": node_id, "branch": branch})

        register_graph_mutation_listener(cb)
        try:
            graph_store.delete_node(node.id)
        finally:
            unregister_graph_mutation_listener(cb)

        delete_events = [e for e in events if e["action"] == "delete"]
        assert delete_events
        assert delete_events[-1]["branch"] == BRANCH_WORLD

    def test_listener_exception_does_not_break_write(self, graph_store):
        def boom(*, action, node_id, branch):
            raise RuntimeError("listener should not break writes")

        register_graph_mutation_listener(boom)
        try:
            # Must complete despite the listener raising.
            node = graph_store.create_node(
                "Bob", "another user fact", parent_id=BRANCH_USER,
            )
            assert graph_store.get_node(node.id) is not None
        finally:
            unregister_graph_mutation_listener(boom)

    def test_unregister_is_idempotent(self):
        def cb(**_):
            pass

        register_graph_mutation_listener(cb)
        unregister_graph_mutation_listener(cb)
        unregister_graph_mutation_listener(cb)  # second remove must not raise

    def test_deep_descendant_resolves_to_branch(self, graph_store):
        """A grandchild several levels deep under user must resolve to the
        ``user`` branch so the listener can scope correctly even for nested
        nodes.
        """
        parent = graph_store.create_node("Profile", "child", parent_id=BRANCH_USER)
        child = graph_store.create_node("Tastes", "grandchild", parent_id=parent.id)
        events: list[dict] = []

        def cb(*, action, node_id, branch):
            events.append({"action": action, "node_id": node_id, "branch": branch})

        register_graph_mutation_listener(cb)
        try:
            graph_store.append_to_node(child.id, "loves jazz")
        finally:
            unregister_graph_mutation_listener(cb)

        # append_to_node calls update_node internally → at least one update.
        update_events = [e for e in events if e["action"] == "update"]
        assert update_events
        assert update_events[-1]["branch"] == BRANCH_USER


@pytest.mark.unit
class TestWarmProfileInvalidationHook:
    """End-to-end: the wiring done in ``daemon.py`` invalidates the warm
    profile entry on User / Directives writes but ignores World writes.
    Re-create that wiring here so the test does not depend on daemon
    start-up.
    """

    def _wire(self, dm: DialogueMemory):
        relevant = {BRANCH_USER, BRANCH_DIRECTIVES}

        def cb(*, action, node_id, branch):
            del action, node_id
            if branch in relevant:
                dm.invalidate_warm_profile()

        register_graph_mutation_listener(cb)
        return cb

    def test_user_write_invalidates_warm_profile(self, graph_store):
        dm = DialogueMemory()
        dm.hot_cache_put(dm.WARM_PROFILE_CACHE_KEY, "stale-block")
        dm.hot_cache_put("router:abc", ["webSearch"])
        cb = self._wire(dm)
        try:
            graph_store.create_node("Eve", "user fact", parent_id=BRANCH_USER)
        finally:
            unregister_graph_mutation_listener(cb)

        assert dm.hot_cache_get(dm.WARM_PROFILE_CACHE_KEY) is None
        # Other cache entries are untouched.
        assert dm.hot_cache_get("router:abc") == ["webSearch"]

    def test_directives_write_invalidates_warm_profile(self, graph_store):
        dm = DialogueMemory()
        dm.hot_cache_put(dm.WARM_PROFILE_CACHE_KEY, "stale-block")
        cb = self._wire(dm)
        try:
            graph_store.create_node(
                "be concise", "rule", parent_id=BRANCH_DIRECTIVES,
            )
        finally:
            unregister_graph_mutation_listener(cb)

        assert dm.hot_cache_get(dm.WARM_PROFILE_CACHE_KEY) is None

    def test_world_write_does_not_invalidate_warm_profile(self, graph_store):
        dm = DialogueMemory()
        dm.hot_cache_put(dm.WARM_PROFILE_CACHE_KEY, "fresh-block")
        cb = self._wire(dm)
        try:
            graph_store.create_node(
                "Paris", "world fact", parent_id=BRANCH_WORLD,
            )
        finally:
            unregister_graph_mutation_listener(cb)

        # World-branch writes are noise for the warm profile.
        assert dm.hot_cache_get(dm.WARM_PROFILE_CACHE_KEY) == "fresh-block"
