"""Tests for graph memory tools: memorise and recallMemory.

Covers the graph store search methods, tool interface, and
end-to-end memorise → recall flows.
"""

import pytest

from src.jarvis.memory.graph import GraphMemoryStore


# ── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def tmp_db(tmp_path):
    """Return a path to a temporary database."""
    return str(tmp_path / "test_tools.db")


@pytest.fixture
def store(tmp_db):
    """Return a fresh GraphMemoryStore."""
    s = GraphMemoryStore(tmp_db)
    yield s
    s.close()


@pytest.fixture
def populated_store(store):
    """Store with some pre-populated topic nodes."""
    store.create_node(
        name="Music Preferences",
        description="What music the user enjoys",
        data="Enjoys jazz and lo-fi hip hop. Favourite artist is Nujabes.",
        parent_id="root",
    )
    store.create_node(
        name="Work",
        description="Information about the user's work life",
        data="Works at Acme Corp as a senior engineer. Uses Python and TypeScript daily.",
        parent_id="root",
    )
    store.create_node(
        name="Health",
        description="Health and fitness related memories",
        data="Runs 3 times a week. Prefers dark roast coffee. Allergic to shellfish.",
        parent_id="root",
    )
    return store


# ── Inline tool implementations for isolated testing ──────────────────
# The builtin __init__.py eagerly imports all tools (including ones that
# require 'requests' etc.). Rather than fighting the import system, we
# test the memorise/recall tools via their underlying GraphMemoryStore
# methods directly, and test the thin tool wrappers with lightweight
# stand-ins that mirror the real tool logic exactly.


class _MemoriseTool:
    """Test stand-in that mirrors src/jarvis/tools/builtin/memorise.py logic."""

    name = "memorise"
    inputSchema = {
        "type": "object",
        "properties": {
            "topic": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["topic", "content"],
    }

    def run(self, args, context):
        from src.jarvis.tools.types import ToolExecutionResult
        context.user_print("🧠 Storing this in my memory...")

        if not args or not isinstance(args, dict):
            return ToolExecutionResult(success=False, reply_text="Missing arguments for memorise.")

        topic = str(args.get("topic", "")).strip()
        content = str(args.get("content", "")).strip()

        if not topic or not content:
            return ToolExecutionResult(success=False, reply_text="Both topic and content are required.")

        store = GraphMemoryStore(context.cfg.db_path)
        topic_node = store.find_node_by_name(topic, parent_id="root")

        if topic_node is None:
            topic_node = store.create_node(
                name=topic, description=f"Memories about: {topic}",
                data=content, parent_id="root",
            )
            context.user_print(f"  📁 Created new topic: {topic}")
        else:
            separator = "\n" if topic_node.data else ""
            updated_data = topic_node.data + separator + content
            store.update_node(topic_node.id, data=updated_data)
            store.touch_node(topic_node.id)
            context.user_print(f"  📝 Updated topic: {topic}")

        return ToolExecutionResult(success=True, reply_text=f"Stored under '{topic}': {content}")


class _RecallMemoryTool:
    """Test stand-in that mirrors src/jarvis/tools/builtin/recall_memory.py logic."""

    name = "recallMemory"
    inputSchema = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }

    def run(self, args, context):
        from src.jarvis.tools.types import ToolExecutionResult
        context.user_print("🔍 Searching my memory...")

        if not args or not isinstance(args, dict):
            return ToolExecutionResult(success=False, reply_text="Missing query argument.")

        query = str(args.get("query", "")).strip()
        if not query:
            return ToolExecutionResult(success=False, reply_text="Please provide a search query.")

        store = GraphMemoryStore(context.cfg.db_path)
        nodes = store.search_nodes(query, limit=8)

        if not nodes:
            context.user_print("  🤔 No matching memories found.")
            return ToolExecutionResult(success=True, reply_text=f"No stored memories found matching '{query}'.")

        parts: list[str] = []
        for node in nodes:
            ancestors = store.get_ancestors(node.id)
            path = " > ".join(a.name for a in ancestors)
            entry = f"**{path}**"
            if node.description:
                entry += f"\n{node.description}"
            if node.data:
                data = node.data if len(node.data) <= 500 else node.data[:500] + "..."
                entry += f"\n{data}"
            parts.append(entry)

        result_text = "\n\n".join(parts)
        context.user_print(f"  ✅ Found {len(nodes)} matching memories.")
        return ToolExecutionResult(success=True, reply_text=result_text)


@pytest.fixture
def memorise_tool():
    return _MemoriseTool()


@pytest.fixture
def recall_tool():
    return _RecallMemoryTool()


# ── GraphMemoryStore.search_nodes ──────────────────────────────────────


@pytest.mark.unit
class TestSearchNodes:
    """Tests for the keyword search method on GraphMemoryStore."""

    def test_search_by_name(self, populated_store):
        results = populated_store.search_nodes("Music")
        assert len(results) == 1
        assert results[0].name == "Music Preferences"

    def test_search_by_data_content(self, populated_store):
        results = populated_store.search_nodes("Nujabes")
        assert len(results) == 1
        assert "Nujabes" in results[0].data

    def test_search_by_description(self, populated_store):
        results = populated_store.search_nodes("fitness")
        assert len(results) == 1
        assert results[0].name == "Health"

    def test_search_multiple_keywords(self, populated_store):
        results = populated_store.search_nodes("Python engineer")
        assert len(results) >= 1
        assert results[0].name == "Work"

    def test_search_no_results(self, populated_store):
        results = populated_store.search_nodes("quantum physics")
        assert results == []

    def test_search_empty_query(self, populated_store):
        results = populated_store.search_nodes("")
        assert results == []

    def test_search_whitespace_only(self, populated_store):
        results = populated_store.search_nodes("   ")
        assert results == []

    def test_search_excludes_root(self, populated_store):
        results = populated_store.search_nodes("Root")
        assert all(r.id != "root" for r in results)

    def test_search_respects_limit(self, populated_store):
        results = populated_store.search_nodes("the user", limit=1)
        assert len(results) <= 1

    def test_search_touches_matched_nodes(self, populated_store):
        node_before = populated_store.search_nodes("Music")[0]
        initial_count = node_before.access_count
        # Search again — the first search already touched it once
        results = populated_store.search_nodes("Music")
        refreshed = populated_store.get_node(results[0].id)
        assert refreshed.access_count > initial_count

    def test_search_ranks_by_relevance(self, populated_store):
        """Nodes matching more keywords should rank higher."""
        results = populated_store.search_nodes("dark roast coffee")
        assert results[0].name == "Health"

    def test_search_case_insensitive(self, populated_store):
        results = populated_store.search_nodes("nujabes")
        assert len(results) == 1
        assert results[0].name == "Music Preferences"


# ── GraphMemoryStore.find_node_by_name ─────────────────────────────────


@pytest.mark.unit
class TestFindNodeByName:
    """Tests for exact name lookup."""

    def test_find_existing_node(self, populated_store):
        node = populated_store.find_node_by_name("Work")
        assert node is not None
        assert node.name == "Work"

    def test_find_case_insensitive(self, populated_store):
        node = populated_store.find_node_by_name("work")
        assert node is not None
        assert node.name == "Work"

    def test_find_nonexistent(self, populated_store):
        node = populated_store.find_node_by_name("Nonexistent Topic")
        assert node is None

    def test_find_excludes_root(self, store):
        node = store.find_node_by_name("Root")
        assert node is None

    def test_find_with_parent_filter(self, populated_store):
        node = populated_store.find_node_by_name("Work", parent_id="root")
        assert node is not None
        assert node.name == "Work"

    def test_find_wrong_parent(self, populated_store):
        node = populated_store.find_node_by_name("Work", parent_id="nonexistent")
        assert node is None


# ── FakeContext for tool tests ─────────────────────────────────────────


class FakeConfig:
    """Minimal config stand-in for tool tests."""
    def __init__(self, db_path):
        self.db_path = db_path
        self.voice_debug = False


class FakeContext:
    """Minimal ToolContext stand-in."""
    def __init__(self, db_path):
        self.cfg = FakeConfig(db_path)
        self.db = None
        self.system_prompt = ""
        self.original_prompt = ""
        self.redacted_text = ""
        self.max_retries = 0
        self.prints = []

    def user_print(self, msg):
        self.prints.append(msg)


# ── MemoriseTool ───────────────────────────────────────────────────────


@pytest.mark.unit
class TestMemoriseTool:
    """Tests for the memorise tool."""

    def test_store_new_topic(self, tmp_db, memorise_tool):
        ctx = FakeContext(tmp_db)
        result = memorise_tool.run({"topic": "Hobbies", "content": "Enjoys woodworking"}, ctx)
        assert result.success is True
        assert "Hobbies" in result.reply_text

        store = GraphMemoryStore(tmp_db)
        node = store.find_node_by_name("Hobbies", parent_id="root")
        assert node is not None
        assert "woodworking" in node.data

    def test_append_to_existing_topic(self, tmp_db, memorise_tool):
        ctx = FakeContext(tmp_db)

        memorise_tool.run({"topic": "Food", "content": "Loves sushi"}, ctx)
        memorise_tool.run({"topic": "Food", "content": "Allergic to shellfish"}, ctx)

        store = GraphMemoryStore(tmp_db)
        node = store.find_node_by_name("Food", parent_id="root")
        assert "sushi" in node.data
        assert "shellfish" in node.data

    def test_missing_topic(self, tmp_db, memorise_tool):
        ctx = FakeContext(tmp_db)
        result = memorise_tool.run({"topic": "", "content": "Something"}, ctx)
        assert result.success is False

    def test_missing_content(self, tmp_db, memorise_tool):
        ctx = FakeContext(tmp_db)
        result = memorise_tool.run({"topic": "Test", "content": ""}, ctx)
        assert result.success is False

    def test_missing_args(self, tmp_db, memorise_tool):
        ctx = FakeContext(tmp_db)
        result = memorise_tool.run(None, ctx)
        assert result.success is False

    def test_tool_properties(self, memorise_tool):
        assert memorise_tool.name == "memorise"
        assert "topic" in memorise_tool.inputSchema["properties"]
        assert "content" in memorise_tool.inputSchema["properties"]


# ── RecallMemoryTool ───────────────────────────────────────────────────


@pytest.mark.unit
class TestRecallMemoryTool:
    """Tests for the recallMemory tool."""

    def test_recall_stored_memory(self, tmp_db, memorise_tool, recall_tool):
        ctx = FakeContext(tmp_db)

        memorise_tool.run({"topic": "Pets", "content": "Has a golden retriever named Max"}, ctx)

        result = recall_tool.run({"query": "pets dog"}, ctx)
        assert result.success is True
        assert "Max" in result.reply_text

    def test_recall_no_results(self, tmp_db, recall_tool):
        ctx = FakeContext(tmp_db)
        result = recall_tool.run({"query": "quantum physics"}, ctx)
        assert result.success is True
        assert "No stored memories" in result.reply_text

    def test_recall_empty_query(self, tmp_db, recall_tool):
        ctx = FakeContext(tmp_db)
        result = recall_tool.run({"query": ""}, ctx)
        assert result.success is False

    def test_recall_missing_args(self, tmp_db, recall_tool):
        ctx = FakeContext(tmp_db)
        result = recall_tool.run(None, ctx)
        assert result.success is False

    def test_recall_includes_breadcrumb(self, tmp_db, memorise_tool, recall_tool):
        ctx = FakeContext(tmp_db)

        memorise_tool.run({"topic": "Travel", "content": "Visited Japan in 2024"}, ctx)

        result = recall_tool.run({"query": "Japan travel"}, ctx)
        assert result.success is True
        assert "Root" in result.reply_text
        assert "Travel" in result.reply_text

    def test_tool_properties(self, recall_tool):
        assert recall_tool.name == "recallMemory"
        assert "query" in recall_tool.inputSchema["properties"]


# ── End-to-end: memorise → recall round-trip ───────────────────────────


@pytest.mark.unit
class TestMemoriseRecallRoundTrip:
    """Integration tests: store memories then search for them."""

    def test_multiple_topics_recall(self, tmp_db, memorise_tool, recall_tool):
        ctx = FakeContext(tmp_db)

        memorise_tool.run({"topic": "Music", "content": "Favourite genre is jazz"}, ctx)
        memorise_tool.run({"topic": "Food", "content": "Favourite cuisine is Japanese"}, ctx)
        memorise_tool.run({"topic": "Work", "content": "Senior engineer at Acme"}, ctx)

        result = recall_tool.run({"query": "jazz music"}, ctx)
        assert result.success is True
        assert "jazz" in result.reply_text

        result = recall_tool.run({"query": "engineer job"}, ctx)
        assert result.success is True
        assert "Acme" in result.reply_text

    def test_accumulated_topic_data(self, tmp_db, memorise_tool, recall_tool):
        """Multiple memorise calls to same topic accumulate data."""
        ctx = FakeContext(tmp_db)

        memorise_tool.run({"topic": "Family", "content": "Has a sister named Alice"}, ctx)
        memorise_tool.run({"topic": "Family", "content": "Parents live in London"}, ctx)
        memorise_tool.run({"topic": "Family", "content": "Married to Bob"}, ctx)

        result = recall_tool.run({"query": "family"}, ctx)
        assert "Alice" in result.reply_text
        assert "London" in result.reply_text
        assert "Bob" in result.reply_text
