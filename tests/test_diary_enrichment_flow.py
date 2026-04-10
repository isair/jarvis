"""
Diary-to-Enrichment Flow Integration Tests

Tests the critical flow where dialogue memory is saved to the diary, cleaned
up from in-memory, and then retrieved via FTS search on a follow-up query.

This validates that after the unified RECENT_WINDOW_SEC = MAX_UNSAVED_AGE_SEC
change, context is not lost when messages are cleaned from memory — the
FTS pipeline successfully retrieves just-saved diary entries.
"""

import time
import pytest
from unittest.mock import patch


@pytest.mark.integration
class TestDiaryToEnrichmentFlow:
    """Test the full diary save → cleanup → enrichment retrieval pipeline."""

    def _create_dialogue_memory(self, timeout: float = 5.0):
        """Create a DialogueMemory with a short timeout for testing."""
        from jarvis.memory.conversation import DialogueMemory
        return DialogueMemory(inactivity_timeout=timeout)

    def _force_messages_old(self, dm, age_seconds: float):
        """Make all messages in dialogue memory appear old."""
        with dm._lock:
            now = time.time()
            dm._messages = [
                (now - age_seconds, role, content)
                for _, role, content in dm._messages
            ]
            dm._last_activity_time = now - age_seconds

    def test_diary_save_then_enrichment_retrieval_fts(self, db):
        """After diary save + cleanup, FTS enrichment finds the saved context.

        This is the core scenario: user discusses a topic, diary update fires,
        messages are cleaned from memory, then a follow-up query successfully
        retrieves the context from the diary via FTS search.
        """
        from jarvis.memory.conversation import (
            DialogueMemory,
            update_diary_from_dialogue_memory,
            search_conversation_memory_by_keywords,
        )

        dm = self._create_dialogue_memory(timeout=5.0)

        # Step 1: Simulate a conversation about a specific topic
        dm.add_message("user", "I've been working on a Python migration to async/await")
        dm.add_message("assistant", "That's a big refactor. Are you using asyncio or trio?")
        dm.add_message("user", "asyncio, and we're converting the database layer first")
        dm.add_message("assistant", "Good approach — the database layer benefits most from async")

        assert dm.has_pending_chunks(), "Should have pending chunks"

        # Step 2: Force diary update with mocked LLM summarisation
        mock_summary = (
            "User is working on migrating a Python codebase to async/await "
            "using asyncio. They are starting with the database layer conversion. "
            "The assistant recommended this approach as the database layer benefits "
            "most from async patterns."
        )
        mock_topics = "python, asyncio, async/await, database, migration, refactoring"

        with patch(
            "jarvis.memory.conversation.generate_conversation_summary",
            return_value=(mock_summary, mock_topics),
        ):
            summary_id = update_diary_from_dialogue_memory(
                db=db,
                dialogue_memory=dm,
                ollama_base_url="http://localhost:11434",
                ollama_chat_model="test",
                ollama_embed_model="test",
                force=True,
                timeout_sec=5.0,
            )

        assert summary_id is not None, "Diary update should succeed"
        print(f"\n  📝 Diary entry saved with ID: {summary_id}")

        # Step 3: Force messages old and trigger cleanup
        self._force_messages_old(dm, dm.RECENT_WINDOW_SEC + 60)
        dm.mark_saved_up_to(time.time())

        # Verify messages were cleaned up
        recent = dm.get_recent_messages()
        assert len(recent) == 0, "Messages should be cleaned from memory after save"
        print("  🧹 In-memory messages cleaned up")

        # Step 4: Search via FTS (no embeddings — simulates fallback path)
        results = search_conversation_memory_by_keywords(
            db=db,
            keywords=["asyncio", "database", "migration"],
            max_results=5,
        )

        print(f"  🔍 FTS search results: {len(results)} found")
        for i, r in enumerate(results):
            preview = r[:120] + "..." if len(r) > 120 else r
            print(f"    {i + 1}. {preview}")

        # Step 5: Verify enrichment finds the diary entry
        assert len(results) > 0, (
            "Enrichment should find the just-saved diary entry via FTS. "
            "This means context is NOT lost after cleanup."
        )

        # Verify the content is relevant
        combined = " ".join(results).lower()
        assert any(kw in combined for kw in ["asyncio", "async", "database", "migration"]), (
            f"Search results should contain relevant keywords. Got: {combined[:200]}"
        )
        print("  ✅ Enrichment successfully retrieved diary context after cleanup")

    def test_followup_query_finds_recent_diary_entry(self, db):
        """Simulate the exact flow: conversation → diary save → follow-up query.

        The follow-up query exercises the enrichment keyword extraction
        (mocked) and diary search (real FTS) to verify the full pipeline.
        """
        from jarvis.memory.conversation import (
            DialogueMemory,
            update_diary_from_dialogue_memory,
            search_conversation_memory_by_keywords,
        )

        dm = self._create_dialogue_memory(timeout=5.0)

        # User discusses their holiday plans
        dm.add_message("user", "I'm planning a trip to Tokyo in November")
        dm.add_message("assistant", "November is a great time for Tokyo — autumn foliage season!")
        dm.add_message("user", "I want to visit Shibuya and Akihabara")
        dm.add_message("assistant", "Both excellent choices. Shibuya for the crossing and shopping, Akihabara for electronics and anime culture.")

        # Save to diary
        mock_summary = (
            "User is planning a trip to Tokyo in November during autumn foliage season. "
            "They want to visit Shibuya for the famous crossing and shopping, and "
            "Akihabara for electronics and anime culture."
        )
        mock_topics = "tokyo, travel, japan, november, shibuya, akihabara, autumn"

        with patch(
            "jarvis.memory.conversation.generate_conversation_summary",
            return_value=(mock_summary, mock_topics),
        ):
            summary_id = update_diary_from_dialogue_memory(
                db=db,
                dialogue_memory=dm,
                ollama_base_url="http://localhost:11434",
                ollama_chat_model="test",
                ollama_embed_model="test",
                force=True,
            )

        assert summary_id is not None

        # Clean up in-memory messages (simulates the unified window expiry)
        self._force_messages_old(dm, dm.RECENT_WINDOW_SEC + 60)
        dm.mark_saved_up_to(time.time())
        assert len(dm.get_recent_messages()) == 0, "Memory should be empty"

        # User comes back and asks a follow-up
        # (Enrichment would extract keywords like: tokyo, trip, travel)
        followup_keywords = ["tokyo", "trip", "travel"]

        results = search_conversation_memory_by_keywords(
            db=db,
            keywords=followup_keywords,
            max_results=5,
        )

        print(f"\n  🗣️ Follow-up: 'what were my Tokyo plans again?'")
        print(f"  🔍 Enrichment keywords: {followup_keywords}")
        print(f"  📋 Results: {len(results)} found")

        assert len(results) > 0, (
            "Follow-up query should find the Tokyo trip diary entry via enrichment"
        )

        combined = " ".join(results).lower()
        assert "tokyo" in combined, "Results should mention Tokyo"
        assert any(kw in combined for kw in ["shibuya", "akihabara", "november"]), (
            "Results should include specific trip details"
        )
        print("  ✅ Follow-up successfully retrieved trip plans from diary")

    def test_multiple_diary_entries_searchable(self, db):
        """Multiple diary entries from different conversations are all searchable."""
        from jarvis.memory.conversation import (
            DialogueMemory,
            update_diary_from_dialogue_memory,
            search_conversation_memory_by_keywords,
        )

        dm = self._create_dialogue_memory(timeout=5.0)

        # First conversation: cooking
        dm.add_message("user", "Can you suggest a good pasta recipe?")
        dm.add_message("assistant", "Try a carbonara — eggs, pecorino, guanciale, and black pepper.")

        with patch(
            "jarvis.memory.conversation.generate_conversation_summary",
            return_value=(
                "User asked for a pasta recipe. Suggested carbonara with eggs, pecorino, guanciale, and black pepper.",
                "cooking, pasta, carbonara, recipe",
            ),
        ):
            id1 = update_diary_from_dialogue_memory(
                db=db, dialogue_memory=dm,
                ollama_base_url="http://localhost:11434",
                ollama_chat_model="test", ollama_embed_model="test",
                force=True,
            )

        assert id1 is not None
        self._force_messages_old(dm, dm.RECENT_WINDOW_SEC + 60)
        dm.mark_saved_up_to(time.time())

        # Second conversation: fitness
        dm.add_message("user", "What's a good strength training routine for beginners?")
        dm.add_message("assistant", "Start with compound lifts: squats, deadlifts, bench press, and overhead press.")

        # Second summary includes first conversation (LLM appends to previous)
        with patch(
            "jarvis.memory.conversation.generate_conversation_summary",
            return_value=(
                "User asked for a pasta recipe. Suggested carbonara with eggs, pecorino, guanciale, and black pepper. "
                "Later, user asked about beginner strength training. Recommended compound lifts: squats, deadlifts, bench press, and overhead press.",
                "cooking, pasta, carbonara, recipe, fitness, strength training, exercise, beginner, workout",
            ),
        ):
            id2 = update_diary_from_dialogue_memory(
                db=db, dialogue_memory=dm,
                ollama_base_url="http://localhost:11434",
                ollama_chat_model="test", ollama_embed_model="test",
                force=True,
            )

        assert id2 is not None
        self._force_messages_old(dm, dm.RECENT_WINDOW_SEC + 60)
        dm.mark_saved_up_to(time.time())

        # Both should be empty from memory
        assert len(dm.get_recent_messages()) == 0

        # Search for cooking — should find first entry
        cooking_results = search_conversation_memory_by_keywords(
            db=db, keywords=["pasta", "recipe", "cooking"], max_results=5,
        )
        assert len(cooking_results) > 0, "Should find cooking diary entry"
        assert "carbonara" in " ".join(cooking_results).lower()

        # Search for fitness — should find second entry
        fitness_results = search_conversation_memory_by_keywords(
            db=db, keywords=["strength", "training", "exercise"], max_results=5,
        )
        assert len(fitness_results) > 0, "Should find fitness diary entry"
        assert any(kw in " ".join(fitness_results).lower() for kw in ["squat", "deadlift", "bench"])

        print(f"\n  📝 Saved 2 diary entries (IDs: {id1}, {id2})")
        print(f"  🔍 Cooking search: {len(cooking_results)} results")
        print(f"  🔍 Fitness search: {len(fitness_results)} results")
        print("  ✅ Multiple diary entries independently searchable after cleanup")

    def test_concurrent_message_during_diary_update_preserved(self, db):
        """Messages arriving during diary update are NOT lost.

        While the diary update (slow LLM call) is processing, new messages
        arrive. These must survive cleanup and appear in the next diary update.
        """
        from jarvis.memory.conversation import (
            DialogueMemory,
            update_daily_conversation_summary,
        )

        dm = self._create_dialogue_memory(timeout=5.0)

        # Add initial messages
        dm.add_message("user", "Tell me about quantum computing")
        dm.add_message("assistant", "Quantum computing uses qubits instead of classical bits.")

        # Simulate the diary update flow manually to inject a concurrent message
        snapshot_timestamp = time.time()
        pending_chunks = dm.get_pending_chunks()
        assert len(pending_chunks) > 0

        # Simulate a new message arriving DURING the slow LLM summarisation
        time.sleep(0.01)  # Ensure timestamp differs
        dm.add_message("user", "What about quantum error correction?")

        # Mock the LLM summarisation result
        with patch(
            "jarvis.memory.conversation.generate_conversation_summary",
            return_value=(
                "Discussed quantum computing basics — qubits vs classical bits.",
                "quantum, computing, qubits",
            ),
        ):
            summary_id = update_daily_conversation_summary(
                db=db,
                new_chunks=pending_chunks,
                ollama_base_url="http://localhost:11434",
                ollama_chat_model="test",
                ollama_embed_model="test",
            )

        assert summary_id is not None

        # Mark saved up to the snapshot (NOT the current time)
        dm.mark_saved_up_to(snapshot_timestamp)

        # The concurrent message should still be pending
        assert dm.has_pending_chunks(), (
            "Message that arrived during diary update should still be pending"
        )

        new_pending = dm.get_pending_chunks()
        combined = " ".join(new_pending).lower()
        assert "quantum error correction" in combined, (
            "The concurrent message about error correction should be preserved"
        )

        print("\n  📝 Diary saved initial conversation")
        print("  ⏱️ New message arrived during save")
        print(f"  📋 Pending after save: {len(new_pending)} chunks")
        print("  ✅ Concurrent message preserved — no data loss")
