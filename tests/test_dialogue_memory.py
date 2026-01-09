"""Tests for dialogue memory and diary redaction functionality."""

import pytest
import time
import threading
from unittest.mock import Mock, patch
from datetime import datetime, timezone

from src.jarvis.memory.conversation import (
    DialogueMemory,
    update_daily_conversation_summary,
    update_diary_from_dialogue_memory,
)
from src.jarvis.reply.engine import run_reply_engine
from src.jarvis.utils.redact import redact


@pytest.mark.unit
class TestDialogueMemory:
    """Test dialogue memory conversation flow preservation."""
    
    def test_add_interaction_basic(self):
        """Test basic interaction storage."""
        dm = DialogueMemory()
        dm.add_interaction("Hello", "Hi there!")
        
        chunks = dm.get_pending_chunks()
        assert len(chunks) == 2
        assert "User: Hello" in chunks
        assert "Assistant: Hi there!" in chunks
    
    def test_add_interaction_preserves_order(self):
        """Test that multiple interactions preserve chronological order."""
        dm = DialogueMemory()
        dm.add_interaction("First message", "First response")
        dm.add_interaction("Second message", "Second response")
        
        chunks = dm.get_pending_chunks()
        assert len(chunks) == 4
        assert chunks[0] == "User: First message"
        assert chunks[1] == "Assistant: First response"
        assert chunks[2] == "User: Second message"
        assert chunks[3] == "Assistant: Second response"
    
    def test_add_interaction_with_conversation_flow(self):
        """Test storing full conversation flow in user_text."""
        dm = DialogueMemory()
        conversation_flow = "User: london, please\nAssistant: I'll check London weather\nUser: what's the temperature?\nAssistant: It's 18°C in London"
        dm.add_interaction(conversation_flow, "")
        
        chunks = dm.get_pending_chunks()
        assert len(chunks) == 1
        assert chunks[0] == f"User: {conversation_flow}"
    
    def test_should_update_diary_logic(self):
        """Test diary update timing logic."""
        dm = DialogueMemory(inactivity_timeout=1.0)  # 1 second timeout
        
        # No interactions yet
        assert not dm.should_update_diary()
        
        # Add interaction
        dm.add_interaction("Hello", "Hi")
        assert not dm.should_update_diary()  # Too soon
        
        # Mock time passage
        import time
        with patch('time.time', return_value=time.time() + 2.0):
            assert dm.should_update_diary()  # Timeout passed
    
    def test_clear_pending_updates(self):
        """Test clearing pending diary updates."""
        dm = DialogueMemory(inactivity_timeout=0.1)  # Short timeout for testing
        dm.add_interaction("Hello", "Hi")
        
        # Mock time passage to trigger diary update
        import time
        with patch('time.time', return_value=time.time() + 1.0):
            assert dm.should_update_diary()
            dm.clear_pending_updates()
            assert not dm.should_update_diary()


class TestReplyEngineDialogueMemory:
    """Test reply engine dialogue memory integration."""
    
    @patch('src.jarvis.reply.engine.chat_with_messages')
    @patch('src.jarvis.reply.engine.extract_text_from_response')
    @patch('src.jarvis.profile.profiles.select_profile_llm')
    def test_dialogue_memory_preserves_message_order(self, mock_profile, mock_extract, mock_chat):
        """Test that reply engine stores conversation in correct order."""
        # Mock dependencies
        mock_profile.return_value = "developer"
        mock_extract.return_value = "Final response"
        mock_chat.return_value = {"message": {"content": "Final response"}}
        
        # Mock database and config
        mock_db = Mock()
        mock_cfg = Mock()
        mock_cfg.ollama_base_url = "http://localhost:11434"
        mock_cfg.ollama_chat_model = "test"
        mock_cfg.active_profiles = ["developer"]
        mock_cfg.voice_debug = False
        mock_cfg.llm_profile_select_timeout_sec = 30.0
        mock_cfg.llm_tools_timeout_sec = 8.0
        mock_cfg.llm_embed_timeout_sec = 10.0
        mock_cfg.llm_chat_timeout_sec = 45.0
        mock_cfg.memory_enrichment_max_results = 5
        mock_cfg.location_ip_address = None
        mock_cfg.location_auto_detect = False
        mock_cfg.agentic_max_turns = 8
        
        # Create dialogue memory
        dialogue_memory = DialogueMemory()
        
        # Run reply engine
        result = run_reply_engine(
            db=mock_db,
            cfg=mock_cfg,
            tts=None,
            text="What's the weather in London?",
            dialogue_memory=dialogue_memory
        )
        
        # Check that dialogue memory was updated
        chunks = dialogue_memory.get_pending_chunks()
        assert len(chunks) == 2  # Now stores individual messages
        
        # Check that both messages are stored correctly
        assert "User: What's the weather in London?" in chunks
        assert "Assistant: Final response" in chunks
    
    @patch('src.jarvis.reply.engine.chat_with_messages')
    @patch('src.jarvis.reply.engine.extract_text_from_response')
    @patch('src.jarvis.profile.profiles.select_profile_llm')
    @patch('src.jarvis.reply.engine.run_tool_with_retries')
    def test_dialogue_memory_filters_tool_calls(self, mock_tool, mock_profile, mock_extract, mock_chat):
        """Test that JSON tool calls are filtered from dialogue memory."""
        # Mock dependencies
        mock_profile.return_value = "developer"
        mock_tool.return_value = Mock(reply_text="Weather data", error_message=None)
        
        # Mock multi-turn conversation: structured tool call then final response
        mock_chat.side_effect = [
            {
                "message": {
                    "content": "",
                    "tool_calls": [{
                        "id": "call_12345",
                        "function": {
                            "name": "webSearch",
                            "arguments": {"query": "London weather"}
                        }
                    }]
                }
            },
            {"message": {"content": "It's sunny in London today!"}}
        ]
        mock_extract.side_effect = [
            "",  # Empty content for tool call
            "It's sunny in London today!"
        ]
        
        # Mock database and config
        mock_db = Mock()
        mock_cfg = Mock()
        mock_cfg.ollama_base_url = "http://localhost:11434"
        mock_cfg.ollama_chat_model = "test"
        mock_cfg.active_profiles = ["developer"]
        mock_cfg.voice_debug = False
        mock_cfg.llm_profile_select_timeout_sec = 30.0
        mock_cfg.llm_tools_timeout_sec = 8.0
        mock_cfg.llm_embed_timeout_sec = 10.0
        mock_cfg.llm_chat_timeout_sec = 45.0
        mock_cfg.memory_enrichment_max_results = 5
        mock_cfg.location_ip_address = None
        mock_cfg.location_auto_detect = False
        mock_cfg.agentic_max_turns = 8
        
        # Create dialogue memory
        dialogue_memory = DialogueMemory()
        
        # Run reply engine
        result = run_reply_engine(
            db=mock_db,
            cfg=mock_cfg,
            tts=None,
            text="What's the weather in London?",
            dialogue_memory=dialogue_memory
        )
        
        # Check that dialogue memory was updated
        chunks = dialogue_memory.get_pending_chunks()
        assert len(chunks) == 2  # User message and assistant response stored separately
        
        # Should include user input and final response
        assert "User: What's the weather in London?" in chunks
        assert "Assistant: It's sunny in London today!" in chunks

        # Should NOT include the tool call
        for chunk in chunks:
            assert 'call_12345' not in chunk


class TestDiaryRedaction:
    """Test diary redaction functionality."""
    
    def test_redact_sensitive_info(self):
        """Test that sensitive information is properly redacted."""
        sensitive_text = "My email is user@example.com and my apikey: sk-abcd1234567890abcdef"
        redacted = redact(sensitive_text)
        
        assert "[REDACTED_EMAIL]" in redacted
        assert "[REDACTED]" in redacted  # API key pattern uses different format
        assert "user@example.com" not in redacted
        assert "sk-abcd1234567890abcdef" not in redacted
    
    @patch('src.jarvis.memory.conversation.generate_conversation_summary')
    def test_diary_update_redacts_chunks(self, mock_summary):
        """Test that diary updates redact sensitive information from chunks."""
        # Mock summary generation
        mock_summary.return_value = ("Daily summary", ["topic1", "topic2"])
        
        # Mock database
        mock_db = Mock()
        mock_db.get_conversation_summary.return_value = None
        mock_db.upsert_conversation_summary.return_value = 1
        
        # Create chunks with sensitive information
        sensitive_chunks = [
            "User: My email is sensitive@example.com",
            "Assistant: I'll help you with that",
            "User: Here's my apikey: sk-abcdef123456"
        ]
        
        # Call diary update function
        result = update_daily_conversation_summary(
            db=mock_db,
            new_chunks=sensitive_chunks,
            ollama_base_url="http://localhost:11434",
            ollama_chat_model="test",
            ollama_embed_model="test",
            source_app="test"
        )
        
        # Verify summary was called with redacted chunks
        mock_summary.assert_called_once()
        redacted_chunks = mock_summary.call_args[0][0]  # First argument to generate_conversation_summary
        
        # Check that sensitive info was redacted
        redacted_text = " ".join(redacted_chunks)
        assert "[REDACTED_EMAIL]" in redacted_text
        assert "[REDACTED]" in redacted_text  # API key pattern uses different format
        assert "sensitive@example.com" not in redacted_text
        assert "sk-abcdef123456" not in redacted_text
    
    @patch('src.jarvis.memory.conversation.generate_conversation_summary')
    def test_diary_update_preserves_conversation_flow(self, mock_summary):
        """Test that diary updates preserve conversation order after redaction."""
        # Mock summary generation
        mock_summary.return_value = ("Daily summary", ["topic1", "topic2"])
        
        # Mock database
        mock_db = Mock()
        mock_db.get_conversation_summary.return_value = None
        mock_db.upsert_conversation_summary.return_value = 1
        
        # Create ordered conversation chunks
        chunks = [
            "User: Hello there",
            "Assistant: Hi! How can I help?",
            "User: What's the weather?",
            "Assistant: Let me check for you"
        ]
        
        # Call diary update function
        result = update_daily_conversation_summary(
            db=mock_db,
            new_chunks=chunks,
            ollama_base_url="http://localhost:11434",
            ollama_chat_model="test",
            ollama_embed_model="test",
            source_app="test"
        )
        
        # Verify summary was called with chunks in correct order
        mock_summary.assert_called_once()
        processed_chunks = mock_summary.call_args[0][0]  # First argument
        
        assert len(processed_chunks) == 4
        assert processed_chunks[0] == "User: Hello there"
        assert processed_chunks[1] == "Assistant: Hi! How can I help?"
        assert processed_chunks[2] == "User: What's the weather?"
        assert processed_chunks[3] == "Assistant: Let me check for you"


class TestDialogueMemoryIntegration:
    """Integration tests for dialogue memory with redaction."""
    
    def test_full_flow_with_sensitive_data(self):
        """Test complete flow from dialogue memory to redacted diary."""
        # Create dialogue memory with sensitive information
        dm = DialogueMemory()
        sensitive_conversation = (
            "User: My email is test@example.com\n"
            "Assistant: I can help with that\n"
            "User: Here's my apikey: sk-1234567890\n"
            "Assistant: Thanks, I'll process that securely"
        )
        dm.add_interaction(sensitive_conversation, "")
        
        # Get chunks (should contain sensitive info)
        chunks = dm.get_pending_chunks()
        assert len(chunks) == 1
        chunk_content = chunks[0]
        assert "test@example.com" in chunk_content
        assert "sk-1234567890" in chunk_content
        
        # Simulate diary update redaction
        from src.jarvis.utils.redact import redact
        redacted_chunks = [redact(chunk) for chunk in chunks]
        redacted_content = redacted_chunks[0]
        
        # Verify redaction worked
        assert "[REDACTED_EMAIL]" in redacted_content
        assert "[REDACTED]" in redacted_content  # API key pattern uses different format
        assert "test@example.com" not in redacted_content
        assert "sk-1234567890" not in redacted_content
        
        # Verify conversation flow is preserved
        assert "User: My email is [REDACTED_EMAIL]" in redacted_content
        assert "Assistant: I can help with that" in redacted_content
        assert "apikey=[REDACTED]" in redacted_content
        assert "Assistant: Thanks, I'll process that securely" in redacted_content


@pytest.mark.unit
class TestDialogueMemoryEdgeCases:
    """Test edge cases for dialogue memory thread safety and long conversations."""

    def test_thread_safety_concurrent_add_and_read(self):
        """Test that concurrent add and read operations don't cause race conditions."""
        dm = DialogueMemory()
        errors = []
        iterations = 100

        def add_messages():
            for i in range(iterations):
                try:
                    dm.add_message("user", f"Message {i}")
                except Exception as e:
                    errors.append(f"add_message error: {e}")

        def read_messages():
            for _ in range(iterations):
                try:
                    dm.get_recent_messages()
                    dm.get_pending_chunks()
                    dm.has_recent_messages()
                except Exception as e:
                    errors.append(f"read error: {e}")

        # Run concurrent operations
        threads = [
            threading.Thread(target=add_messages),
            threading.Thread(target=read_messages),
            threading.Thread(target=add_messages),
            threading.Thread(target=read_messages),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread safety errors: {errors}"

    def test_new_message_during_diary_update_not_lost(self):
        """Test that messages added during diary update are not incorrectly marked as saved."""
        dm = DialogueMemory(inactivity_timeout=0.1)

        # Add initial message
        dm.add_message("user", "First message")
        time.sleep(0.01)  # Small delay to ensure different timestamp
        dm.add_message("assistant", "First response")

        # Get current timestamp (simulating what update_diary_from_dialogue_memory does)
        snapshot_timestamp = time.time()

        # Get pending chunks (2 messages)
        chunks_before = dm.get_pending_chunks()
        assert len(chunks_before) == 2

        # Simulate new message arriving during LLM summarization
        time.sleep(0.01)
        dm.add_message("user", "New message during update")

        # Mark saved up to snapshot (not including new message)
        dm.mark_saved_up_to(snapshot_timestamp)

        # New message should still be pending
        chunks_after = dm.get_pending_chunks()
        assert len(chunks_after) == 1
        assert "New message during update" in chunks_after[0]

    def test_mark_saved_up_to_preserves_new_messages(self):
        """Test that mark_saved_up_to only marks messages up to the given timestamp."""
        dm = DialogueMemory()

        # Add messages at different times
        dm.add_message("user", "Old message 1")
        time.sleep(0.05)
        cutoff_time = time.time()
        time.sleep(0.05)
        dm.add_message("user", "New message 2")
        time.sleep(0.05)
        dm.add_message("user", "New message 3")

        # Mark only old messages as saved
        dm.mark_saved_up_to(cutoff_time)

        # New messages should still be pending
        pending = dm.get_pending_chunks()
        assert len(pending) == 2
        assert any("New message 2" in chunk for chunk in pending)
        assert any("New message 3" in chunk for chunk in pending)

    def test_long_conversation_forces_diary_update(self):
        """Test that very long conversations force diary update to prevent data loss."""
        dm = DialogueMemory(inactivity_timeout=300.0)  # 5 minute inactivity timeout

        # Add a message and simulate it being old (older than MAX_UNSAVED_AGE_SEC)
        dm.add_message("user", "Old message")

        # Manually adjust the message timestamp to be old
        with dm._lock:
            ts, role, content = dm._messages[0]
            # Make it 11 minutes old (beyond MAX_UNSAVED_AGE_SEC of 10 minutes)
            old_ts = time.time() - (DialogueMemory.MAX_UNSAVED_AGE_SEC + 60)
            dm._messages[0] = (old_ts, role, content)

        # Should trigger diary update even though user is "active" (recent _last_activity_time)
        assert dm.should_update_diary()

    def test_long_conversation_does_not_force_if_recent(self):
        """Test that recent messages don't trigger forced diary update."""
        dm = DialogueMemory(inactivity_timeout=300.0)

        # Add a recent message
        dm.add_message("user", "Recent message")

        # Should not trigger diary update (not inactive and not too old)
        assert not dm.should_update_diary()

    def test_cleanup_removes_old_saved_messages(self):
        """Test that old saved messages are cleaned up from memory."""
        dm = DialogueMemory()

        # Add messages
        dm.add_message("user", "Message 1")
        time.sleep(0.01)
        dm.add_message("user", "Message 2")

        # Mark all as saved
        dm.clear_pending_updates()

        # Manually make messages old (beyond RECENT_WINDOW_SEC)
        with dm._lock:
            old_ts = time.time() - (DialogueMemory.RECENT_WINDOW_SEC + 60)
            dm._messages = [
                (old_ts, role, content) for _, role, content in dm._messages
            ]
            dm._cleanup_old_messages()

        # Old saved messages should be removed
        assert len(dm._messages) == 0

    def test_cleanup_keeps_unsaved_old_messages(self):
        """Test that old unsaved messages are NOT cleaned up (needed for diary)."""
        dm = DialogueMemory()

        # Add messages
        dm.add_message("user", "Unsaved message")

        # Manually make message old but don't mark as saved
        with dm._lock:
            old_ts = time.time() - (DialogueMemory.RECENT_WINDOW_SEC + 60)
            dm._messages = [
                (old_ts, role, content) for _, role, content in dm._messages
            ]
            dm._cleanup_old_messages()

        # Old unsaved messages should still exist (needed for diary update)
        assert len(dm._messages) == 1

    def test_has_pending_chunks(self):
        """Test has_pending_chunks method."""
        dm = DialogueMemory()

        # No messages yet
        assert not dm.has_pending_chunks()

        # Add message
        dm.add_message("user", "Hello")
        assert dm.has_pending_chunks()

        # Mark as saved
        dm.clear_pending_updates()
        assert not dm.has_pending_chunks()

    def test_should_update_diary_returns_false_when_no_pending(self):
        """Test that should_update_diary returns False when no pending chunks."""
        dm = DialogueMemory(inactivity_timeout=0.1)

        # No messages
        assert not dm.should_update_diary()

        # Add and save messages
        dm.add_message("user", "Hello")
        dm.clear_pending_updates()

        # Even after timeout, should return False if no pending
        time.sleep(0.15)
        assert not dm.should_update_diary()

    @patch('src.jarvis.memory.conversation.update_daily_conversation_summary')
    def test_update_diary_preserves_new_messages_during_slow_llm(self, mock_summary):
        """Integration test: messages arriving during slow LLM call are preserved."""
        dm = DialogueMemory(inactivity_timeout=0.1)
        mock_db = Mock()

        # Add initial messages
        dm.add_message("user", "Initial message")
        dm.add_message("assistant", "Initial response")

        # Simulate slow LLM call that takes time
        def slow_summary(*args, **kwargs):
            # Simulate user sending new message during LLM call
            dm.add_message("user", "Message during LLM call")
            return 123  # Return summary ID

        mock_summary.return_value = 123
        mock_summary.side_effect = slow_summary

        # Wait for inactivity timeout
        time.sleep(0.15)

        # Run diary update
        result = update_diary_from_dialogue_memory(
            db=mock_db,
            dialogue_memory=dm,
            ollama_base_url="http://localhost",
            ollama_chat_model="test",
            ollama_embed_model="test",
            force=True,
        )

        assert result == 123

        # New message should still be pending
        pending = dm.get_pending_chunks()
        assert len(pending) == 1
        assert "Message during LLM call" in pending[0]


@pytest.mark.unit
class TestDialogueMemoryConstants:
    """Test that DialogueMemory constants are reasonable."""

    def test_recent_window_is_5_minutes(self):
        """Verify RECENT_WINDOW_SEC is 5 minutes (300 seconds)."""
        assert DialogueMemory.RECENT_WINDOW_SEC == 300.0

    def test_max_unsaved_age_is_10_minutes(self):
        """Verify MAX_UNSAVED_AGE_SEC is 10 minutes (600 seconds)."""
        assert DialogueMemory.MAX_UNSAVED_AGE_SEC == 600.0

    def test_max_unsaved_age_exceeds_recent_window(self):
        """Verify MAX_UNSAVED_AGE_SEC > RECENT_WINDOW_SEC (otherwise data loss possible)."""
        assert DialogueMemory.MAX_UNSAVED_AGE_SEC > DialogueMemory.RECENT_WINDOW_SEC
