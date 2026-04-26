from __future__ import annotations
import json
import time
import threading
from datetime import datetime, timezone
from typing import Optional, List, Tuple, Union, Callable
from .db import Database
from ..llm import call_llm_direct
from .embeddings import get_embedding
from ..debug import debug_log
from ..utils.redact import redact, scrub_secrets


_UNTRUSTED_FENCE_BEGIN = "<<<BEGIN UNTRUSTED WEB EXTRACT>>>"
_UNTRUSTED_FENCE_END = "<<<END UNTRUSTED WEB EXTRACT>>>"


def _filter_contexts_by_time(
    contexts: List[str],
    from_time: Optional[str],
    to_time: Optional[str],
    voice_debug: bool = False
) -> List[str]:
    """Helper to filter context strings by time range."""
    if not from_time and not to_time:
        return contexts

    filtered = []
    from_dt = None
    to_dt = None

    try:
        if from_time:
            from_dt = datetime.fromisoformat(from_time.replace('Z', '+00:00'))
        if to_time:
            to_dt = datetime.fromisoformat(to_time.replace('Z', '+00:00'))
    except Exception as e:
        if voice_debug:
            debug_log(f"      📋 Error parsing time: {e}", "memory")
        return contexts

    import re
    for ctx in contexts:
        # Extract date from formatted text like "[2025-08-27] ..."
        date_match = re.match(r'\[(\d{4}-\d{2}-\d{2})\]', ctx)
        if date_match:
            date_str = date_match.group(1)
            try:
                ctx_date = datetime.fromisoformat(date_str + 'T00:00:00+00:00')

                in_range = True
                if from_dt and ctx_date.date() < from_dt.date():
                    in_range = False
                if to_dt and ctx_date.date() > to_dt.date():
                    in_range = False

                if in_range:
                    filtered.append(ctx)
            except Exception:
                filtered.append(ctx)  # Keep if can't parse date
        else:
            filtered.append(ctx)  # Keep non-dated entries

    return filtered


class DialogueMemory:
    """
    In-memory storage for recent dialogue interactions.
    Provides short-term context for the configured dialogue memory window.

    Thread-safe: uses a lock to protect against concurrent diary updates.
    Tracks saved messages by timestamp to prevent data loss when new messages
    arrive during diary update.

    The dialogue memory window and the forced diary update interval share the
    same duration (dialogue_memory_timeout). After a diary update, saved messages
    older than this window are cleaned up; the enrichment feature retrieves any
    relevant earlier context from the diary. The rolling transcript buffer is
    separate (ambient speech for intent judging).
    """

    def __init__(self, inactivity_timeout: float = 300.0, max_interactions: int = 20):
        """Initialize dialogue memory.

        The inactivity_timeout drives two unified durations:
        - RECENT_WINDOW_SEC: how long messages are kept in memory for context
        - MAX_UNSAVED_AGE_SEC: how old unsaved messages can get before forcing
          a diary update (same as the window, since enrichment covers older context)
        """
        self._messages: List[Tuple[float, str, str]] = []  # (timestamp, role, content)
        # Tool carryover: in-loop assistant-with-tool_calls + tool-role messages
        # from prior replies, so follow-up turns within the hot window can reuse
        # the prior tool output instead of re-fetching. Stored as a list of
        # (timestamp, [msg_dict, ...]) where each entry is one reply's worth of
        # tool-related messages. Excluded from `get_pending_chunks` so raw tool
        # payloads never reach the diary summariser.
        self._tool_turns: List[Tuple[float, List[dict]]] = []
        self._last_activity_time: float = time.time()
        self._inactivity_timeout = inactivity_timeout
        # Unified window: context retention = forced diary update interval
        self.RECENT_WINDOW_SEC = inactivity_timeout
        self.MAX_UNSAVED_AGE_SEC = inactivity_timeout
        # Track the timestamp up to which messages have been saved to diary
        # Messages with timestamp <= this value have been processed
        self._last_saved_timestamp: float = 0.0
        self._lock = threading.RLock()  # Reentrant lock for thread safety
        # Track the last profile used for follow-up detection
        self._last_profile: Optional[str] = None

    def add_message(self, role: str, content: str) -> None:
        """Add a message to recent memory. Thread-safe."""
        with self._lock:
            timestamp = time.time()
            self._messages.append((timestamp, role.strip(), content.strip()))
            self._last_activity_time = timestamp

    def get_recent_context(self) -> List[str]:
        """Get recent messages formatted as context strings."""
        messages = self.get_recent_messages()
        return [f"{msg['role'].title()}: {msg['content']}" for msg in messages]

    def get_recent_messages(self) -> List[dict]:
        """
        Get recent messages (last 5 minutes) formatted for LLM API.

        Returns:
            List of message dictionaries with 'role' and 'content' keys
        """
        with self._lock:
            if not self._messages:
                return []

            # Filter to last 5 minutes
            cutoff = time.time() - self.RECENT_WINDOW_SEC
            recent_messages = [msg for msg in self._messages if msg[0] >= cutoff]

            return [{"role": role, "content": content} for _, role, content in recent_messages]

    def record_tool_turn(self, tool_msgs: List[dict]) -> None:
        """Store in-loop tool-call/tool-role messages from a just-finished reply.

        Called once per reply with the tool-related messages extracted from the
        engine's messages array. These interleave with text messages on
        subsequent `get_recent_turns_with_tools` calls so follow-ups can see
        the prior tool output.
        """
        if not tool_msgs:
            return
        with self._lock:
            ts = time.time()
            scrubbed: List[dict] = []
            for m in tool_msgs:
                mm = dict(m)
                c = mm.get("content")
                if isinstance(c, str) and c:
                    # Tool outputs may contain PII or secrets (email bodies,
                    # API responses, scraped pages). Scrub before persisting
                    # so re-injection on the next turn can't leak them.
                    mm["content"] = scrub_secrets(c)
                scrubbed.append(mm)
            self._tool_turns.append((ts, scrubbed))

    def clear_tool_carryover(self) -> None:
        """Drop all stored tool-turn messages. Text messages are untouched."""
        with self._lock:
            self._tool_turns = []

    def get_recent_turns_with_tools(
        self,
        max_tool_turns: int = 2,
        per_entry_chars: int = 1200,
    ) -> List[dict]:
        """Like `get_recent_messages`, but interleaves stored tool turns in
        timestamp order. Only the most recent `max_tool_turns` tool groups
        survive; older ones are dropped wholesale (avoids orphan
        assistant-with-tool_calls without a matching tool result, which would
        break native tool calling).
        """
        with self._lock:
            if not self._messages and not self._tool_turns:
                return []
            cutoff = time.time() - self.RECENT_WINDOW_SEC
            # Build timeline of (ts, payload) where payload is either a single
            # text message dict or a list of tool messages.
            timeline: list = []
            for ts, role, content in self._messages:
                if ts >= cutoff:
                    timeline.append((ts, "msg", {"role": role, "content": content}))
            # Keep only the last N tool turns within the window.
            live_tool_turns = [(ts, msgs) for ts, msgs in self._tool_turns if ts >= cutoff]
            for ts, msgs in live_tool_turns[-max_tool_turns:]:
                truncated: list[dict] = []
                for m in msgs:
                    mm = dict(m)
                    c = mm.get("content")
                    if isinstance(c, str) and len(c) > per_entry_chars:
                        cut = c[:per_entry_chars].rstrip() + "…"
                        # If truncation sliced away the closing marker of an
                        # UNTRUSTED WEB EXTRACT fence, re-append it so the
                        # injection-defence fence stays intact downstream.
                        if (
                            _UNTRUSTED_FENCE_BEGIN in cut
                            and _UNTRUSTED_FENCE_END not in cut
                        ):
                            cut = cut + "\n" + _UNTRUSTED_FENCE_END
                        mm["content"] = cut
                    truncated.append(mm)
                timeline.append((ts, "group", truncated))
            timeline.sort(key=lambda t: t[0])
            flat: List[dict] = []
            for _, kind, payload in timeline:
                if kind == "msg":
                    flat.append(payload)
                else:
                    flat.extend(payload)
            return flat

    def has_recent_messages(self) -> bool:
        """Check if there are any messages in the last 5 minutes."""
        with self._lock:
            cutoff = time.time() - self.RECENT_WINDOW_SEC
            return any(ts >= cutoff for ts, _, _ in self._messages)

    def set_last_profile(self, profile: str) -> None:
        """Track the last profile used for follow-up detection."""
        with self._lock:
            self._last_profile = profile

    def get_last_profile(self) -> Optional[str]:
        """Get the last profile used, if within the recent window."""
        with self._lock:
            # Only return profile if we have recent messages
            cutoff = time.time() - self.RECENT_WINDOW_SEC
            if any(ts >= cutoff for ts, _, _ in self._messages):
                return self._last_profile
            return None

    # Compatibility and diary functionality
    def add_interaction(self, user_text: str, assistant_text: str) -> None:
        """Compatibility method - use add_message() instead."""
        if user_text.strip():
            self.add_message("user", user_text.strip())
        if assistant_text.strip():
            self.add_message("assistant", assistant_text.strip())

    def get_pending_chunks(self) -> List[str]:
        """Get unsaved messages as formatted chunks for diary update.

        Returns messages that haven't been saved to diary yet (timestamp > _last_saved_timestamp).
        Thread-safe.
        """
        with self._lock:
            # Get messages that haven't been saved yet
            unsaved_messages = [
                (ts, role, content) for ts, role, content in self._messages
                if ts > self._last_saved_timestamp
            ]
            return [f"{role.title()}: {content}" for _, role, content in unsaved_messages]

    def has_pending_chunks(self) -> bool:
        """Check if there are unsaved messages. Thread-safe."""
        with self._lock:
            return any(ts > self._last_saved_timestamp for ts, _, _ in self._messages)

    def should_update_diary(self) -> bool:
        """Check if diary should be updated based on inactivity timeout.

        Returns True if:
        1. There are unsaved messages AND user has been inactive for inactivity_timeout, OR
        2. There are unsaved messages older than MAX_UNSAVED_AGE_SEC (prevents data loss
           in very long conversations)
        """
        with self._lock:
            if not self.has_pending_chunks():
                return False

            current_time = time.time()

            # Standard inactivity check
            if (current_time - self._last_activity_time) >= self._inactivity_timeout:
                return True

            # Edge case: very long conversation - force update if old messages exist
            # This prevents context loss when a conversation exceeds the recent window
            oldest_unsaved = None
            for ts, _, _ in self._messages:
                if ts > self._last_saved_timestamp:
                    oldest_unsaved = ts
                    break  # First unsaved message is the oldest

            if oldest_unsaved is not None:
                unsaved_age = current_time - oldest_unsaved
                if unsaved_age >= self.MAX_UNSAVED_AGE_SEC:
                    return True

            return False

    def mark_saved_up_to(self, timestamp: float) -> None:
        """Mark all messages up to the given timestamp as saved.

        Thread-safe. Also cleans up old messages that have been saved.
        """
        with self._lock:
            self._last_saved_timestamp = max(self._last_saved_timestamp, timestamp)
            self._cleanup_old_messages()

    def _cleanup_old_messages(self) -> None:
        """Remove messages that are both saved and older than the recent window.

        Must be called while holding the lock.
        """
        current_time = time.time()
        # Keep messages that are either:
        # 1. Recent (within RECENT_WINDOW_SEC) - needed for LLM context
        # 2. Not yet saved (timestamp > _last_saved_timestamp) - needed for diary
        cutoff = current_time - self.RECENT_WINDOW_SEC
        self._messages = [
            (ts, role, content) for ts, role, content in self._messages
            if ts >= cutoff or ts > self._last_saved_timestamp
        ]

    def clear_pending_updates(self) -> None:
        """Mark all current messages as saved. Thread-safe.

        DEPRECATED: Use mark_saved_up_to() instead for proper timestamp tracking.
        Kept for backward compatibility.
        """
        with self._lock:
            if self._messages:
                # Mark all current messages as saved
                max_ts = max(ts for ts, _, _ in self._messages)
                self._last_saved_timestamp = max_ts
            self._cleanup_old_messages()


def generate_conversation_summary(
    recent_chunks: List[str],
    previous_summary: Optional[str],
    ollama_base_url: str,
    ollama_chat_model: str,
    timeout_sec: float = 30.0,
    on_token: Optional[Callable[[str], None]] = None,
    thinking: bool = False,
) -> Tuple[str, str]:
    """
    Generate a concise conversation summary from recent chunks and previous summary.

    Args:
        recent_chunks: List of conversation chunks to summarize
        previous_summary: Previous summary for today (if any)
        ollama_base_url: Ollama API base URL
        ollama_chat_model: Model to use
        timeout_sec: Request timeout
        on_token: Optional callback for streaming tokens (for live UI updates)

    Returns:
        Tuple of (summary, topics) where topics is comma-separated
    """
    from ..llm import call_llm_direct, call_llm_streaming

    chunks_text = "\n".join(recent_chunks[-10:])  # Last 10 chunks to keep context manageable

    system_prompt = """You are a conversation summarizer for a personal AI assistant. Your job is to create concise daily summaries of conversations that will be stored in a diary for future reference.

Create a summary that:
1. Captures the key topics discussed and important information shared
2. Is concise but informative (max 200 words)
3. Focuses on facts, decisions, and context that would be useful for future conversations
4. Includes any personal information, preferences, or important events mentioned
5. Maintains a neutral, factual tone
6. Records what the user asked about and what was established as true — NOT the assistant's conversational missteps. Do NOT narrate moments where the assistant failed to answer, said it lacked information, offered to search, deflected, declined, or clarified its own limitations. Those events are transient and do not generalise; preserving them in the diary trains future sessions to repeat the same failures.
   - If the assistant eventually answered (e.g. after calling a tool), summarise the final answer only.
   - If the topic was raised but never resolved, record only the topic and the user's intent — strip out every phrase describing the assistant's inability, uncertainty, or offers to help.
   - Never write phrases like "the assistant did not have/does not have information", "the assistant could not", "the assistant offered to search", "the assistant was unable", "the assistant clarified that it…", "the assistant lacks…", or their equivalents in any language.

   Example of what NOT to write:
     BAD: "The user asked about the book Piranesi. The assistant stated it did not have specific information on that book."
   Example of the correct output:
     GOOD: "The user asked about the book Piranesi."

   This rule applies in any language.
7. CRITICAL attribution rule — record what was SAID faithfully, but make clear WHO said it. The diary is a log of the conversation, not a fact sheet, so preserve the actual content (including the assistant's answers, because a later session may need them — and because the user may later correct a wrong answer, and we want the whole chain on record). What must not happen is quietly promoting an assistant claim into an unattributed fact, because the assistant may hallucinate.
   - When the assistant states something about a third-party entity (film, book, product, company, person, place, event, scientific fact, definition), always attribute it in the summary: write "the assistant said/stated/explained X" rather than "X". The attribution lets downstream readers treat the claim with appropriate skepticism.
   - Never paraphrase an attributed claim into an unattributed assertion. "The assistant said Possessor is a 2006 film by Brandon Cronenberg" is fine (attribution preserved). "Possessor is a 2006 film by Brandon Cronenberg" is NOT (attribution stripped — now reads as established fact).
   - If the user later corrects the assistant, record both: the initial claim AND the correction. That's how the final state becomes recoverable — never delete earlier claims when a correction comes in.
   - Weather, time, location, calculator results, and other clearly tool-grounded data can be recorded as fact without attribution caveats — the tool output is the authority.
   - User-stated facts about themselves (preferences, biography, plans, decisions) are always safe to record verbatim as user facts.

   Example — attributed assistant claim (preserves information, flags provenance):
     GOOD: "The user asked about the movie Possessor; the assistant said it is a 2006 science fiction film directed by Brandon Cronenberg."
     BAD (unattributed — reads as established fact, will poison downstream): "The user asked about the movie Possessor. It is a 2006 science fiction film directed by Brandon Cronenberg."

   Example — correction chain preserved:
     GOOD: "The user asked about Possessor; the assistant said it is a 2006 film, the user corrected that it is from 2020."

   Example — tool-grounded + user-stated, no attribution caveats needed:
     OK: "The weather in Hackney was 10.6°C and partly cloudy. The user said they prefer Thai over Indian food."

   This rule applies in any language.
8. CRITICAL topic-separation rule — do NOT weld unrelated topics into one grammatical clause. If the conversation covered two distinct subjects (e.g. a film and a person, a recipe and a weather query, two different named entities), write a separate sentence for each, each with its own subject and verb. A welded clause reads to downstream retrievers — and to other LLMs enriching future replies — as a single claim about both referents, and silently corrupts the record.
   - One topic per sentence. Never join two unrelated topics with "and", a shared appositive, or a shared relative clause.
   - Never let an appositive or relative clause dangle over more than one topic. "X and Y, identified as Z" reads as Z describing both X and Y — this is the exact failure mode.

   Example — two distinct topics raised in the same conversation (a film, and the name "Jarvis" meaning the MCU character). The BAD version welded them so downstream readers treated the MCU description as pertaining to the film:
     BAD: "The conversation focused on the movie Possessor and the character Jarvis, identified as the artificial intelligence from the Marvel Cinematic Universe, created by Tony Stark and later embodied by Vision."
     GOOD: "The user asked about the movie Possessor; the assistant said it is a 2020 science-fiction horror film directed by Brandon Cronenberg. Separately, the user asked about the name Jarvis; the assistant said the MCU character Jarvis is an AI created by Tony Stark and later embodied by Vision."

   This rule applies in any language.

Also extract 3-5 main topics as comma-separated keywords."""

    if previous_summary:
        user_prompt = f"""Previous summary for today: {previous_summary}

Recent conversation chunks:
{chunks_text}

Update the summary to include the new information. Provide:
1. Updated summary (max 200 words)
2. Main topics (comma-separated)

Format your response as:
SUMMARY: [your summary here]
TOPICS: [topic1, topic2, topic3]"""
    else:
        user_prompt = f"""Conversation chunks from today:
{chunks_text}

Create a summary of today's conversations. Provide:
1. Summary (max 200 words)
2. Main topics (comma-separated)

Format your response as:
SUMMARY: [your summary here]
TOPICS: [topic1, topic2, topic3]"""

    try:
        # Use streaming if callback provided, otherwise use direct call
        if on_token:
            response = call_llm_streaming(
                ollama_base_url, ollama_chat_model, system_prompt, user_prompt,
                on_token=on_token, timeout_sec=timeout_sec, thinking=thinking,
            )
        else:
            response = call_llm_direct(
                ollama_base_url, ollama_chat_model, system_prompt, user_prompt,
                timeout_sec=timeout_sec, thinking=thinking,
            )

        if not response:
            # No fallback - if LLM fails to respond, skip summarization
            return None, None

        # Parse the response
        lines = response.strip().split('\n')
        summary = ""
        topics = ""

        for line in lines:
            if line.startswith("SUMMARY:"):
                summary = line[8:].strip()
            elif line.startswith("TOPICS:"):
                topics = line[7:].strip()

        # No fallback - if parsing fails, skip summarization
        if not summary or not topics:
            return None, None

        return summary, topics

    except Exception:
        # No fallback - if LLM fails, skip summarization entirely
        return None, None


def update_daily_conversation_summary(
    db: Database,
    new_chunks: List[str],
    ollama_base_url: str,
    ollama_chat_model: str,
    ollama_embed_model: str,
    source_app: str = "jarvis",
    voice_debug: bool = False,
    timeout_sec: float = 30.0,
    on_token: Optional[Callable[[str], None]] = None,
    thinking: bool = False,
) -> Optional[int]:
    """
    Update the conversation summary for today with new chunks.

    Args:
        on_token: Optional callback for streaming tokens (for live UI updates)

    Returns the summary ID if successful, None otherwise.
    """
    if not new_chunks:
        return None

    today = datetime.now(timezone.utc).date().isoformat()  # YYYY-MM-DD format

    try:
        # Redact sensitive information from chunks before processing
        from ..utils.redact import redact
        redacted_chunks = [redact(chunk) for chunk in new_chunks]

        # Debug: Log the redacted chunks being processed
        debug_log(f"updating conversation memory with {len(redacted_chunks)} new chunks:", "memory")
        for i, chunk in enumerate(redacted_chunks):
            chunk_preview = chunk[:100] + "..." if len(chunk) > 100 else chunk
            debug_log(f"  chunk {i+1}: {chunk_preview}", "memory")

        # Get existing summary for today
        existing = db.get_conversation_summary(today, source_app)
        previous_summary = existing['summary'] if existing else None

        # Generate updated summary using redacted chunks
        summary, topics = generate_conversation_summary(
            redacted_chunks, previous_summary, ollama_base_url, ollama_chat_model,
            timeout_sec=timeout_sec, on_token=on_token, thinking=thinking,
        )

        # Skip summarization if LLM failed
        if summary is None or topics is None:
            debug_log("conversation summary skipped - LLM failed to generate summary", "memory")
            return  # Skip summarization entirely

        # Debug: Log the generated summary and topics
        summary_preview = summary[:200] + "..." if len(summary) > 200 else summary
        debug_log("conversation memory updated to:", "memory")
        debug_log(f"  summary: {summary_preview}", "memory")
        debug_log(f"  topics: {topics}", "memory")
        if previous_summary:
            prev_preview = previous_summary[:100] + "..." if len(previous_summary) > 100 else previous_summary
            debug_log(f"  previous summary: {prev_preview}", "memory")
        else:
            debug_log("  previous summary: (none)", "memory")

        # Store the summary
        summary_id = db.upsert_conversation_summary(
            date_utc=today,
            summary=summary,
            topics=topics,
            source_app=source_app,
        )

        # Generate and store embedding for semantic search
        if db.is_vss_enabled:
            # Combine summary and topics for embedding
            text_for_embedding = f"{summary} {topics}"
            vec = get_embedding(text_for_embedding, ollama_base_url, ollama_embed_model, timeout_sec=15.0)  # Use shorter timeout for embeddings
            if vec is not None:
                db.upsert_summary_embedding(summary_id, vec)

        return summary_id

    except Exception:
        return None


def search_conversation_memory_by_keywords(
    db: Database,
    keywords: List[str],
    from_time: Optional[str] = None,
    to_time: Optional[str] = None,
    ollama_base_url: Optional[str] = None,
    ollama_embed_model: Optional[str] = None,
    timeout_sec: float = 60.0,
    voice_debug: bool = False,
    max_results: int = 10,
) -> List[str]:
    """
    Search conversation memory using multiple keywords with OR logic.
    This is optimized for memory enrichment where we have extracted topic keywords.

    Args:
        db: Database instance
        keywords: List of keywords to search for (will be OR'd together)
        from_time: Start timestamp (ISO format)
        to_time: End timestamp (ISO format)
        ollama_base_url: Base URL for embeddings
        ollama_embed_model: Model for embeddings
        timeout_sec: Timeout for embedding generation
        voice_debug: Enable debug output
        max_results: Maximum number of results to return (default: 10)

    Returns:
        List of formatted context strings (limited to max_results)
    """
    contexts = []

    if not keywords:
        return contexts

    # Clean keywords
    clean_keywords = [k.strip() for k in keywords if k and k.strip()]
    if not clean_keywords:
        return contexts

    try:
        debug_log(f"      🔍 Keyword-based search for: {clean_keywords}", "memory")

        # Build FTS OR query for better recall
        fts_query = " OR ".join(clean_keywords[:5])  # Limit to 5 keywords

        # For embedding, combine keywords to get semantic meaning of the topic cluster
        embed_query = " ".join(clean_keywords)

        debug_log(f"      📝 FTS query: '{fts_query}'", "memory")
        debug_log(f"      📝 Embed query: '{embed_query}'", "memory")

        if ollama_base_url and ollama_embed_model:
            try:
                vec = get_embedding(embed_query, ollama_base_url, ollama_embed_model, timeout_sec=timeout_sec)
                vec_json = json.dumps(vec) if vec is not None else None

                if vec_json:
                    # Hybrid search with OR query for FTS and combined embedding
                    search_results = db.search_hybrid(fts_query, vec_json, top_k=max_results)
                else:
                    # Fallback: FTS-only with OR query
                    search_results = db.search_hybrid(fts_query, None, top_k=max_results)
            except Exception as e:
                debug_log(f"      ❌ Embedding failed, using FTS only: {e}", "memory")
                # Fallback to FTS-only
                search_results = db.search_hybrid(fts_query, None, top_k=max_results)
        else:
            # No embedding service available, use FTS-only
            search_results = db.search_hybrid(fts_query, None, top_k=max_results)

        # Collect results with scores and dates for recency-aware ordering
        scored_results: list[tuple[float, str, str]] = []  # (score, date, text)
        for result in search_results:
            if isinstance(result, dict):
                result_text = result.get('text', '')
                score = result.get('score', 0.0)
            else:
                result_text = result[2] if len(result) > 2 else ''
                score = result[1] if len(result) > 1 else 0.0
            if isinstance(result_text, str) and result_text:
                # Extract date from "[YYYY-MM-DD] ..." prefix for recency tiebreaking
                date_str = result_text[1:11] if result_text.startswith('[') and len(result_text) > 11 else ''
                scored_results.append((float(score) if score else 0.0, date_str, result_text))

        # Sort newest-first so recency-superseding works at the injection site:
        # when two entries disagree, the model sees the newer one first and the
        # preamble in the reply engine tells it to treat the newer entry as the
        # user's current understanding. Fall back to relevance score as tiebreak.
        scored_results.sort(key=lambda x: (x[1], x[0]), reverse=True)
        contexts = [text for _, _, text in scored_results]

        debug_log(f"      ✅ found {len(contexts)} keyword search results", "memory")
        if contexts:
            # Show preview of first result
            preview = contexts[0][:150] + "..." if len(contexts[0]) > 150 else contexts[0]
            debug_log(f"      📋 First result: {preview}", "memory")

    except Exception as e:
        debug_log(f"keyword search failed: {e}", "memory")

    # Apply time filtering if needed
    if from_time or to_time:
        contexts = _filter_contexts_by_time(contexts, from_time, to_time, voice_debug)

    return contexts[:max_results]


def search_conversation_memory(
    db: Database,
    search_query: Optional[str] = None,
    from_time: Optional[str] = None,
    to_time: Optional[str] = None,
    ollama_base_url: Optional[str] = None,
    ollama_embed_model: Optional[str] = None,
    timeout_sec: float = 60.0,
    voice_debug: bool = False,
    max_results: int = 15,
) -> List[str]:
    """
    Search conversation memory with a natural language query or phrase.
    This is optimized for direct user queries and tool usage.

    Args:
        db: Database instance
        search_query: Natural language query or phrase to search for
        from_time: Start timestamp (ISO format)
        to_time: End timestamp (ISO format)
        ollama_base_url: Base URL for embeddings (required if search_query provided)
        ollama_embed_model: Model for embeddings (required if search_query provided)
        timeout_sec: Timeout for embedding generation
        voice_debug: Enable debug output
        max_results: Maximum number of results to return (default: 15)

    Returns:
        List of formatted context strings (limited to max_results)
    """
    contexts = []

    try:
        if search_query and search_query.strip() and ollama_base_url and ollama_embed_model:
            # Primary: Use vector search for semantic similarity
            try:
                vec = get_embedding(search_query, ollama_base_url, ollama_embed_model, timeout_sec=timeout_sec)
                vec_json = json.dumps(vec) if vec is not None else None

                if vec_json:
                    # Use database hybrid search (combines vector similarity with FTS)
                    search_results = db.search_hybrid(search_query, vec_json, top_k=max_results)
                else:
                    # Fallback: Pure FTS if embedding fails
                    search_results = db.search_hybrid(search_query, None, top_k=max_results)

                # Add search results to context
                for result in search_results:
                    # Handle both tuple (sqlite-vss) and dict (python vector store) results
                    if isinstance(result, dict):
                        result_text = result.get('text', '')
                    else:
                        result_text = result[2] if len(result) > 2 else ''
                    if isinstance(result_text, str) and result_text:
                        contexts.append(result_text)

            except Exception as e:
                if voice_debug:
                    debug_log(f"memory search failed: {e}", "memory")

        # Apply time filtering if provided
        debug_log(f"      📋 Checking time filtering: from_time={from_time}, to_time={to_time}", "memory")

        if from_time or to_time:
            filtered_contexts = []
            from_dt = None
            to_dt = None

            try:
                if from_time:
                    from_dt = datetime.fromisoformat(from_time.replace('Z', '+00:00'))
                if to_time:
                    to_dt = datetime.fromisoformat(to_time.replace('Z', '+00:00'))
            except Exception as e:
                debug_log(f"      📋 Error parsing time: {e}", "memory")

            debug_log(f"      📋 Time filtering: search_query='{search_query}', from_dt={from_dt}, to_dt={to_dt}", "memory")

            # If we have time constraints but no search query, get all summaries in range
            if (not search_query or not search_query.strip()) and (from_dt or to_dt):
                recent_summaries = db.get_recent_conversation_summaries(days=30)
                debug_log(f"      📋 Time filter: from={from_dt.date() if from_dt else None} to={to_dt.date() if to_dt else None}", "memory")
                debug_log(f"      📋 Found {len(recent_summaries)} summaries to check", "memory")

                for summary_row in recent_summaries:
                    date_str = summary_row['date_utc']
                    summary_date = datetime.fromisoformat(date_str + 'T00:00:00+00:00')

                    in_range = True
                    if from_dt and summary_date.date() < from_dt.date():
                        in_range = False
                        debug_log(f"      📋 Skipping {date_str}: before from_dt", "memory")
                    if to_dt and summary_date.date() > to_dt.date():
                        in_range = False
                        debug_log(f"      📋 Skipping {date_str}: after to_dt", "memory")

                    if in_range:
                        summary_text = summary_row['summary']
                        topics = summary_row['topics'] or ""
                        context_str = f"[{date_str}] {summary_text}"
                        if topics:
                            context_str += f" (Topics: {topics})"
                        contexts.append(context_str)
                        debug_log(f"      📋 Including summary from {date_str} (length: {len(summary_text)})", "memory")

            else:
                # Filter existing search results by time
                import re
                for ctx in contexts:
                    if ctx.startswith("---"):  # Skip headers
                        filtered_contexts.append(ctx)
                        continue

                    # Extract date from formatted text
                    date_match = re.match(r'\[(\d{4}-\d{2}-\d{2})\]', ctx)
                    if date_match:
                        date_str = date_match.group(1)
                        try:
                            summary_date = datetime.fromisoformat(date_str + 'T00:00:00+00:00')

                            in_range = True
                            if from_dt and summary_date < from_dt:
                                in_range = False
                            if to_dt and summary_date > to_dt:
                                in_range = False

                            if in_range:
                                filtered_contexts.append(ctx)
                        except Exception:
                            filtered_contexts.append(ctx)  # Keep if can't parse date
                    else:
                        filtered_contexts.append(ctx)  # Keep non-dated entries

                contexts = filtered_contexts

        return contexts[:max_results]  # Limit results

    except Exception:
        return contexts[:max_results] if contexts else []


def get_relevant_conversation_context(
    db: Database,
    query: str,
    ollama_base_url: str,
    ollama_embed_model: str,
    timeout_sec: float = 60.0,
    max_results: int = 15,
) -> List[str]:
    """
    Get relevant conversation summaries that might provide context for the current query.

    Returns list of formatted context strings.

    This is a wrapper around search_conversation_memory for backward compatibility.
    """
    return search_conversation_memory(
        db=db,
        search_query=query,
        ollama_base_url=ollama_base_url,
        ollama_embed_model=ollama_embed_model,
        timeout_sec=timeout_sec,
        voice_debug=False,
        max_results=max_results
    )


def update_diary_from_dialogue_memory(
    db: Database,
    dialogue_memory: DialogueMemory,
    ollama_base_url: str,
    ollama_chat_model: str,
    ollama_embed_model: str,
    source_app: str = "jarvis",
    voice_debug: bool = False,
    timeout_sec: float = 30.0,
    force: bool = False,
    on_token: Optional[Callable[[str], None]] = None,
    thinking: bool = False,
    graph_picker_model: Optional[str] = None,
) -> Optional[int]:
    """
    Update the diary with pending interactions from dialogue memory.

    Thread-safe: captures the timestamp of messages being processed before
    LLM summarization starts, so new messages arriving during summarization
    won't be incorrectly marked as saved.

    Args:
        on_token: Optional callback for streaming tokens (for live UI updates)

    Returns the summary ID if successful, None otherwise.
    """
    debug_log(f"update_diary_from_dialogue_memory called: force={force}", "memory")

    if not force and not dialogue_memory.should_update_diary():
        debug_log("diary update skipped: should_update_diary=False and force=False", "memory")
        return None

    try:
        # CRITICAL: Capture the current timestamp BEFORE getting chunks
        # This ensures that any new messages arriving during LLM summarization
        # (which can take 30-45 seconds) won't be incorrectly marked as saved.
        snapshot_timestamp = time.time()

        # Get pending chunks from dialogue memory
        pending_chunks = dialogue_memory.get_pending_chunks()
        debug_log(f"diary update: got {len(pending_chunks)} pending chunks from dialogue_memory", "memory")

        if not pending_chunks:
            debug_log("diary update skipped: no pending chunks in dialogue_memory", "memory")
            return None

        # Update the daily conversation summary
        # This is the slow operation (LLM call) during which new messages might arrive
        debug_log("calling update_daily_conversation_summary...", "memory")
        summary_id = update_daily_conversation_summary(
            db=db,
            new_chunks=pending_chunks,
            ollama_base_url=ollama_base_url,
            ollama_chat_model=ollama_chat_model,
            ollama_embed_model=ollama_embed_model,
            source_app=source_app,
            voice_debug=voice_debug,
            timeout_sec=timeout_sec,
            on_token=on_token,
            thinking=thinking,
        )

        debug_log(f"update_daily_conversation_summary returned: {summary_id}", "memory")

        # Mark only the messages that existed at snapshot time as saved
        # New messages that arrived during summarization remain pending
        if summary_id is not None:
            dialogue_memory.mark_saved_up_to(snapshot_timestamp)
            debug_log(f"marked messages saved up to timestamp {snapshot_timestamp}", "memory")

            # Graph memory (v2): extract facts and store in the node graph.
            # Non-blocking — if this fails, the diary update still succeeded.
            # Uses a dedicated timeout (30s) rather than the diary chat timeout,
            # so graph updates don't inflate the diary flush wall time.
            try:
                from .graph import GraphMemoryStore
                from .graph_ops import update_graph_from_dialogue

                graph_store = GraphMemoryStore(db.db_path)
                # Retrieve the summary we just stored to use for extraction
                today = datetime.now(timezone.utc).date().isoformat()
                existing = db.get_conversation_summary(today, source_app)
                summary_text = existing['summary'] if existing else None

                if summary_text:
                    # Use a shorter timeout for graph operations — extraction (30s),
                    # placement (15s/fact), and split (45s) each have their own budgets
                    # inside update_graph_from_dialogue.
                    graph_timeout = min(timeout_sec, 30.0)
                    result = update_graph_from_dialogue(
                        store=graph_store,
                        summary=summary_text,
                        ollama_base_url=ollama_base_url,
                        ollama_chat_model=ollama_chat_model,
                        timeout_sec=graph_timeout,
                        thinking=thinking,
                        date_utc=today,
                        picker_model=graph_picker_model,
                    )
                    stored = result.stored
                    skipped = result.skipped
                    # Print whenever extraction produced anything — including
                    # all-duplicate flushes. Without the skipped count this
                    # line went silent after #282's dedupe (cumulative diary
                    # re-extracts the same facts on every flush), making it
                    # look like the memory pipeline had stopped working.
                    if stored or skipped:
                        dup_suffix = (
                            f"{skipped} duplicate{'' if skipped == 1 else 's'} skipped"
                        )
                        if stored:
                            fact_count = (
                                f"{len(stored)} new fact"
                                f"{'' if len(stored) == 1 else 's'}"
                            )
                            tail = f" ({dup_suffix})" if skipped else ""
                            print(
                                f"  🧠 Knowledge graph: learned {fact_count}{tail}",
                                flush=True,
                            )
                            # Show each new fact with the node it landed in so
                            # the user can eyeball extraction/placement. Cap
                            # preview length per fact.
                            for fact, node_name in stored[:6]:
                                preview = fact.replace("\n", " ").strip()
                                if len(preview) > 90:
                                    preview = preview[:90].rstrip() + "…"
                                print(f"     · {preview} → {node_name}", flush=True)
                            if len(stored) > 6:
                                print(f"     · …and {len(stored) - 6} more", flush=True)
                        else:
                            print(
                                f"  🧠 Knowledge graph: nothing new ({dup_suffix})",
                                flush=True,
                            )
                    debug_log(
                        f"graph memory: stored {len(stored)} facts, "
                        f"{skipped} duplicates skipped",
                        "memory",
                    )
            except Exception as e:
                debug_log(f"graph memory update failed (non-fatal): {e}", "memory")

        return summary_id

    except Exception as e:
        debug_log(f"update_diary_from_dialogue_memory error: {e}", "memory")
        return None
