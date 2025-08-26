from __future__ import annotations
import json
import time
from datetime import datetime, timezone
from typing import Optional, List, Tuple
from .db import Database
from .coach import ask_coach
from .embed import get_embedding


class DialogueMemory:
    """
    In-memory storage for recent dialogue interactions.
    Provides short-term context without immediately updating the diary.
    """
    
    def __init__(self, inactivity_timeout: float = 300.0, max_interactions: int = 20):
        """
        Initialize dialogue memory.
        
        Args:
            inactivity_timeout: Seconds of inactivity before triggering diary update
            max_interactions: Maximum number of interactions to keep in memory
        """
        self._interactions: List[Tuple[float, str, str]] = []  # (timestamp, user_text, assistant_text)
        self._last_activity_time: float = time.time()
        self._inactivity_timeout = inactivity_timeout
        self._max_interactions = max_interactions  # kept for backward-compat; not used for cropping
        self._is_pending_diary_update = False
    
    def add_interaction(self, user_text: str, assistant_text: str) -> None:
        """Add a new interaction to memory."""
        timestamp = time.time()
        self._interactions.append((timestamp, user_text.strip(), assistant_text.strip()))
        self._last_activity_time = timestamp
        self._is_pending_diary_update = True
        # Do not crop by count; retrieval applies a 5-minute window
    
    def get_recent_context(self, max_interactions: Optional[int] = None) -> List[str]:
        """
        Get recent interactions formatted for context.
        
        Args:
            max_interactions: Deprecated; ignored. Context is limited by time window only.
            
        Returns:
            List of formatted interaction strings
        """
        if not self._interactions:
            return []
        # Filter to last 5 minutes
        cutoff = time.time() - 300.0
        interactions_to_return = [it for it in self._interactions if it[0] >= cutoff]
        context = []
        for timestamp, user_text, assistant_text in interactions_to_return:
            if user_text:
                context.append(f"User: {user_text}")
            if assistant_text:
                context.append(f"Assistant: {assistant_text}")
        
        return context
    
    def get_recent_messages(self, max_interactions: Optional[int] = None) -> List[dict]:
        """
        Get recent interactions formatted as conversation messages for LLM API.
        
        Args:
            max_interactions: Deprecated; ignored. Messages are limited by time window only.
            
        Returns:
            List of message dictionaries with 'role' and 'content' keys
        """
        if not self._interactions:
            return []
        # Filter to last 5 minutes
        cutoff = time.time() - 300.0
        interactions_to_return = [it for it in self._interactions if it[0] >= cutoff]
        messages = []
        for timestamp, user_text, assistant_text in interactions_to_return:
            if user_text:
                messages.append({"role": "user", "content": user_text})
            if assistant_text:
                messages.append({"role": "assistant", "content": assistant_text})
        
        return messages
    
    def should_update_diary(self) -> bool:
        """Check if diary should be updated based on inactivity timeout."""
        if not self._is_pending_diary_update:
            return False
        
        current_time = time.time()
        return (current_time - self._last_activity_time) >= self._inactivity_timeout
    
    def get_pending_chunks(self) -> List[str]:
        """Get all pending interactions as chunks for diary update."""
        if not self._interactions:
            return []
        
        chunks = []
        for timestamp, user_text, assistant_text in self._interactions:
            if user_text:
                chunks.append(f"User: {user_text}")
            if assistant_text:
                chunks.append(f"Assistant: {assistant_text}")
        
        return chunks
    
    def clear_pending_updates(self) -> None:
        """Clear the pending diary update flag and remove processed interactions."""
        self._is_pending_diary_update = False
        # Keep the last few interactions for immediate context but mark as processed
        # We'll keep them for context but they won't trigger another diary update
        
    def has_recent_interactions(self) -> bool:
        """Check if there are any interactions in the last 5 minutes."""
        cutoff = time.time() - 300.0
        return any(ts >= cutoff for ts, _, _ in self._interactions)
    
    def get_time_since_last_activity(self) -> float:
        """Get seconds since last activity."""
        return time.time() - self._last_activity_time


def generate_conversation_summary(
    recent_chunks: List[str],
    previous_summary: Optional[str],
    ollama_base_url: str,
    ollama_chat_model: str,
    timeout_sec: float = 30.0,
) -> Tuple[str, str]:
    """
    Generate a concise conversation summary from recent chunks and previous summary.
    
    Returns:
        Tuple of (summary, topics) where topics is comma-separated
    """
    chunks_text = "\n".join(recent_chunks[-10:])  # Last 10 chunks to keep context manageable
    
    system_prompt = """You are a conversation summarizer for a personal AI assistant. Your job is to create concise daily summaries of conversations that will be stored in a diary for future reference.

Create a summary that:
1. Captures the key topics discussed and important information shared
2. Is concise but informative (max 200 words)
3. Focuses on facts, decisions, and context that would be useful for future conversations
4. Includes any personal information, preferences, or important events mentioned
5. Maintains a neutral, factual tone

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
        response = ask_coach(ollama_base_url, ollama_chat_model, system_prompt, user_prompt, timeout_sec=timeout_sec, include_location=False)
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
) -> Optional[int]:
    """
    Update the conversation summary for today with new chunks.
    
    Returns the summary ID if successful, None otherwise.
    """
    if not new_chunks:
        return None
        
    today = datetime.now(timezone.utc).date().isoformat()  # YYYY-MM-DD format
    
    try:
        # Debug: Log the new chunks being processed
        if voice_debug:
            try:
                import sys
                print(f"[debug] updating conversation memory with {len(new_chunks)} new chunks:", file=sys.stderr)
                for i, chunk in enumerate(new_chunks):
                    chunk_preview = chunk[:100] + "..." if len(chunk) > 100 else chunk
                    print(f"[debug]   chunk {i+1}: {chunk_preview}", file=sys.stderr)
            except Exception:
                pass
        
        # Get existing summary for today
        existing = db.get_conversation_summary(today, source_app)
        previous_summary = existing['summary'] if existing else None
        
        # Generate updated summary
        summary, topics = generate_conversation_summary(
            new_chunks, previous_summary, ollama_base_url, ollama_chat_model, timeout_sec=timeout_sec
        )
        
        # Skip summarization if LLM failed
        if summary is None or topics is None:
            if voice_debug:
                try:
                    print(f"[debug] conversation summary skipped - LLM failed to generate summary", file=sys.stderr)
                except Exception:
                    pass
            return  # Skip summarization entirely
        
        # Debug: Log the generated summary and topics
        if voice_debug:
            try:
                summary_preview = summary[:200] + "..." if len(summary) > 200 else summary
                print(f"[debug] conversation memory updated to:", file=sys.stderr)
                print(f"[debug]   summary: {summary_preview}", file=sys.stderr)
                print(f"[debug]   topics: {topics}", file=sys.stderr)
                if previous_summary:
                    prev_preview = previous_summary[:100] + "..." if len(previous_summary) > 100 else previous_summary
                    print(f"[debug]   previous summary: {prev_preview}", file=sys.stderr)
                else:
                    print(f"[debug]   previous summary: (none)", file=sys.stderr)
            except Exception:
                pass
        
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


def get_relevant_conversation_context(
    db: Database,
    query: str,
    ollama_base_url: str,
    ollama_embed_model: str,
    days_back: int = 7,
    dialogue_memory: Optional[DialogueMemory] = None,
    timeout_sec: float = 15.0,
) -> List[str]:
    """
    Get relevant conversation summaries that might provide context for the current query.
    Also includes recent dialogue memory for immediate context.
    
    Returns list of formatted context strings.
    """
    try:
        # Start with recent dialogue memory for immediate context
        contexts = []
        if dialogue_memory and dialogue_memory.has_recent_interactions():
            recent_dialogue = dialogue_memory.get_recent_context()
            if recent_dialogue:
                # Add a header to distinguish recent vs historical context
                contexts.append("--- Recent Conversation ---")
                contexts.extend(recent_dialogue)
                contexts.append("--- Historical Context ---")
        
        # Continue with existing logic for historical context
        # Use fuzzy search for better matching
        try:
            from .fuzzy_search import fuzzy_search_summaries
            fuzzy_results = fuzzy_search_summaries(
                db=db,
                query=query,
                top_k=8,
                fuzzy_threshold=50  # Moderate threshold for context retrieval
            )
            
            # Convert to context format and add to existing contexts
            for summary_id, formatted_text, fuzzy_score in fuzzy_results:
                contexts.append(formatted_text)
            
            if len(contexts) > 1:  # We have both recent and historical context
                return contexts[:15]  # Return more to include both recent and historical
                
        except ImportError:
            # Fallback to hybrid search if fuzzy search not available
            pass
        
        # Fallback: Use the database's hybrid search
        try:
            # Get vector embedding for semantic search if possible
            vec = get_embedding(query, ollama_base_url, ollama_embed_model, timeout_sec=timeout_sec)
            vec_json = json.dumps(vec) if vec is not None else None
        except Exception:
            vec_json = None
        
        # Use the database's hybrid search which now includes summaries
        search_results = db.search_hybrid(query, vec_json, top_k=10)
        
        # Filter for summary results only and add to existing contexts
        for result in search_results:
            result_text = result[2]  # text column
            result_type = result[3] if len(result) > 3 else 'chunk'  # result_type if available
            
            # Check if this is a summary result (contains date format)
            if result_text.startswith('[') and ']' in result_text and '(Topics:' in result_text:
                contexts.append(result_text)
            elif result_type == 'summary':
                contexts.append(result_text)
        
        if contexts:
            return contexts[:15]  # Return combined recent and historical context
        
        # Final fallback: get recent summaries with simple matching
        recent_summaries = db.get_recent_conversation_summaries(days_back)
        if recent_summaries:
            query_words = query.lower().split()
            
            for summary_row in recent_summaries:
                date_str = summary_row['date_utc']
                summary_text = summary_row['summary']
                topics = summary_row['topics'] or ""
                
                # Check if any query words are in the summary or topics
                full_text = f"{summary_text} {topics}".lower()
                if any(word in full_text for word in query_words):
                    context_str = f"[{date_str}] {summary_text}"
                    if topics:
                        context_str += f" (Topics: {topics})"
                    contexts.append(context_str)
        
        return contexts[:15]  # Return combined recent dialogue and historical context
        
    except Exception:
        return []


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
) -> Optional[int]:
    """
    Update the diary with pending interactions from dialogue memory.
    
    Returns the summary ID if successful, None otherwise.
    """
    if not force and not dialogue_memory.should_update_diary():
        return None
        
    try:
        # Get pending chunks from dialogue memory
        pending_chunks = dialogue_memory.get_pending_chunks()
        if not pending_chunks:
            return None
        
        # Update the daily conversation summary
        summary_id = update_daily_conversation_summary(
            db=db,
            new_chunks=pending_chunks,
            ollama_base_url=ollama_base_url,
            ollama_chat_model=ollama_chat_model,
            ollama_embed_model=ollama_embed_model,
            source_app=source_app,
            voice_debug=voice_debug,
            timeout_sec=timeout_sec,
        )
        
        # Clear the pending updates flag
        if summary_id is not None:
            dialogue_memory.clear_pending_updates()
        
        return summary_id
        
    except Exception:
        return None
