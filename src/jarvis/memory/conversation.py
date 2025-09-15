from __future__ import annotations
import json
import time
from datetime import datetime, timezone
from typing import Optional, List, Tuple, Union
from .db import Database
from ..llm import call_llm_direct
from .embeddings import get_embedding
from ..debug import debug_log


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
    Provides short-term context for the last 5 minutes of conversation.
    """
    
    def __init__(self, inactivity_timeout: float = 300.0, max_interactions: int = 20):
        """Initialize dialogue memory."""
        self._messages: List[Tuple[float, str, str]] = []  # (timestamp, role, content)
        self._last_activity_time: float = time.time()
        self._inactivity_timeout = inactivity_timeout
        self._is_pending_diary_update = False
    
    def add_message(self, role: str, content: str) -> None:
        """Add a message to recent memory."""
        timestamp = time.time()
        self._messages.append((timestamp, role.strip(), content.strip()))
        self._last_activity_time = timestamp
        self._is_pending_diary_update = True
    
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
        if not self._messages:
            return []
        
        # Filter to last 5 minutes
        cutoff = time.time() - 300.0
        recent_messages = [msg for msg in self._messages if msg[0] >= cutoff]
        
        return [{"role": role, "content": content} for _, role, content in recent_messages]
    
    def has_recent_messages(self) -> bool:
        """Check if there are any messages in the last 5 minutes."""
        cutoff = time.time() - 300.0
        return any(ts >= cutoff for ts, _, _ in self._messages)

    # Compatibility and diary functionality
    def add_interaction(self, user_text: str, assistant_text: str) -> None:
        """Compatibility method - use add_message() instead."""
        if user_text.strip():
            self.add_message("user", user_text.strip())
        if assistant_text.strip():
            self.add_message("assistant", assistant_text.strip())
    
    def get_pending_chunks(self) -> List[str]:
        """Get recent messages as formatted chunks for diary update."""
        messages = self.get_recent_messages()
        return [f"{msg['role'].title()}: {msg['content']}" for msg in messages]
    
    def should_update_diary(self) -> bool:
        """Check if diary should be updated based on inactivity timeout."""
        if not self._is_pending_diary_update:
            return False
        
        current_time = time.time()
        return (current_time - self._last_activity_time) >= self._inactivity_timeout
    
    def clear_pending_updates(self) -> None:
        """Clear the pending diary update flag."""
        self._is_pending_diary_update = False


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
        response = call_llm_direct(ollama_base_url, ollama_chat_model, system_prompt, user_prompt, timeout_sec=timeout_sec)
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
            redacted_chunks, previous_summary, ollama_base_url, ollama_chat_model, timeout_sec=timeout_sec
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
        
        # Collect results
        for result in search_results:
            if isinstance(result, dict):
                result_text = result.get('text', '')
            else:
                result_text = result[2] if len(result) > 2 else ''
            if isinstance(result_text, str) and result_text:
                contexts.append(result_text)
        
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
