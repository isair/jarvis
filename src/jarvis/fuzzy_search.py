from __future__ import annotations
import re
from typing import List, Tuple, Optional
try:
    from rapidfuzz import fuzz, process
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False


def generate_flexible_fts_query(query: str, field_names: List[str] = None) -> str:
    """
    Generate a more flexible FTS5 query that handles variations and partial matches.
    
    Args:
        query: The search query
        field_names: Optional list of field names to search in (for multi-column FTS)
    
    Returns:
        FTS5 query string with flexible matching
    """
    if not query.strip():
        return ""
    
    # Clean and tokenize the query
    tokens = re.findall(r"[A-Za-z0-9_]+", query.lower())
    if not tokens:
        return ""
    
    # Build flexible FTS5 query components
    query_parts = []
    
    # For short queries (1-2 words), use OR logic with prefix matching
    if len(tokens) <= 2:
        prefix_terms = [f"{token}*" for token in tokens]
        exact_terms = tokens.copy()
        
        # Add both exact and prefix matches with OR
        all_terms = exact_terms + prefix_terms
        if field_names:
            # Multi-column search: search in any field
            field_parts = []
            for field in field_names:
                field_parts.extend([f"{field}:{term}" for term in all_terms])
            query_parts.append("(" + " OR ".join(field_parts) + ")")
        else:
            query_parts.append("(" + " OR ".join(all_terms) + ")")
    
    # For longer queries, use NEAR operator for phrase-like matching
    elif len(tokens) <= 5:
        # Try exact phrase first
        phrase_query = " ".join(tokens)
        
        # Add NEAR variants for flexible word order
        near_queries = []
        if len(tokens) >= 2:
            # NEAR/3 allows up to 3 words between terms
            near_queries.append(f"NEAR({' '.join(tokens)}, 3)")
        
        # Add prefix matching for the last word (common for incomplete typing)
        prefix_variant = " ".join(tokens[:-1] + [f"{tokens[-1]}*"])
        
        if field_names:
            # Multi-column search
            field_parts = []
            for field in field_names:
                field_parts.append(f'{field}:"{phrase_query}"')
                field_parts.extend([f"{field}:{nq}" for nq in near_queries])
                field_parts.append(f"{field}:{prefix_variant}")
            query_parts.append("(" + " OR ".join(field_parts) + ")")
        else:
            all_variants = [f'"{phrase_query}"'] + near_queries + [prefix_variant]
            query_parts.append("(" + " OR ".join(all_variants) + ")")
    
    # For very long queries, fall back to AND logic with some OR alternatives
    else:
        # Use first few words with AND, rest with OR
        primary_terms = tokens[:3]
        secondary_terms = tokens[3:]
        
        primary_and = " ".join(primary_terms)
        secondary_or = " OR ".join(secondary_terms)
        
        if field_names:
            field_parts = []
            for field in field_names:
                field_parts.append(f"{field}:({primary_and}) AND ({field}:({secondary_or}))")
            query_parts.append("(" + " OR ".join(field_parts) + ")")
        else:
            query_parts.append(f"({primary_and}) AND ({secondary_or})")
    
    return " OR ".join(query_parts) if query_parts else ""


def fuzzy_match_results(query: str, candidates: List[Tuple[any, str]], threshold: int = 60) -> List[Tuple[any, str, int]]:
    """
    Post-process search results with fuzzy matching to catch partial matches.
    
    Args:
        query: Original search query
        candidates: List of (id/data, text) tuples to match against
        threshold: Minimum fuzzy match score (0-100)
    
    Returns:
        List of (id/data, text, fuzzy_score) tuples sorted by fuzzy score
    """
    if not RAPIDFUZZ_AVAILABLE or not query.strip() or not candidates:
        # Fallback: return candidates with score 100 (exact match assumed)
        return [(item[0], item[1], 100) for item in candidates]
    
    query_lower = query.lower().strip()
    scored_results = []
    
    for item_data, text in candidates:
        text_lower = text.lower()
        
        # Try different fuzzy matching strategies
        scores = []
        
        # 1. Partial ratio (good for substring matches)
        scores.append(fuzz.partial_ratio(query_lower, text_lower))
        
        # 2. Token sort ratio (good for word order differences)
        scores.append(fuzz.token_sort_ratio(query_lower, text_lower))
        
        # 3. Token set ratio (good for subset matches)
        scores.append(fuzz.token_set_ratio(query_lower, text_lower))
        
        # 4. WRatio (weighted combination)
        scores.append(fuzz.WRatio(query_lower, text_lower))
        
        # Use the best score
        best_score = max(scores)
        
        if best_score >= threshold:
            scored_results.append((item_data, text, best_score))
    
    # Sort by fuzzy score (descending)
    scored_results.sort(key=lambda x: x[2], reverse=True)
    return scored_results


def fuzzy_search_summaries(db, query: str, top_k: int = 10, fuzzy_threshold: int = 50) -> List[Tuple[int, str, int]]:
    """
    Perform fuzzy search on conversation summaries.
    
    Args:
        db: Database instance
        query: Search query
        top_k: Maximum number of results
        fuzzy_threshold: Minimum fuzzy match score
    
    Returns:
        List of (summary_id, formatted_text, fuzzy_score) tuples
    """
    with db._lock:
        cursor = db.conn.cursor()
        
        # First, try FTS5 search with flexible query
        fts_query = generate_flexible_fts_query(query, ["summary", "topics"])
        
        if fts_query:
            # Use the improved FTS5 query
            fts_sql = """
            SELECT s.id, s.date_utc, s.summary, s.topics, bm25(summaries_fts) as fts_score
            FROM summaries_fts
            JOIN conversation_summaries s ON s.id = summaries_fts.rowid
            WHERE summaries_fts MATCH ?
            ORDER BY fts_score
            LIMIT ?
            """
            try:
                fts_results = cursor.execute(fts_sql, (fts_query, top_k * 2)).fetchall()
            except Exception:
                # FTS query failed, fall back to simple search
                fts_results = []
        else:
            fts_results = []
        
        # If FTS5 didn't find enough results, get recent summaries for fuzzy matching
        if len(fts_results) < top_k:
            recent_sql = """
            SELECT id, date_utc, summary, topics, 0 as fts_score
            FROM conversation_summaries
            ORDER BY date_utc DESC
            LIMIT ?
            """
            recent_results = cursor.execute(recent_sql, (top_k * 2,)).fetchall()
            
            # Combine FTS and recent results, avoiding duplicates
            fts_ids = {row[0] for row in fts_results}
            combined_results = list(fts_results)
            for row in recent_results:
                if row[0] not in fts_ids:
                    combined_results.append(row)
        else:
            combined_results = fts_results
    
    # Prepare candidates for fuzzy matching
    candidates = []
    for row in combined_results:
        summary_id, date_utc, summary, topics, fts_score = row
        # Format the text for display and matching
        formatted_text = f"[{date_utc}] {summary}"
        if topics:
            formatted_text += f" (Topics: {topics})"
        candidates.append((summary_id, formatted_text))
    
    # Apply fuzzy matching
    fuzzy_results = fuzzy_match_results(query, candidates, fuzzy_threshold)
    
    # Return top results
    return fuzzy_results[:top_k]


def fuzzy_search_chunks(db, query: str, top_k: int = 10, fuzzy_threshold: int = 50) -> List[Tuple[int, str, int]]:
    """
    Perform fuzzy search on text chunks.
    
    Args:
        db: Database instance
        query: Search query
        top_k: Maximum number of results
        fuzzy_threshold: Minimum fuzzy match score
    
    Returns:
        List of (chunk_id, text, fuzzy_score) tuples
    """
    with db._lock:
        cursor = db.conn.cursor()
        
        # Try FTS5 search with flexible query
        fts_query = generate_flexible_fts_query(query, ["text"])
        
        if fts_query:
            fts_sql = """
            SELECT c.id, c.text, bm25(chunks_fts) as fts_score
            FROM chunks_fts
            JOIN chunks c ON c.id = chunks_fts.rowid
            WHERE chunks_fts MATCH ?
            ORDER BY fts_score
            LIMIT ?
            """
            try:
                fts_results = cursor.execute(fts_sql, (fts_query, top_k * 2)).fetchall()
            except Exception:
                fts_results = []
        else:
            fts_results = []
        
        # If FTS5 didn't find enough, get recent chunks
        if len(fts_results) < top_k:
            recent_sql = """
            SELECT id, text, 0 as fts_score
            FROM chunks
            ORDER BY id DESC
            LIMIT ?
            """
            recent_results = cursor.execute(recent_sql, (top_k * 2,)).fetchall()
            
            # Combine results
            fts_ids = {row[0] for row in fts_results}
            combined_results = list(fts_results)
            for row in recent_results:
                if row[0] not in fts_ids:
                    combined_results.append(row)
        else:
            combined_results = fts_results
    
    # Prepare candidates for fuzzy matching
    candidates = [(row[0], row[1]) for row in combined_results]
    
    # Apply fuzzy matching
    fuzzy_results = fuzzy_match_results(query, candidates, fuzzy_threshold)
    
    return fuzzy_results[:top_k]
