from __future__ import annotations
import json
from typing import List, Dict, Optional, Tuple, Set
from .db import Database
from .memory import DialogueMemory, get_relevant_conversation_context
from .retrieve import retrieve_top_chunks
from .keyword_extraction import extract_query_keywords, generate_search_variants
from .embed import get_embedding

try:
    from .fuzzy_search import fuzzy_search_summaries, fuzzy_search_chunks
    FUZZY_SEARCH_AVAILABLE = True
except ImportError:
    FUZZY_SEARCH_AVAILABLE = False


class EnhancedRAGResult:
    """Container for enhanced RAG results with metadata."""
    
    def __init__(self):
        self.chunk_contexts: List[str] = []
        self.memory_contexts: List[str] = []
        self.keyword_contexts: List[str] = []
        self.extracted_keywords: Dict[str, any] = {}
        self.query_type: str = "unknown"
        self.total_contexts: int = 0
        self.search_strategies_used: List[str] = []
    
    def get_all_contexts(self, max_total: int = 15) -> List[str]:
        """Get all contexts combined, prioritizing by relevance."""
        all_contexts = []
        
        # Add memory contexts first (most relevant to recent conversation)
        all_contexts.extend(self.memory_contexts)
        
        # Add keyword-enhanced contexts
        all_contexts.extend(self.keyword_contexts)
        
        # Add chunk contexts (general document retrieval)
        all_contexts.extend(self.chunk_contexts)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_contexts = []
        for context in all_contexts:
            # Use first 100 chars as deduplication key to catch similar contexts
            context_key = context[:100].strip()
            if context_key not in seen:
                unique_contexts.append(context)
                seen.add(context_key)
                
        return unique_contexts[:max_total]
    
    def get_context_summary(self) -> str:
        """Get a summary of what contexts were found."""
        summary_parts = []
        
        if self.memory_contexts:
            summary_parts.append(f"{len(self.memory_contexts)} memory contexts")
        
        if self.keyword_contexts:
            summary_parts.append(f"{len(self.keyword_contexts)} keyword-enhanced contexts")
            
        if self.chunk_contexts:
            summary_parts.append(f"{len(self.chunk_contexts)} document chunks")
        
        if self.extracted_keywords.get('all_keywords'):
            keywords_str = ', '.join(self.extracted_keywords['all_keywords'][:5])
            summary_parts.append(f"keywords: {keywords_str}")
        
        return f"Retrieved {' + '.join(summary_parts)}"


def enhance_query_with_keywords(
    db: Database,
    original_query: str,
    keywords: List[str],
    search_variants: List[str],
    ollama_base_url: str,
    ollama_embed_model: str,
    top_k: int = 5,
    timeout_sec: float = 10.0
) -> Tuple[List[str], List[str]]:
    """
    Enhance query retrieval using extracted keywords and search variants.
    
    Args:
        db: Database instance
        original_query: Original user query
        keywords: Extracted keywords
        search_variants: Generated search variants
        ollama_base_url: Ollama base URL
        ollama_embed_model: Embedding model name
        top_k: Number of results per search strategy
        timeout_sec: Timeout for embedding requests
        
    Returns:
        Tuple of (enhanced_chunk_contexts, search_strategies_used)
    """
    enhanced_contexts = []
    strategies_used = []
    
    try:
        # Strategy 1: Fuzzy search on keywords (if available)
        if FUZZY_SEARCH_AVAILABLE and keywords:
            try:
                # Search summaries with top keywords
                primary_keywords = ' '.join(keywords[:3])
                if primary_keywords:
                    fuzzy_results = fuzzy_search_summaries(
                        db=db,
                        query=primary_keywords,
                        top_k=top_k,
                        fuzzy_threshold=40  # Lower threshold for keyword search
                    )
                    
                    for summary_id, formatted_text, score in fuzzy_results:
                        enhanced_contexts.append(f"[memory-keyword] {formatted_text}")
                    
                    if fuzzy_results:
                        strategies_used.append("fuzzy_summary_keywords")
                
                # Search chunks with keywords
                for keyword in keywords[:2]:  # Top 2 keywords
                    if len(keyword) >= 4:  # Only search with longer keywords
                        chunk_results = fuzzy_search_chunks(
                            db=db,
                            query=keyword,
                            top_k=2,  # Fewer results per keyword
                            fuzzy_threshold=50
                        )
                        
                        for chunk_id, text, score in chunk_results:
                            enhanced_contexts.append(f"[chunk-keyword] {text}")
                        
                        if chunk_results:
                            strategies_used.append(f"fuzzy_chunk_{keyword}")
                            
            except Exception:
                pass  # Fuzzy search failed, continue with other strategies
        
        # Strategy 2: Semantic search with search variants
        for i, variant in enumerate(search_variants[:3]):  # Use top 3 variants
            try:
                # Get embedding for search variant
                vec = get_embedding(variant, ollama_base_url, ollama_embed_model, timeout_sec=timeout_sec)
                vec_json = json.dumps(vec) if vec is not None else None
                
                if vec_json:
                    # Hybrid search with the variant
                    search_results = db.search_hybrid(variant, vec_json, top_k=max(2, top_k // 2))
                    
                    for result in search_results[:2]:  # Limit results per variant
                        result_text = result[2]  # text column
                        result_type = result[3] if len(result) > 3 else 'chunk'
                        
                        if result_type == 'summary':
                            enhanced_contexts.append(f"[memory-variant-{i+1}] {result_text}")
                        else:
                            enhanced_contexts.append(f"[chunk-variant-{i+1}] {result_text}")
                    
                    if search_results:
                        strategies_used.append(f"semantic_variant_{i+1}")
                        
            except Exception:
                continue  # Skip this variant if embedding fails
        
        # Strategy 3: Direct database search with keywords
        try:
            for keyword in keywords[:2]:  # Search with top 2 individual keywords
                if len(keyword) >= 4:
                    # Simple hybrid search with individual keyword
                    search_results = db.search_hybrid(keyword, None, top_k=2)
                    
                    for result in search_results:
                        result_text = result[2]
                        enhanced_contexts.append(f"[direct-{keyword}] {result_text}")
                    
                    if search_results:
                        strategies_used.append(f"direct_{keyword}")
                        
        except Exception:
            pass
        
    except Exception:
        # If all keyword enhancement fails, fall back gracefully
        pass
    
    return enhanced_contexts, strategies_used


def retrieve_enhanced_context(
    db: Database,
    query: str,
    ollama_base_url: str,
    ollama_embed_model: str,
    ollama_chat_model: str,
    dialogue_memory: Optional[DialogueMemory] = None,
    max_total_contexts: int = 15,
    chunk_timeout_sec: float = 15.0,
    keyword_timeout_sec: float = 10.0
) -> EnhancedRAGResult:
    """
    Enhanced RAG retrieval that combines multiple strategies:
    1. Extract keywords from query
    2. Retrieve conversation memory context
    3. Enhance retrieval with keyword-based searches
    4. Combine with traditional chunk retrieval
    
    Args:
        db: Database instance
        query: User query
        ollama_base_url: Ollama base URL
        ollama_embed_model: Embedding model name
        ollama_chat_model: Chat model name for keyword extraction
        dialogue_memory: Optional dialogue memory for recent context
        max_total_contexts: Maximum total contexts to return
        chunk_timeout_sec: Timeout for chunk retrieval
        keyword_timeout_sec: Timeout for keyword-enhanced searches
        
    Returns:
        EnhancedRAGResult with all retrieved contexts and metadata
    """
    result = EnhancedRAGResult()
    
    try:
        # Step 1: Extract keywords from query using LLM (with fallback to rule-based)
        result.extracted_keywords = extract_query_keywords(
            query=query, 
            ollama_base_url=ollama_base_url,
            ollama_chat_model=ollama_chat_model,
            max_keywords=12,
            timeout_sec=keyword_timeout_sec,
            prefer_llm=True
        )
        result.query_type = result.extracted_keywords.get('query_type', 'unknown')
        
        keywords = result.extracted_keywords.get('all_keywords', [])
        search_variants = generate_search_variants(keywords, result.query_type)
        
        # Step 2: Get conversation memory context (existing functionality)
        try:
            memory_contexts = get_relevant_conversation_context(
                db=db,
                query=query,
                ollama_base_url=ollama_base_url,
                ollama_embed_model=ollama_embed_model,
                days_back=7,
                dialogue_memory=dialogue_memory,
                timeout_sec=chunk_timeout_sec
            )
            result.memory_contexts = memory_contexts
            if memory_contexts:
                result.search_strategies_used.append("conversation_memory")
        except Exception:
            pass
        
        # Step 3: Enhanced keyword-based retrieval
        if keywords:
            try:
                keyword_contexts, keyword_strategies = enhance_query_with_keywords(
                    db=db,
                    original_query=query,
                    keywords=keywords,
                    search_variants=search_variants,
                    ollama_base_url=ollama_base_url,
                    ollama_embed_model=ollama_embed_model,
                    top_k=4,
                    timeout_sec=keyword_timeout_sec
                )
                result.keyword_contexts = keyword_contexts
                result.search_strategies_used.extend(keyword_strategies)
            except Exception:
                pass
        
        # Step 4: Traditional chunk retrieval (existing functionality)
        try:
            # Truncate query for chunk retrieval to avoid very long queries
            chunk_query = query[:1024] if len(query) > 1024 else query
            top_chunks = retrieve_top_chunks(
                db=db,
                query=chunk_query,
                ollama_base_url=ollama_base_url,
                embed_model=ollama_embed_model,
                top_k=6,
                timeout_sec=chunk_timeout_sec
            )
            
            chunk_contexts = []
            for chunk_id, score, text in top_chunks:
                chunk_contexts.append(f"[chunk {chunk_id}] {text}")
            
            result.chunk_contexts = chunk_contexts
            if chunk_contexts:
                result.search_strategies_used.append("traditional_chunks")
                
        except Exception:
            pass
        
        # Calculate total contexts found
        result.total_contexts = (len(result.memory_contexts) + 
                               len(result.keyword_contexts) + 
                               len(result.chunk_contexts))
        
    except Exception:
        # If everything fails, ensure we return a valid result
        pass
    
    return result


def format_enhanced_context_for_llm(
    rag_result: EnhancedRAGResult,
    query: str,
    max_contexts: int = 15,
    include_metadata: bool = False
) -> str:
    """
    Format the enhanced RAG results for inclusion in LLM prompt.
    
    Args:
        rag_result: EnhancedRAGResult from retrieve_enhanced_context
        query: Original user query
        max_contexts: Maximum contexts to include
        include_metadata: Whether to include metadata about search strategies
        
    Returns:
        Formatted context string for LLM prompt
    """
    all_contexts = rag_result.get_all_contexts(max_contexts)
    
    if not all_contexts:
        return "Context: No relevant context found."
    
    context_parts = []
    
    # Add metadata header if requested
    if include_metadata and rag_result.extracted_keywords.get('all_keywords'):
        keywords = ', '.join(rag_result.extracted_keywords['all_keywords'][:5])
        context_parts.append(f"Query Analysis: Type='{rag_result.query_type}', Keywords=[{keywords}]")
    
    # Add contexts
    context_parts.append("Context (relevant information):")
    context_parts.extend(all_contexts)
    
    # Add summary if metadata requested
    if include_metadata:
        context_parts.append(f"\n[Retrieved using: {', '.join(rag_result.search_strategies_used)}]")
    
    return "\n".join(context_parts)


def get_enhanced_rag_for_query(
    db: Database,
    query: str,
    ollama_base_url: str,
    ollama_embed_model: str,
    ollama_chat_model: str,
    dialogue_memory: Optional[DialogueMemory] = None,
    max_contexts: int = 15,
    include_metadata: bool = False,
    voice_debug: bool = False
) -> str:
    """
    Convenience function that performs enhanced RAG and returns formatted context.
    
    This is the main function to use for integrating enhanced RAG into existing code.
    
    Args:
        db: Database instance
        query: User query
        ollama_base_url: Ollama base URL
        ollama_embed_model: Embedding model name
        ollama_chat_model: Chat model name for keyword extraction
        dialogue_memory: Optional dialogue memory
        max_contexts: Maximum contexts to return
        include_metadata: Whether to include search metadata
        voice_debug: Whether to print debug information
        
    Returns:
        Formatted context string ready for LLM prompt
    """
    try:
        # Perform enhanced RAG retrieval
        rag_result = retrieve_enhanced_context(
            db=db,
            query=query,
            ollama_base_url=ollama_base_url,
            ollama_embed_model=ollama_embed_model,
            ollama_chat_model=ollama_chat_model,
            dialogue_memory=dialogue_memory,
            max_total_contexts=max_contexts
        )
        
        # Debug output
        if voice_debug:
            try:
                import sys
                keywords = ', '.join(rag_result.extracted_keywords.get('all_keywords', [])[:5])
                extraction_method = rag_result.extracted_keywords.get('extraction_method', 'unknown')
                intent = rag_result.extracted_keywords.get('intent_summary', '')
                print(f"[debug] Enhanced RAG: extracted keywords ({extraction_method}): {keywords}", file=sys.stderr)
                if intent:
                    print(f"[debug] Enhanced RAG: intent: {intent}", file=sys.stderr)
                print(f"[debug] Enhanced RAG: found {rag_result.total_contexts} total contexts", file=sys.stderr)
                print(f"[debug] Enhanced RAG: strategies used: {', '.join(rag_result.search_strategies_used)}", file=sys.stderr)
            except Exception:
                pass
        
        # Format for LLM
        return format_enhanced_context_for_llm(
            rag_result=rag_result,
            query=query,
            max_contexts=max_contexts,
            include_metadata=include_metadata
        )
        
    except Exception:
        # Fallback to simple message if everything fails
        return "Context: Unable to retrieve relevant context at this time."
