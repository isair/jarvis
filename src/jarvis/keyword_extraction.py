from __future__ import annotations
import re
import string
import json
from typing import List, Dict, Set, Tuple, Optional
from collections import Counter
import math

# Language-neutral patterns for technical content
TECHNICAL_PATTERNS = [
    r'\b\w+\.\w+\b',  # file.extension or module.function
    r'\b[A-Z][a-zA-Z0-9_]*\b',  # CamelCase (likely class names)
    r'\b[a-z_]+[a-z0-9_]*\(\)',  # function_calls()
    r'\b\w+_\w+\b',  # snake_case variables
    r'\b[A-Z_]{2,}\b',  # CONSTANTS
    r'\b\d{4}-\d{2}-\d{2}\b',  # dates
    r'\b\d+\.\d+\.\d+\b',  # version numbers
    r'\b\w+://\w+\b',  # URLs/protocols
    r'\b\w+@\w+\.\w+\b',  # email addresses
    r'\b\w+\.\w{2,4}\b',  # file extensions
]

# Minimal language-agnostic filter patterns
FILTER_PATTERNS = [
    r'^\d+$',  # Pure numbers
    r'^[^\w]+$',  # Pure punctuation
    r'^.{1,2}$',  # Very short words (1-2 chars)
]



def extract_keywords_simple(text: str, max_keywords: int = 10) -> List[str]:
    """
    Simple language-agnostic keyword extraction using basic text processing.
    
    Args:
        text: Input text to extract keywords from
        max_keywords: Maximum number of keywords to return
        
    Returns:
        List of extracted keywords sorted by relevance
    """
    if not text or not text.strip():
        return []
    
    # Keep original case for technical terms, but normalize for filtering
    original_text = text.strip()
    
    # Split into words while preserving technical patterns
    words = re.findall(r'\b\w+(?:[._]\w+)*\b', original_text)
    
    # Filter words using language-agnostic patterns
    keywords = []
    for word in words:
        # Skip if matches filter patterns
        if any(re.match(pattern, word) for pattern in FILTER_PATTERNS):
            continue
        
        # Keep words that are 3+ characters or match technical patterns
        if len(word) >= 3 or any(re.match(pattern, word) for pattern in TECHNICAL_PATTERNS):
            keywords.append(word)
    
    # Count frequency and get most common
    word_counts = Counter(keywords)
    common_words = [word for word, count in word_counts.most_common(max_keywords)]
    
    return common_words


def extract_keywords_enhanced(text: str, max_keywords: int = 15) -> Dict[str, List[str]]:
    """
    Enhanced language-agnostic keyword extraction with categorization.
    Simplified to focus on technical patterns and word frequency without language-specific lists.
    
    Args:
        text: Input text to extract keywords from
        max_keywords: Maximum total keywords to return
        
    Returns:
        Dictionary with categorized keywords:
        - 'entities': Technical patterns and quoted terms
        - 'actions': Most frequent longer words (likely verbs)
        - 'topics': General keywords
        - 'questions': Empty (language-agnostic)
    """
    if not text or not text.strip():
        return {'entities': [], 'actions': [], 'topics': [], 'questions': []}
    
    result = {
        'entities': [],
        'actions': [],
        'topics': [],
        'questions': []  # Will remain empty for language neutrality
    }
    
    original_text = text.strip()
    
    # Extract technical patterns and entities (case-sensitive)
    for pattern in TECHNICAL_PATTERNS:
        matches = re.findall(pattern, original_text)
        for match in matches:
            if match and len(match) > 2:
                result['entities'].append(match)
    
    # Extract quoted strings as entities
    quoted_matches = re.findall(r'"([^"]+)"', original_text)
    quoted_matches.extend(re.findall(r"'([^']+)'", original_text))
    for match in quoted_matches:
        if match and len(match) > 2:
            result['entities'].append(match)
    
    # Extract and categorize general words
    words = re.findall(r'\b\w+\b', original_text)
    
    # Filter and categorize by length and frequency
    word_counts = Counter(word for word in words if len(word) >= 3)
    
    # Actions: shorter frequent words (3-6 chars, likely verbs)
    # Topics: longer or less frequent words (likely nouns, concepts)
    for word, count in word_counts.most_common():
        if len(word) <= 6 and count > 1:
            result['actions'].append(word)
        else:
            result['topics'].append(word)
    
    # Limit results for each category
    result['entities'] = result['entities'][:max(2, max_keywords // 3)]
    result['actions'] = result['actions'][:max(2, max_keywords // 4)]
    result['topics'] = result['topics'][:max(3, max_keywords // 2)]
    
    return result


def calculate_tfidf_keywords(text: str, corpus_stats: Optional[Dict[str, int]] = None, max_keywords: int = 10) -> List[Tuple[str, float]]:
    """
    Calculate TF-IDF scores for keywords (language-agnostic version).
    
    Args:
        text: Input text
        corpus_stats: Optional dictionary of word frequencies in corpus
        max_keywords: Maximum number of keywords to return
        
    Returns:
        List of (word, tfidf_score) tuples sorted by score
    """
    if not text or not text.strip():
        return []
    
    # Tokenize using language-agnostic patterns
    words = re.findall(r'\b\w+\b', text)
    
    # Filter using language-agnostic patterns
    filtered_words = []
    for word in words:
        if not any(re.match(pattern, word) for pattern in FILTER_PATTERNS):
            filtered_words.append(word)
    
    if not filtered_words:
        return []
    
    # Calculate term frequency
    word_counts = Counter(filtered_words)
    total_words = len(filtered_words)
    tf_scores = {word: count / total_words for word, count in word_counts.items()}
    
    # Calculate IDF (inverse document frequency)
    idf_scores = {}
    for word in tf_scores:
        if corpus_stats and word in corpus_stats:
            # Real IDF calculation if corpus stats available
            idf = math.log(len(corpus_stats) / (1 + corpus_stats[word]))
        else:
            # Heuristic IDF: favor longer, less common words
            length_bonus = min(2.0, len(word) / 5.0)
            frequency_penalty = math.log(1 + word_counts[word])
            idf = length_bonus / frequency_penalty
        
        idf_scores[word] = max(0.1, idf)  # Minimum IDF of 0.1
    
    # Calculate TF-IDF
    tfidf_scores = []
    for word in tf_scores:
        score = tf_scores[word] * idf_scores[word]
        tfidf_scores.append((word, score))
    
    # Sort by score and return top keywords
    tfidf_scores.sort(key=lambda x: x[1], reverse=True)
    return tfidf_scores[:max_keywords]


def extract_keywords_llm(
    query: str, 
    ollama_base_url: str, 
    ollama_chat_model: str, 
    max_keywords: int = 10,
    timeout_sec: float = 15.0
) -> Dict[str, any]:
    """
    Extract keywords using LLM for better context understanding.
    
    Args:
        query: User query text
        ollama_base_url: Ollama base URL
        ollama_chat_model: Chat model name
        max_keywords: Maximum number of keywords to extract
        timeout_sec: Timeout for LLM request
        
    Returns:
        Dictionary containing extracted keywords and metadata
    """
    if not query or not query.strip():
        return {
            'all_keywords': [],
            'categorized': {'entities': [], 'actions': [], 'topics': [], 'questions': []},
            'query_type': 'unknown',
            'extraction_method': 'none'
        }
    
    try:
        # Import here to avoid circular imports
        from .coach import ask_coach
        
        system_prompt = """You are a keyword extraction specialist for information retrieval. Extract the most important keywords from user queries to find relevant context and information.

Extract keywords that would help find:
- Related conversation history
- Relevant documents and information
- Technical concepts and terms

Focus on:
- Technical terms, names, and specific concepts
- Action verbs that indicate user intent
- Domain-specific terminology
- Important entities and topics
- File names, functions, or system components

Respond ONLY with a JSON object:
{
    "keywords": ["keyword1", "keyword2", ...],
    "entities": ["technical_term1", "file_name", ...],
    "actions": ["action1", "action2", ...],
    "topics": ["concept1", "topic2", ...],
    "query_type": "question|command|search|conversation",
    "intent_summary": "brief description of user intent"
}

Extract maximum """ + str(max_keywords) + """ total keywords. Preserve original language and technical terms exactly."""

        user_prompt = f"Extract keywords from this query: \"{query}\""
        
        response = ask_coach(
            base_url=ollama_base_url,
            chat_model=ollama_chat_model,
            system_prompt=system_prompt,
            user_content=user_prompt,
            timeout_sec=timeout_sec,
            include_location=False  # Don't need location for keyword extraction
        )
        
        if not response:
            # Fallback to rule-based extraction
            return extract_query_keywords_fallback(query, max_keywords)
        
        # Try to parse JSON response
        try:
            # Extract JSON from response (handle cases where LLM adds extra text)
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                parsed = json.loads(json_str)
                
                # Extract and validate fields
                keywords = parsed.get('keywords', [])[:max_keywords]
                entities = parsed.get('entities', [])
                actions = parsed.get('actions', [])
                topics = parsed.get('topics', [])
                query_type = parsed.get('query_type', 'unknown')
                intent_summary = parsed.get('intent_summary', '')
                
                # Combine all keywords for backward compatibility
                all_keywords = []
                seen = set()
                
                # Prioritize entities and actions
                for word in entities + actions + keywords + topics:
                    if word and word.lower() not in seen and len(word) >= 2:
                        all_keywords.append(word)
                        seen.add(word.lower())
                
                return {
                    'all_keywords': all_keywords[:max_keywords],
                    'categorized': {
                        'entities': entities,
                        'actions': actions,
                        'topics': topics,
                        'questions': []  # LLM doesn't need to identify question words
                    },
                    'query_type': query_type,
                    'intent_summary': intent_summary,
                    'extraction_method': 'llm'
                }
                
            else:
                # JSON parsing failed, fall back to rule-based
                return extract_query_keywords_fallback(query, max_keywords)
                
        except json.JSONDecodeError:
            # JSON parsing failed, fall back to rule-based
            return extract_query_keywords_fallback(query, max_keywords)
            
    except Exception:
        # LLM extraction failed, fall back to rule-based
        return extract_query_keywords_fallback(query, max_keywords)


def extract_query_keywords_fallback(query: str, max_keywords: int = 12) -> Dict[str, any]:
    """
    Language-agnostic fallback keyword extraction using multiple strategies.
    
    Args:
        query: User query text  
        max_keywords: Maximum number of keywords to extract
        
    Returns:
        Dictionary containing extracted keywords and metadata
    """
    if not query or not query.strip():
        return {
            'all_keywords': [],
            'categorized': {'entities': [], 'actions': [], 'topics': [], 'questions': []},
            'tfidf_keywords': [],
            'query_type': 'unknown',
            'extraction_method': 'none'
        }
    
    # Extract using different methods
    simple_keywords = extract_keywords_simple(query, max_keywords)
    categorized_keywords = extract_keywords_enhanced(query, max_keywords)
    tfidf_keywords = calculate_tfidf_keywords(query, max_keywords=max_keywords)
    
    # Combine all keywords (remove duplicates, preserve order)
    all_keywords = []
    seen = set()
    
    # Priority: entities (technical terms) > tfidf (relevant) > simple (frequent)
    for word in categorized_keywords['entities']:
        if word.lower() not in seen:
            all_keywords.append(word)
            seen.add(word.lower())
    
    # Add TF-IDF keywords (good relevance indicators)
    for word, score in tfidf_keywords:
        if word not in seen and len(all_keywords) < max_keywords:
            all_keywords.append(word)
            seen.add(word)
    
    # Add remaining simple keywords
    for word in simple_keywords:
        if word not in seen and len(all_keywords) < max_keywords:
            all_keywords.append(word)
            seen.add(word)
    
    # Add actions and topics 
    for word in categorized_keywords['actions'] + categorized_keywords['topics']:
        if word not in seen and len(all_keywords) < max_keywords:
            all_keywords.append(word)
            seen.add(word)
    
    # Infer query type
    query_type = infer_query_type(query, categorized_keywords)
    
    return {
        'all_keywords': all_keywords[:max_keywords],
        'categorized': categorized_keywords,
        'tfidf_keywords': tfidf_keywords,
        'query_type': query_type,
        'extraction_method': 'rule_based'
    }


def infer_query_type(query: str, categorized_keywords: Dict[str, List[str]]) -> str:
    """
    Infer the type of query based on universal patterns (language-agnostic).
    
    Args:
        query: Original query text
        categorized_keywords: Categorized keywords from extract_keywords_enhanced
        
    Returns:
        String indicating query type: 'question', 'command', 'search', 'conversation'
    """
    query = query.strip()
    
    # Check for question patterns (universal indicators)
    if query.endswith('?'):
        return 'question'
    
    # Check for command patterns (based on presence of actions)
    if categorized_keywords['actions']:
        return 'command'
    
    # Check for technical patterns (likely search)
    if categorized_keywords['entities']:
        return 'search'
    
    # Default to conversation
    return 'conversation'


def generate_search_variants(keywords: List[str], query_type: str = 'search') -> List[str]:
    """
    Generate language-agnostic search query variants based on extracted keywords.
    
    Args:
        keywords: List of extracted keywords
        query_type: Type of query (affects variant generation strategy)
        
    Returns:
        List of search query variants optimized for retrieval
    """
    if not keywords:
        return []
    
    variants = []
    
    # Combined query with top keywords
    if len(keywords) >= 2:
        variants.append(' '.join(keywords[:4]))  # Top 4 keywords combined
    
    # Individual high-value keywords (longer terms, technical patterns)
    for keyword in keywords[:3]:
        if len(keyword) >= 4 or any(re.match(pattern, keyword) for pattern in TECHNICAL_PATTERNS):
            variants.append(keyword)
    
    # Keyword pairs for moderate specificity
    if len(keywords) >= 2:
        for i in range(min(2, len(keywords) - 1)):
            variants.append(f"{keywords[i]} {keywords[i + 1]}")
    
    # Technical terms grouped together
    technical_terms = [kw for kw in keywords if any(re.match(pattern, kw) for pattern in TECHNICAL_PATTERNS)]
    if technical_terms:
        variants.append(' '.join(technical_terms[:2]))
    
    # Remove duplicates while preserving order
    seen = set()
    unique_variants = []
    for variant in variants:
        if variant and variant not in seen:
            unique_variants.append(variant)
            seen.add(variant)
    
    return unique_variants[:5]  # Return up to 5 variants


def extract_query_keywords(
    query: str, 
    ollama_base_url: Optional[str] = None, 
    ollama_chat_model: Optional[str] = None, 
    max_keywords: int = 12,
    timeout_sec: float = 15.0,
    prefer_llm: bool = True
) -> Dict[str, any]:
    """
    Main function to extract keywords from a user query.
    Uses LLM extraction when available, falls back to rule-based methods.
    
    Args:
        query: User query text
        ollama_base_url: Optional Ollama base URL for LLM extraction
        ollama_chat_model: Optional chat model name for LLM extraction
        max_keywords: Maximum number of keywords to extract
        timeout_sec: Timeout for LLM requests
        prefer_llm: Whether to prefer LLM extraction over rule-based
        
    Returns:
        Dictionary containing keywords and metadata
    """
    # Try LLM extraction first if available and preferred
    if (prefer_llm and ollama_base_url and ollama_chat_model):
        try:
            llm_result = extract_keywords_llm(
                query=query,
                ollama_base_url=ollama_base_url,
                ollama_chat_model=ollama_chat_model,
                max_keywords=max_keywords,
                timeout_sec=timeout_sec
            )
            
            # If LLM extraction was successful, return it
            if llm_result.get('extraction_method') == 'llm':
                return llm_result
                
        except Exception:
            pass  # Fall through to rule-based extraction
    
    # Fall back to rule-based extraction
    return extract_query_keywords_fallback(query, max_keywords)
