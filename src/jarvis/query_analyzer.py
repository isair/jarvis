"""
Query analysis system for determining user intent and extracting relevant information.
This drives the multi-tool execution system.
"""

from __future__ import annotations
import json
import re
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone, timedelta
from .coach import ask_coach


def analyze_user_query(
    query: str,
    ollama_base_url: str,
    ollama_chat_model: str,
    timeout_sec: float = 15.0
) -> Dict[str, Any]:
    """
    Analyze user query to extract intent, keywords, time references, and determine needed tools.
    
    Args:
        query: User's query text
        ollama_base_url: Ollama base URL
        ollama_chat_model: Chat model name
        timeout_sec: Timeout for LLM call
        
    Returns:
        Dictionary with analysis results:
        {
            'intent': str,  # conversation, food, weather, screenshot, etc.
            'keywords': List[str],  # relevant search terms
            'time_range': Optional[Dict],  # {from: ISO, to: ISO} if temporal query
            'suggested_tools': List[str],  # tools that should be called
            'confidence': float,  # 0-1 confidence in analysis
            'analysis_method': str  # 'llm' or 'fallback'
        }
    """
    
    system_prompt = """You are a query analyzer. Analyze user queries to determine:
1. PRIMARY INTENT (conversation/memory, food/nutrition, weather/web, screenshot, general)
2. RELEVANT KEYWORDS for searching
3. TIME REFERENCE if present (convert to specific dates)
4. NEEDED TOOLS based on intent

Current date/time for reference: {current_time}

RESPOND ONLY with a JSON object in this exact format:
{{
    "intent": "conversation|food|weather|screenshot|web|general",
    "keywords": ["keyword1", "keyword2"],
    "time_range": {{"from": "2025-08-22T00:00:00Z", "to": "2025-08-22T23:59:59Z"}} or null,
    "suggested_tools": ["TOOL_NAME1", "TOOL_NAME2"],
    "confidence": 0.85
}}

RULES:
- intent: conversation (asking about past conversations/memory), food (eating/meals), weather (weather/news), screenshot (visual help needed), web (current info), general (other)
- keywords: 2-5 most relevant search terms, avoid stop words
- time_range: only if query mentions time (today, yesterday, last week, etc.) - convert to exact ISO timestamps
- suggested_tools: RECALL_CONVERSATION for memory, FETCH_MEALS for food history, WEB_SEARCH for current info, SCREENSHOT for visual help
- confidence: how certain you are about the analysis (0.0-1.0)

Examples:
"what did I eat yesterday" → {{"intent": "food", "keywords": ["eat", "food", "meals"], "time_range": {{"from": "2025-08-21T00:00:00Z", "to": "2025-08-21T23:59:59Z"}}, "suggested_tools": ["FETCH_MEALS"], "confidence": 0.95}}
"do you remember that warhammer keyword?" → {{"intent": "conversation", "keywords": ["warhammer", "keyword", "remember"], "time_range": null, "suggested_tools": ["RECALL_CONVERSATION"], "confidence": 0.9}}
"what's the weather like today?" → {{"intent": "weather", "keywords": ["weather", "today"], "time_range": null, "suggested_tools": ["WEB_SEARCH"], "confidence": 0.95}}
"""
    
    # Add current time context
    now = datetime.now(timezone.utc)
    current_time = now.strftime("%A, %Y-%m-%d %H:%M UTC")
    formatted_system_prompt = system_prompt.format(current_time=current_time)
    
    try:
        # Call LLM for analysis
        response = ask_coach(
            base_url=ollama_base_url,
            chat_model=ollama_chat_model,
            system_prompt=formatted_system_prompt,
            user_content=f"Analyze this query: {query}",
            timeout_sec=timeout_sec,
            include_location=False  # Don't need location for analysis
        )
        
        if response:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    analysis = json.loads(json_match.group())
                    # Validate required fields
                    required_fields = ['intent', 'keywords', 'suggested_tools', 'confidence']
                    if all(field in analysis for field in required_fields):
                        analysis['analysis_method'] = 'llm'
                        return analysis
                except json.JSONDecodeError:
                    pass
    
    except Exception:
        pass
    
    # Fallback to rule-based analysis
    return _fallback_query_analysis(query)


def _fallback_query_analysis(query: str) -> Dict[str, Any]:
    """
    Rule-based fallback query analysis when LLM fails.
    """
    query_lower = query.lower().strip()
    
    # Extract basic keywords (remove common stop words)
    stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can'}
    words = [w.strip('.,!?;:()[]{}"\'-') for w in query_lower.split()]
    keywords = [w for w in words if len(w) > 2 and w not in stop_words][:5]
    
    # Determine intent based on patterns
    intent = "general"
    suggested_tools = []
    confidence = 0.6  # Lower confidence for rule-based
    
    # Weather/web patterns (check first since they're more specific)
    if any(word in query_lower for word in ['weather', 'temperature', 'forecast', 'rain', 'sunny', 'cloudy']):
        intent = "weather"
        suggested_tools = ["WEB_SEARCH"]
        confidence = 0.8
    
    # Conversation/memory patterns
    elif any(phrase in query_lower for phrase in ['remember', 'recall', 'what did we', 'what have we', 'talked about', 'discussed', 'conversation', 'mentioned', 'keyword', 'password']):
        intent = "conversation"
        suggested_tools = ["RECALL_CONVERSATION"]
        confidence = 0.8
    
    # Food/meal patterns
    elif any(word in query_lower for word in ['eat', 'ate', 'food', 'meal', 'breakfast', 'lunch', 'dinner', 'snack', 'drink', 'drank']):
        intent = "food"
        if any(word in query_lower for word in ['yesterday', 'today', 'what did i', 'what have i']):
            suggested_tools = ["FETCH_MEALS"]
        else:
            suggested_tools = ["RECALL_CONVERSATION"]
        confidence = 0.75
    
    # Screenshot patterns
    elif any(phrase in query_lower for phrase in ['see', 'screen', 'image', 'visual', 'show me', 'look at']):
        intent = "screenshot"
        suggested_tools = ["SCREENSHOT"]
        confidence = 0.7
    
    # Check for time references
    time_range = _extract_time_range(query_lower)
    
    return {
        'intent': intent,
        'keywords': keywords,
        'time_range': time_range,
        'suggested_tools': suggested_tools,
        'confidence': confidence,
        'analysis_method': 'fallback'
    }


def _extract_time_range(query: str) -> Optional[Dict[str, str]]:
    """
    Extract time range from query using simple pattern matching.
    """
    now = datetime.now(timezone.utc)
    
    # Yesterday
    if 'yesterday' in query:
        yesterday = now - timedelta(days=1)
        return {
            'from': yesterday.replace(hour=0, minute=0, second=0, microsecond=0).isoformat() + 'Z',
            'to': yesterday.replace(hour=23, minute=59, second=59, microsecond=999999).isoformat() + 'Z'
        }
    
    # Today
    if 'today' in query:
        return {
            'from': now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat() + 'Z',
            'to': now.replace(hour=23, minute=59, second=59, microsecond=999999).isoformat() + 'Z'
        }
    
    # Last week
    if 'last week' in query:
        week_ago = now - timedelta(days=7)
        yesterday = now - timedelta(days=1)
        return {
            'from': week_ago.replace(hour=0, minute=0, second=0, microsecond=0).isoformat() + 'Z',
            'to': yesterday.replace(hour=23, minute=59, second=59, microsecond=999999).isoformat() + 'Z'
        }
    
    # This week
    if 'this week' in query:
        week_start = now - timedelta(days=now.weekday())
        return {
            'from': week_start.replace(hour=0, minute=0, second=0, microsecond=0).isoformat() + 'Z',
            'to': now.replace(hour=23, minute=59, second=59, microsecond=999999).isoformat() + 'Z'
        }
    
    return None


def determine_tools_from_analysis(analysis: Dict[str, Any], available_tools: List[str]) -> List[str]:
    """
    Determine which tools should be called based on query analysis.
    
    Args:
        analysis: Result from analyze_user_query
        available_tools: List of tools available for the current profile
        
    Returns:
        List of tool names to execute
    """
    suggested = analysis.get('suggested_tools', [])
    intent = analysis.get('intent', 'general')
    
    # Filter by available tools
    tools_to_call = [tool for tool in suggested if tool in available_tools]
    
    # Add additional tools based on intent and context
    if intent == 'conversation' and 'RECALL_CONVERSATION' not in tools_to_call and 'RECALL_CONVERSATION' in available_tools:
        tools_to_call.append('RECALL_CONVERSATION')
    
    # For complex queries, might want multiple tools
    if intent == 'food' and analysis.get('time_range'):
        # Food query with time range - use FETCH_MEALS for structured data
        if 'FETCH_MEALS' in available_tools and 'FETCH_MEALS' not in tools_to_call:
            tools_to_call.append('FETCH_MEALS')
        # But also include conversation memory for context
        if 'RECALL_CONVERSATION' in available_tools and 'RECALL_CONVERSATION' not in tools_to_call:
            tools_to_call.append('RECALL_CONVERSATION')
    
    return tools_to_call


if __name__ == "__main__":
    # Test the analyzer
    test_queries = [
        "what did I eat yesterday?",
        "do you remember that warhammer 40k keyword I told you?",
        "what's the weather like today?",
        "what have we talked about this week?",
        "show me what's on my screen"
    ]
    
    for query in test_queries:
        result = _fallback_query_analysis(query)
        print(f"Query: {query}")
        print(f"Analysis: {result}")
        print()
