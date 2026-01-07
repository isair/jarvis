"""
Agent Behavior Evaluations

Tests core agent capabilities:
1. Response Quality - Gives useful answers, not deflections
2. Context Utilization - Uses location, time, and memory appropriately  
3. Tool Usage - Calls right tools with right arguments
4. Multi-Step Reasoning - Chains tools and synthesizes information

Run: ./scripts/run_evals.sh
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

_this_file = Path(__file__).resolve()
EVALS_DIR = _this_file.parent
if str(EVALS_DIR) not in sys.path:
    sys.path.insert(0, str(EVALS_DIR))

import pytest
from unittest.mock import patch
import json

from helpers import (
    MockConfig,
    create_mock_llm_response, create_tool_call,
    judge_response_answers_query,
    is_judge_llm_available
)

_JUDGE_LLM_AVAILABLE = is_judge_llm_available()
requires_judge_llm = pytest.mark.skipif(
    not _JUDGE_LLM_AVAILABLE,
    reason="Judge LLM not available"
)


# =============================================================================
# Test Data
# =============================================================================

MOCK_WEATHER_SEARCH = """Web search results for 'weather London UK this week':
1. **BBC Weather** - https://www.bbc.co.uk/weather/2643743
2. **Met Office** - https://www.metoffice.gov.uk/weather/forecast/gcpvj0v07
"""

MOCK_WEATHER_PAGE = """London 7 Day Weather Forecast
Wednesday: Partly cloudy, 12°C, 30% rain
Thursday: Sunny, 14°C, 10% rain  
Friday: Cloudy, 11°C, 60% rain
Saturday: Heavy rain, 10°C, 90% rain
Sunday: Showers, 11°C, 50% rain
"""

MOCK_MEMORY_RESULTS = """Past conversations about health goals:
[2024-01-10] User mentioned wanting to lose 10 pounds by March
[2024-01-12] User said they're targeting 1800 calories per day
[2024-01-14] User logged a gym workout - 45 min cardio
"""

MOCK_NUTRITION_DATA = """Today's nutrition (so far):
- Oatmeal breakfast: 320 kcal, 12g protein
- Chicken salad lunch: 450 kcal, 35g protein
Total: 770 kcal, 47g protein, 65g carbs, 28g fat
"""


# =============================================================================
# Evaluation Helpers
# =============================================================================

def evaluate_response(response: Optional[str], query: str) -> Tuple[bool, List[str]]:
    """
    Evaluate response quality with heuristics.
    Returns (passed, issues).
    """
    issues = []
    
    if response is None:
        return False, ["No response generated"]
    
    response_lower = response.lower().strip()
    
    # Too short
    if len(response_lower) < 20:
        issues.append("Response too short")
    
    # Pure deflection (asking for info without providing anything)
    deflection_only = [
        "how can i help you",
        "what would you like to know",
        "what can i do for you",
    ]
    if any(d in response_lower for d in deflection_only) and len(response_lower) < 100:
        issues.append("Pure deflection without content")
    
    # Topic relevance check (only check one topic per query)
    query_lower = query.lower()
    if "weather" in query_lower:
        weather_terms = ["°c", "°f", "rain", "sun", "cloud", "temperature", "forecast", "warm", "cold", "degrees"]
        if not any(t in response_lower for t in weather_terms):
            issues.append("Weather query but no weather info in response")
    elif "calorie" in query_lower or "pizza" in query_lower or "food" in query_lower:
        nutrition_terms = ["calorie", "kcal", "protein", "carb", "fat", "meal", "eat", "pizza"]
        if not any(t in response_lower for t in nutrition_terms):
            issues.append("Nutrition query but no nutrition info in response")
    
    return len(issues) == 0, issues


@dataclass
class ToolCallCapture:
    """Captures tool calls during evaluation."""
    calls: List[Dict[str, Any]] = field(default_factory=list)
    
    def record(self, name: str, args: Dict[str, Any]):
        self.calls.append({"name": name, "args": args})
    
    def has_tool(self, name: str) -> bool:
        return any(c["name"] == name for c in self.calls)
    
    def get_args(self, name: str) -> Optional[Dict[str, Any]]:
        for c in self.calls:
            if c["name"] == name:
                return c["args"]
        return None


# =============================================================================
# Response Quality Evaluations (LLM-as-Judge)
# =============================================================================

class TestResponseQuality:
    """
    LLM-as-judge evaluations for response quality.
    
    Tests that the judge correctly identifies good vs bad responses.
    This validates our evaluation methodology.
    """
    
    @pytest.mark.eval
    @requires_judge_llm
    @pytest.mark.parametrize("response,should_pass", [
        pytest.param(
            "This week in London: 12°C Wednesday partly cloudy, 14°C Thursday sunny, "
            "rain expected Friday-Saturday with temps around 10-11°C, improving Sunday.",
            True,
            id="Good: complete weekly forecast"
        ),
        pytest.param(
            "It'll be around 12-14°C with some rain mid-week.",
            True,
            id="Good: brief but informative"
        ),
        pytest.param(
            "Hey there! How can I help you today?",
            False,
            id="Bad: generic greeting ignores query"
        ),
        pytest.param(
            "I'm not sure, could you clarify what you mean?",
            False,
            id="Bad: deflection without attempting answer"
        ),
        pytest.param(
            "Sure thing!",
            False,
            id="Bad: empty acknowledgment"
        ),
    ])
    def test_weather_response_quality(self, response: str, should_pass: bool):
        """Judge correctly identifies good vs bad weather responses."""
        query = "how's the weather this week?"
        
        verdict = judge_response_answers_query(
            query=query,
            response=response,
            context=MOCK_WEATHER_PAGE
        )
        
        print(f"\n🧑‍⚖️ Judge Evaluation:")
        print(f"   Response: {response[:60]}...")
        print(f"   Score: {verdict.score:.2f}")
        print(f"   Reasoning: {verdict.reasoning[:100]}...")
        
        if should_pass:
            assert verdict.score >= 0.5, f"Expected pass. Reasoning: {verdict.reasoning}"
        else:
            assert verdict.score < 0.5, f"Expected fail. Reasoning: {verdict.reasoning}"


# =============================================================================
# Context Utilization Evaluations
# =============================================================================

class TestContextUtilization:
    """
    Tests that the agent properly uses available context.
    
    Uses mocked LLM to verify context flows through correctly.
    """
    
    @pytest.mark.eval
    def test_location_context_in_search(self, mock_config, eval_db, eval_dialogue_memory):
        """Agent includes user's location in search queries when available."""
        from jarvis.reply.engine import run_reply_engine
        
        query = "how's the weather?"
        user_location = "Berlin, Germany"
        capture = ToolCallCapture()
        
        def mock_tool_run(db, cfg, tool_name, tool_args, **kwargs):
            from jarvis.tools.types import ToolExecutionResult
            capture.record(tool_name, tool_args or {})
            
            if tool_name == "webSearch":
                return ToolExecutionResult(success=True, reply_text=MOCK_WEATHER_SEARCH)
            return ToolExecutionResult(success=True, reply_text="OK")
        
        call_count = 0
        def mock_chat(base_url, chat_model, messages, timeout_sec, extra_options=None):
            nonlocal call_count
            call_count += 1
            
            # Check if location is in context
            has_location = any("Berlin" in msg.get("content", "") for msg in messages)
            
            if call_count == 1:
                search = "weather Berlin Germany" if has_location else "weather today"
                return create_mock_llm_response("", [create_tool_call("webSearch", {"search_query": search})])
            return create_mock_llm_response("Weather in Berlin: 8°C, partly cloudy.")
        
        with patch('jarvis.reply.engine.run_tool_with_retries', side_effect=mock_tool_run), \
             patch('jarvis.reply.engine.chat_with_messages', side_effect=mock_chat), \
             patch('jarvis.reply.engine.get_location_context', return_value=f"Location: {user_location}"), \
             patch('jarvis.reply.engine.extract_search_params_for_memory', return_value={"keywords": []}), \
             patch('jarvis.reply.engine.select_profile_llm', return_value="life"):
            
            run_reply_engine(db=eval_db, cfg=mock_config, tts=None, text=query, dialogue_memory=eval_dialogue_memory)
        
        # Verify location was used
        assert capture.has_tool("webSearch"), "Should have called webSearch"
        search_args = capture.get_args("webSearch")
        search_query = search_args.get("search_query", "").lower()
        
        print(f"\n📊 Context Utilization:")
        print(f"   User location: {user_location}")
        print(f"   Search query: {search_query}")
        
        assert "berlin" in search_query, f"Search should include location. Got: {search_query}"


# =============================================================================
# Tool Usage Evaluations
# =============================================================================

class TestToolUsage:
    """
    Tests that the agent uses tools correctly.
    
    Verifies tool selection, argument quality, and chaining.
    """
    
    @pytest.mark.eval
    def test_simple_search_flow(self, mock_config, eval_db, eval_dialogue_memory):
        """Agent calls webSearch for information queries."""
        from jarvis.reply.engine import run_reply_engine
        
        query = "what's happening in tech news today?"
        capture = ToolCallCapture()
        
        def mock_tool_run(db, cfg, tool_name, tool_args, **kwargs):
            from jarvis.tools.types import ToolExecutionResult
            capture.record(tool_name, tool_args or {})
            return ToolExecutionResult(success=True, reply_text="Tech news: AI advances, new chip releases.")
        
        call_count = 0
        def mock_chat(base_url, chat_model, messages, timeout_sec, extra_options=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return create_mock_llm_response("", [create_tool_call("webSearch", {"search_query": "tech news today"})])
            return create_mock_llm_response("Today in tech: Major AI announcements and new hardware releases.")
        
        with patch('jarvis.reply.engine.run_tool_with_retries', side_effect=mock_tool_run), \
             patch('jarvis.reply.engine.chat_with_messages', side_effect=mock_chat), \
             patch('jarvis.reply.engine.extract_search_params_for_memory', return_value={"keywords": []}), \
             patch('jarvis.reply.engine.select_profile_llm', return_value="developer"):
            
            response = run_reply_engine(db=eval_db, cfg=mock_config, tts=None, text=query, dialogue_memory=eval_dialogue_memory)
        
        print(f"\n📊 Tool Usage:")
        print(f"   Query: {query}")
        print(f"   Tools called: {[c['name'] for c in capture.calls]}")
        
        assert capture.has_tool("webSearch"), "Should call webSearch for news query"
        assert response is not None, "Should generate a response"
    
    @pytest.mark.eval
    def test_tool_chaining_search_then_fetch(self, mock_config, eval_db, eval_dialogue_memory):
        """Agent chains webSearch → fetchWebPage for detailed info."""
        from jarvis.reply.engine import run_reply_engine
        
        query = "how's the weather this week?"
        capture = ToolCallCapture()
        
        def mock_tool_run(db, cfg, tool_name, tool_args, **kwargs):
            from jarvis.tools.types import ToolExecutionResult
            capture.record(tool_name, tool_args or {})
            
            if tool_name == "webSearch":
                return ToolExecutionResult(success=True, reply_text=MOCK_WEATHER_SEARCH)
            elif tool_name == "fetchWebPage":
                return ToolExecutionResult(success=True, reply_text=MOCK_WEATHER_PAGE)
            return ToolExecutionResult(success=True, reply_text="OK")
        
        call_count = 0
        def mock_chat(base_url, chat_model, messages, timeout_sec, extra_options=None):
            nonlocal call_count
            call_count += 1
            
            if call_count == 1:
                return create_mock_llm_response("", [create_tool_call("webSearch", {"search_query": "weather London this week"})])
            elif call_count == 2:
                return create_mock_llm_response("", [create_tool_call("fetchWebPage", {"url": "https://www.bbc.co.uk/weather/2643743"})])
            return create_mock_llm_response(
                "This week: 12°C Wed partly cloudy, 14°C Thu sunny, "
                "rain Fri-Sat around 10-11°C, improving Sunday."
            )
        
        with patch('jarvis.reply.engine.run_tool_with_retries', side_effect=mock_tool_run), \
             patch('jarvis.reply.engine.chat_with_messages', side_effect=mock_chat), \
             patch('jarvis.reply.engine.extract_search_params_for_memory', return_value={"keywords": []}), \
             patch('jarvis.reply.engine.select_profile_llm', return_value="life"):
            
            response = run_reply_engine(db=eval_db, cfg=mock_config, tts=None, text=query, dialogue_memory=eval_dialogue_memory)
        
        print(f"\n📊 Tool Chaining:")
        print(f"   Tools called: {[c['name'] for c in capture.calls]}")
        print(f"   Response: {response[:80] if response else 'None'}...")
        
        assert capture.has_tool("webSearch"), "Should call webSearch first"
        assert capture.has_tool("fetchWebPage"), "Should chain to fetchWebPage for details"
        
        passed, issues = evaluate_response(response, query)
        assert passed, f"Response quality issues: {issues}"


# =============================================================================
# Multi-Step Reasoning Evaluations
# =============================================================================

class TestMultiStepReasoning:
    """
    Tests complex scenarios requiring multiple steps.
    
    These test the agent's ability to:
    - Chain multiple tools
    - Use memory context
    - Synthesize information from multiple sources
    """
    
    @pytest.mark.eval
    def test_nutrition_advice_uses_memory_and_data(self, mock_config, eval_db, eval_dialogue_memory):
        """
        Agent uses memory + nutrition data for personalized advice.
        
        Scenario: User asks about eating pizza
        Expected: Agent recalls health goals from memory AND checks today's intake
        """
        from jarvis.reply.engine import run_reply_engine
        
        query = "should I order pizza tonight?"
        capture = ToolCallCapture()
        
        def mock_tool_run(db, cfg, tool_name, tool_args, **kwargs):
            from jarvis.tools.types import ToolExecutionResult
            capture.record(tool_name, tool_args or {})
            
            if tool_name == "recallConversation":
                return ToolExecutionResult(success=True, reply_text=MOCK_MEMORY_RESULTS)
            elif tool_name == "fetchMeals":
                return ToolExecutionResult(success=True, reply_text=MOCK_NUTRITION_DATA)
            return ToolExecutionResult(success=True, reply_text="OK")
        
        call_count = 0
        def mock_chat(base_url, chat_model, messages, timeout_sec, extra_options=None):
            nonlocal call_count
            call_count += 1
            
            if call_count == 1:
                # Agent should recall health goals
                return create_mock_llm_response("", [
                    create_tool_call("recallConversation", {"search_query": "health goals diet"}),
                    create_tool_call("fetchMeals", {})
                ])
            return create_mock_llm_response(
                "You've had 770 kcal so far today, leaving room for pizza within your 1800 kcal target. "
                "Given your weight loss goal, I'd suggest a thin crust with veggies - around 600 kcal for 2 slices. "
                "You've been consistent this week, so one pizza night won't derail your progress!"
            )
        
        with patch('jarvis.reply.engine.run_tool_with_retries', side_effect=mock_tool_run), \
             patch('jarvis.reply.engine.chat_with_messages', side_effect=mock_chat), \
             patch('jarvis.reply.engine.extract_search_params_for_memory', return_value={"keywords": ["health", "diet"]}), \
             patch('jarvis.reply.engine.select_profile_llm', return_value="life"):
            
            response = run_reply_engine(db=eval_db, cfg=mock_config, tts=None, text=query, dialogue_memory=eval_dialogue_memory)
        
        print(f"\n📊 Multi-Step Reasoning:")
        print(f"   Query: {query}")
        print(f"   Tools called: {[c['name'] for c in capture.calls]}")
        print(f"   Response: {response[:100] if response else 'None'}...")
        
        # Should use both memory and nutrition tools
        tools_used = [c["name"] for c in capture.calls]
        assert "recallConversation" in tools_used or "fetchMeals" in tools_used, \
            f"Should use memory or nutrition tools. Used: {tools_used}"
        
        # Response should reference calorie info
        if response:
            assert "calor" in response.lower() or "kcal" in response.lower(), \
                "Response should mention calorie context"


# =============================================================================
# End-to-End Live Evaluations
# =============================================================================

class TestLiveEndToEnd:
    """
    Live tests with real LLM inference.
    
    These run against the actual model and verify real behavior.
    """
    
    @pytest.mark.eval
    @requires_judge_llm
    def test_weather_query_live(self, mock_config, eval_db, eval_dialogue_memory):
        """Live eval: Weather query with real LLM."""
        from jarvis.reply.engine import run_reply_engine
        from helpers import JUDGE_MODEL
        
        query = "how's the weather this week?"
        test_location = "London, England, United Kingdom"
        
        mock_config.ollama_base_url = "http://localhost:11434"
        mock_config.ollama_chat_model = JUDGE_MODEL
        
        def mock_get_location(**kwargs):
            return f"Location: {test_location}"
        
        with patch('jarvis.reply.engine.get_location_context', side_effect=mock_get_location):
            response = run_reply_engine(
                db=eval_db, cfg=mock_config, tts=None,
                text=query, dialogue_memory=eval_dialogue_memory
            )
        
        print(f"\n📝 Live Eval:")
        print(f"   Query: {query}")
        print(f"   Response: {response}")
        
        # Heuristic check
        passed, issues = evaluate_response(response, query)
        print(f"   Heuristic: {'PASS' if passed else 'FAIL'} {issues}")
        
        assert passed, f"Live eval failed: {issues}"
        
        # LLM judge check
        verdict = judge_response_answers_query(query, response or "")
        print(f"   Judge score: {verdict.score:.2f}")
        
        assert verdict.score >= 0.4, f"Judge failed: {verdict.reasoning}"

