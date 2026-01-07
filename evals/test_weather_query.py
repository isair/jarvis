"""
Eval: Weather query response quality.

Tests that when a user asks about weather, the assistant:
1. Uses appropriate tools (webSearch, fetchWebPage)
2. Returns actual weather information instead of generic greetings
3. Addresses the specific query (e.g., "this week" means multiple days)
4. Includes location context when known

Run with: pytest evals/test_weather_query.py -v
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple

# Ensure evals directory is in path for imports
_this_file = Path(__file__).resolve()
EVALS_DIR = _this_file.parent
if str(EVALS_DIR) not in sys.path:
    sys.path.insert(0, str(EVALS_DIR))

import pytest
from unittest.mock import patch
import json

from helpers import (
    MockConfig, EvalResult,
    is_generic_greeting, response_addresses_topic,
    create_mock_llm_response, create_tool_call,
    judge_response_answers_query, judge_search_query_quality,
    is_judge_llm_available
)

# Check LLM availability once at module load
_JUDGE_LLM_AVAILABLE = is_judge_llm_available()
requires_judge_llm = pytest.mark.skipif(
    not _JUDGE_LLM_AVAILABLE,
    reason="Judge LLM not available (Ollama not running or model missing)"
)


# =============================================================================
# Test Data
# =============================================================================

MOCK_WEATHER_SEARCH_RESULTS = """Web search results for 'weather London UK this week':

1. **London Weather Forecast - Met Office**
   Link: https://www.metoffice.gov.uk/weather/forecast/gcpvj0v07

2. **London 7-Day Weather Forecast - BBC Weather**
   Link: https://www.bbc.co.uk/weather/2643743

3. **London Weather - AccuWeather**
   Link: https://www.accuweather.com/en/gb/london/ec4a-2/weather-forecast/328328
"""

MOCK_WEATHER_PAGE_CONTENT = """**Title:** London 7 Day Weather Forecast - BBC Weather

**URL:** https://www.bbc.co.uk/weather/2643743

**Content:**
London Weather Forecast
Tonight: Cloudy with light rain, 8°C
Wednesday: Partly cloudy, High 12°C, Low 7°C, 30% chance of rain
Thursday: Sunny intervals, High 14°C, Low 8°C, 10% chance of rain
Friday: Cloudy, High 11°C, Low 6°C, 60% chance of rain
Saturday: Heavy rain, High 10°C, Low 5°C, 90% chance of rain
Sunday: Light showers, High 11°C, Low 6°C, 50% chance of rain
Monday: Partly cloudy, High 13°C, Low 7°C, 20% chance of rain

Humidity: 75%
Wind: SW 15 mph
"""


# =============================================================================
# Tool Call Capture Infrastructure
# =============================================================================

@dataclass
class CapturedToolCall:
    """Captured tool call with arguments."""
    name: str
    args: Dict[str, Any]
    result: str
    is_success: bool


@dataclass
class EvalCapture:
    """Captures all relevant data from an eval run."""
    query: str
    response: Optional[str]
    tool_calls: List[CapturedToolCall] = field(default_factory=list)
    llm_turn_count: int = 0
    messages_sent_to_llm: List[Dict] = field(default_factory=list)
    location_context: Optional[str] = None

    def get_search_queries(self) -> List[str]:
        """Extract all search queries from captured tool calls."""
        return [
            tc.args.get("search_query", "")
            for tc in self.tool_calls
            if tc.name == "webSearch" and "search_query" in tc.args
        ]

    def has_tool_results(self) -> bool:
        """Check if any tools returned successful results."""
        return any(tc.is_success for tc in self.tool_calls)


# =============================================================================
# Heuristic Evaluators (No LLM Required)
# =============================================================================

def evaluate_response_heuristically(
    query: str,
    response: Optional[str],
    tool_results_available: bool = False
) -> Tuple[bool, List[str]]:
    """
    Evaluate a response using heuristic rules.

    Returns: (is_passed, list_of_issues)
    """
    issues = []

    if response is None:
        issues.append("Response is None")
        return False, issues

    response_lower = response.lower().strip()

    # Check for empty response
    if len(response_lower) < 10:
        issues.append(f"Response too short: {len(response_lower)} chars")

    # Check for generic greetings when tool results were available
    if tool_results_available and is_generic_greeting(response):
        issues.append("Generic greeting despite having tool results")

    # Check if response addresses the topic
    query_lower = query.lower()
    if "weather" in query_lower:
        weather_terms = ["weather", "temperature", "rain", "sun", "cloud", "°c", "°f", "forecast", "warm", "cold", "hot"]
        if not any(term in response_lower for term in weather_terms):
            issues.append("Weather query but response has no weather-related terms")

    # Check for deflection patterns
    deflection_patterns = [
        "how can i help",
        "what would you like",
        "let me know",
        "feel free to ask",
        "is there something",
        "what can i do for you",
    ]
    for pattern in deflection_patterns:
        if pattern in response_lower:
            issues.append(f"Contains deflection pattern: '{pattern}'")
            break

    return len(issues) == 0, issues


def evaluate_search_query_heuristically(
    user_query: str,
    search_query: str,
    location: Optional[str] = None,
) -> Tuple[bool, List[str], Dict[str, float]]:
    """
    Evaluate a search query using heuristic rules.

    Returns: (is_passed, list_of_issues, scores_dict)
    """
    issues = []
    scores = {}

    search_lower = search_query.lower()
    user_lower = user_query.lower()

    # Check if search captures user intent
    if "weather" in user_lower and "weather" not in search_lower:
        issues.append("Weather query but search doesn't mention weather")
        scores["intent_match"] = 0.3
    else:
        scores["intent_match"] = 0.8

    # Check time awareness
    time_terms = ["today", "tomorrow", "week", "weekend", "monday", "tuesday", "wednesday",
                  "thursday", "friday", "saturday", "sunday"]
    user_has_time = any(t in user_lower for t in time_terms)
    search_has_time = any(t in search_lower for t in time_terms)

    if user_has_time and not search_has_time:
        issues.append("User specified time but search doesn't include it")
        scores["time_awareness"] = 0.4
    else:
        scores["time_awareness"] = 0.9

    # Check location awareness
    if location:
        location_parts = [p.strip().lower() for p in location.split(",")]
        # Check for city or country
        location_found = any(
            part in search_lower
            for part in location_parts
            if len(part) > 2  # Skip short terms
        )
        if not location_found:
            issues.append(f"Location '{location}' not reflected in search query")
            scores["location_awareness"] = 0.2
        else:
            scores["location_awareness"] = 0.9
    else:
        scores["location_awareness"] = 0.5  # Neutral if no location known

    # Overall pass if no major issues
    is_passed = len(issues) == 0 or all(s >= 0.6 for s in scores.values())

    return is_passed, issues, scores


# =============================================================================
# Test Classes
# =============================================================================

class TestWeatherResponseQuality:
    """
    Eval tests for weather response quality using heuristic evaluation.

    These tests work WITHOUT LLM and catch common failure patterns.
    """

    @pytest.mark.eval
    @pytest.mark.parametrize("response,should_pass,expected_issue", [
        # Good responses
        (
            "This week expect temperatures around 12-15°C with rain on Saturday.",
            True,
            None
        ),
        (
            "The forecast shows partly cloudy conditions with highs of 14°C tomorrow.",
            True,
            None
        ),
        # Bad responses - generic greetings
        (
            "Hey there! How can I help you today?",
            False,
            "Generic greeting"
        ),
        (
            "Sure thing! What would you like to know?",
            False,
            "deflection"
        ),
        # Bad responses - no weather content
        (
            "I've searched for that information.",
            False,
            "no weather-related terms"
        ),
        # Edge cases
        (
            "",
            False,
            "too short"
        ),
        (
            "It's nice",
            False,
            "too short"
        ),
    ])
    def test_response_quality_heuristics(self, response, should_pass, expected_issue):
        """
        Eval: Test heuristic response evaluation catches common issues.
        """
        query = "how's the weather this week?"
        is_passed, issues = evaluate_response_heuristically(
            query=query,
            response=response,
            tool_results_available=True  # Simulate that tools were called
        )

        if should_pass:
            assert is_passed, f"Expected PASS but got issues: {issues}"
        else:
            assert not is_passed, f"Expected FAIL but got PASS for: {response[:50]}"
            if expected_issue:
                assert any(expected_issue.lower() in issue.lower() for issue in issues), (
                    f"Expected issue containing '{expected_issue}' but got: {issues}"
                )


class TestSearchQueryQuality:
    """
    Eval tests for search query quality using heuristic evaluation.
    """

    @pytest.mark.eval
    @pytest.mark.parametrize("user_query,search_query,location,should_pass", [
        # Good search queries with location
        ("how's the weather?", "weather London UK today", "London, UK", True),
        ("weather this week", "weather forecast London this week", "London, UK", True),
        # Missing location
        ("how's the weather?", "weather today", "London, UK", False),
        ("weather this week", "weather forecast", "Kensington, London, UK", False),
        # Good queries without known location
        ("how's the weather in Paris?", "weather Paris today", None, True),
        # Missing time context
        ("weather for the weekend", "weather forecast", None, False),
    ])
    def test_search_query_heuristics(self, user_query, search_query, location, should_pass):
        """
        Eval: Test search query quality catches location/time issues.
        """
        is_passed, issues, scores = evaluate_search_query_heuristically(
            user_query=user_query,
            search_query=search_query,
            location=location
        )

        print(f"\n📝 User query: '{user_query}'")
        print(f"🔍 Search query: '{search_query}'")
        print(f"📍 Location: {location}")
        print(f"📊 Scores: {scores}")
        print(f"⚠️  Issues: {issues}")

        if should_pass:
            assert is_passed, f"Expected PASS but got issues: {issues}"
        else:
            assert not is_passed, f"Expected FAIL with search: '{search_query}'"


class TestToolCallCapture:
    """
    Eval tests that capture and validate tool call behavior.

    These tests mock only the tool execution, letting the engine logic run.
    """

    @pytest.mark.eval
    def test_captures_search_query_arguments(self, mock_config, eval_db, eval_dialogue_memory):
        """
        Eval: Verify we capture what search queries are generated.

        This test runs the actual engine logic with mocked tools and LLM,
        capturing the exact arguments passed to tools.
        """
        from jarvis.reply.engine import run_reply_engine

        query = "how's the weather this week?"
        capture = EvalCapture(query=query, response=None)
        llm_call_count = 0

        def mock_tool_run(db, cfg, tool_name, tool_args, **kwargs):
            from jarvis.tools.types import ToolExecutionResult

            # Capture the tool call
            result_text = ""
            is_success = True

            if tool_name == "webSearch":
                result_text = MOCK_WEATHER_SEARCH_RESULTS
            elif tool_name == "fetchWebPage":
                result_text = MOCK_WEATHER_PAGE_CONTENT
            else:
                result_text = f"Unknown tool: {tool_name}"
                is_success = False

            capture.tool_calls.append(CapturedToolCall(
                name=tool_name,
                args=tool_args or {},
                result=result_text,
                is_success=is_success
            ))

            return ToolExecutionResult(success=is_success, reply_text=result_text)

        def mock_chat(base_url, chat_model, messages, timeout_sec, extra_options=None):
            nonlocal llm_call_count
            llm_call_count += 1
            capture.llm_turn_count = llm_call_count

            # Capture messages for debugging
            capture.messages_sent_to_llm = messages.copy()

            # Extract location from context if present
            for msg in messages:
                if msg.get("role") == "system" and "Location:" in msg.get("content", ""):
                    content = msg["content"]
                    loc_start = content.find("Location:") + 9
                    loc_end = content.find("\n", loc_start) if "\n" in content[loc_start:] else len(content)
                    capture.location_context = content[loc_start:loc_end].strip()

            has_tool_results = any(msg.get("role") == "tool" for msg in messages)

            if llm_call_count == 1:
                # First call - should search with location if available
                search_query = "weather London this week" if capture.location_context else "weather this week"
                return create_mock_llm_response(
                    "",
                    [create_tool_call("webSearch", {"search_query": search_query})]
                )
            elif llm_call_count == 2 and has_tool_results:
                return create_mock_llm_response(
                    "",
                    [create_tool_call("fetchWebPage", {"url": "https://www.bbc.co.uk/weather/2643743"})]
                )
            else:
                return create_mock_llm_response(
                    "This week in London: Wednesday 12°C partly cloudy, Thursday 14°C sunny, "
                    "Friday 11°C cloudy with 60% rain, Saturday heavy rain 10°C, "
                    "improving Sunday-Monday 11-13°C."
                )

        with patch('jarvis.reply.engine.run_tool_with_retries', side_effect=mock_tool_run), \
             patch('jarvis.reply.engine.chat_with_messages', side_effect=mock_chat), \
             patch('jarvis.reply.engine.extract_search_params_for_memory', return_value={"keywords": []}), \
             patch('jarvis.reply.engine.select_profile_llm', return_value="life"):

            capture.response = run_reply_engine(
                db=eval_db,
                cfg=mock_config,
                tts=None,
                text=query,
                dialogue_memory=eval_dialogue_memory
            )

        # Validate capture
        print(f"\n📊 Eval Capture Summary:")
        print(f"   Query: {capture.query}")
        print(f"   Response: {capture.response[:100] if capture.response else 'None'}...")
        print(f"   LLM turns: {capture.llm_turn_count}")
        print(f"   Tool calls: {len(capture.tool_calls)}")
        for tc in capture.tool_calls:
            print(f"     - {tc.name}: {json.dumps(tc.args)}")
        print(f"   Location context: {capture.location_context}")

        # Assertions
        assert capture.response is not None, "Should have a response"
        assert capture.has_tool_results(), "Should have successful tool calls"

        search_queries = capture.get_search_queries()
        assert len(search_queries) > 0, "Should have made at least one search"

        # Validate response quality
        is_passed, issues = evaluate_response_heuristically(
            query=query,
            response=capture.response,
            tool_results_available=capture.has_tool_results()
        )
        assert is_passed, f"Response failed heuristic evaluation: {issues}"

    @pytest.mark.eval
    def test_location_included_in_search_when_available(self, mock_config, eval_db, eval_dialogue_memory):
        """
        Eval: When location context is available, verify it's used in search.
        """
        from jarvis.reply.engine import run_reply_engine

        query = "how's the weather?"
        user_location = "Kensington, Royal Kensington and Chelsea, United Kingdom"
        captured_search_queries = []

        def mock_tool_run(db, cfg, tool_name, tool_args, **kwargs):
            from jarvis.tools.types import ToolExecutionResult
            if tool_name == "webSearch":
                captured_search_queries.append(tool_args.get("search_query", ""))
                return ToolExecutionResult(success=True, reply_text=MOCK_WEATHER_SEARCH_RESULTS)
            elif tool_name == "fetchWebPage":
                return ToolExecutionResult(success=True, reply_text=MOCK_WEATHER_PAGE_CONTENT)
            return ToolExecutionResult(success=False, reply_text="Unknown")

        call_count = 0
        def mock_chat(base_url, chat_model, messages, timeout_sec, extra_options=None):
            nonlocal call_count
            call_count += 1

            # Check if location was injected into context
            has_location = any(
                "Kensington" in msg.get("content", "") or "London" in msg.get("content", "")
                for msg in messages
            )

            if call_count == 1:
                # If LLM sees location, it should include it in search
                if has_location:
                    search = "weather Kensington London UK"
                else:
                    search = "weather today"  # Bad - no location
                return create_mock_llm_response("", [create_tool_call("webSearch", {"search_query": search})])
            else:
                return create_mock_llm_response(
                    "In Kensington this week: 12-14°C, rain expected Saturday."
                )

        def mock_get_location(**kwargs):
            return f"Location: {user_location}"

        with patch('jarvis.reply.engine.run_tool_with_retries', side_effect=mock_tool_run), \
             patch('jarvis.reply.engine.chat_with_messages', side_effect=mock_chat), \
             patch('jarvis.reply.engine.get_location_context', side_effect=mock_get_location), \
             patch('jarvis.reply.engine.extract_search_params_for_memory', return_value={"keywords": []}), \
             patch('jarvis.reply.engine.select_profile_llm', return_value="life"):

            response = run_reply_engine(
                db=eval_db,
                cfg=mock_config,
                tts=None,
                text=query,
                dialogue_memory=eval_dialogue_memory
            )

        # Validate
        assert len(captured_search_queries) > 0, "Should have made a search"

        search_query = captured_search_queries[0]
        is_passed, issues, scores = evaluate_search_query_heuristically(
            user_query=query,
            search_query=search_query,
            location=user_location
        )

        print(f"\n📊 Location Awareness Eval:")
        print(f"   User location: {user_location}")
        print(f"   Search query: '{search_query}'")
        print(f"   Scores: {scores}")
        print(f"   Issues: {issues}")

        # This assertion documents expected behavior
        assert scores.get("location_awareness", 0) >= 0.7, (
            f"Search should include location. Query: '{search_query}', Issues: {issues}"
        )


class TestToolResultGuidance:
    """
    Tests for the tool result guidance fix.

    After each tool result, the engine adds a reminder that data is available.
    The LLM can then decide to: answer, chain another tool, or clarify.
    """

    @pytest.mark.eval
    def test_tool_result_guidance_prompts_proper_response(self, mock_config, eval_db, eval_dialogue_memory):
        """
        Eval: Verifies that tool result guidance helps the LLM decide next steps.

        Scenario:
        1. User asks: "how's the weather this week?"
        2. webSearch called → returns weather results ✓
        3. Engine adds: "Tool result received. Decide: answer, call another tool, or clarify."
        4. LLM decides to fetch more details → fetchWebPage called ✓
        5. Engine adds guidance again after fetchWebPage result
        6. LLM now has enough data → responds with weather info ✓

        The guidance is non-prescriptive - it reminds the LLM it has data
        but lets it intelligently decide the next step.
        """
        from jarvis.reply.engine import run_reply_engine

        query = "how's the weather this week?"

        def mock_tool_run(db, cfg, tool_name, tool_args, **kwargs):
            from jarvis.tools.types import ToolExecutionResult
            if tool_name == "webSearch":
                return ToolExecutionResult(success=True, reply_text=MOCK_WEATHER_SEARCH_RESULTS)
            elif tool_name == "fetchWebPage":
                return ToolExecutionResult(success=True, reply_text=MOCK_WEATHER_PAGE_CONTENT)
            return ToolExecutionResult(success=False, reply_text="Unknown")

        call_count = 0
        def mock_chat_with_fix(base_url, chat_model, messages, timeout_sec, extra_options=None):
            """
            Simulates LLM behavior with the prompting fix applied.

            The fix adds "Use the tool result above to answer" after each tool result.
            When the LLM sees this guidance, it properly uses the tool results.
            """
            nonlocal call_count
            call_count += 1

            # Check if we have the tool guidance prompt (the fix)
            has_tool_guidance = any(
                "tool result received" in msg.get("content", "").lower()
                for msg in messages if msg.get("role") == "system"
            )

            # Check if we have actual tool results
            has_tool_results = any(msg.get("role") == "tool" for msg in messages)

            if call_count == 1:
                return create_mock_llm_response("", [create_tool_call("webSearch", {"search_query": "weather"})])
            elif call_count == 2:
                return create_mock_llm_response("", [create_tool_call("fetchWebPage", {"url": "https://example.com"})])
            elif has_tool_results and has_tool_guidance:
                # THE FIX: With explicit guidance, LLM uses tool results
                return create_mock_llm_response(
                    "This week's weather: Wednesday 12°C partly cloudy, Thursday 14°C sunny, "
                    "Friday 11°C with 60% rain, Saturday heavy rain at 10°C, improving Sunday-Monday."
                )
            else:
                # WITHOUT the fix: LLM might return generic greeting
                return create_mock_llm_response(
                    "Hey there! How can I help you today? Whether it's a quick question, "
                    "a reminder, or some guidance, just let me know."
                )

        with patch('jarvis.reply.engine.run_tool_with_retries', side_effect=mock_tool_run), \
             patch('jarvis.reply.engine.chat_with_messages', side_effect=mock_chat_with_fix), \
             patch('jarvis.reply.engine.extract_search_params_for_memory', return_value={"keywords": []}), \
             patch('jarvis.reply.engine.select_profile_llm', return_value="life"):

            response = run_reply_engine(
                db=eval_db,
                cfg=mock_config,
                tts=None,
                text=query,
                dialogue_memory=eval_dialogue_memory
            )

        # With the prompting fix, this should pass
        is_passed, issues = evaluate_response_heuristically(
            query=query,
            response=response,
            tool_results_available=True
        )

        assert is_passed, (
            f"❌ BUG CONFIRMED: LLM returned generic greeting despite weather data.\n"
            f"   Response: {response}\n"
            f"   Issues: {issues}"
        )


class TestLLMAsJudge:
    """
    Eval tests using LLM-as-judge for deeper semantic evaluation.

    Requires Ollama running with the judge model.
    """

    @pytest.mark.eval
    @requires_judge_llm
    @pytest.mark.parametrize("response,expected_pass", [
        # Good weather responses
        ("This week: 12°C Wednesday, 14°C Thursday sunny, rain Saturday 10°C.", True),
        ("Expect partly cloudy skies with temperatures 10-14°C and 60% rain chance Friday.", True),
        # Bad responses
        ("Hey there! How can I help you today?", False),
        ("I'm not sure, can you clarify?", False),
        ("Sure thing!", False),
    ])
    def test_judge_evaluates_weather_responses(self, response, expected_pass):
        """
        Eval: LLM judge should correctly identify good vs bad weather responses.
        """
        query = "how's the weather this week?"

        verdict = judge_response_answers_query(
            query=query,
            response=response,
            context=MOCK_WEATHER_PAGE_CONTENT
        )

        print(f"\n🧑‍⚖️ LLM Judge Evaluation:")
        print(f"   Query: {query}")
        print(f"   Response: {response[:60]}...")
        print(f"   Expected: {'PASS' if expected_pass else 'FAIL'}")
        print(f"   Verdict: {'PASS' if verdict.is_passed else 'FAIL'}")
        print(f"   Score: {verdict.score:.2f}")
        print(f"   Reasoning: {verdict.reasoning[:100]}...")

        if expected_pass:
            assert verdict.is_passed or verdict.score >= 0.5, (
                f"Expected PASS. Reasoning: {verdict.reasoning}"
            )
        else:
            assert not verdict.is_passed or verdict.score < 0.6, (
                f"Expected FAIL. Reasoning: {verdict.reasoning}"
            )

    @pytest.mark.eval
    @requires_judge_llm
    def test_judge_evaluates_search_query_with_location(self):
        """
        Eval: LLM judge should verify search queries include location context.
        """
        user_query = "how's the weather?"
        location = "Kensington, London, UK"

        # Test good search query
        good_search = "weather Kensington London UK today"
        good_verdict = judge_search_query_quality(
            user_query=user_query,
            search_query=good_search,
            location=location
        )

        # Test bad search query
        bad_search = "weather today"
        bad_verdict = judge_search_query_quality(
            user_query=user_query,
            search_query=bad_search,
            location=location
        )

        print(f"\n🧑‍⚖️ Search Query Evaluation:")
        print(f"   Good query: '{good_search}' → Score: {good_verdict.score:.2f}")
        print(f"   Bad query:  '{bad_search}' → Score: {bad_verdict.score:.2f}")

        assert good_verdict.score > bad_verdict.score, (
            "Search with location should score higher than without"
        )


class TestLiveEval:
    """
    Live evaluation tests using real LLM.

    These require Ollama running and are skipped by default.
    Enable with: pytest evals/ -k "Live" --live
    """

    @pytest.mark.eval
    @pytest.mark.skipif(True, reason="Enable with --live flag for real LLM testing")
    def test_weather_query_live_evaluation(self, mock_config, eval_db, eval_dialogue_memory):
        """
        Live eval: Test actual LLM behavior on weather query.
        """
        from jarvis.reply.engine import run_reply_engine

        query = "how's the weather this week?"

        mock_config.ollama_base_url = "http://localhost:11434"
        mock_config.ollama_chat_model = "qwen3"

        response = run_reply_engine(
            db=eval_db,
            cfg=mock_config,
            tts=None,
            text=query,
            dialogue_memory=eval_dialogue_memory
        )

        print(f"\n📝 Live Eval:")
        print(f"   Query: {query}")
        print(f"   Response: {response}")

        # Heuristic evaluation
        is_passed, issues = evaluate_response_heuristically(
            query=query,
            response=response,
            tool_results_available=True
        )

        print(f"   Heuristic: {'PASS' if is_passed else 'FAIL'}")
        print(f"   Issues: {issues}")

        assert is_passed, f"Live eval failed: {issues}"

        # LLM judge evaluation if available
        if _JUDGE_LLM_AVAILABLE:
            verdict = judge_response_answers_query(query, response or "")
            print(f"   Judge: {'PASS' if verdict.is_passed else 'FAIL'} (score: {verdict.score:.2f})")
            assert verdict.is_passed, f"Judge failed: {verdict.reasoning}"
