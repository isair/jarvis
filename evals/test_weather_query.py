"""
Eval: Weather query response quality.

Tests that when a user asks about weather, the assistant:
1. Uses appropriate tools (webSearch, fetchWebPage)
2. Returns actual weather information instead of generic greetings
3. Addresses the specific query (e.g., "this week" means multiple days)

Run with: pytest evals/test_weather_query.py -v
"""

import sys
from pathlib import Path

# Ensure evals directory is in path for imports
_this_file = Path(__file__).resolve()
EVALS_DIR = _this_file.parent
if str(EVALS_DIR) not in sys.path:
    sys.path.insert(0, str(EVALS_DIR))

import pytest
from unittest.mock import patch, MagicMock
from typing import Dict, Any, List, Optional

from helpers import (
    MockConfig, EvalCase, EvalResult,
    assert_response_quality, is_generic_greeting, response_addresses_topic,
    create_mock_llm_response, create_tool_call,
    JudgeVerdict, judge_response_answers_query, judge_search_query_quality,
    judge_tool_usage_appropriateness, is_judge_llm_available
)

# Check LLM availability once at module load
_JUDGE_LLM_AVAILABLE = is_judge_llm_available()
requires_judge_llm = pytest.mark.skipif(
    not _JUDGE_LLM_AVAILABLE,
    reason="Judge LLM not available (Ollama not running)"
)


# Sample tool results that simulate what the real tools would return
MOCK_WEATHER_SEARCH_RESULTS = """Web search results for 'weather London UK this week':

1. **London Weather Forecast - Met Office**
   Link: https://www.metoffice.gov.uk/weather/forecast/gcpvj0v07

2. **London 7-Day Weather Forecast - BBC Weather**
   Link: https://www.bbc.co.uk/weather/2643743

3. **London Weather - AccuWeather**
   Link: https://www.accuweather.com/en/gb/london/ec4a-2/weather-forecast/328328

4. **Weather Underground London Forecast**
   Link: https://www.wunderground.com/weather/gb/london

5. **Time and Date London Weather**
   Link: https://www.timeanddate.com/weather/uk/london
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
UV Index: Low
"""


class TestWeatherQueryEval:
    """Eval tests for weather query handling."""

    @pytest.mark.eval
    def test_weather_query_uses_tools_and_returns_weather_info(self, mock_config, eval_db, eval_dialogue_memory):
        """
        Eval: Weather query should use web search and return actual weather.

        This test simulates the failing scenario where the LLM:
        1. Successfully calls webSearch (returns results)
        2. Successfully calls fetchWebPage (returns weather data)
        3. But then returns a generic greeting instead of weather info

        The eval checks that the final response contains weather information.
        """
        from jarvis.reply.engine import run_reply_engine

        query = "how's the weather this week, all the rest of this week?"

        # Track what tools were called
        tools_called = []

        def mock_tool_run(db, cfg, tool_name, tool_args, **kwargs):
            """Mock tool execution that returns realistic results."""
            tools_called.append(tool_name)

            from jarvis.tools.types import ToolExecutionResult

            if tool_name == "webSearch":
                return ToolExecutionResult(
                    success=True,
                    reply_text=MOCK_WEATHER_SEARCH_RESULTS
                )
            elif tool_name == "fetchWebPage":
                return ToolExecutionResult(
                    success=True,
                    reply_text=MOCK_WEATHER_PAGE_CONTENT
                )
            else:
                return ToolExecutionResult(
                    success=False,
                    reply_text=f"Unknown tool: {tool_name}"
                )

        # Track LLM calls to understand the conversation flow
        llm_call_count = 0

        def mock_chat_with_messages(base_url, chat_model, messages, timeout_sec, extra_options=None):
            """
            Mock LLM that simulates a GOOD response pattern:
            Turn 1: Call webSearch
            Turn 2: Call fetchWebPage
            Turn 3: Provide weather summary
            """
            nonlocal llm_call_count
            llm_call_count += 1

            # Check if we have tool results in messages
            has_web_search_result = any(
                msg.get("role") == "tool" and "weather" in msg.get("content", "").lower()
                for msg in messages
            )
            has_page_content = any(
                msg.get("role") == "tool" and "Wednesday" in msg.get("content", "")
                for msg in messages
            )

            if llm_call_count == 1:
                # First call: search for weather
                return create_mock_llm_response(
                    "",
                    [create_tool_call("webSearch", {"search_query": "weather London UK this week"})]
                )
            elif llm_call_count == 2 and has_web_search_result:
                # Second call: fetch a weather page
                return create_mock_llm_response(
                    "",
                    [create_tool_call("fetchWebPage", {"url": "https://www.bbc.co.uk/weather/2643743"})]
                )
            elif has_page_content:
                # Final: provide weather summary based on tool results
                return create_mock_llm_response(
                    "Here's the weather forecast for London this week:\n\n"
                    "• **Wednesday**: Partly cloudy, 12°C high, 30% rain chance\n"
                    "• **Thursday**: Sunny intervals, 14°C high - great day!\n"
                    "• **Friday**: Getting cloudier, 11°C high, 60% rain\n"
                    "• **Saturday**: Heavy rain expected, 10°C high\n"
                    "• **Sunday-Monday**: Improving, 11-13°C with occasional showers\n\n"
                    "Looks like Thursday is your best bet for outdoor activities!"
                )
            else:
                # Fallback
                return create_mock_llm_response(
                    "I'd be happy to help with the weather, but I need to search for current conditions first."
                )

        with patch('jarvis.reply.engine.run_tool_with_retries', side_effect=mock_tool_run), \
             patch('jarvis.reply.engine.chat_with_messages', side_effect=mock_chat_with_messages), \
             patch('jarvis.reply.engine.extract_search_params_for_memory', return_value={"keywords": []}), \
             patch('jarvis.reply.engine.select_profile_llm', return_value="life"):

            response = run_reply_engine(
                db=eval_db,
                cfg=mock_config,
                tts=None,
                text=query,
                dialogue_memory=eval_dialogue_memory
            )

        # Build eval result
        result = EvalResult(
            query=query,
            response=response,
            is_passed=False,
            tool_calls_made=tools_called,
            turn_count=llm_call_count
        )

        # Assertions
        assert response is not None, "Response should not be None"
        assert not is_generic_greeting(response), (
            f"Response should NOT be a generic greeting. Got: {response}"
        )
        assert response_addresses_topic(response, ["weather", "temperature", "rain", "sunny", "cloudy", "°C", "°F"]), (
            f"Response should address weather topic. Got: {response}"
        )
        assert "webSearch" in tools_called, "Should have called webSearch tool"

        result.is_passed = True
        print(f"\n{result}")

    @pytest.mark.eval
    @pytest.mark.xfail(reason="Documents existing bug: LLM returns generic greeting despite having tool results")
    def test_weather_query_fails_with_generic_response(self, mock_config, eval_db, eval_dialogue_memory):
        """
        Eval: Detect when LLM returns generic response despite having tool results.

        This reproduces the exact failure pattern from the user's logs:
        - Tools are called successfully
        - Weather data is retrieved
        - But LLM returns "Hey there! How can I help you today?"

        This test is marked xfail - when the bug is fixed, this test will start passing
        and pytest will report it as XPASS, indicating the fix worked.
        """
        from jarvis.reply.engine import run_reply_engine

        query = "how's the weather this week, all the rest of this week?"
        tools_called = []
        llm_call_count = 0

        def mock_tool_run(db, cfg, tool_name, tool_args, **kwargs):
            tools_called.append(tool_name)
            from jarvis.tools.types import ToolExecutionResult

            if tool_name == "webSearch":
                return ToolExecutionResult(success=True, reply_text=MOCK_WEATHER_SEARCH_RESULTS)
            elif tool_name == "fetchWebPage":
                return ToolExecutionResult(success=True, reply_text=MOCK_WEATHER_PAGE_CONTENT)
            return ToolExecutionResult(success=False, reply_text="Unknown tool")

        def mock_chat_buggy(base_url, chat_model, messages, timeout_sec, extra_options=None):
            """
            Mock LLM that reproduces the bug:
            - Calls tools correctly
            - But returns generic greeting at the end
            """
            nonlocal llm_call_count
            llm_call_count += 1

            has_tool_results = any(msg.get("role") == "tool" for msg in messages)

            if llm_call_count == 1:
                return create_mock_llm_response(
                    "",
                    [create_tool_call("webSearch", {"search_query": "weather this week"})]
                )
            elif llm_call_count == 2:
                return create_mock_llm_response(
                    "",
                    [create_tool_call("fetchWebPage", {"url": "https://example.com/weather"})]
                )
            else:
                # BUG: Returns generic greeting despite having weather data
                return create_mock_llm_response(
                    "Hey there! How can I help you today? Whether it's a quick question, "
                    "a reminder, or some guidance, just let me know."
                )

        with patch('jarvis.reply.engine.run_tool_with_retries', side_effect=mock_tool_run), \
             patch('jarvis.reply.engine.chat_with_messages', side_effect=mock_chat_buggy), \
             patch('jarvis.reply.engine.extract_search_params_for_memory', return_value={"keywords": []}), \
             patch('jarvis.reply.engine.select_profile_llm', return_value="life"):

            response = run_reply_engine(
                db=eval_db,
                cfg=mock_config,
                tts=None,
                text=query,
                dialogue_memory=eval_dialogue_memory
            )

        # This test documents the bug - it should fail
        assert response is not None

        # These assertions demonstrate what SHOULD happen but currently fails
        is_bad_response = is_generic_greeting(response)
        has_weather_content = response_addresses_topic(
            response,
            ["weather", "temperature", "rain", "sunny", "cloudy", "°C", "°F", "forecast"]
        )

        if is_bad_response or not has_weather_content:
            pytest.fail(
                f"❌ BUG DETECTED: LLM returned generic response despite having weather data.\n"
                f"   Query: {query}\n"
                f"   Response: {response}\n"
                f"   Tools called: {tools_called}\n"
                f"   This is the exact failure pattern from production."
            )

    @pytest.mark.eval
    @pytest.mark.parametrize("query,expected_keywords", [
        ("what's the weather like today", ["today", "weather"]),
        ("will it rain tomorrow", ["rain", "tomorrow"]),
        ("weather forecast for the weekend", ["weekend", "forecast"]),
        ("is it going to be sunny this week", ["sunny", "week"]),
    ])
    def test_various_weather_queries(self, mock_config, eval_db, eval_dialogue_memory, query, expected_keywords):
        """
        Eval: Various weather query patterns should return relevant responses.
        """
        from jarvis.reply.engine import run_reply_engine

        call_count = 0

        def mock_chat(base_url, chat_model, messages, timeout_sec, extra_options=None):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                return create_mock_llm_response(
                    "",
                    [create_tool_call("webSearch", {"search_query": f"weather {' '.join(expected_keywords)}"})]
                )
            else:
                # Good response that addresses the query
                return create_mock_llm_response(
                    f"Based on the forecast, here's what to expect: "
                    f"Temperatures around 12-15°C with a mix of sun and clouds. "
                    f"There's a 40% chance of rain later in the period."
                )

        def mock_tool(db, cfg, tool_name, tool_args, **kwargs):
            from jarvis.tools.types import ToolExecutionResult
            return ToolExecutionResult(success=True, reply_text=MOCK_WEATHER_SEARCH_RESULTS)

        with patch('jarvis.reply.engine.run_tool_with_retries', side_effect=mock_tool), \
             patch('jarvis.reply.engine.chat_with_messages', side_effect=mock_chat), \
             patch('jarvis.reply.engine.extract_search_params_for_memory', return_value={"keywords": []}), \
             patch('jarvis.reply.engine.select_profile_llm', return_value="life"):

            response = run_reply_engine(
                db=eval_db,
                cfg=mock_config,
                tts=None,
                text=query,
                dialogue_memory=eval_dialogue_memory
            )

        assert response is not None
        assert not is_generic_greeting(response), f"Generic greeting for: {query}"
        assert response_addresses_topic(response, ["weather", "temperature", "rain", "sun", "°C", "forecast"]), (
            f"Response should be about weather for query: {query}"
        )


class TestSearchQueryLocationAwareness:
    """
    Eval tests for verifying search queries include location when relevant.

    When the assistant knows the user's location, weather/local queries should
    include that location in the search to get relevant results.
    """

    @pytest.mark.eval
    def test_weather_search_includes_location(self, mock_config, eval_db, eval_dialogue_memory):
        """
        Eval: Weather search should include user's known location.

        Given: User is in "Kensington, London, UK"
        When: User asks "how's the weather this week?"
        Then: Search query should include location (e.g., "weather Kensington London this week")
        """
        from jarvis.reply.engine import run_reply_engine

        query = "how's the weather this week?"
        user_location = "Kensington, Royal Kensington and Chelsea, United Kingdom"

        # Track search queries made
        captured_search_queries = []

        def mock_tool_run(db, cfg, tool_name, tool_args, **kwargs):
            from jarvis.tools.types import ToolExecutionResult

            if tool_name == "webSearch":
                search_query = tool_args.get("search_query", "")
                captured_search_queries.append(search_query)
                return ToolExecutionResult(success=True, reply_text=MOCK_WEATHER_SEARCH_RESULTS)
            elif tool_name == "fetchWebPage":
                return ToolExecutionResult(success=True, reply_text=MOCK_WEATHER_PAGE_CONTENT)
            return ToolExecutionResult(success=False, reply_text="Unknown tool")

        call_count = 0
        def mock_chat(base_url, chat_model, messages, timeout_sec, extra_options=None):
            nonlocal call_count
            call_count += 1

            # Check for location in system context
            has_location_context = any(
                "Kensington" in msg.get("content", "") or "London" in msg.get("content", "")
                for msg in messages if msg.get("role") == "system"
            )

            if call_count == 1:
                # LLM should include location in search query
                search_query = "weather Kensington London UK this week" if has_location_context else "weather this week"
                return create_mock_llm_response(
                    "",
                    [create_tool_call("webSearch", {"search_query": search_query})]
                )
            else:
                return create_mock_llm_response(
                    "This week in Kensington, expect temperatures around 10-14°C with rain on Saturday."
                )

        # Mock location context to be injected
        def mock_get_location_context(**kwargs):
            return f"Location: {user_location}"

        with patch('jarvis.reply.engine.run_tool_with_retries', side_effect=mock_tool_run), \
             patch('jarvis.reply.engine.chat_with_messages', side_effect=mock_chat), \
             patch('jarvis.reply.engine.get_location_context', side_effect=mock_get_location_context), \
             patch('jarvis.reply.engine.extract_search_params_for_memory', return_value={"keywords": []}), \
             patch('jarvis.reply.engine.select_profile_llm', return_value="life"):

            response = run_reply_engine(
                db=eval_db,
                cfg=mock_config,
                tts=None,
                text=query,
                dialogue_memory=eval_dialogue_memory
            )

        # Verify search query included location
        assert len(captured_search_queries) > 0, "Should have made at least one search"
        search_query = captured_search_queries[0].lower()

        # Check if location or nearby area is mentioned
        location_terms = ["kensington", "london", "uk", "united kingdom"]
        has_location = any(term in search_query for term in location_terms)

        assert has_location, (
            f"Search query should include location context.\n"
            f"   User location: {user_location}\n"
            f"   Search query: {captured_search_queries[0]}\n"
            f"   Expected one of: {location_terms}"
        )

    @pytest.mark.eval
    @pytest.mark.xfail(reason="Tests if search queries properly use location - may fail if LLM doesn't include location")
    def test_weather_search_location_awareness_with_judge(self, mock_config, eval_db, eval_dialogue_memory):
        """
        Eval with LLM-as-judge: Verify search query quality includes location.

        Uses LLM judge to evaluate if the search query is appropriate.
        """
        query = "how's the weather this week?"
        user_location = "Kensington, Royal Kensington and Chelsea, United Kingdom"

        # Simulate a search query that DOESN'T include location (the bug)
        bad_search_query = "weather this week"

        verdict = judge_search_query_quality(
            user_query=query,
            search_query=bad_search_query,
            location=user_location,
            time_context="this week"
        )

        print(f"\n🧑‍⚖️ Judge Verdict:")
        print(f"   Search query: '{bad_search_query}'")
        print(f"   User location: {user_location}")
        print(f"   Score: {verdict.score:.2f}")
        print(f"   Passed: {verdict.is_passed}")
        print(f"   Criteria: {verdict.criteria_scores}")
        print(f"   Reasoning: {verdict.reasoning}")

        # Location awareness should fail if location isn't in search
        assert verdict.criteria_scores.get("location_awareness", 0) >= 0.7, (
            f"Search query should include location. Got location_awareness score: "
            f"{verdict.criteria_scores.get('location_awareness', 0)}"
        )


class TestResponseQualityWithJudge:
    """
    Eval tests using LLM-as-judge for semantic evaluation.

    These tests use a separate LLM call to judge response quality,
    providing more nuanced evaluation than keyword matching.

    Requires Ollama running locally. Skip with: pytest -k "not Judge"
    """

    @pytest.mark.eval
    @requires_judge_llm
    def test_judge_good_weather_response(self):
        """
        Eval: LLM judge should pass a good weather response.
        """
        query = "how's the weather this week?"
        good_response = (
            "This week in London, expect partly cloudy skies with temperatures "
            "ranging from 10-14°C. Wednesday and Thursday look best with sunshine. "
            "Rain is likely on Saturday, so plan indoor activities. "
            "Bring a light jacket for the cooler evenings."
        )
        context = MOCK_WEATHER_PAGE_CONTENT

        verdict = judge_response_answers_query(query, good_response, context)

        print(f"\n🧑‍⚖️ Judge Verdict for GOOD response:")
        print(f"   Score: {verdict.score:.2f}")
        print(f"   Passed: {verdict.is_passed}")
        print(f"   Criteria: {verdict.criteria_scores}")
        print(f"   Reasoning: {verdict.reasoning}")

        assert verdict.is_passed, f"Good response should pass. Reasoning: {verdict.reasoning}"
        assert verdict.score >= 0.6, f"Good response should score well. Got: {verdict.score}"

    @pytest.mark.eval
    @requires_judge_llm
    def test_judge_generic_greeting_fails(self):
        """
        Eval: LLM judge should fail a generic greeting response.
        """
        query = "how's the weather this week?"
        bad_response = (
            "Hey there! How can I help you today? Whether it's a quick question, "
            "a reminder, or some guidance, just let me know."
        )
        context = MOCK_WEATHER_PAGE_CONTENT  # Weather data WAS available

        verdict = judge_response_answers_query(query, bad_response, context)

        print(f"\n🧑‍⚖️ Judge Verdict for BAD response:")
        print(f"   Score: {verdict.score:.2f}")
        print(f"   Passed: {verdict.is_passed}")
        print(f"   Criteria: {verdict.criteria_scores}")
        print(f"   Reasoning: {verdict.reasoning}")

        assert not verdict.is_passed, f"Generic greeting should fail. Reasoning: {verdict.reasoning}"
        assert verdict.criteria_scores.get("no_deflection", 1) < 0.5, (
            "Should detect deflection in generic greeting"
        )

    @pytest.mark.eval
    @requires_judge_llm
    def test_judge_tool_usage_for_weather(self):
        """
        Eval: LLM judge should verify appropriate tool usage for weather queries.
        """
        query = "how's the weather this week?"

        # Good tool usage
        good_tools = ["webSearch", "fetchWebPage"]
        good_args = [
            {"search_query": "weather London this week"},
            {"url": "https://www.bbc.co.uk/weather/2643743"}
        ]

        verdict = judge_tool_usage_appropriateness(
            query=query,
            tools_called=good_tools,
            tool_args=good_args,
            expected_tools=["webSearch"]
        )

        print(f"\n🧑‍⚖️ Judge Verdict for tool usage:")
        print(f"   Tools: {good_tools}")
        print(f"   Score: {verdict.score:.2f}")
        print(f"   Passed: {verdict.is_passed}")
        print(f"   Reasoning: {verdict.reasoning}")

        assert verdict.is_passed, f"Good tool usage should pass. Reasoning: {verdict.reasoning}"

    @pytest.mark.eval
    @requires_judge_llm
    @pytest.mark.parametrize("query,response,should_pass", [
        (
            "what's the weather like?",
            "Currently it's 12°C and cloudy with a light breeze. Expect similar conditions through the afternoon.",
            True
        ),
        (
            "will it rain tomorrow?",
            "Yes, there's a 70% chance of rain tomorrow. I'd recommend bringing an umbrella.",
            True
        ),
        (
            "how's the weather?",
            "I'm not sure what you're asking about. Can you clarify?",
            False
        ),
        (
            "weather forecast for the week",
            "Sure thing! What else would you like to know?",
            False
        ),
    ])
    def test_judge_various_responses(self, query, response, should_pass):
        """
        Eval: Test LLM judge on various response patterns.
        """
        verdict = judge_response_answers_query(query, response)

        print(f"\n🧑‍⚖️ Query: '{query}'")
        print(f"   Response: '{response[:80]}...'")
        print(f"   Expected: {'PASS' if should_pass else 'FAIL'}")
        print(f"   Got: {'PASS' if verdict.is_passed else 'FAIL'}")
        print(f"   Score: {verdict.score:.2f}")

        if should_pass:
            assert verdict.is_passed or verdict.score >= 0.5, (
                f"Expected PASS for: {response[:50]}... Got: {verdict.reasoning}"
            )
        else:
            assert not verdict.is_passed or verdict.score < 0.7, (
                f"Expected FAIL for: {response[:50]}... Got: {verdict.reasoning}"
            )


class TestWeatherQueryLive:
    """
    Live eval tests that use actual LLM.

    These tests require:
    - Ollama running locally
    - A capable model (qwen3 or similar)

    Run with: pytest evals/test_weather_query.py::TestWeatherQueryLive -v
    Skip with: pytest evals/test_weather_query.py -v -k "not Live"
    """

    @pytest.mark.eval
    @pytest.mark.skipif(True, reason="Enable manually when testing with live LLM")
    def test_weather_query_live(self, mock_config, eval_db, eval_dialogue_memory):
        """
        Live eval: Test weather query against actual LLM.

        Enable by removing the skipif decorator and ensuring Ollama is running.
        """
        from jarvis.reply.engine import run_reply_engine

        query = "how's the weather this week?"

        # Use real config with actual LLM
        mock_config.ollama_base_url = "http://localhost:11434"
        mock_config.ollama_chat_model = "qwen3"

        response = run_reply_engine(
            db=eval_db,
            cfg=mock_config,
            tts=None,
            text=query,
            dialogue_memory=eval_dialogue_memory
        )

        print(f"\n📝 Query: {query}")
        print(f"🤖 Response: {response}")

        assert response is not None
        assert not is_generic_greeting(response), f"Got generic greeting: {response}"

    @pytest.mark.eval
    @pytest.mark.skipif(True, reason="Enable manually when testing with live LLM")
    def test_weather_query_live_with_judge(self, mock_config, eval_db, eval_dialogue_memory):
        """
        Live eval with LLM-as-judge: Full end-to-end evaluation.

        This test:
        1. Runs the actual reply engine with real LLM
        2. Uses a separate LLM call to judge the response quality
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

        # Use LLM-as-judge to evaluate
        verdict = judge_response_answers_query(query, response or "")

        print(f"\n📝 Query: {query}")
        print(f"🤖 Response: {response}")
        print(f"\n🧑‍⚖️ Judge Verdict:")
        print(f"   Score: {verdict.score:.2f}")
        print(f"   Passed: {verdict.is_passed}")
        print(f"   Criteria: {verdict.criteria_scores}")
        print(f"   Reasoning: {verdict.reasoning}")

        assert verdict.is_passed, f"Response should pass judge evaluation. Reasoning: {verdict.reasoning}"

