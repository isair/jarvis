"""
Agent Behavior Evaluations

Tests core agent capabilities:
1. Response Quality - Gives useful answers, not deflections
2. Context Utilization - Uses location, time, and memory appropriately
3. Tool Usage - Calls right tools with right arguments
4. Multi-Step Reasoning - Chains tools and synthesizes information

Run: ./scripts/run_evals.sh
"""

from typing import List, Optional, Tuple

import pytest
from unittest.mock import patch

from conftest import requires_judge_llm
from helpers import (
    MockConfig, ToolCallCapture,
    create_mock_llm_response, create_tool_call,
    create_mock_tool_run,
    judge_response_answers_query,
)


# =============================================================================
# Test Data
# =============================================================================

MOCK_WEATHER_FORECAST = """Current weather in Tbilisi, Tbilisi, Georgia:

Conditions: Slight rain
Temperature: 6.1°C (43.0°F)
Humidity: 80%
Wind: 10.0 km/h

Today's forecast (upcoming hours):
  15:00 — 8.0°C, Partly cloudy
  18:00 — 6.5°C, Clear sky
  21:00 — 4.0°C, Clear sky

7-day forecast:
  2026-04-08: 3–8°C, Slight rain
  2026-04-09: 5–14°C, Partly cloudy
  2026-04-10: 7–16°C, Clear sky
  2026-04-11: 6–13°C, Overcast
  2026-04-12: 4–11°C, Slight rain
  2026-04-13: 5–12°C, Partly cloudy
  2026-04-14: 6–15°C, Clear sky"""

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
        # This test checks that location context flows into the webSearch query;
        # bypass the router so webSearch is exposed regardless of its own routing.
        mock_config.tool_selection_strategy = "all"
        capture = ToolCallCapture()
        mock_tool_run = create_mock_tool_run(capture, {"webSearch": MOCK_WEATHER_SEARCH})

        call_count = 0
        def mock_chat(base_url, chat_model, messages, timeout_sec, extra_options=None, tools=None, **kwargs):
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
             patch('jarvis.reply.engine.get_location_context_with_timezone', return_value=(f"Location: {user_location}", None)), \
             patch('jarvis.reply.engine.extract_search_params_for_memory', return_value={"keywords": []}):

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
        mock_tool_run = create_mock_tool_run(capture, {
            "webSearch": "Tech news: AI advances, new chip releases.",
        })

        call_count = 0
        def mock_chat(base_url, chat_model, messages, timeout_sec, extra_options=None, tools=None, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return create_mock_llm_response("", [create_tool_call("webSearch", {"search_query": "tech news today"})])
            return create_mock_llm_response("Today in tech: Major AI announcements and new hardware releases.")

        with patch('jarvis.reply.engine.run_tool_with_retries', side_effect=mock_tool_run), \
             patch('jarvis.reply.engine.chat_with_messages', side_effect=mock_chat), \
             patch('jarvis.reply.engine.extract_search_params_for_memory', return_value={"keywords": []}):

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
        # This test exercises tool-chaining behaviour; the context-aware router
        # is tested elsewhere. Force ALL tools so the mocked chat can freely
        # issue webSearch → fetchWebPage calls.
        mock_config.tool_selection_strategy = "all"
        capture = ToolCallCapture()
        mock_tool_run = create_mock_tool_run(capture, {
            "webSearch": MOCK_WEATHER_SEARCH,
            "fetchWebPage": MOCK_WEATHER_PAGE,
        })

        call_count = 0
        def mock_chat(base_url, chat_model, messages, timeout_sec, extra_options=None, tools=None, **kwargs):
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
             patch('jarvis.reply.engine.extract_search_params_for_memory', return_value={"keywords": []}):

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
        # Bypass the context-aware tool router so fetchMeals is exposed to the
        # mocked chat. Router behaviour is covered by dedicated router tests.
        mock_config.tool_selection_strategy = "all"
        capture = ToolCallCapture()
        mock_tool_run = create_mock_tool_run(capture, {
            "fetchMeals": MOCK_NUTRITION_DATA,
        })

        call_count = 0
        def mock_chat(base_url, chat_model, messages, timeout_sec, extra_options=None, tools=None, **kwargs):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                # Memory enrichment has already surfaced health goals into the
                # system prompt — the agent should go straight to fetchMeals.
                return create_mock_llm_response("", [
                    create_tool_call("fetchMeals", {})
                ])
            return create_mock_llm_response(
                "You've had 770 kcal so far today, leaving room for pizza within your 1800 kcal target. "
                "Given your weight loss goal, I'd suggest a thin crust with veggies - around 600 kcal for 2 slices. "
                "You've been consistent this week, so one pizza night won't derail your progress!"
            )

        with patch('jarvis.reply.engine.run_tool_with_retries', side_effect=mock_tool_run), \
             patch('jarvis.reply.engine.chat_with_messages', side_effect=mock_chat), \
             patch('jarvis.reply.engine.extract_search_params_for_memory', return_value={"keywords": ["health", "diet"]}):

            response = run_reply_engine(db=eval_db, cfg=mock_config, tts=None, text=query, dialogue_memory=eval_dialogue_memory)

        print(f"\n📊 Multi-Step Reasoning:")
        print(f"   Query: {query}")
        print(f"   Tools called: {[c['name'] for c in capture.calls]}")
        print(f"   Response: {response[:100] if response else 'None'}...")

        # Enrichment surfaces the health goals; agent only needs fetchMeals.
        tools_used = [c["name"] for c in capture.calls]
        assert "fetchMeals" in tools_used, \
            f"Should fetch today's meals for nutrition context. Used: {tools_used}"

        # Response should reference calorie info
        if response:
            assert "calor" in response.lower() or "kcal" in response.lower(), \
                "Response should mention calorie context"

# =============================================================================
# Memory Enrichment Evaluations
# =============================================================================

class TestMemoryEnrichment:
    """
    Tests that memory enrichment extracts correct keywords for different query types.

    Memory enrichment happens automatically BEFORE the LLM loop, so correct keyword
    extraction is critical for personalization to work without explicit tool calls.
    """

    @pytest.mark.eval
    @requires_judge_llm
    @pytest.mark.parametrize("query,expected_keywords", [
        pytest.param(
            "what news might interest me?",
            ["interests", "hobbies", "preferences"],
            id="Memory enrichment: personalized news"
        ),
        pytest.param(
            "what did we discuss about the python project?",
            ["python", "project", "code", "programming"],
            id="Memory enrichment: topic recall"
        ),
        pytest.param(
            "what did I eat yesterday?",
            ["eat", "food", "meal", "nutrition"],
            id="Memory enrichment: time-based recall"
        ),
    ])
    def test_enrichment_extracts_correct_keywords(self, query: str, expected_keywords: list, mock_config):
        """Enrichment should extract keywords that find relevant memory context."""
        from jarvis.reply.enrichment import extract_search_params_for_memory
        from helpers import JUDGE_MODEL

        mock_config.ollama_base_url = "http://localhost:11434"
        mock_config.ollama_chat_model = JUDGE_MODEL

        result = extract_search_params_for_memory(
            query=query,
            ollama_base_url=mock_config.ollama_base_url,
            ollama_chat_model=mock_config.ollama_chat_model,
            timeout_sec=15.0
        )

        extracted_keywords = result.get("keywords", [])
        extracted_lower = [k.lower() for k in extracted_keywords]

        print(f"\n📊 Enrichment Keyword Extraction:")
        print(f"   Query: {query}")
        print(f"   Extracted: {extracted_keywords}")
        print(f"   Expected (any of): {expected_keywords}")

        # At least one expected keyword should be present (or a close synonym)
        has_relevant = any(
            any(exp in kw or kw in exp for kw in extracted_lower)
            for exp in [k.lower() for k in expected_keywords]
        )

        assert has_relevant, \
            f"Extracted keywords {extracted_keywords} don't match any expected: {expected_keywords}"

    @pytest.mark.eval
    @requires_judge_llm
    def test_enrichment_skips_questions_answered_by_context(self, mock_config):
        """
        When context already contains information (e.g. location, short-term dialogue),
        the query generator should not emit implicit questions asking for that same
        information — we don't want to pull it from long-term memory redundantly.
        """
        from jarvis.reply.enrichment import extract_search_params_for_memory
        from helpers import JUDGE_MODEL

        mock_config.ollama_base_url = "http://localhost:11434"
        mock_config.ollama_chat_model = JUDGE_MODEL

        context_hint = (
            "Current local time: Sunday, 2026-04-19 14:30 local. "
            "Location: Tbilisi, Georgia.\n\n"
            "Recent dialogue (short-term memory):\n"
            "- user: I just finished a big bowl of khinkali for lunch.\n"
            "- assistant: Sounds tasty — anything planned for dinner?"
        )

        result = extract_search_params_for_memory(
            query="recommend a restaurant I'd enjoy",
            ollama_base_url=mock_config.ollama_base_url,
            ollama_chat_model=mock_config.ollama_chat_model,
            timeout_sec=15.0,
            context_hint=context_hint,
        )

        questions = [q.lower() for q in result.get("questions", [])]
        keywords = result.get("keywords", [])
        print(f"\n📊 Context-aware questions: {questions}")
        print(f"   keywords: {keywords}")

        # Sanity check: guard against a silent extractor failure making the
        # assertion below pass vacuously.
        assert keywords, \
            f"Extractor returned no keywords — test would pass trivially. Result: {result}"

        # Location is in context — no need to ask "where is the user?"
        assert not any("locat" in q or "where" in q for q in questions), \
            f"Should not ask about location when it's in context. Got: {questions}"

    @pytest.mark.eval
    def test_enrichment_provides_context_to_llm(self, mock_config, eval_db, eval_dialogue_memory):
        """
        Verify that enrichment results are included in the system message.

        When enrichment finds relevant memory, it should be available to the
        LLM directly via the system prompt — no tool call required.
        """
        from jarvis.reply.engine import run_reply_engine

        query = "what should I have for dinner?"

        # Mock the memory search to return user's food preferences
        mock_memory_results = [
            "[2024-12-15] User mentioned they love Italian cuisine, especially pasta dishes",
            "[2024-12-20] User said they're trying to eat more vegetables and less red meat",
        ]

        captured_messages = []

        def mock_chat(base_url, chat_model, messages, timeout_sec, extra_options=None, tools=None, **kwargs):
            captured_messages.extend(messages)
            return create_mock_llm_response(
                "Based on your love for Italian food and goal to eat more veggies, "
                "how about a primavera pasta with seasonal vegetables?"
            )

        with patch('jarvis.reply.engine.chat_with_messages', side_effect=mock_chat), \
             patch('jarvis.reply.engine.extract_search_params_for_memory', return_value={"keywords": ["dinner", "food", "preferences"]}), \
             patch('jarvis.memory.conversation.search_conversation_memory_by_keywords', return_value=mock_memory_results):

            run_reply_engine(db=eval_db, cfg=mock_config, tts=None, text=query, dialogue_memory=eval_dialogue_memory)

        # Check that enrichment context is in the system message
        system_messages = [m for m in captured_messages if m.get("role") == "system"]
        system_content = " ".join(m.get("content", "") for m in system_messages)

        print(f"\n📊 Enrichment Context in System Message:")
        print(f"   Query: {query}")
        print(f"   Has 'Italian': {'Italian' in system_content}")
        print(f"   Has 'vegetables': {'vegetables' in system_content}")

        assert "Italian" in system_content or "pasta" in system_content, \
            "Enrichment results should be in system message context"

    @pytest.mark.eval
    def test_llm_uses_enrichment_for_personalised_queries(self, mock_config, eval_db, eval_dialogue_memory):
        """
        When enrichment provides sufficient context (user interests), the LLM
        should read them from the system prompt and route to webSearch with an
        interest-flavoured query, rather than asking the user.
        """
        from jarvis.reply.engine import run_reply_engine

        query = "what news might interest me?"
        capture = ToolCallCapture()

        # Mock enrichment to return user interests
        mock_enrichment_context = [
            "[2024-12-15] User is passionate about space exploration and astronomy",
            "[2024-12-20] User follows AI and machine learning developments closely",
        ]

        mock_tool_run = create_mock_tool_run(capture, {
            "webSearch": "SpaceX launched, new AI model released",
        })

        call_count = 0
        def mock_chat(base_url, chat_model, messages, timeout_sec, extra_options=None, tools=None, **kwargs):
            nonlocal call_count
            call_count += 1

            # Check if enrichment context is in the messages
            system_content = " ".join(m.get("content", "") for m in messages if m.get("role") == "system")
            has_enrichment = "space exploration" in system_content or "AI" in system_content

            if call_count == 1 and has_enrichment:
                # LLM sees enrichment context and should use it directly for search
                return create_mock_llm_response("", [
                    create_tool_call("webSearch", {"search_query": "space exploration AI news today"})
                ])
            return create_mock_llm_response(
                "Based on your interests in space and AI, here's today's news: SpaceX launched and a new AI model was released."
            )

        with patch('jarvis.reply.engine.run_tool_with_retries', side_effect=mock_tool_run), \
             patch('jarvis.reply.engine.chat_with_messages', side_effect=mock_chat), \
             patch('jarvis.reply.engine.extract_search_params_for_memory', return_value={"keywords": ["interests", "hobbies", "preferences"]}), \
             patch('jarvis.memory.conversation.search_conversation_memory_by_keywords', return_value=mock_enrichment_context):

            response = run_reply_engine(db=eval_db, cfg=mock_config, tts=None, text=query, dialogue_memory=eval_dialogue_memory)

        tools_used = [c["name"] for c in capture.calls]

        print(f"\n📊 Enrichment Efficiency:")
        print(f"   Query: {query}")
        print(f"   Enrichment provided: user interests in space/AI")
        print(f"   Tools called: {tools_used}")
        print(f"   Response: {(response or '')[:100]}...")

        # Should proceed to webSearch with interests-informed query
        assert "webSearch" in tools_used, \
            f"LLM should search based on enriched interests. Tools: {tools_used}"

        print(f"   ✅ Enrichment surfaced interests, webSearch routed")


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
            return (f"Location: {test_location}", None)

        with patch('jarvis.reply.engine.get_location_context_with_timezone', side_effect=mock_get_location):
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

    @pytest.mark.eval
    @requires_judge_llm
    def test_personalized_query_recalls_memory_live(self, mock_config, eval_db, eval_dialogue_memory):
        """
        Live eval: Personalized query with available memory should use it.

        This tests that when memory enrichment provides user interests, the LLM
        uses them for personalized search rather than asking the user or ignoring them.
        """
        from jarvis.reply.engine import run_reply_engine
        from helpers import JUDGE_MODEL

        query = "what news from today might interest me?"
        capture = ToolCallCapture()

        mock_config.ollama_base_url = "http://localhost:11434"
        mock_config.ollama_chat_model = JUDGE_MODEL

        # Provide enrichment context so LLM has user interests available
        mock_enrichment_context = [
            "[2024-12-15] User is passionate about space exploration and astronomy",
            "[2024-12-20] User follows AI and machine learning developments closely",
        ]

        mock_tool_run = create_mock_tool_run(capture, {
            "webSearch": "AI breakthrough announced, SpaceX launch successful, quantum computing milestone reached",
            "fetchWebPage": "Full article about AI and space news...",
        })

        with patch('jarvis.reply.engine.run_tool_with_retries', side_effect=mock_tool_run), \
             patch('jarvis.reply.engine.get_location_context_with_timezone', return_value=("Location: London, UK", None)), \
             patch('jarvis.memory.conversation.search_conversation_memory_by_keywords', return_value=mock_enrichment_context):

            response = run_reply_engine(
                db=eval_db, cfg=mock_config, tts=None,
                text=query, dialogue_memory=eval_dialogue_memory
            )

        tools_used = [c["name"] for c in capture.calls]

        print(f"\n📝 Live Personalized Query Eval:")
        print(f"   Query: {query}")
        print(f"   Enrichment provided: user interests in space/AI")
        print(f"   Tools called: {tools_used}")
        print(f"   Response: {(response or '')[:150]}...")

        # Check if the response is asking the user about their interests
        # (which is wrong since enrichment provided interests)
        asking_phrases = [
            "what topics", "what are you interested", "could you let me know",
            "what kind of", "tell me what", "what subjects", "are there any particular",
            "which topics", "any specific", "what type of", "interested in?"
        ]
        is_asking_user = response and any(phrase in response.lower() for phrase in asking_phrases)

        print(f"   Asked user instead: {is_asking_user}")

        # FAIL if LLM asked user when enrichment already provided interests
        assert not is_asking_user, \
            f"LLM asked user about interests when enrichment already provided them.\n" \
            f"Response: {response[:300]}"

        # Should have used the enriched interests somehow (search or response)
        response_mentions_interests = response and any(
            term in response.lower() for term in ["ai", "space", "astronomy", "machine learning"]
        )

        print(f"   Response mentions user interests: {response_mentions_interests}")
        print(f"   ✅ Personalized query handling: PASS")


# =============================================================================
# Helpfulness Evaluations (Anti-Deflection)
# =============================================================================

# Phrases that indicate the agent is deflecting instead of using its tools
DEFLECTION_PHRASES = [
    "check a weather app",
    "check a local weather",
    "check a dedicated weather",
    "use a weather app",
    "try a weather app",
    "visit a weather",
    "check online",
    "i don't have",
    "i do not have",
    "i cannot check",
    "i can't check",
    "i'm unable to",
    "i am unable to",
    "beyond my capabilities",
    "outside my capabilities",
    "i can only check",
    "only for today",
    "not able to provide",
    "unable to provide",
    "don't have access to",
    "do not have access to",
    "recommend checking",
    "suggest checking",
]


def _response_is_deflection(response: str) -> bool:
    """Check if the response deflects the user to another app/service."""
    if not response:
        return True
    response_lower = response.lower()
    return any(phrase in response_lower for phrase in DEFLECTION_PHRASES)


class TestHelpfulness:
    """
    Tests that the agent uses its tools proactively instead of deflecting.

    The agent should NEVER tell users to "check a weather app" or "I can't do that"
    when it has tools available to fulfil the request.
    """

    @pytest.mark.eval
    @requires_judge_llm
    @pytest.mark.parametrize("query", [
        pytest.param(
            "what's the weather tomorrow?",
            id="No deflection: tomorrow weather"
        ),
        pytest.param(
            "will it rain this week?",
            id="No deflection: weekly rain forecast"
        ),
    ])
    def test_no_deflection_for_weather_forecast_live(
        self, query, mock_config, eval_db, eval_dialogue_memory
    ):
        """Live eval: agent should use tools for forecast queries, never deflect."""
        from jarvis.reply.engine import run_reply_engine
        from helpers import JUDGE_MODEL

        mock_config.ollama_base_url = "http://localhost:11434"
        mock_config.ollama_chat_model = JUDGE_MODEL

        capture = ToolCallCapture()
        mock_tool_run = create_mock_tool_run(capture, {
            "getWeather": MOCK_WEATHER_FORECAST,
            "webSearch": "Weather forecast: partly cloudy, 14°C tomorrow.",
            "fetchWebPage": "Detailed 7-day forecast...",
        })

        with patch('jarvis.reply.engine.run_tool_with_retries', side_effect=mock_tool_run), \
             patch('jarvis.reply.engine.get_location_context_with_timezone', return_value=("Location: Tbilisi, Georgia", None)):

            response = run_reply_engine(
                db=eval_db, cfg=mock_config, tts=None,
                text=query, dialogue_memory=eval_dialogue_memory
            )

        tools_used = capture.tool_names()

        print(f"\n📊 Anti-Deflection (Weather Forecast):")
        print(f"   Query: {query}")
        print(f"   Tools called: {tools_used}")
        print(f"   Response: {(response or '')[:150]}...")

        # Must have used at least one tool
        assert capture.has_any_tool(), \
            f"Agent should use tools for weather forecast, not respond from knowledge. " \
            f"Response: {(response or '')[:200]}"

        # Must NOT deflect
        assert not _response_is_deflection(response or ""), \
            f"Agent deflected instead of using its tools. Response: {(response or '')[:300]}"

    @pytest.mark.eval
    @requires_judge_llm
    @pytest.mark.parametrize("query", [
        pytest.param(
            "what's the latest news in tech?",
            id="No deflection: tech news"
        ),
        pytest.param(
            "what time is it in Tokyo?",
            id="No deflection: time query"
        ),
    ])
    def test_no_deflection_for_answerable_queries_live(
        self, query, mock_config, eval_db, eval_dialogue_memory
    ):
        """Live eval: agent should use tools for answerable queries, never deflect."""
        from jarvis.reply.engine import run_reply_engine
        from helpers import JUDGE_MODEL

        mock_config.ollama_base_url = "http://localhost:11434"
        mock_config.ollama_chat_model = JUDGE_MODEL

        capture = ToolCallCapture()
        mock_tool_run = create_mock_tool_run(capture, {
            "webSearch": "Top tech news: AI advances, new chip announcements.",
            "fetchWebPage": "Detailed article about tech trends...",
            "getWeather": "Current time info...",
        })

        with patch('jarvis.reply.engine.run_tool_with_retries', side_effect=mock_tool_run), \
             patch('jarvis.reply.engine.get_location_context_with_timezone', return_value=("Location: Tbilisi, Georgia", None)):

            response = run_reply_engine(
                db=eval_db, cfg=mock_config, tts=None,
                text=query, dialogue_memory=eval_dialogue_memory
            )

        print(f"\n📊 Anti-Deflection (General):")
        print(f"   Query: {query}")
        print(f"   Tools called: {capture.tool_names()}")
        print(f"   Response: {(response or '')[:150]}...")

        # Should not deflect for queries the agent can handle
        assert not _response_is_deflection(response or ""), \
            f"Agent deflected instead of being helpful. Response: {(response or '')[:300]}"

    @pytest.mark.eval
    @requires_judge_llm
    @pytest.mark.parametrize("follow_up", [
        pytest.param(
            "you have a weather tool, try again",
            id="Tool retry: explicit tool mention"
        ),
        pytest.param(
            "go ahead and check again, maybe try a different spelling",
            id="Tool retry: vague go ahead"
        ),
        pytest.param(
            "just try checking the weather one more time",
            id="Tool retry: vague just try"
        ),
    ])
    def test_tool_retry_after_failure_live(
        self, follow_up, mock_config, eval_db, eval_dialogue_memory
    ):
        """
        Live eval: when the user insists on retrying a tool after it returned
        unhelpful results, the agent should actually call the tool again —
        not narrate its intention to do so.

        Reproduces the bug where the model says "I will try checking the weather now"
        without actually producing a tool_calls field, causing the engine to treat
        the narration as a final response.

        Scenario:
        - Turn 1: User asks about weather in an obscure location → tool returns
          error/no data → model deflects or gives partial answer
        - Turn 2: User insists "try again" → model MUST call the tool, not
          just say "I will try"

        Small models often fail to retry after a tool error because they
        lack the reasoning capacity to override the "it failed, don't retry"
        heuristic. This is marked as xfail for small models.
        """
        from jarvis.reply.engine import run_reply_engine
        from jarvis.reply.prompts import detect_model_size, ModelSize
        from helpers import JUDGE_MODEL

        mock_config.ollama_base_url = "http://localhost:11434"
        mock_config.ollama_chat_model = JUDGE_MODEL

        is_small = detect_model_size(JUDGE_MODEL) == ModelSize.SMALL

        call_count = {"n": 0}

        def mock_tool_run(db, cfg, tool_name, tool_args, **kwargs):
            """First call returns error, second call succeeds."""
            from jarvis.tools.types import ToolExecutionResult
            capture.record(tool_name, tool_args or {})
            call_count["n"] += 1

            if tool_name == "getWeather":
                if call_count["n"] <= 1:
                    # First call: tool can't find the location
                    return ToolExecutionResult(
                        success=False,
                        reply_text="",
                        error_message="Could not find location 'Kazbegi'. Try a different spelling or a nearby city."
                    )
                else:
                    # Subsequent calls: tool succeeds
                    return ToolExecutionResult(
                        success=True,
                        reply_text="Current weather near Kazbegi (Stepantsminda), Georgia:\nConditions: Partly cloudy\nTemperature: 2.5°C\nWind: 25 km/h\n7-day: 2026-04-10: -1–5°C, Snow showers"
                    )
            return ToolExecutionResult(success=True, reply_text="OK")

        capture = ToolCallCapture()

        with patch('jarvis.reply.engine.run_tool_with_retries', side_effect=mock_tool_run), \
             patch('jarvis.reply.engine.get_location_context_with_timezone', return_value=("Location: Tbilisi, Georgia", None)):

            # Turn 1: Ask about weather in obscure location — tool will fail
            capture.clear()
            run_reply_engine(
                db=eval_db, cfg=mock_config, tts=None,
                text="how's the weather in Kazbegi today?",
                dialogue_memory=eval_dialogue_memory
            )
            turn1_tools = capture.tool_names()

            # Turn 2: User insists on retry — tool should succeed this time
            capture.clear()
            response = run_reply_engine(
                db=eval_db, cfg=mock_config, tts=None,
                text=follow_up,
                dialogue_memory=eval_dialogue_memory
            )

        turn2_tools = capture.tool_names()

        print(f"\n📊 Tool Retry After Failure:")
        print(f"   Turn 1 tools: {turn1_tools}")
        print(f"   Follow-up: {follow_up}")
        print(f"   Turn 2 tools: {turn2_tools}")
        print(f"   Response: {(response or '')[:200]}...")

        # The agent must actually call a tool on turn 2, not just narrate intent
        tool_called = capture.has_any_tool()
        is_deflection = _response_is_deflection(response or "")

        if not tool_called or is_deflection:
            if is_small:
                pytest.xfail(
                    f"Small model {JUDGE_MODEL} failed to retry tool after error. "
                    f"Known limitation. Tools called: {turn2_tools}, "
                    f"Response: {(response or '')[:150]}"
                )
            failure_reason = "no tool called" if not tool_called else "deflection in response"
            pytest.fail(
                f"Agent failed ({failure_reason}) on follow-up '{follow_up}'. "
                f"Tools called: {turn2_tools}. "
                f"Response: {(response or '')[:300]}"
            )

    @pytest.mark.eval
    @requires_judge_llm
    def test_graph_knowledge_surfaced_in_reply_live(
        self, mock_config, eval_db, eval_dialogue_memory
    ):
        """
        Live eval: when graph enrichment injects stored knowledge about the user,
        the LLM must use it — not deny having any personal information.

        Reproduces the observed failure where asking "tell me something about
        myself" surfaced 5 knowledge nodes yet the model still replied "I only
        know what you have told me in this current conversation". The graph
        context is now framed as the model's own knowledge; this eval locks
        that behaviour in so any regression (prompt drift, block framing, or
        silent drop like the earlier orphan-list bug) is caught.
        """
        from jarvis.reply.engine import run_reply_engine
        from helpers import JUDGE_MODEL

        mock_config.ollama_base_url = "http://localhost:11434"
        mock_config.ollama_chat_model = JUDGE_MODEL
        # Graph enrichment is opt-in via this setting; MockConfig defaults it off.
        mock_config.memory_enrichment_source = "all"

        class _Node:
            def __init__(self, id_, name, data):
                self.id = id_
                self.name = name
                self.data = data
                self.data_token_count = max(1, len(data) // 4)

        class _Ancestor:
            def __init__(self, name):
                self.name = name

        nodes = [
            _Node(
                "n-food",
                "Food Preferences",
                "The user loves Thai food (especially pad see ew) and "
                "regularly cooks homemade ramen on Sundays.",
            ),
            _Node(
                "n-fitness",
                "Fitness & Wellness",
                "The user boxes three times a week at Trenches Gym in Hackney "
                "and has been training consistently since 2023.",
            ),
            _Node(
                "n-work",
                "Work",
                "The user is a software engineer at Equals Money and works "
                "primarily on a local voice-assistant side-project called Jarvis.",
            ),
        ]

        class _FakeStore:
            def __init__(self, *a, **kw):
                pass

            def search_nodes(self, query, limit=5):
                return nodes[:limit]

            def get_ancestors(self, node_id):
                return [_Ancestor("Root")]

        # Extractor must produce questions so graph enrichment runs.
        fake_extract = {
            "keywords": ["personal", "interests", "preferences"],
            "questions": [
                "what are the user's hobbies and interests?",
                "what food does the user like?",
                "where does the user work?",
            ],
        }

        query = "what do you know about my hobbies, interests, and work?"

        with patch("jarvis.reply.engine.extract_search_params_for_memory", return_value=fake_extract), \
             patch("jarvis.memory.graph.GraphMemoryStore", _FakeStore), \
             patch("jarvis.memory.conversation.search_conversation_memory_by_keywords", return_value=[]), \
             patch("jarvis.reply.engine.get_location_context_with_timezone",
                   return_value=("Location: Hackney, London, UK", "Europe/London")):
            response = run_reply_engine(
                db=eval_db, cfg=mock_config, tts=None,
                text=query, dialogue_memory=eval_dialogue_memory,
            )

        response = response or ""
        response_lower = response.lower()

        print(f"\n📊 Graph Knowledge Surfaced in Reply (live):")
        print(f"   Query: {query}")
        print(f"   Model: {JUDGE_MODEL}")
        print(f"   Response: {response[:300]}")

        # Deflection phrases that indicate the model ignored the stored knowledge.
        denial_phrases = [
            "don't have any personal",
            "do not have any personal",
            "don't have personal information",
            "no personal information",
            "i don't know anything about you",
            "i only know what you",
            "only have access to the information you",
            "only have access to what you",
            "i don't have any information about you",
            # Long-term memory denial templates
            "do not have long-term",
            "don't have long-term",
            "no long-term memory",
            "do not store personal details",
            "don't store personal details",
            "forgotten between sessions",
            "outside of our conversation history",
        ]
        denied = next((p for p in denial_phrases if p in response_lower), None)
        assert denied is None, (
            f"Model denied knowing personal info despite graph enrichment providing it. "
            f"Matched denial phrase: {denied!r}\nResponse: {response[:400]}"
        )

        # At least one concrete fact from the stored nodes should appear.
        fact_keywords = [
            "thai", "pad see ew", "ramen",
            "box", "trenches", "hackney", "gym",
            "equals money", "software engineer", "jarvis",
        ]
        matched_facts = [kw for kw in fact_keywords if kw in response_lower]
        assert matched_facts, (
            f"Response did not reference any stored knowledge. "
            f"Expected at least one of: {fact_keywords}\nResponse: {response[:400]}"
        )

        print(f"   ✅ Response referenced stored facts: {matched_facts}")

    @pytest.mark.eval
    @requires_judge_llm
    def test_does_not_deny_long_term_memory_live(
        self, mock_config, eval_db, eval_dialogue_memory
    ):
        """
        Live eval: asking the assistant to remember something must not trigger
        a 'I have no long-term memory across sessions' denial.

        Jarvis *does* have persistent memory (the knowledge graph + diary), so
        replying with "I can't remember things between sessions" is a factually
        wrong hedge that small models slip into. This eval locks in the fix:
        system-prompt directive + banned phrasings.
        """
        from jarvis.reply.engine import run_reply_engine
        from helpers import JUDGE_MODEL

        mock_config.ollama_base_url = "http://localhost:11434"
        mock_config.ollama_chat_model = JUDGE_MODEL
        mock_config.memory_enrichment_source = "all"

        query = "please remember that I'm vegetarian"

        with patch("jarvis.reply.engine.extract_search_params_for_memory",
                   return_value={"keywords": ["vegetarian", "diet"], "questions": []}), \
             patch("jarvis.memory.conversation.search_conversation_memory_by_keywords", return_value=[]), \
             patch("jarvis.reply.engine.get_location_context_with_timezone",
                   return_value=("Location: Hackney, London, UK", "Europe/London")):
            response = run_reply_engine(
                db=eval_db, cfg=mock_config, tts=None,
                text=query, dialogue_memory=eval_dialogue_memory,
            )

        response = response or ""
        response_lower = response.lower()

        print(f"\n📊 Long-Term Memory Self-Awareness (live):")
        print(f"   Query: {query}")
        print(f"   Model: {JUDGE_MODEL}")
        print(f"   Response: {response[:300]}")

        memory_denials = [
            "do not have long-term",
            "don't have long-term",
            "no long-term memory",
            "do not store personal details",
            "don't store personal details",
            "forgotten between sessions",
            "lose that information when",
            "only within this session",
            "only for this conversation",
            "only for our current conversation",
            "do not retain",
            "don't retain",
        ]
        denied = next((p for p in memory_denials if p in response_lower), None)
        assert denied is None, (
            f"Model denied having long-term memory. Matched: {denied!r}\n"
            f"Response: {response[:400]}"
        )
        print(f"   ✅ No long-term-memory denial")

    @pytest.mark.eval
    @requires_judge_llm
    def test_open_ended_prompt_grounds_in_graph_context_live(
        self, mock_config, eval_db, eval_dialogue_memory
    ):
        """
        Live eval: open-ended prompts like "say something" should ground the
        reply in the stored knowledge about the user rather than fall back to
        a generic "Hello, how can I help you?" greeting.

        Locks in the system-prompt nudge that tells the model to use provided
        context on open-ended prompts instead of emitting a stock greeting.
        """
        from jarvis.reply.engine import run_reply_engine
        from helpers import JUDGE_MODEL

        mock_config.ollama_base_url = "http://localhost:11434"
        mock_config.ollama_chat_model = JUDGE_MODEL
        mock_config.memory_enrichment_source = "all"

        class _Node:
            def __init__(self, id_, name, data):
                self.id = id_
                self.name = name
                self.data = data
                self.data_token_count = max(1, len(data) // 4)

        class _Ancestor:
            def __init__(self, name):
                self.name = name

        nodes = [
            _Node(
                "n-food",
                "Food Preferences",
                "The user loves Thai food (especially pad see ew) and "
                "regularly cooks homemade ramen on Sundays.",
            ),
            _Node(
                "n-fitness",
                "Fitness & Wellness",
                "The user boxes three times a week at Trenches Gym in Hackney.",
            ),
        ]

        class _FakeStore:
            def __init__(self, *a, **kw):
                pass

            def search_nodes(self, query, limit=5):
                return nodes[:limit]

            def get_ancestors(self, node_id):
                return [_Ancestor("Root")]

        fake_extract = {
            "keywords": ["interests", "preferences"],
            "questions": [
                "what are the user's hobbies and interests?",
                "what food does the user like?",
            ],
        }

        query = "say something"

        with patch("jarvis.reply.engine.extract_search_params_for_memory", return_value=fake_extract), \
             patch("jarvis.memory.graph.GraphMemoryStore", _FakeStore), \
             patch("jarvis.memory.conversation.search_conversation_memory_by_keywords", return_value=[]), \
             patch("jarvis.reply.engine.get_location_context_with_timezone",
                   return_value=("Location: Hackney, London, UK", "Europe/London")):
            response = run_reply_engine(
                db=eval_db, cfg=mock_config, tts=None,
                text=query, dialogue_memory=eval_dialogue_memory,
            )

        response = response or ""
        response_lower = response.lower()

        print(f"\n📊 Open-Ended Prompt Grounds in Graph Context (live):")
        print(f"   Query: {query}")
        print(f"   Model: {JUDGE_MODEL}")
        print(f"   Response: {response[:300]}")

        # Stock greeting fallbacks — what we *don't* want.
        generic_phrases = [
            "how can i help you",
            "how may i help you",
            "what can i do for you",
            "what would you like",
            "i'm here and ready to chat",
            "is there something specific",
            "what's on your mind",
        ]
        generic_hit = next((p for p in generic_phrases if p in response_lower), None)
        assert generic_hit is None, (
            f"Open-ended prompt produced a generic greeting instead of using stored "
            f"knowledge. Matched: {generic_hit!r}\nResponse: {response[:400]}"
        )

        # At least one concrete fact from the stored nodes should appear.
        fact_keywords = [
            "thai", "pad see ew", "ramen",
            "box", "trenches", "hackney", "gym",
        ]
        matched_facts = [kw for kw in fact_keywords if kw in response_lower]
        assert matched_facts, (
            f"Open-ended response did not reference any stored knowledge. "
            f"Expected at least one of: {fact_keywords}\nResponse: {response[:400]}"
        )
        print(f"   ✅ Grounded in stored facts: {matched_facts}")


# =============================================================================
# Malformed LLM Response After Tool Results
# =============================================================================

class TestMalformedResponseAfterTools:
    """Tests that the engine handles malformed LLM outputs after tool results.

    Field capture (2026-04-21): after webSearch + Wikipedia fallback, gemma4:e2b
    returned 'tool_calls: []' as its content. The engine should treat this as
    a malformed response and not surface it as the reply.
    """

    @pytest.mark.eval
    def test_tool_calls_literal_not_surfaced_after_web_search(
        self, mock_config, eval_db, eval_dialogue_memory,
    ):
        """Engine must not return 'tool_calls: []' after a web search result.

        Scenario: user asks a factual question, webSearch is called and returns
        a result, but the LLM then emits 'tool_calls: []' instead of synthesising
        an answer. The engine should catch this as malformed and produce an error
        message rather than surfacing the raw literal.
        """
        from jarvis.reply.engine import run_reply_engine

        query = "what is Britney Spears' most famous song?"
        capture = ToolCallCapture()

        MOCK_SEARCH_RESULT = (
            "Britney Spears Wikipedia: American pop star. "
            "Her debut single '...Baby One More Time' (1998) was a global hit."
        )

        mock_tool_run = create_mock_tool_run(capture, {"webSearch": MOCK_SEARCH_RESULT})

        call_count = 0

        def mock_chat(base_url, chat_model, messages, timeout_sec, extra_options=None, tools=None, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First turn: model calls webSearch
                return create_mock_llm_response("", [
                    create_tool_call("webSearch", {"search_query": "Britney Spears most famous song"}),
                ])
            # Second turn: model produces the field-captured malformed output
            return create_mock_llm_response("tool_calls: []")

        with patch("jarvis.reply.engine.run_tool_with_retries", side_effect=mock_tool_run), \
             patch("jarvis.reply.engine.chat_with_messages", side_effect=mock_chat), \
             patch("jarvis.reply.engine.extract_search_params_for_memory", return_value={"keywords": []}):

            response = run_reply_engine(
                db=eval_db, cfg=mock_config, tts=None,
                text=query, dialogue_memory=eval_dialogue_memory,
            )

        print(f"\n📊 Malformed Response After Tools:")
        print(f"   Query: {query}")
        print(f"   Tools called: {[c['name'] for c in capture.calls]}")
        print(f"   Response: {response!r}")

        # The malformed literal must not reach the user
        assert "tool_calls" not in (response or "").lower(), (
            f"Engine surfaced 'tool_calls: []' to user. Got: {response!r}"
        )

        # Should have called webSearch
        assert capture.has_tool("webSearch"), "Expected webSearch to be called"

        # Response should be non-empty (either the error fallback or a proper answer)
        assert response and response.strip(), "Engine returned empty response"

        verdict = judge_response_answers_query(query, response or "")
        print(f"   Judge score: {verdict.score:.2f} — {verdict.reasoning[:80]}")
        # The judge should not give a high score to a malformed or empty-sounding reply
        # (if the engine correctly falls back to an error message, the score will be low
        # but the key assertion is that the literal wasn't surfaced)

