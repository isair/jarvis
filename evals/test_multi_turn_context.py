"""
Multi-Turn Context Evaluations

Tests the agent's ability to handle multi-turn conversations correctly:
1. Topic Switching - Selecting correct tool when conversation topic changes
2. Context Anchoring - Not getting "stuck" on previous turn's tool
3. Follow-up Handling - Using context from previous turns when relevant

These evals are critical for catching regressions where the model might:
- Call the wrong tool after a topic change (e.g., getWeather for store hours)
- Ignore context from previous turns
- Fail to follow up on established conversation context

Run: ./scripts/run_evals.sh
"""

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

_this_file = Path(__file__).resolve()
EVALS_DIR = _this_file.parent
if str(EVALS_DIR) not in sys.path:
    sys.path.insert(0, str(EVALS_DIR))

import pytest
from unittest.mock import patch

from helpers import (
    MockConfig,
    is_judge_llm_available,
    JUDGE_MODEL,
)

_JUDGE_LLM_AVAILABLE = is_judge_llm_available()
requires_judge_llm = pytest.mark.skipif(
    not _JUDGE_LLM_AVAILABLE,
    reason="Judge LLM not available"
)


# =============================================================================
# Test Data - Consistent tool responses for reproducibility
# =============================================================================

MOCK_WEATHER_RESPONSE = """Current weather in Kensington, Royal Kensington and Chelsea, United Kingdom:
Conditions: Overcast
Temperature: 7.8°C
Feels like: 5°C
Humidity: 75%
Wind: 12 km/h from the west
"""

MOCK_STORE_HOURS_SEARCH = """Web search results for 'CEX store hours Kensington':

**Content from top result:**
CEX Kensington High Street
Opening Hours:
Monday - Saturday: 10:00 AM - 6:00 PM
Sunday: 11:00 AM - 5:00 PM

**Other search results:**
1. **CEX Kensington - Store Info** - https://uk.webuy.com/store/kensington
2. **CEX Store Locator** - https://uk.webuy.com/stores
"""

MOCK_RESTAURANT_SEARCH = """Web search results for 'Italian restaurants Kensington':

**Content from top result:**
Best Italian Restaurants in Kensington:
1. Zafferano - Fine Italian dining, Michelin recommended
2. Scalini - Traditional Italian cuisine since 1988
3. Lucio - Modern Italian with great pasta

**Other search results:**
1. **TripAdvisor Italian Restaurants** - https://tripadvisor.com/...
2. **Time Out London** - https://timeout.com/...
"""

MOCK_NEWS_SEARCH = """Web search results for 'tech news today':

**Content from top result:**
Today's Tech Headlines:
- Apple announces new M4 chip
- OpenAI releases GPT-5
- SpaceX Starship completes orbital test

**Other search results:**
1. **TechCrunch** - https://techcrunch.com
2. **The Verge** - https://theverge.com
"""


# =============================================================================
# Tool Call Capture Helper
# =============================================================================

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

    def tool_sequence(self) -> List[str]:
        return [c["name"] for c in self.calls]

    def clear(self):
        self.calls = []


# =============================================================================
# Topic Switching Evaluations (Live LLM)
# =============================================================================

class TestTopicSwitching:
    """
    Tests that the agent selects the correct tool when the conversation
    topic changes between turns.

    Uses real LLM inference to test actual model behavior.
    Tool execution is mocked for consistent responses.
    """

    @pytest.mark.eval
    @requires_judge_llm
    def test_weather_then_store_hours(self, mock_config, eval_db, eval_dialogue_memory):
        """
        After weather query, asking about store hours should use webSearch.

        Scenario:
        - Turn 1: "How's the weather?" -> getWeather (correct)
        - Turn 2: "Can you check when CEX closes?" -> webSearch (NOT getWeather!)

        This tests the exact bug scenario where llama3.2:3b called getWeather
        for a store hours query because it got anchored on the previous tool.
        """
        from jarvis.reply.engine import run_reply_engine

        mock_config.ollama_base_url = "http://localhost:11434"
        mock_config.ollama_chat_model = JUDGE_MODEL

        capture = ToolCallCapture()

        def mock_tool_run(db, cfg, tool_name, tool_args, **kwargs):
            from jarvis.tools.types import ToolExecutionResult
            capture.record(tool_name, tool_args or {})

            if tool_name == "getWeather":
                return ToolExecutionResult(success=True, reply_text=MOCK_WEATHER_RESPONSE)
            elif tool_name == "webSearch":
                return ToolExecutionResult(success=True, reply_text=MOCK_STORE_HOURS_SEARCH)
            return ToolExecutionResult(success=True, reply_text="OK")

        with patch('jarvis.reply.engine.run_tool_with_retries', side_effect=mock_tool_run), \
             patch('jarvis.reply.engine.get_location_context', return_value="Location: Kensington, Royal Kensington and Chelsea, United Kingdom"):

            # Turn 1: Weather query
            capture.clear()
            response1 = run_reply_engine(
                db=eval_db, cfg=mock_config, tts=None,
                text="How's the weather today?",
                dialogue_memory=eval_dialogue_memory
            )
            turn1_tools = capture.tool_sequence()

            # Turn 2: Store hours query (topic change)
            capture.clear()
            response2 = run_reply_engine(
                db=eval_db, cfg=mock_config, tts=None,
                text="Yeah, I could do but can you check how long CEX is open for?",
                dialogue_memory=eval_dialogue_memory
            )
            turn2_tools = capture.tool_sequence()

        print(f"\n📊 Topic Switching - Weather → Store Hours:")
        print(f"   Turn 1 query: 'How's the weather today?'")
        print(f"   Turn 1 tools: {turn1_tools}")
        print(f"   Turn 1 response: {response1[:100] if response1 else 'None'}...")
        print(f"   Turn 2 query: 'can you check how long CEX is open for?'")
        print(f"   Turn 2 tools: {turn2_tools}")
        print(f"   Turn 2 response: {response2[:100] if response2 else 'None'}...")

        # Turn 1 should use getWeather
        assert "getWeather" in turn1_tools, \
            f"Turn 1 should use getWeather for weather query. Used: {turn1_tools}"

        # Turn 2 MUST use webSearch, NOT getWeather
        # This is the critical assertion - the model should recognize topic change
        used_wrong_tool = "getWeather" in turn2_tools and "webSearch" not in turn2_tools

        if used_wrong_tool:
            pytest.fail(
                f"❌ CONTEXT ANCHORING BUG: Model used getWeather for store hours!\n"
                f"   Turn 2 tools: {turn2_tools}\n"
                f"   Expected: webSearch\n"
                f"   The model got 'stuck' on the previous turn's tool.\n"
                f"   Response: {response2[:200] if response2 else 'None'}"
            )

        assert "webSearch" in turn2_tools, \
            f"Turn 2 should use webSearch for store hours. Used: {turn2_tools}"

        print(f"   ✅ Correctly switched from getWeather to webSearch")

    @pytest.mark.eval
    @requires_judge_llm
    def test_weather_then_restaurant_search(self, mock_config, eval_db, eval_dialogue_memory):
        """
        After weather query, asking for restaurant recommendations should use webSearch.
        """
        from jarvis.reply.engine import run_reply_engine

        mock_config.ollama_base_url = "http://localhost:11434"
        mock_config.ollama_chat_model = JUDGE_MODEL

        capture = ToolCallCapture()

        def mock_tool_run(db, cfg, tool_name, tool_args, **kwargs):
            from jarvis.tools.types import ToolExecutionResult
            capture.record(tool_name, tool_args or {})

            if tool_name == "getWeather":
                return ToolExecutionResult(success=True, reply_text=MOCK_WEATHER_RESPONSE)
            elif tool_name == "webSearch":
                return ToolExecutionResult(success=True, reply_text=MOCK_RESTAURANT_SEARCH)
            return ToolExecutionResult(success=True, reply_text="OK")

        with patch('jarvis.reply.engine.run_tool_with_retries', side_effect=mock_tool_run), \
             patch('jarvis.reply.engine.get_location_context', return_value="Location: Kensington, UK"):

            # Turn 1: Weather
            capture.clear()
            run_reply_engine(
                db=eval_db, cfg=mock_config, tts=None,
                text="What's the weather like?",
                dialogue_memory=eval_dialogue_memory
            )
            turn1_tools = capture.tool_sequence()

            # Turn 2: Restaurant search
            capture.clear()
            response2 = run_reply_engine(
                db=eval_db, cfg=mock_config, tts=None,
                text="Any good Italian restaurants nearby?",
                dialogue_memory=eval_dialogue_memory
            )
            turn2_tools = capture.tool_sequence()

        print(f"\n📊 Topic Switching - Weather → Restaurants:")
        print(f"   Turn 1 tools: {turn1_tools}")
        print(f"   Turn 2 tools: {turn2_tools}")

        assert "getWeather" in turn1_tools, \
            f"Turn 1 should use getWeather. Used: {turn1_tools}"

        # Check for context anchoring bug
        if "getWeather" in turn2_tools and "webSearch" not in turn2_tools:
            pytest.fail(
                f"❌ CONTEXT ANCHORING BUG: Model used getWeather for restaurant query!\n"
                f"   Turn 2 tools: {turn2_tools}\n"
                f"   Response: {response2[:200] if response2 else 'None'}"
            )

        assert "webSearch" in turn2_tools, \
            f"Turn 2 should use webSearch for restaurant query. Used: {turn2_tools}"

        print(f"   ✅ Correctly switched from getWeather to webSearch")

    @pytest.mark.eval
    @requires_judge_llm
    def test_search_then_weather(self, mock_config, eval_db, eval_dialogue_memory):
        """
        After a web search, asking about weather should use getWeather.

        Tests the reverse direction - ensuring the model doesn't stay stuck
        on webSearch when weather is asked.
        """
        from jarvis.reply.engine import run_reply_engine

        mock_config.ollama_base_url = "http://localhost:11434"
        mock_config.ollama_chat_model = JUDGE_MODEL

        capture = ToolCallCapture()

        def mock_tool_run(db, cfg, tool_name, tool_args, **kwargs):
            from jarvis.tools.types import ToolExecutionResult
            capture.record(tool_name, tool_args or {})

            if tool_name == "getWeather":
                return ToolExecutionResult(success=True, reply_text=MOCK_WEATHER_RESPONSE)
            elif tool_name == "webSearch":
                return ToolExecutionResult(success=True, reply_text=MOCK_NEWS_SEARCH)
            return ToolExecutionResult(success=True, reply_text="OK")

        with patch('jarvis.reply.engine.run_tool_with_retries', side_effect=mock_tool_run), \
             patch('jarvis.reply.engine.get_location_context', return_value="Location: Kensington, UK"):

            # Turn 1: News search
            capture.clear()
            run_reply_engine(
                db=eval_db, cfg=mock_config, tts=None,
                text="What's the latest tech news?",
                dialogue_memory=eval_dialogue_memory
            )
            turn1_tools = capture.tool_sequence()

            # Turn 2: Weather
            capture.clear()
            response2 = run_reply_engine(
                db=eval_db, cfg=mock_config, tts=None,
                text="How's the weather outside?",
                dialogue_memory=eval_dialogue_memory
            )
            turn2_tools = capture.tool_sequence()

        print(f"\n📊 Topic Switching - News → Weather:")
        print(f"   Turn 1 tools: {turn1_tools}")
        print(f"   Turn 2 tools: {turn2_tools}")

        assert "webSearch" in turn1_tools, \
            f"Turn 1 should use webSearch for news. Used: {turn1_tools}"

        # Check for reverse anchoring
        if "webSearch" in turn2_tools and "getWeather" not in turn2_tools:
            pytest.fail(
                f"❌ CONTEXT ANCHORING BUG: Model used webSearch for weather query!\n"
                f"   Turn 2 tools: {turn2_tools}\n"
                f"   Response: {response2[:200] if response2 else 'None'}"
            )

        assert "getWeather" in turn2_tools, \
            f"Turn 2 should use getWeather for weather query. Used: {turn2_tools}"

        print(f"   ✅ Correctly switched from webSearch to getWeather")


# =============================================================================
# Follow-Up Context Evaluations (Live LLM)
# =============================================================================

class TestFollowUpContext:
    """
    Tests that the agent maintains context from previous turns
    when handling follow-up questions.
    """

    @pytest.mark.eval
    @requires_judge_llm
    def test_follow_up_references_previous_context(self, mock_config, eval_db, eval_dialogue_memory):
        """
        Follow-up questions should reference information from previous turns.

        Scenario:
        - Turn 1: "How's the weather?" -> (gets weather data showing overcast, 7.8°C)
        - Turn 2: "Should I bring an umbrella?" -> Response should reference weather

        The model should use the weather context to inform the umbrella advice.
        """
        from jarvis.reply.engine import run_reply_engine

        mock_config.ollama_base_url = "http://localhost:11434"
        mock_config.ollama_chat_model = JUDGE_MODEL

        capture = ToolCallCapture()

        def mock_tool_run(db, cfg, tool_name, tool_args, **kwargs):
            from jarvis.tools.types import ToolExecutionResult
            capture.record(tool_name, tool_args or {})

            if tool_name == "getWeather":
                return ToolExecutionResult(success=True, reply_text=MOCK_WEATHER_RESPONSE)
            return ToolExecutionResult(success=True, reply_text="OK")

        with patch('jarvis.reply.engine.run_tool_with_retries', side_effect=mock_tool_run), \
             patch('jarvis.reply.engine.get_location_context', return_value="Location: Kensington, UK"):

            # Turn 1: Weather query
            capture.clear()
            response1 = run_reply_engine(
                db=eval_db, cfg=mock_config, tts=None,
                text="How's the weather today?",
                dialogue_memory=eval_dialogue_memory
            )
            turn1_tools = capture.tool_sequence()

            # Turn 2: Follow-up about umbrella
            capture.clear()
            response2 = run_reply_engine(
                db=eval_db, cfg=mock_config, tts=None,
                text="Should I bring an umbrella?",
                dialogue_memory=eval_dialogue_memory
            )
            turn2_tools = capture.tool_sequence()

        print(f"\n📊 Follow-Up Context - Weather → Umbrella:")
        print(f"   Turn 1 tools: {turn1_tools}")
        print(f"   Turn 1 response: {response1[:80] if response1 else 'None'}...")
        print(f"   Turn 2 tools: {turn2_tools}")
        print(f"   Turn 2 response: {response2[:120] if response2 else 'None'}...")

        # Turn 1 should fetch weather
        assert "getWeather" in turn1_tools, "Turn 1 should fetch weather"

        # Turn 2: Check if response references weather context
        # (It may or may not call getWeather again - both are acceptable)
        if response2:
            weather_terms = ["overcast", "cloud", "rain", "weather", "chilly", "cold", "7", "8"]
            references_weather = any(term in response2.lower() for term in weather_terms)
            print(f"   References weather context: {references_weather}")

            # The response should acknowledge or use the weather context
            # Not a hard fail if it doesn't, but we log it
            if not references_weather:
                print(f"   ⚠️ Response doesn't seem to reference weather context")


# =============================================================================
# Extended Multi-Turn Evaluations (Live LLM)
# =============================================================================

class TestMultiTurnExtended:
    """
    Extended multi-turn scenarios testing longer conversations
    and more complex topic changes.
    """

    @pytest.mark.eval
    @requires_judge_llm
    def test_three_turn_topic_changes(self, mock_config, eval_db, eval_dialogue_memory):
        """
        Three-turn conversation with multiple topic changes.

        Turn 1: Weather query
        Turn 2: Store hours query (topic change from weather)
        Turn 3: News query (topic change from store)

        Each turn should select the appropriate tool.
        """
        from jarvis.reply.engine import run_reply_engine

        mock_config.ollama_base_url = "http://localhost:11434"
        mock_config.ollama_chat_model = JUDGE_MODEL

        capture = ToolCallCapture()
        all_turns = []

        def mock_tool_run(db, cfg, tool_name, tool_args, **kwargs):
            from jarvis.tools.types import ToolExecutionResult
            capture.record(tool_name, tool_args or {})

            if tool_name == "getWeather":
                return ToolExecutionResult(success=True, reply_text=MOCK_WEATHER_RESPONSE)
            elif tool_name == "webSearch":
                # Return appropriate content based on query
                args_str = str(tool_args).lower() if tool_args else ""
                if "cex" in args_str or "store" in args_str or "hour" in args_str:
                    return ToolExecutionResult(success=True, reply_text=MOCK_STORE_HOURS_SEARCH)
                else:
                    return ToolExecutionResult(success=True, reply_text=MOCK_NEWS_SEARCH)
            return ToolExecutionResult(success=True, reply_text="OK")

        with patch('jarvis.reply.engine.run_tool_with_retries', side_effect=mock_tool_run), \
             patch('jarvis.reply.engine.get_location_context', return_value="Location: Kensington, UK"):

            queries = [
                ("How's the weather today?", "getWeather"),
                ("What time does CEX close?", "webSearch"),
                ("What's happening in tech news?", "webSearch"),
            ]

            for query, expected_tool in queries:
                capture.clear()
                response = run_reply_engine(
                    db=eval_db, cfg=mock_config, tts=None,
                    text=query,
                    dialogue_memory=eval_dialogue_memory
                )
                all_turns.append({
                    "query": query,
                    "expected": expected_tool,
                    "tools": capture.tool_sequence().copy(),
                    "response": response
                })

        print(f"\n📊 Three-Turn Topic Changes:")
        failures = []
        for i, turn in enumerate(all_turns, 1):
            tools = turn["tools"]
            expected = turn["expected"]
            has_expected = expected in tools

            status = "✅" if has_expected else "❌"
            print(f"   Turn {i}: '{turn['query'][:35]}...'")
            print(f"      Expected: {expected}, Got: {tools} {status}")

            if not has_expected:
                # Check for context anchoring specifically
                if i > 1 and all_turns[i-2]["expected"] in tools:
                    failures.append(
                        f"Turn {i}: Context anchoring bug - used {tools} (previous turn's tool) "
                        f"instead of {expected}"
                    )
                else:
                    failures.append(f"Turn {i}: Expected {expected}, got {tools}")

        if failures:
            pytest.fail(
                f"❌ Multi-turn tool selection failures:\n" +
                "\n".join(f"   - {f}" for f in failures)
            )

        print(f"   ✅ All turns selected correct tools")

    @pytest.mark.eval
    @requires_judge_llm
    def test_rapid_topic_switching(self, mock_config, eval_db, eval_dialogue_memory):
        """
        Rapid back-and-forth between weather and search topics.

        This stress-tests the model's ability to quickly switch context.
        """
        from jarvis.reply.engine import run_reply_engine

        mock_config.ollama_base_url = "http://localhost:11434"
        mock_config.ollama_chat_model = JUDGE_MODEL

        capture = ToolCallCapture()

        def mock_tool_run(db, cfg, tool_name, tool_args, **kwargs):
            from jarvis.tools.types import ToolExecutionResult
            capture.record(tool_name, tool_args or {})

            if tool_name == "getWeather":
                return ToolExecutionResult(success=True, reply_text=MOCK_WEATHER_RESPONSE)
            elif tool_name == "webSearch":
                return ToolExecutionResult(success=True, reply_text=MOCK_NEWS_SEARCH)
            return ToolExecutionResult(success=True, reply_text="OK")

        with patch('jarvis.reply.engine.run_tool_with_retries', side_effect=mock_tool_run), \
             patch('jarvis.reply.engine.get_location_context', return_value="Location: Kensington, UK"):

            # Rapid switches between weather and search
            queries = [
                ("What's the weather?", "getWeather"),
                ("Search for coffee shops", "webSearch"),
                ("Is it going to rain?", "getWeather"),
                ("Find me a gym nearby", "webSearch"),
            ]

            results = []
            for query, expected in queries:
                capture.clear()
                run_reply_engine(
                    db=eval_db, cfg=mock_config, tts=None,
                    text=query,
                    dialogue_memory=eval_dialogue_memory
                )
                tools = capture.tool_sequence()
                results.append({
                    "query": query,
                    "expected": expected,
                    "tools": tools,
                    "correct": expected in tools
                })

        print(f"\n📊 Rapid Topic Switching:")
        correct_count = sum(1 for r in results if r["correct"])
        total = len(results)

        for r in results:
            status = "✅" if r["correct"] else "❌"
            print(f"   {status} '{r['query'][:30]}...' → {r['tools']} (expected: {r['expected']})")

        print(f"\n   Score: {correct_count}/{total} correct")

        # Allow some flexibility - at least 3/4 should be correct
        assert correct_count >= 3, \
            f"Rapid topic switching: Only {correct_count}/{total} correct tool selections"

        if correct_count == total:
            print(f"   ✅ Perfect score on rapid topic switching")
        else:
            print(f"   ⚠️ Some errors in rapid switching (acceptable: {correct_count}/{total})")
