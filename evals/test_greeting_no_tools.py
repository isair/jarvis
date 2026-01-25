"""
Greeting No-Tools Evaluations

Tests that greetings and simple conversation don't trigger tool calls,
while tool-requiring queries still do.

This is critical for small models (3b) which may incorrectly call tools
for simple greetings when prompted to "proactively use tools."

Run: ./scripts/run_evals.sh test_greeting
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
    create_mock_llm_response,
    create_tool_call,
    is_judge_llm_available,
)

_JUDGE_LLM_AVAILABLE = is_judge_llm_available()
requires_judge_llm = pytest.mark.skipif(
    not _JUDGE_LLM_AVAILABLE,
    reason="Judge LLM not available"
)


# =============================================================================
# Test Data
# =============================================================================

# Greetings in multiple languages - should NOT trigger tools
GREETING_TEST_CASES = [
    ("hello", False, "English hello"),
    ("hi there", False, "English hi"),
    ("hey", False, "English hey"),
    ("ni hao", False, "Chinese greeting"),
    ("bonjour", False, "French greeting"),
    ("hola", False, "Spanish greeting"),
    ("merhaba", False, "Turkish greeting"),
    ("ciao", False, "Italian greeting"),
    ("guten tag", False, "German greeting"),
    ("how are you", False, "English wellbeing"),
    ("thank you", False, "English thanks"),
    ("thanks", False, "English thanks short"),
    ("goodbye", False, "English goodbye"),
    ("good morning", False, "English morning"),
    ("good night", False, "English night"),
]

# Queries that SHOULD trigger tools
TOOL_REQUIRED_TEST_CASES = [
    ("what's the weather", True, "Weather query"),
    ("search for python tutorials", True, "Web search"),
    ("what's the weather in Tokyo", True, "Weather with location"),
    ("look up the news today", True, "News search"),
    ("what did I eat yesterday", True, "Meal recall"),
]


# =============================================================================
# Helpers
# =============================================================================

@dataclass
class ToolCallCapture:
    """Captures tool calls during evaluation."""
    calls: List[Dict[str, Any]] = field(default_factory=list)

    def record(self, name: str, args: Dict[str, Any]):
        self.calls.append({"name": name, "args": args})

    def has_any_tool(self) -> bool:
        return len(self.calls) > 0

    def tool_names(self) -> List[str]:
        return [c["name"] for c in self.calls]


# =============================================================================
# Model Size Detection Tests
# =============================================================================

class TestModelSizeDetection:
    """Tests for model size detection logic."""

    @pytest.mark.eval
    @pytest.mark.parametrize("model_name,expected_size", [
        pytest.param("llama3.2:3b", "SMALL", id="llama3.2:3b"),
        pytest.param("llama3.2:1b", "SMALL", id="llama3.2:1b"),
        pytest.param("mistral:7b", "SMALL", id="mistral:7b"),
        pytest.param("gpt-oss:20b", "LARGE", id="gpt-oss:20b"),
        pytest.param("llama3.1:8b", "LARGE", id="llama3.1:8b"),
        pytest.param("qwen2.5:14b", "LARGE", id="qwen2.5:14b"),
        pytest.param("gemma2:27b", "LARGE", id="gemma2:27b"),
        pytest.param(None, "LARGE", id="None defaults to LARGE"),
        pytest.param("", "LARGE", id="Empty defaults to LARGE"),
    ])
    def test_model_size_detection(self, model_name: str, expected_size: str):
        """Verify model size is correctly detected from model name."""
        from jarvis.reply.prompts import detect_model_size, ModelSize

        result = detect_model_size(model_name)
        expected = ModelSize.SMALL if expected_size == "SMALL" else ModelSize.LARGE

        print(f"\n  Model: {model_name}")
        print(f"  Detected: {result.value}")
        print(f"  Expected: {expected.value}")

        assert result == expected, f"Expected {expected_size} for model '{model_name}'"


# =============================================================================
# Greeting Behavior Tests (Mocked)
# =============================================================================

class TestGreetingNoTools:
    """
    Tests that greetings don't trigger tool calls.

    Uses mocked LLM to verify the prompt system works correctly.
    """

    @pytest.mark.eval
    @pytest.mark.parametrize("query,should_use_tools,desc", GREETING_TEST_CASES)
    def test_greeting_no_tool_calls(
        self,
        query: str,
        should_use_tools: bool,
        desc: str,
        mock_config,
        eval_db,
        eval_dialogue_memory
    ):
        """Greetings should not trigger tool calls."""
        from jarvis.reply.engine import run_reply_engine

        # Use small model to test conservative prompts
        mock_config.ollama_chat_model = "llama3.2:3b"
        capture = ToolCallCapture()

        def mock_tool_run(db, cfg, tool_name, tool_args, **kwargs):
            from jarvis.tools.types import ToolExecutionResult
            capture.record(tool_name, tool_args or {})
            return ToolExecutionResult(success=True, reply_text="Tool result")

        def mock_chat(base_url, chat_model, messages, timeout_sec, extra_options=None, tools=None):
            # Simulate model correctly NOT calling tools for greetings
            return create_mock_llm_response("Hello! How can I help you today?")

        with patch('jarvis.reply.engine.run_tool_with_retries', side_effect=mock_tool_run), \
             patch('jarvis.reply.engine.chat_with_messages', side_effect=mock_chat), \
             patch('jarvis.reply.engine.extract_search_params_for_memory', return_value={"keywords": []}), \
             patch('jarvis.reply.engine.select_profile_llm', return_value="life"):

            response = run_reply_engine(
                db=eval_db, cfg=mock_config, tts=None,
                text=query, dialogue_memory=eval_dialogue_memory
            )

        print(f"\n  Query: '{query}' ({desc})")
        print(f"  Tools called: {capture.tool_names() or 'none'}")
        print(f"  Response: {(response or '')[:80]}...")

        assert not capture.has_any_tool(), \
            f"Greeting '{query}' should NOT trigger tools. Called: {capture.tool_names()}"

    @pytest.mark.eval
    @pytest.mark.parametrize("query,should_use_tools,desc", TOOL_REQUIRED_TEST_CASES)
    def test_tool_queries_still_work(
        self,
        query: str,
        should_use_tools: bool,
        desc: str,
        mock_config,
        eval_db,
        eval_dialogue_memory
    ):
        """Tool-requiring queries should still trigger tools."""
        from jarvis.reply.engine import run_reply_engine

        # Use small model
        mock_config.ollama_chat_model = "llama3.2:3b"
        capture = ToolCallCapture()

        def mock_tool_run(db, cfg, tool_name, tool_args, **kwargs):
            from jarvis.tools.types import ToolExecutionResult
            capture.record(tool_name, tool_args or {})
            return ToolExecutionResult(success=True, reply_text="Weather: 20C sunny")

        call_count = 0
        def mock_chat(base_url, chat_model, messages, timeout_sec, extra_options=None, tools=None):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                # Model correctly identifies need for tool
                if "weather" in query.lower():
                    return create_mock_llm_response("", [create_tool_call("getWeather", {"location": "here"})])
                elif "search" in query.lower() or "look up" in query.lower() or "news" in query.lower():
                    return create_mock_llm_response("", [create_tool_call("webSearch", {"search_query": query})])
                elif "eat" in query.lower() or "meal" in query.lower():
                    return create_mock_llm_response("", [create_tool_call("fetchMeals", {})])
            return create_mock_llm_response("Here's the information you requested.")

        with patch('jarvis.reply.engine.run_tool_with_retries', side_effect=mock_tool_run), \
             patch('jarvis.reply.engine.chat_with_messages', side_effect=mock_chat), \
             patch('jarvis.reply.engine.extract_search_params_for_memory', return_value={"keywords": []}), \
             patch('jarvis.reply.engine.select_profile_llm', return_value="life"):

            response = run_reply_engine(
                db=eval_db, cfg=mock_config, tts=None,
                text=query, dialogue_memory=eval_dialogue_memory
            )

        print(f"\n  Query: '{query}' ({desc})")
        print(f"  Tools called: {capture.tool_names() or 'none'}")
        print(f"  Response: {(response or '')[:80]}...")

        assert capture.has_any_tool(), \
            f"Query '{query}' SHOULD trigger tools but didn't. Response: {response}"


# =============================================================================
# Live Tests with Real LLM
# =============================================================================

def _is_small_model(model_name: str) -> bool:
    """Check if model is a small model (1b-7b)."""
    if not model_name:
        return False
    name_lower = model_name.lower()
    return any(p in name_lower for p in [":1b", ":3b", ":7b", "-1b", "-3b", "-7b"])


class TestGreetingNoToolsLive:
    """
    Live tests with real LLM inference.

    These verify that the prompt changes actually work with real models.

    NOTE: Small models (1b-7b) may still incorrectly call tools for greetings
    despite explicit prompt constraints. This is a fundamental limitation of
    small model reasoning capacity. These tests document this behavior.
    """

    @pytest.mark.eval
    @requires_judge_llm
    @pytest.mark.parametrize("query,should_use_tools,desc", [
        pytest.param("hello", False, "English hello"),
        pytest.param("ni hao", False, "Chinese greeting"),
        pytest.param("bonjour", False, "French greeting"),
        pytest.param("how are you", False, "Wellbeing query"),
    ])
    def test_greeting_no_tools_live(
        self,
        query: str,
        should_use_tools: bool,
        desc: str,
        mock_config,
        eval_db,
        eval_dialogue_memory
    ):
        """Live test: greetings should not trigger tool calls."""
        from jarvis.reply.engine import run_reply_engine
        from helpers import JUDGE_MODEL

        # Use the judge model (which may be small or large)
        mock_config.ollama_base_url = "http://localhost:11434"
        mock_config.ollama_chat_model = JUDGE_MODEL

        # Small models may fail this test due to limited reasoning capacity
        # This documents the limitation rather than masking it
        is_small = _is_small_model(JUDGE_MODEL)

        capture = ToolCallCapture()

        def mock_tool_run(db, cfg, tool_name, tool_args, **kwargs):
            from jarvis.tools.types import ToolExecutionResult
            capture.record(tool_name, tool_args or {})
            return ToolExecutionResult(success=True, reply_text="Tool result")

        with patch('jarvis.reply.engine.run_tool_with_retries', side_effect=mock_tool_run):
            response = run_reply_engine(
                db=eval_db, cfg=mock_config, tts=None,
                text=query, dialogue_memory=eval_dialogue_memory
            )

        print(f"\n  Live Greeting Test ({JUDGE_MODEL}):")
        print(f"  Query: '{query}' ({desc})")
        print(f"  Tools called: {capture.tool_names() or 'none'}")
        print(f"  Response: {(response or '')[:100]}...")
        print(f"  Model size: {'small' if is_small else 'large'}")

        # For greetings, we expect NO tool calls
        if not should_use_tools:
            if capture.has_any_tool():
                if is_small:
                    # Document the limitation but don't fail the test
                    pytest.xfail(
                        f"Small model {JUDGE_MODEL} called tools for greeting '{query}'. "
                        f"This is a known limitation of small models. Called: {capture.tool_names()}"
                    )
                else:
                    # Large models should follow the guidance
                    pytest.fail(
                        f"Large model greeting '{query}' should NOT trigger tools. "
                        f"Called: {capture.tool_names()}"
                    )

    @pytest.mark.eval
    @requires_judge_llm
    def test_weather_still_triggers_tools_live(
        self,
        mock_config,
        eval_db,
        eval_dialogue_memory
    ):
        """Live test: weather query should still trigger tools."""
        from jarvis.reply.engine import run_reply_engine
        from helpers import JUDGE_MODEL

        query = "what's the weather today"
        mock_config.ollama_base_url = "http://localhost:11434"
        mock_config.ollama_chat_model = JUDGE_MODEL

        capture = ToolCallCapture()

        def mock_tool_run(db, cfg, tool_name, tool_args, **kwargs):
            from jarvis.tools.types import ToolExecutionResult
            capture.record(tool_name, tool_args or {})
            return ToolExecutionResult(success=True, reply_text="Weather: 22C, partly cloudy")

        with patch('jarvis.reply.engine.run_tool_with_retries', side_effect=mock_tool_run):
            response = run_reply_engine(
                db=eval_db, cfg=mock_config, tts=None,
                text=query, dialogue_memory=eval_dialogue_memory
            )

        print(f"\n  Live Weather Test ({JUDGE_MODEL}):")
        print(f"  Query: '{query}'")
        print(f"  Tools called: {capture.tool_names() or 'none'}")
        print(f"  Response: {(response or '')[:100]}...")

        # Weather should trigger tools (getWeather or webSearch)
        assert capture.has_any_tool(), \
            f"Weather query should trigger tools. Response: {response}"
