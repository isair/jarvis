"""
Unit tests for greeting and instruction handling in the reply engine.

Verifies that the model-size-aware prompt system correctly prevents
tool calls for greetings and user instructions, while still allowing
tools for queries that genuinely require them.

These tests use a mocked LLM and do not require a real Ollama instance.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _disable_planner():
    """These tests verify greeting/instruction routing against a mocked
    chat LLM. The planner uses its own LLM call (`call_llm_direct`) which
    is not mocked here, so disable it to keep the test hermetic."""
    with patch("jarvis.reply.engine.plan_query", return_value=[]):
        yield


# =============================================================================
# Test Data
# =============================================================================

# Greetings in multiple languages - should NOT trigger tools
GREETING_TEST_CASES = [
    pytest.param("hello", False, id="Greeting: hello"),
    pytest.param("hi there", False, id="Greeting: hi there"),
    pytest.param("hey", False, id="Greeting: hey"),
    pytest.param("ni hao", False, id="Greeting: ni hao (Chinese)"),
    pytest.param("bonjour", False, id="Greeting: bonjour (French)"),
    pytest.param("hola", False, id="Greeting: hola (Spanish)"),
    pytest.param("merhaba", False, id="Greeting: merhaba (Turkish)"),
    pytest.param("ciao", False, id="Greeting: ciao (Italian)"),
    pytest.param("guten tag", False, id="Greeting: guten tag (German)"),
    pytest.param("how are you", False, id="Greeting: how are you"),
    pytest.param("thank you", False, id="Greeting: thank you"),
    pytest.param("thanks", False, id="Greeting: thanks"),
    pytest.param("goodbye", False, id="Greeting: goodbye"),
    pytest.param("good morning", False, id="Greeting: good morning"),
    pytest.param("good night", False, id="Greeting: good night"),
]

# User instructions about behaviour - should NOT trigger tools
USER_INSTRUCTION_TEST_CASES = [
    pytest.param("always use Celsius when telling me temperatures", False, id="Instruction: use Celsius"),
    pytest.param("remember to always tell me things in Celsius", False, id="Instruction: remember Celsius"),
    pytest.param("be more brief in your responses", False, id="Instruction: be more brief"),
    pytest.param("speak in French from now on", False, id="Instruction: speak in French"),
    pytest.param("always give me the short version", False, id="Instruction: short version"),
    pytest.param("don't use emojis in your responses", False, id="Instruction: no emojis"),
    pytest.param("note that I prefer metric units", False, id="Instruction: prefer metric"),
]

# Queries that SHOULD trigger tools
TOOL_REQUIRED_TEST_CASES = [
    pytest.param("what's the weather", True, id="Tool query: weather"),
    pytest.param("search for python tutorials", True, id="Tool query: web search"),
    pytest.param("what's the weather in Tokyo", True, id="Tool query: weather with location"),
    pytest.param("look up the news today", True, id="Tool query: news search"),
    pytest.param("what did I eat yesterday", True, id="Tool query: meal recall"),
]


# =============================================================================
# Helpers
# =============================================================================

@dataclass
class ToolCallCapture:
    """Captures tool calls made during a test run."""
    calls: List[Dict[str, Any]] = field(default_factory=list)

    def record(self, name: str, args: Dict[str, Any]):
        self.calls.append({"name": name, "args": args})

    def has_any_tool(self) -> bool:
        return len(self.calls) > 0

    def tool_names(self) -> List[str]:
        return [c["name"] for c in self.calls]


def _mock_llm_response(content: str, tool_calls=None):
    """Build a minimal mock Ollama response dict."""
    message = {"content": content, "role": "assistant"}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {"message": message}


def _tool_call(name: str, args: Dict[str, Any]):
    """Build a mock tool-call entry in OpenAI format."""
    return {
        "id": f"call_{name}_001",
        "function": {"name": name, "arguments": args},
    }


# =============================================================================
# Tests
# =============================================================================

class TestGreetingNoTools:
    """
    Verifies that the model-size-aware prompt system does not trigger tool
    calls for greetings or user instructions when using a mocked LLM.
    """

    @pytest.mark.unit
    @pytest.mark.parametrize("query,should_use_tools", GREETING_TEST_CASES + USER_INSTRUCTION_TEST_CASES)
    def test_greeting_no_tool_calls(
        self,
        query: str,
        should_use_tools: bool,
        mock_config,
        db,
        dialogue_memory,
    ):
        """Greetings and user instructions should not trigger tool calls."""
        from jarvis.reply.engine import run_reply_engine

        mock_config.ollama_chat_model = "gemma4:e2b"
        capture = ToolCallCapture()

        def mock_tool_run(db, cfg, tool_name, tool_args, **kwargs):  # noqa: F841 (shadows fixture)
            from jarvis.tools.types import ToolExecutionResult
            capture.record(tool_name, tool_args or {})
            return ToolExecutionResult(success=True, reply_text="Tool result")

        def mock_chat(base_url, chat_model, messages, timeout_sec, extra_options=None, tools=None, thinking=False):
            return _mock_llm_response("Hello! How can I help you today?")

        with patch('jarvis.reply.engine.run_tool_with_retries', side_effect=mock_tool_run), \
             patch('jarvis.reply.engine.chat_with_messages', side_effect=mock_chat), \
             patch('jarvis.reply.engine.extract_search_params_for_memory', return_value={"keywords": []}):

            run_reply_engine(
                db=db, cfg=mock_config, tts=None,
                text=query, dialogue_memory=dialogue_memory,
            )

        assert not capture.has_any_tool(), \
            f"Greeting '{query}' should NOT trigger tools. Called: {capture.tool_names()}"

    @pytest.mark.unit
    @pytest.mark.parametrize("query,should_use_tools", TOOL_REQUIRED_TEST_CASES)
    def test_tool_queries_still_work(
        self,
        query: str,
        should_use_tools: bool,
        mock_config,
        db,
        dialogue_memory,
    ):
        """Queries that require tools should still trigger them."""
        from jarvis.reply.engine import run_reply_engine

        mock_config.ollama_chat_model = "gemma4:e2b"
        capture = ToolCallCapture()

        def mock_tool_run(db, cfg, tool_name, tool_args, **kwargs):  # noqa: F841 (shadows fixture)
            from jarvis.tools.types import ToolExecutionResult
            capture.record(tool_name, tool_args or {})
            return ToolExecutionResult(success=True, reply_text="Weather: 20C sunny")

        call_count = 0

        def mock_chat(base_url, chat_model, messages, timeout_sec, extra_options=None, tools=None, thinking=False):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                if "weather" in query.lower():
                    return _mock_llm_response("", [_tool_call("getWeather", {"location": "here"})])
                elif "search" in query.lower() or "look up" in query.lower() or "news" in query.lower():
                    return _mock_llm_response("", [_tool_call("webSearch", {"search_query": query})])
                elif "eat" in query.lower() or "meal" in query.lower():
                    return _mock_llm_response("", [_tool_call("fetchMeals", {})])
            return _mock_llm_response("Here's the information you requested.")

        with patch('jarvis.reply.engine.run_tool_with_retries', side_effect=mock_tool_run), \
             patch('jarvis.reply.engine.chat_with_messages', side_effect=mock_chat), \
             patch('jarvis.reply.engine.extract_search_params_for_memory', return_value={"keywords": []}), \
             patch('jarvis.reply.engine.select_tools',
                   return_value=["webSearch", "getWeather", "fetchMeals", "stop"]):

            response = run_reply_engine(
                db=db, cfg=mock_config, tts=None,
                text=query, dialogue_memory=dialogue_memory,
            )

        assert capture.has_any_tool(), \
            f"Query '{query}' SHOULD trigger tools but didn't. Response: {response}"

    @pytest.mark.unit
    def test_thinking_only_response_continues_loop(
        self,
        mock_config,
        db,
        dialogue_memory,
    ):
        """A thinking-only response (no content, no tool call) should continue the loop, not break it."""
        from jarvis.reply.engine import run_reply_engine

        mock_config.ollama_chat_model = "gemma4:12b"
        call_count = 0

        def mock_chat(base_url, chat_model, messages, timeout_sec, extra_options=None, tools=None, thinking=False):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First turn: thinking only, no content, no tool call
                return {"message": {"content": "", "role": "assistant", "thinking": "Let me think about this..."}}
            # Second turn: actual response
            return _mock_llm_response("The answer is 42.")

        with patch('jarvis.reply.engine.chat_with_messages', side_effect=mock_chat), \
             patch('jarvis.reply.engine.extract_search_params_for_memory', return_value={"keywords": []}):

            response = run_reply_engine(
                db=db, cfg=mock_config, tts=None,
                text="what is the meaning of life",
                dialogue_memory=dialogue_memory,
            )

        assert call_count == 2, f"Expected 2 LLM calls (thinking + response), got {call_count}"
        assert response is not None
        assert "42" in response

    @pytest.mark.unit
    def test_all_tools_available_regardless_of_profile(
        self,
        mock_config,
        db,
        dialogue_memory,
    ):
        """All builtin tools should be available regardless of which profile is selected."""
        from jarvis.reply.engine import run_reply_engine

        mock_config.ollama_chat_model = "gemma4:e2b"
        capture = ToolCallCapture()

        def mock_tool_run(db, cfg, tool_name, tool_args, **kwargs):
            from jarvis.tools.types import ToolExecutionResult
            capture.record(tool_name, tool_args or {})
            return ToolExecutionResult(success=True, reply_text="Logged: pizza")

        call_count = 0

        def mock_chat(base_url, chat_model, messages, timeout_sec, extra_options=None, tools=None, thinking=False):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _mock_llm_response("", [_tool_call("logMeal", {"description": "pizza"})])
            return _mock_llm_response("Logged your meal!")

        # logMeal was previously restricted to "life" profile only — now all tools are always available
        with patch('jarvis.reply.engine.run_tool_with_retries', side_effect=mock_tool_run), \
             patch('jarvis.reply.engine.chat_with_messages', side_effect=mock_chat), \
             patch('jarvis.reply.engine.extract_search_params_for_memory', return_value={"keywords": []}):

            run_reply_engine(
                db=db, cfg=mock_config, tts=None,
                text="log that I had pizza for lunch",
                dialogue_memory=dialogue_memory,
            )

        assert capture.has_any_tool(), "logMeal should always be callable"
        assert "logMeal" in capture.tool_names()
