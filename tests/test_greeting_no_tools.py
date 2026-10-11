"""Reply-loop controls with synthetic planner, router and chat responses.

Multilingual greetings and instructions exercise responses without tool calls.
Tool-bearing responses execute, and thinking-only turns continue to an answer.
These checks cover orchestration, not live-model intent accuracy.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List
from unittest.mock import patch

import pytest


pytestmark = pytest.mark.usefixtures("no_network")


@pytest.fixture(autouse=True)
def _isolate_decision_models(monkeypatch):
    from jarvis.reply import engine

    monkeypatch.setattr(engine, "plan_query", lambda *args, **kwargs: [])
    monkeypatch.setattr(engine, "select_tools", lambda **kwargs: list(kwargs["builtin_tools"]))


# =============================================================================
# Test Data
# =============================================================================

# Greetings in multiple languages - should NOT trigger tools
GREETING_TEST_CASES = [
    pytest.param("hello", id="Greeting: hello"),
    pytest.param("hi there", id="Greeting: hi there"),
    pytest.param("hey", id="Greeting: hey"),
    pytest.param("ni hao", id="Greeting: ni hao (Chinese)"),
    pytest.param("bonjour", id="Greeting: bonjour (French)"),
    pytest.param("hola", id="Greeting: hola (Spanish)"),
    pytest.param("merhaba", id="Greeting: merhaba (Turkish)"),
    pytest.param("ciao", id="Greeting: ciao (Italian)"),
    pytest.param("guten tag", id="Greeting: guten tag (German)"),
    pytest.param("how are you", id="Greeting: how are you"),
    pytest.param("thank you", id="Greeting: thank you"),
    pytest.param("thanks", id="Greeting: thanks"),
    pytest.param("goodbye", id="Greeting: goodbye"),
    pytest.param("good morning", id="Greeting: good morning"),
    pytest.param("good night", id="Greeting: good night"),
]

# User instructions about behaviour - should NOT trigger tools
USER_INSTRUCTION_TEST_CASES = [
    pytest.param("always use Celsius when telling me temperatures", id="Instruction: use Celsius"),
    pytest.param("remember to always tell me things in Celsius", id="Instruction: remember Celsius"),
    pytest.param("be more brief in your responses", id="Instruction: be more brief"),
    pytest.param("speak in French from now on", id="Instruction: speak in French"),
    pytest.param("always give me the short version", id="Instruction: short version"),
    pytest.param("don't use emojis in your responses", id="Instruction: no emojis"),
    pytest.param("note that I prefer metric units", id="Instruction: prefer metric"),
]

# Queries that SHOULD trigger tools
TOOL_REQUIRED_TEST_CASES = [
    pytest.param("what's the weather", id="Tool query: weather"),
    pytest.param("search for python tutorials", id="Tool query: web search"),
    pytest.param("what's the weather in Tokyo", id="Tool query: weather with location"),
    pytest.param("look up the news today", id="Tool query: news search"),
    pytest.param("what did I eat yesterday", id="Tool query: meal recall"),
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
    Responses without tool calls stay conversational; tool-bearing responses execute.
    """

    @pytest.mark.unit
    @pytest.mark.parametrize("query", GREETING_TEST_CASES + USER_INSTRUCTION_TEST_CASES)
    def test_greeting_no_tool_calls(
        self,
        query: str,
        mock_config,
        db,
        dialogue_memory,
    ):
        """Greetings and user instructions should not trigger tool calls."""
        from jarvis.reply.engine import run_reply_engine

        capture = ToolCallCapture()

        def mock_tool_run(db, cfg, tool_name, tool_args, **kwargs):  # noqa: F841 (shadows fixture)
            from jarvis.tools.types import ToolExecutionResult
            capture.record(tool_name, tool_args or {})
            return ToolExecutionResult(success=True, reply_text="Tool result")

        def mock_chat(*args, **kwargs):
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
    @pytest.mark.parametrize("query", TOOL_REQUIRED_TEST_CASES)
    def test_tool_queries_still_work(
        self,
        query: str,
        mock_config,
        db,
        dialogue_memory,
    ):
        """Queries that require tools should still trigger them."""
        from jarvis.reply.engine import run_reply_engine

        capture = ToolCallCapture()

        def mock_tool_run(db, cfg, tool_name, tool_args, **kwargs):  # noqa: F841 (shadows fixture)
            from jarvis.tools.types import ToolExecutionResult
            capture.record(tool_name, tool_args or {})
            return ToolExecutionResult(success=True, reply_text="Weather: 20C sunny")

        call_count = 0

        def mock_chat(*args, **kwargs):
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

        mock_config.llm_chat_model = "gemma4:12b"
        call_count = 0

        def mock_chat(*args, **kwargs):
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

        capture = ToolCallCapture()

        def mock_tool_run(db, cfg, tool_name, tool_args, **kwargs):
            from jarvis.tools.types import ToolExecutionResult
            capture.record(tool_name, tool_args or {})
            return ToolExecutionResult(success=True, reply_text="Logged: pizza")

        call_count = 0

        def mock_chat(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _mock_llm_response("", [_tool_call("logMeal", {"meal": "pizza"})])
            return _mock_llm_response("Logged your meal!")

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
