"""
Greeting No-Tools Evaluations (Live)

Live tests that verify greetings don't trigger tool calls with real LLM inference.
Mocked equivalents live in tests/test_greeting_no_tools.py as unit tests.

Run: ./scripts/run_evals.sh test_greeting
"""

import pytest
from unittest.mock import patch

from conftest import requires_judge_llm
from helpers import MockConfig, ToolCallCapture, create_mock_tool_run


def _assert_no_tools(capture, query, is_small, model_name):
    """Assert no tools were called; xfail for small models."""
    if capture.has_any_tool():
        if is_small:
            pytest.xfail(
                f"Small model {model_name} called tools for '{query}'. "
                f"Known limitation. Called: {capture.tool_names()}"
            )
        else:
            pytest.fail(
                f"Large model '{query}' should NOT trigger tools. "
                f"Called: {capture.tool_names()}"
            )


# =============================================================================
# Live Tests with Real LLM
# =============================================================================

def _is_small_model(model_name: str) -> bool:
    """Check if model is classified as small by the model size detector."""
    from jarvis.reply.prompts import detect_model_size, ModelSize
    return detect_model_size(model_name) == ModelSize.SMALL


class TestGreetingNoToolsLive:
    """
    Live tests with real LLM inference.

    These verify that the prompt changes actually work with real models.

    NOTE: Small models (1b-7b) may still incorrectly call tools for greetings
    despite explicit prompt constraints. This is a fundamental limitation of
    small model reasoning capacity. These tests document this behaviour.
    """

    @pytest.mark.eval
    @requires_judge_llm
    @pytest.mark.parametrize("query,should_use_tools", [
        pytest.param("hello", False, id="Live greeting: hello"),
        pytest.param("ni hao", False, id="Live greeting: ni hao (Chinese)"),
    ])
    def test_greeting_no_tools_live(
        self,
        query: str,
        should_use_tools: bool,
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

        with patch('jarvis.reply.engine.run_tool_with_retries',
                   side_effect=create_mock_tool_run(capture)):
            response = run_reply_engine(
                db=eval_db, cfg=mock_config, tts=None,
                text=query, dialogue_memory=eval_dialogue_memory
            )

        print(f"\n  Live Greeting Test ({JUDGE_MODEL}):")
        print(f"  Query: '{query}'")
        print(f"  Tools called: {capture.tool_names() or 'none'}")
        print(f"  Response: {(response or '')[:100]}...")
        print(f"  Model size: {'small' if is_small else 'large'}")

        # For greetings, we expect NO tool calls
        if not should_use_tools:
            _assert_no_tools(capture, query, is_small, JUDGE_MODEL)

    @pytest.mark.eval
    @requires_judge_llm
    @pytest.mark.parametrize("query,should_use_tools", [
        pytest.param("always use Celsius when telling me temperatures", False, id="Live instruction: use Celsius"),
        pytest.param("be more brief in your responses", False, id="Live instruction: be more brief"),
    ])
    def test_user_instructions_no_tools_live(
        self,
        query: str,
        should_use_tools: bool,
        mock_config,
        eval_db,
        eval_dialogue_memory
    ):
        """Live test: user instructions about behaviour should not trigger tool calls."""
        from jarvis.reply.engine import run_reply_engine
        from helpers import JUDGE_MODEL

        mock_config.ollama_base_url = "http://localhost:11434"
        mock_config.ollama_chat_model = JUDGE_MODEL

        is_small = _is_small_model(JUDGE_MODEL)

        capture = ToolCallCapture()

        with patch('jarvis.reply.engine.run_tool_with_retries',
                   side_effect=create_mock_tool_run(capture)):
            response = run_reply_engine(
                db=eval_db, cfg=mock_config, tts=None,
                text=query, dialogue_memory=eval_dialogue_memory
            )

        print(f"\n  Live User Instruction Test ({JUDGE_MODEL}):")
        print(f"  Query: '{query}'")
        print(f"  Tools called: {capture.tool_names() or 'none'}")
        print(f"  Response: {(response or '')[:100]}...")
        print(f"  Model size: {'small' if is_small else 'large'}")

        _assert_no_tools(capture, query, is_small, JUDGE_MODEL)

    @pytest.mark.eval
    @requires_judge_llm
    @pytest.mark.parametrize("query", [
        pytest.param("what do you know about the Possessor movie", id="Live unknown entity: Possessor (film)"),
        pytest.param("tell me about the book Piranesi", id="Live unknown entity: Piranesi (book)"),
    ])
    def test_unknown_named_entity_triggers_web_search_live(
        self,
        query: str,
        mock_config,
        eval_db,
        eval_dialogue_memory,
    ):
        """Live test: questions about specific named entities should trigger a web lookup.

        The model should recognise it has no concrete facts about the entity and call
        webSearch rather than denying knowledge or asking for a link.
        """
        from jarvis.reply.engine import run_reply_engine
        from helpers import JUDGE_MODEL

        mock_config.ollama_base_url = "http://localhost:11434"
        mock_config.ollama_chat_model = JUDGE_MODEL
        is_small = _is_small_model(JUDGE_MODEL)

        capture = ToolCallCapture()

        with patch('jarvis.reply.engine.run_tool_with_retries',
                   side_effect=create_mock_tool_run(capture, {
                       "webSearch": "Search result: relevant details about the requested entity.",
                       "fetchWebPage": "Page content: relevant details about the requested entity.",
                   })):
            response = run_reply_engine(
                db=eval_db, cfg=mock_config, tts=None,
                text=query, dialogue_memory=eval_dialogue_memory,
            )

        print(f"\n  Live Unknown-Entity Test ({JUDGE_MODEL}):")
        print(f"  Query: '{query}'")
        print(f"  Tools called: {capture.tool_names() or 'none'}")
        print(f"  Response: {(response or '')[:120]}...")
        print(f"  Model size: {'small' if is_small else 'large'}")

        if not capture.has_tool("webSearch"):
            msg = (
                f"Query about unknown named entity should trigger webSearch. "
                f"Called: {capture.tool_names() or 'none'}. Response: {(response or '')[:200]}"
            )
            if is_small:
                pytest.xfail(f"Small model {JUDGE_MODEL} did not call webSearch. {msg}")
            else:
                pytest.fail(msg)

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

        with patch('jarvis.reply.engine.run_tool_with_retries',
                   side_effect=create_mock_tool_run(capture, {
                       "getWeather": "Weather: 22C, partly cloudy",
                   })):
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
