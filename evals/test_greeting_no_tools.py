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
        pytest.param("hello", False, id="Greeting: hello"),
        pytest.param("ni hao", False, id="Greeting: ni hao (Chinese)"),
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
        pytest.param("always use Celsius when telling me temperatures", False, id="Instruction: use Celsius"),
        pytest.param("be more brief in your responses", False, id="Instruction: be more brief"),
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
        pytest.param("what do you know about the Possessor movie", id="Unknown entity: Possessor (film)"),
        pytest.param("tell me about the book Piranesi", id="Unknown entity: Piranesi (book)"),
        # Permission-framed phrasing. Regression: the small model previously
        # read "what can you tell me" as "tell me what you can do" and deflected
        # with "I can search the web if you'd like" instead of calling webSearch.
        pytest.param("what can you tell me about the movie Possessor", id="Unknown entity: permission-framed (Possessor)"),
        # "Have you heard of" is another common permission-framed variant.
        pytest.param("have you heard of the film Piranesi", id="Unknown entity: have-you-heard-of (Piranesi)"),
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
    def test_unknown_entity_with_poisoned_diary_still_triggers_web_search_live(
        self,
        mock_config,
        eval_db,
        eval_dialogue_memory,
    ):
        """Reproduces the Possessor field regression.

        A prior diary entry narrates the assistant's past deflection ("the assistant
        offered to search the web"). When the same entity is asked about again, the
        diary entry is retrieved as enrichment and — without the reference-only
        framing — the small model imitates the narrated deflection instead of
        calling webSearch.

        The defences this test guards:
          1. Summariser should not produce such entries in the first place (the
             seeded entry simulates a legacy poisoned summary from before the fix).
          2. The reply engine must frame the enrichment as reference-only so the
             model doesn't treat "the assistant offered to search" as a template.
        """
        from jarvis.reply.engine import run_reply_engine
        from helpers import JUDGE_MODEL

        mock_config.ollama_base_url = "http://localhost:11434"
        mock_config.ollama_chat_model = JUDGE_MODEL
        is_small = _is_small_model(JUDGE_MODEL)

        # Seed a poisoned diary entry — matches the shape of the real 2026-04-19
        # entry from the field failure. Uses the exact deflection phrasing we're
        # trying to stop the model from imitating.
        poisoned_summary = (
            '[2026-04-19] The conversation began with the user asking for information about '
            'the movie "Possessor." The assistant initially could not provide details. '
            'Subsequently, the user asked for details about "Possessor," prompting the '
            'assistant to state it lacked specific context and offer to search the web.'
        )

        # Also seed short-term dialogue memory with a prior deflection turn —
        # mirrors the real field session where the model had already said it
        # lacked info earlier in the same conversation, which then primes it
        # to repeat the same pattern on the follow-up.
        eval_dialogue_memory.add_message("user", "what do you know about the Possessor movie")
        eval_dialogue_memory.add_message(
            "assistant",
            "I don't have specific information about the film Possessor. "
            "I could search the web for it if you'd like.",
        )

        query = "tell me more about Possessor"
        capture = ToolCallCapture()

        # Patch the keyword search to guarantee the poisoned entry reaches the
        # system prompt. Going through the FTS/vector hybrid would make the test
        # flaky on seeded data that lacks vector embeddings.
        with patch(
            'jarvis.memory.conversation.search_conversation_memory_by_keywords',
            return_value=[poisoned_summary],
        ), patch(
            'jarvis.reply.engine.run_tool_with_retries',
            side_effect=create_mock_tool_run(capture, {
                "webSearch": "Search result: Possessor is a 2020 film directed by Brandon Cronenberg.",
                "fetchWebPage": "Page content: relevant details about the requested entity.",
            }),
        ):
            response = run_reply_engine(
                db=eval_db, cfg=mock_config, tts=None,
                text=query, dialogue_memory=eval_dialogue_memory,
            )

        print(f"\n  Live Poisoned-Diary Test ({JUDGE_MODEL}):")
        print(f"  Query: '{query}'")
        print(f"  Tools called: {capture.tool_names() or 'none'}")
        print(f"  Response: {(response or '')[:200]}...")
        print(f"  Model size: {'small' if is_small else 'large'}")

        if not capture.has_tool("webSearch"):
            msg = (
                f"With a poisoned diary entry narrating past deflection, the model still "
                f"must call webSearch. Called: {capture.tool_names() or 'none'}. "
                f"Response: {(response or '')[:300]}"
            )
            if is_small:
                pytest.xfail(f"Small model {JUDGE_MODEL} regressed under poisoned diary. {msg}")
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
