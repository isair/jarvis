"""
Greeting No-Tools Evaluations (Live)

Live tests that verify greetings don't trigger tool calls with real LLM inference.
Mocked equivalents live in tests/test_greeting_no_tools.py as unit tests.

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
    is_judge_llm_available,
)

_JUDGE_LLM_AVAILABLE = is_judge_llm_available()
requires_judge_llm = pytest.mark.skipif(
    not _JUDGE_LLM_AVAILABLE,
    reason="Judge LLM not available"
)


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
        pytest.param("bonjour", False, id="Live greeting: bonjour (French)"),
        pytest.param("how are you", False, id="Live greeting: how are you"),
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
        print(f"  Query: '{query}'")
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
    @pytest.mark.parametrize("query,should_use_tools", [
        pytest.param("always use Celsius when telling me temperatures", False, id="Live instruction: use Celsius"),
        pytest.param("remember to always tell me things in Celsius", False, id="Live instruction: remember Celsius"),
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

        def mock_tool_run(db, cfg, tool_name, tool_args, **kwargs):
            from jarvis.tools.types import ToolExecutionResult
            capture.record(tool_name, tool_args or {})
            return ToolExecutionResult(success=True, reply_text="Tool result")

        with patch('jarvis.reply.engine.run_tool_with_retries', side_effect=mock_tool_run):
            response = run_reply_engine(
                db=eval_db, cfg=mock_config, tts=None,
                text=query, dialogue_memory=eval_dialogue_memory
            )

        print(f"\n  Live User Instruction Test ({JUDGE_MODEL}):")
        print(f"  Query: '{query}'")
        print(f"  Tools called: {capture.tool_names() or 'none'}")
        print(f"  Response: {(response or '')[:100]}...")
        print(f"  Model size: {'small' if is_small else 'large'}")

        if capture.has_any_tool():
            if is_small:
                pytest.xfail(
                    f"Small model {JUDGE_MODEL} called tools for instruction '{query}'. "
                    f"This is a known limitation of small models. Called: {capture.tool_names()}"
                )
            else:
                pytest.fail(
                    f"Large model instruction '{query}' should NOT trigger tools. "
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
