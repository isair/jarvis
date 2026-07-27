"""End-to-end eval — the project intake trigger phrase must be routed to
the projectIntake tool by normal (LLM) tool selection, since starting a
new intake is the one point in the flow where tool selection depends on
the router/planner rather than the deterministic gate (see
project_intake.spec.md "Trigger detection").

This complements the deterministic unit/integration tests in
tests/tools/builtin/test_project_intake.py, which cover the gate,
template matching, full flow, abandon phrase, and Obsidian write without
needing a live LLM. This eval is the one case that genuinely needs the
router+planner to reliably pick the tool from natural language.

Run: EVAL_JUDGE_MODEL=gemma4:e2b ./scripts/run_evals.sh project_intake_flow
"""

import os

import pytest

from conftest import requires_judge_llm
from helpers import assert_not_fallback_reply, JUDGE_MODEL

_TEMPLATES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "project_templates.json",
)


@pytest.mark.eval
@requires_judge_llm
class TestProjectIntakeFlow:
    """Router must select projectIntake when the user says the trigger
    phrase with no active intake session, and the reply must be exactly
    the type question — not a paraphrase, not extra chatter."""

    def test_trigger_phrase_starts_intake_and_asks_type(
        self, mock_config, eval_db, eval_dialogue_memory,
    ):
        from jarvis.reply.engine import run_reply_engine

        mock_config.ollama_base_url = "http://localhost:11434"
        mock_config.ollama_chat_model = JUDGE_MODEL
        mock_config.project_templates_path = _TEMPLATES_PATH

        response = run_reply_engine(
            db=eval_db, cfg=mock_config, tts=None,
            text="vamos começar um novo projeto",
            dialogue_memory=eval_dialogue_memory,
        )

        print(f"\n  Project Intake Trigger ({JUDGE_MODEL}):")
        print(f"  Response: {(response or '')[:300]}")

        assert_not_fallback_reply(response, context="project-intake-trigger")

        session = eval_db.get_active_intake_session()
        assert session is not None, (
            "No project_intake_sessions row was created — the router did "
            f"not select projectIntake for the trigger phrase. Response: "
            f"{(response or '')[:400]}"
        )
        assert session["status"] == "awaiting_type"

        response_lower = (response or "").lower()
        assert "tipo de projeto" in response_lower, (
            "Reply does not ask which type of project this is, as required "
            f"by the intake flow. Response: {(response or '')[:400]}"
        )

    def test_second_turn_is_gated_deterministically_not_via_llm(
        self, mock_config, eval_db, eval_dialogue_memory,
    ):
        """Once a session is active, the follow-up turn must be forced to
        projectIntake by the gate — this should hold even with a judge
        model in the loop, proving the gate really does short-circuit the
        router/planner rather than relying on the model picking the tool
        again on its own."""
        from jarvis.reply.engine import run_reply_engine

        mock_config.ollama_base_url = "http://localhost:11434"
        mock_config.ollama_chat_model = JUDGE_MODEL
        mock_config.project_templates_path = _TEMPLATES_PATH

        eval_db.insert_intake_session()

        response = run_reply_engine(
            db=eval_db, cfg=mock_config, tts=None,
            text="branding",
            dialogue_memory=eval_dialogue_memory,
        )

        session = eval_db.get_active_intake_session()
        assert session is not None
        assert session["status"] == "in_progress"
        assert session["project_type"] == "branding"
        # Exactly one question relayed verbatim, no LLM paraphrase.
        assert response == response.strip()
