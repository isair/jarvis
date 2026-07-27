"""Engine-level wiring tests for the project intake gate: an active session
must force the projectIntake tool call and skip planner/router entirely;
no active session must leave normal routing untouched; a DB error on
session lookup must fail open. See project_intake.spec.md "The gate".
"""

from unittest.mock import patch

import pytest


class _RouterReached(Exception):
    """Raised by a stubbed select_tools() to prove control reached normal
    routing — i.e. the gate did NOT short-circuit this turn."""


def _raise_router_reached(*args, **kwargs):
    raise _RouterReached("router reached")


class TestProjectIntakeGateWiring:
    def test_active_session_forces_tool_call_and_skips_planner(self, db, mock_config, dialogue_memory):
        from jarvis.reply import engine as engine_mod
        from jarvis.tools.types import ToolExecutionResult

        db.insert_intake_session()

        captured = {}

        def _fake_run_tool_with_retries(db, cfg, tool_name, tool_args, **kwargs):
            captured["tool_name"] = tool_name
            captured["tool_args"] = tool_args
            return ToolExecutionResult(success=True, reply_text="Que tipo de projeto é este?")

        with patch.object(engine_mod, "run_tool_with_retries", side_effect=_fake_run_tool_with_retries), \
             patch.object(engine_mod, "plan_query") as mock_plan, \
             patch.object(engine_mod, "select_tools") as mock_select:
            reply = engine_mod.run_reply_engine(
                db=db, cfg=mock_config, tts=None,
                text="site institucional",
                dialogue_memory=dialogue_memory,
            )

        assert captured["tool_name"] == "projectIntake"
        assert captured["tool_args"] == {"input": "site institucional"}
        assert reply == "Que tipo de projeto é este?"
        mock_plan.assert_not_called()
        mock_select.assert_not_called()

    def test_no_active_session_reaches_normal_routing(self, db, mock_config, dialogue_memory):
        from jarvis.reply import engine as engine_mod

        with patch.object(engine_mod, "select_tools", side_effect=_raise_router_reached):
            with pytest.raises(_RouterReached):
                engine_mod.run_reply_engine(
                    db=db, cfg=mock_config, tts=None,
                    text="olá",
                    dialogue_memory=dialogue_memory,
                )

    def test_db_error_on_session_lookup_fails_open(self, mock_config, dialogue_memory):
        from jarvis.reply import engine as engine_mod

        class _BrokenDB:
            def get_active_intake_session(self):
                raise RuntimeError("db unavailable")

        with patch.object(engine_mod, "select_tools", side_effect=_raise_router_reached):
            with pytest.raises(_RouterReached):
                engine_mod.run_reply_engine(
                    db=_BrokenDB(), cfg=mock_config, tts=None,
                    text="olá",
                    dialogue_memory=dialogue_memory,
                )

    def test_disabled_config_skips_gate_entirely(self, db, mock_config, dialogue_memory):
        from jarvis.reply import engine as engine_mod

        db.insert_intake_session()
        mock_config.project_intake_enabled = False

        with patch.object(engine_mod, "select_tools", side_effect=_raise_router_reached):
            with pytest.raises(_RouterReached):
                engine_mod.run_reply_engine(
                    db=db, cfg=mock_config, tts=None,
                    text="olá",
                    dialogue_memory=dialogue_memory,
                )
