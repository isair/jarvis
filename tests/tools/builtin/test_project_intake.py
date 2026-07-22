"""Tests for the project intake tool: gate, template matching, full flow,
abandon phrase, and Obsidian write behaviour. See
src/jarvis/tools/builtin/project_intake.spec.md.
"""

from unittest.mock import Mock, patch

import pytest

from jarvis.tools.base import ToolContext
from jarvis.tools.builtin import project_intake as pi
from jarvis.tools.builtin.project_intake import (
    ProjectIntakeTool,
    get_gated_session,
    load_templates,
    match_template,
)


def _make_context(db, cfg, redacted_text=""):
    return ToolContext(
        db=db,
        cfg=cfg,
        system_prompt="",
        original_prompt=redacted_text,
        redacted_text=redacted_text,
        max_retries=1,
        user_print=lambda *_: None,
    )


@pytest.fixture(autouse=True)
def _templates_path(mock_config):
    import os
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    mock_config.project_templates_path = os.path.join(here, "project_templates.json")
    return mock_config.project_templates_path


class TestGate:
    """An active session forces the tool call; no session leaves normal
    routing untouched; a DB error on session lookup fails open."""

    def test_no_session_returns_none(self, db):
        assert get_gated_session(db) is None

    def test_active_session_returns_row(self, db):
        db.insert_intake_session()
        session = get_gated_session(db)
        assert session is not None
        assert session["status"] == "awaiting_type"

    def test_completed_session_does_not_gate(self, db):
        session_id = db.insert_intake_session()
        db.update_intake_session(session_id, status="completed")
        assert get_gated_session(db) is None

    def test_db_error_fails_open(self):
        broken_db = Mock()
        broken_db.get_active_intake_session.side_effect = RuntimeError("boom")
        assert get_gated_session(broken_db) is None


class TestTemplateMatching:
    def test_keyword_overlap_resolves_ties_toward_specific_template(self):
        templates = {
            "other": {"label": "Outro", "keywords": [], "questions": ["q"]},
            "generic": {"label": "Generic", "keywords": ["site", "web"], "questions": ["q1"]},
            "specific": {
                "label": "Specific",
                "keywords": ["site", "web", "loja online", "ecommerce"],
                "questions": ["q2"],
            },
        }
        # "site" appears in both — the template with the longer keyword
        # list (more specific) must win.
        assert match_template(templates, "quero um site novo") == "specific"

    def test_unmatched_input_falls_back_to_other(self, _templates_path):
        templates = load_templates(Mock(project_templates_path=_templates_path))
        assert match_template(templates, "xyzzy plugh qwerty") == "other"

    def test_accented_input_normalisation(self):
        templates = {
            "other": {"label": "Outro", "keywords": [], "questions": ["q"]},
            "website": {"label": "Site", "keywords": ["pagina web"], "questions": ["q1"]},
        }
        assert match_template(templates, "quero uma PÁGINA WEB") == "website"

    def test_malformed_config_falls_back_to_minimal_builtin(self):
        cfg = Mock(project_templates_path="/nonexistent/path/does_not_exist.json")
        templates = load_templates(cfg)
        assert "other" in templates
        assert templates["other"]["keywords"] == []
        assert len(templates["other"]["questions"]) == 4


class TestFullFlow:
    def test_awaiting_type_to_completed_one_question_per_turn(self, db, mock_config):
        tool = ProjectIntakeTool()

        # Turn 0: trigger — creates the session, asks for the type.
        result = tool.run({"input": "vamos começar um novo projeto"}, _make_context(db, mock_config))
        assert result.success is True
        assert "tipo de projeto" in result.reply_text.lower()

        # Turn 1: answer with a type — resolves template, returns Q1 only.
        with patch.object(pi, "write_brief_to_obsidian", return_value=True):
            result = tool.run({"input": "quero uma marca nova, branding"}, _make_context(db, mock_config))
        assert result.success is True
        first_question = result.reply_text
        session = db.get_active_intake_session()
        assert session["status"] == "in_progress"
        questions = pi.json.loads(session["questions_json"])
        assert first_question == questions[0]

        # Answer every remaining question one at a time; each turn must
        # return exactly one question until the last answer compiles the brief.
        answers = []
        for i, q in enumerate(questions):
            answers.append(f"resposta {i}")
            with patch.object(pi, "write_brief_to_obsidian", return_value=True):
                result = tool.run({"input": f"resposta {i}"}, _make_context(db, mock_config))
            if i < len(questions) - 1:
                assert result.reply_text == questions[i + 1]
            else:
                assert "concluído" in result.reply_text
                for q_text, a_text in zip(questions, answers):
                    assert f"{q_text}: {a_text}" in result.reply_text

        assert get_gated_session(db) is None


class TestAbandon:
    def test_abandon_phrase_stops_gate_from_firing_next_turn(self, db, mock_config):
        tool = ProjectIntakeTool()
        tool.run({"input": "vamos começar um novo projeto"}, _make_context(db, mock_config))
        assert get_gated_session(db) is not None

        result = tool.run({"input": "esquece o projeto"}, _make_context(db, mock_config))
        assert result.success is True
        assert "cancelei" in result.reply_text.lower()

        # Gate must not fire on the next turn.
        assert get_gated_session(db) is None

    def test_abandon_mid_interview_preserves_partial_answers(self, db, mock_config):
        tool = ProjectIntakeTool()
        tool.run({"input": "site novo"}, _make_context(db, mock_config))
        tool.run({"input": "site institucional"}, _make_context(db, mock_config))  # resolves type, asks Q1
        session_before = db.get_active_intake_session()
        assert session_before["status"] == "in_progress"

        tool.run({"input": "cancela isto"}, _make_context(db, mock_config))
        assert get_gated_session(db) is None


class TestRestartTriggerMidInterview:
    """Saying the start-a-new-project trigger phrase again while a session
    is already awaiting_type/in_progress must not be swallowed as free-text
    input to the current question — see project_intake.spec.md "Restarting
    mid-interview"."""

    def test_restart_phrase_during_awaiting_type_does_not_advance(self, db, mock_config):
        tool = ProjectIntakeTool()
        tool.run({"input": "vamos começar um novo projeto"}, _make_context(db, mock_config))
        session_before = db.get_active_intake_session()
        assert session_before["status"] == "awaiting_type"

        result = tool.run(
            {"input": "vamos começar um novo projeto"}, _make_context(db, mock_config)
        )
        assert result.success is True
        assert "projeto em curso" in result.reply_text.lower()
        assert "esquece o projeto" in result.reply_text.lower()

        session_after = db.get_active_intake_session()
        assert session_after["status"] == "awaiting_type"
        assert session_after["current_index"] == session_before["current_index"]
        assert session_after["answers_json"] == session_before["answers_json"]

    def test_restart_phrase_during_in_progress_does_not_advance(self, db, mock_config):
        tool = ProjectIntakeTool()
        tool.run({"input": "vamos começar um novo projeto"}, _make_context(db, mock_config))
        tool.run({"input": "site institucional"}, _make_context(db, mock_config))  # -> in_progress, Q1
        session_before = db.get_active_intake_session()
        assert session_before["status"] == "in_progress"

        result = tool.run(
            {"input": "vamos começar um novo projeto"}, _make_context(db, mock_config)
        )
        assert result.success is True
        assert "projeto em curso" in result.reply_text.lower()

        session_after = db.get_active_intake_session()
        assert session_after["current_index"] == session_before["current_index"]
        assert session_after["answers_json"] == session_before["answers_json"]


class TestStaleSessionExpiry:
    def test_stale_session_is_auto_abandoned_and_does_not_gate(self, db, mock_config):
        from datetime import datetime, timedelta, timezone

        mock_config.project_intake_stale_minutes = 30
        session_id = db.insert_intake_session()

        stale_time = (datetime.now(timezone.utc) - timedelta(minutes=31)).isoformat()
        with db._lock:
            db.conn.execute(
                "UPDATE project_intake_sessions SET updated_at = ? WHERE id = ?",
                (stale_time, session_id),
            )
            db.conn.commit()

        assert get_gated_session(db, mock_config) is None

        row = db.get_active_intake_session()
        assert row is None  # session was marked completed/abandoned

    def test_fresh_session_within_threshold_still_gates(self, db, mock_config):
        mock_config.project_intake_stale_minutes = 30
        db.insert_intake_session()
        assert get_gated_session(db, mock_config) is not None

    def test_missing_cfg_does_not_crash_and_uses_default_threshold(self, db):
        db.insert_intake_session()
        assert get_gated_session(db) is not None


class TestObsidianWrite:
    def _complete_a_session(self, db, mock_config):
        tool = ProjectIntakeTool()
        tool.run({"input": "novo projeto"}, _make_context(db, mock_config))
        tool.run({"input": "outro"}, _make_context(db, mock_config))
        session = db.get_active_intake_session()
        questions = pi.json.loads(session["questions_json"])
        result = None
        for i in range(len(questions)):
            result = tool.run({"input": f"resposta {i}"}, _make_context(db, mock_config))
        return result

    def test_success_writes_note_with_correct_frontmatter_and_body(self, db, mock_config):
        with patch.object(pi, "MCPClient") as mock_client_cls:
            mock_client = Mock()
            mock_client.invoke_tool.return_value = {"isError": False, "text": "ok"}
            mock_client_cls.return_value = mock_client

            result = self._complete_a_session(db, mock_config)

        assert result.success is True
        assert "falhou a gravação" not in result.reply_text

        call_kwargs = mock_client.invoke_tool.call_args.kwargs
        assert call_kwargs["server_name"] == pi.OBSIDIAN_MCP_SERVER
        assert call_kwargs["tool_name"] == pi.OBSIDIAN_WRITE_TOOL
        content = call_kwargs["arguments"]["content"]
        assert content.startswith("---\nstatus: active\n")
        assert "type: plan" in content
        assert "## Status" in content
        assert "Plano criado, desenvolvimento ainda não iniciado." in content

    def test_write_failure_returns_honest_message(self, db, mock_config):
        with patch.object(pi, "MCPClient") as mock_client_cls:
            mock_client = Mock()
            mock_client.invoke_tool.side_effect = pi.MCPServerSessionError("session lost")
            mock_client_cls.return_value = mock_client

            result = self._complete_a_session(db, mock_config)

        assert result.success is True  # the brief itself is still delivered
        assert "brief guardado localmente" in result.reply_text
        assert "falhou a gravação no Obsidian" in result.reply_text
        # Must not claim success it can't back up.
        assert "concluído" in result.reply_text
