"""Preparation and staged planning respect their configured time budgets."""

from unittest.mock import patch

import pytest

from jarvis.memory.db import Database
from jarvis.reply import engine
from jarvis.reply.preparation import PreparedTurn


pytestmark = pytest.mark.unit


@pytest.fixture
def db(tmp_path):
    database = Database(str(tmp_path / "memory.db"))
    yield database
    database.conn.close()


def _reply(db, mock_config, dialogue_memory):
    with patch.object(
        engine, "chat_with_messages", return_value={"message": {"content": "Hello."}},
    ):
        return engine.run_reply_engine(db, mock_config, None, "hello", dialogue_memory)


@pytest.mark.parametrize("query_budget", [30.0, 2.0])
def test_combined_preparation_uses_routing_budget_capped_by_query(
    mock_config, db, dialogue_memory, query_budget,
):
    mock_config.agentic_preparation = "combined"
    mock_config.tool_selection_strategy = "llm"
    mock_config.llm_chat_model = "test-large"
    mock_config.db_path = db.db_path
    mock_config.llm_tools_timeout_sec = 12.0
    mock_config.planner_timeout_sec = 0.5
    mock_config.agentic_query_timeout_sec = query_budget
    received = []

    def prepare(**kwargs):
        received.append(kwargs["timeout_sec"])
        return PreparedTurn([], [], False, {})

    with patch("jarvis.reply.preparation.prepare_turn", side_effect=prepare):
        assert _reply(db, mock_config, dialogue_memory) == "Hello."

    assert len(received) == 1
    assert mock_config.planner_timeout_sec < received[0]
    assert received[0] <= min(mock_config.llm_tools_timeout_sec, query_budget)


def test_staged_planner_keeps_its_own_timeout(mock_config, db, dialogue_memory):
    mock_config.agentic_preparation = "staged"
    mock_config.tool_selection_strategy = "llm"
    mock_config.llm_chat_model = "test-large"
    mock_config.db_path = db.db_path
    mock_config.llm_tools_timeout_sec = 12.0
    mock_config.planner_timeout_sec = 0.75
    received = []

    def plan(**kwargs):
        received.append(kwargs["timeout_sec"])
        return ["Reply to the user."]

    with patch.object(engine, "select_tools", return_value=["getWeather"]), \
         patch.object(engine, "plan_query", side_effect=plan):
        assert _reply(db, mock_config, dialogue_memory) == "Hello."

    assert len(received) == 1
    assert 0 < received[0] <= mock_config.planner_timeout_sec
