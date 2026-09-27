"""Combined preparation and current memory reach the actual reply loop."""
from unittest.mock import patch
import threading

import pytest

from jarvis.reply import engine
from jarvis.reply.preparation import PreparedTurn
from jarvis.memory.facts import FactStore
from jarvis.reply.task_state import TaskStore
from jarvis.tools.types import ToolExecutionResult

pytestmark = pytest.mark.unit


def test_resumed_plan_exposes_original_tools(mock_config, db, dialogue_memory):
    mock_config.agentic_preparation = "combined"
    mock_config.tool_selection_strategy = "llm"
    mock_config.llm_chat_model = "test-large"
    mock_config.db_path = db.db_path
    store = TaskStore(db.db_path)
    task = store.begin("Check London weather", ["getWeather location=London"])
    store.finish(task.task_id, status="partial", missing_info=["Weather still needed"])
    decision = PreparedTurn([], [], False, {}, resume_task_id=task.task_id)
    requests = []

    def chat(**kwargs):
        requests.append(kwargs)
        return {"message": {"content": "Which forecast date?"}}

    with patch("jarvis.reply.preparation.prepare_turn", return_value=decision), \
         patch.object(engine, "chat_with_messages", side_effect=chat):
        assert engine.run_reply_engine(db, mock_config, None, "continue", dialogue_memory)
    assert "getWeather" in {tool["function"]["name"] for tool in requests[0]["tools"]}


def test_resumed_dependent_step_resolves_from_bounded_prior_evidence(
    mock_config, db, dialogue_memory,
):
    mock_config.agentic_preparation = "combined"
    mock_config.tool_selection_strategy = "llm"
    mock_config.llm_chat_model = "gemma4:e2b"
    mock_config.agentic_tool_result_chars = 400
    mock_config.db_path = db.db_path
    store = TaskStore(db.db_path)
    task = store.begin("Check the weather where Alice lives", [
        "webSearch query='where Alice lives'",
        "getWeather location='<city from step 1>'",
    ])
    store.record_result(
        task.task_id, step_index=0, tool_name="webSearch", success=True,
        full_text="Alice lives in Paris. " + "Further source detail. " * 400,
        signature='webSearch:{"query":"where Alice lives"}', mutating=False,
    )
    store.finish(task.task_id, status="partial", missing_info=["Weather still needed"])
    decision = PreparedTurn([], [], False, {}, resume_task_id=task.task_id)
    resolver_contexts = []

    def resolve(**kwargs):
        resolver_contexts.append(kwargs["prior_results"])
        return None

    with patch("jarvis.reply.preparation.prepare_turn", return_value=decision), \
         patch.object(engine, "_resolve_plan_step", side_effect=resolve), \
         patch.object(engine, "chat_with_messages", return_value={"message": {"content": "I need the weather forecast."}}):
        engine.run_reply_engine(db, mock_config, None, "continue", dialogue_memory)

    assert resolver_contexts
    prior = resolver_contexts[0]
    assert any(name == "webSearch" and "Paris" in result for name, _, result in prior)
    assert all(len(result) <= mock_config.agentic_tool_result_chars for _, _, result in prior)


def test_out_of_order_tool_result_nudges_first_pending_plan_step(
    mock_config, db, dialogue_memory,
):
    mock_config.agentic_preparation = "combined"
    mock_config.tool_selection_strategy = "llm"
    mock_config.llm_chat_model = "gemma4:e2b"
    mock_config.db_path = db.db_path
    decision = PreparedTurn(["getWeather"], [
        "getWeather location=London",
        "getWeather location=Paris",
    ], False, {})
    follow_up_prompts = []

    def chat(**kwargs):
        if not follow_up_prompts:
            follow_up_prompts.append(None)
            return {"message": {"content": "", "tool_calls": [{
                "id": "call_paris", "type": "function",
                "function": {"name": "getWeather", "arguments": {"location": "Paris"}},
            }]}}
        follow_up_prompts.append(kwargs["messages"][-1]["content"])
        return {"message": {"content": "Paris weather checked."}}

    with patch("jarvis.reply.preparation.prepare_turn", return_value=decision), \
         patch.object(engine, "_resolve_plan_step", return_value=None), \
         patch.object(engine, "chat_with_messages", side_effect=chat), \
         patch.object(engine, "run_tool_with_retries", return_value=ToolExecutionResult(True, "Sunny in Paris")):
        engine.run_reply_engine(db, mock_config, None, "Compare London and Paris weather", dialogue_memory)

    assert len(follow_up_prompts) >= 2
    assert 'NEXT STEP: "getWeather location=London"' in follow_up_prompts[1]


@pytest.mark.parametrize("extra", [[], [None, {"function": {}}, {"function": {"name": "localFiles", "arguments": "not json"}}]])
def test_native_missing_ids_and_malformed_entries_have_complete_history(mock_config, db, dialogue_memory, extra):
    mock_config.llm_chat_model = "test-large"
    mock_config.db_path = db.db_path
    requests = []
    calls = []

    def chat(**kwargs):
        requests.append(kwargs)
        if len(requests) == 1:
            return {"message": {"content": "", "tool_calls": [
                {"function": {"name": "getWeather", "arguments": {"location": "London"}}}, *extra,
            ]}}
        return {"message": {"content": "Weather checked."}}

    def tool(**kwargs):
        calls.append(kwargs["tool_name"])
        return ToolExecutionResult(True, "Sunny")

    with patch.object(engine, "chat_with_messages", side_effect=chat), \
         patch.object(engine, "select_tools", return_value=["getWeather", "localFiles"]), \
         patch.object(engine, "plan_query", return_value=[]), \
         patch.object(engine, "run_tool_with_retries", side_effect=tool):
        reply = engine.run_reply_engine(db, mock_config, None, "London weather", dialogue_memory)
    assert reply == "Weather checked."
    assert calls == ["getWeather"]
    history = requests[1]["messages"]
    native = next(message["tool_calls"] for message in history if message.get("tool_calls"))
    results = [message for message in history if message["role"] == "tool"]
    assert len(native) == len(results) == 1
    assert native[0]["id"] == results[0]["tool_call_id"]


@pytest.fixture
def db(tmp_path):
    from jarvis.memory.db import Database
    database = Database(str(tmp_path / "memory.db"))
    yield database
    database.conn.close()


def test_combined_pass_skips_three_separate_inferences(mock_config, db, dialogue_memory):
    mock_config.agentic_preparation = "combined"
    mock_config.tool_selection_strategy = "llm"
    mock_config.llm_chat_model = "test-large"
    mock_config.db_path = db.db_path
    decision = PreparedTurn([], [], False, {})
    with patch("jarvis.reply.preparation.prepare_turn", return_value=decision) as prepare, \
         patch.object(engine, "select_tools") as route, \
         patch.object(engine, "plan_query") as plan, \
         patch.object(engine, "extract_search_params_for_memory") as extract, \
         patch.object(engine, "chat_with_messages", return_value={"message": {"content": "Hello."}}):
        result = engine.run_reply_engine(db, mock_config, None, "hello", dialogue_memory)
    assert result == "Hello."
    assert prepare.call_count == 1
    route.assert_not_called()
    plan.assert_not_called()
    extract.assert_not_called()


def test_combined_failure_uses_deterministic_recall_without_extra_model_passes(mock_config, db, dialogue_memory):
    mock_config.agentic_preparation = "combined"
    mock_config.tool_selection_strategy = "llm"
    mock_config.llm_chat_model = "test-large"
    mock_config.db_path = db.db_path
    mock_config.memory_enrichment_source = "all"
    with patch("jarvis.reply.preparation.prepare_turn", return_value=None), \
         patch.object(engine, "select_tools", return_value=[]) as route, \
         patch.object(engine, "plan_query") as plan, \
         patch.object(engine, "extract_search_params_for_memory") as extract, \
         patch("jarvis.memory.facts.recall_evidence", return_value="[User statement] My city is Paris.") as recall, \
         patch.object(engine, "chat_with_messages", return_value={"message": {"content": "Paris."}}) as chat:
        result = engine.run_reply_engine(db, mock_config, None, "what city did I mention", dialogue_memory)
    assert result == "Paris."
    assert route.call_args.kwargs["strategy"].value == "keyword"
    plan.assert_not_called()
    extract.assert_not_called()
    assert recall.called
    assert "My city is Paris" in chat.call_args.kwargs["messages"][0]["content"]


def test_cancel_during_preparation_prevents_main_model_and_memory(mock_config, db, dialogue_memory):
    mock_config.agentic_preparation = "combined"
    mock_config.tool_selection_strategy = "llm"
    mock_config.db_path = db.db_path
    cancelled = threading.Event()

    def prepare(**kwargs):
        cancelled.set()
        return PreparedTurn([], [], False, {})

    with patch("jarvis.reply.preparation.prepare_turn", side_effect=prepare), \
         patch.object(engine, "chat_with_messages") as chat:
        result = engine.run_reply_engine(db, mock_config, None, "hello", dialogue_memory,
                                        cancel_event=cancelled)
    assert result is None
    chat.assert_not_called()
    assert dialogue_memory.get_recent_messages() == []


def test_external_correction_invalidates_profile_in_same_conversation(mock_config, db, dialogue_memory):
    mock_config.db_path = db.db_path
    mock_config.llm_chat_model = "test-large"
    store = FactStore(db.db_path)
    fact = store.add_fact("I live in Paris", kind="user", owner="user", subject="user",
                          source_ref="test:old", source_type="dialogue", source_role="user",
                          source_channel="text", source_text="I live in Paris", evidence="I live in Paris",
                          observed_at="2026-01-01T00:00:00Z")
    prompts = []

    def chat(*args, **kwargs):
        prompts.append(kwargs["messages"][0]["content"])
        return {"message": {"content": "Hello."}}

    with patch.object(engine, "chat_with_messages", side_effect=chat), \
         patch.object(engine, "select_tools", return_value=[]), \
         patch.object(engine, "plan_query", return_value=["Reply to the user."]):
        engine.run_reply_engine(db, mock_config, None, "hello", dialogue_memory)
        store.correct_fact(fact["id"], text="I live in London", evidence="I live in London",
                           source_text="I live in London", source_ref="test:new")
        engine.run_reply_engine(db, mock_config, None, "hello again", dialogue_memory)
    store.close()
    assert "I live in Paris" in prompts[0]
    assert "I live in London" in prompts[1]
    assert "I live in Paris" not in prompts[1]
