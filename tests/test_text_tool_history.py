"""Recognised textual tool calls retain complete request/result history."""
import copy
import json
from unittest.mock import patch

import pytest

from jarvis.reply import engine
from jarvis.tools.types import ToolExecutionResult

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("no_network")]


def _response(content, **fields):
    return {"message": {"role": "assistant", "content": content, **fields}}


def _text_call(arguments, shape, call_id="source-call", tool="getWeather"):
    if shape == "fence":
        return "```tool_call\n" + json.dumps({"name": tool, "arguments": arguments}) + "\n```"
    if shape == "function":
        return tool + "(" + json.dumps(arguments) + ")"
    return "tool_calls: " + json.dumps([{
        "id": call_id, "function": {"name": tool, "arguments": arguments},
    }])


def _assert_paired_results(messages):
    for index, message in enumerate(messages):
        if message.get("role") == "assistant" and message.get("tool_calls"):
            assert index + 1 < len(messages), "Request lost its outcome"
            following = messages[index + 1]
            assert following.get("role") == "tool" or (
                following.get("role") == "user" and following.get("tool_name")
            ), "Request lost its tagged outcome"
    outcomes = [(index, message) for index, message in enumerate(messages)
                if message.get("role") == "tool" or
                (message.get("role") == "user" and message.get("tool_name"))]
    for index, result in outcomes:
        assert index > 0, "Result has no preceding request"
        request = messages[index - 1]
        assert request["role"] == "assistant"
        calls = request.get("tool_calls")
        assert calls, "Result lost its recognised tool request"
        if result["role"] == "tool":
            matching = [call for call in calls if call.get("id") == result.get("tool_call_id")]
        else:
            matching = [call for call in calls if call["function"]["name"] == result["tool_name"]]
        assert len(matching) == 1, "Result does not identify exactly one request"
        assert isinstance(matching[0]["id"], str) and matching[0]["id"]
        assert result.get("tool_call_id") == matching[0]["id"]
    return outcomes


def _run(mock_config, db, dialogue_memory, responses, *, followup=False, success=True, route=None, parts=None):
    mock_config.planner_enabled = False
    mock_config.tool_result_digest_enabled = False
    mock_config.memory_digest_enabled = False
    mock_config.agentic_max_turns = 4
    snapshots, executed = [], []
    outputs = iter(responses)

    def chat(**kwargs):
        messages = copy.deepcopy(kwargs["messages"])
        snapshots.append(messages)
        # A native provider requires every tool result to answer a request.
        if any(message.get("role") == "tool" for message in messages):
            _assert_paired_results(messages)
        return next(outputs)

    def execute(db, cfg, tool_name, tool_args, **kwargs):
        executed.append((tool_name, tool_args))
        return ToolExecutionResult(success=success, reply_text="Recorded reading: 14 C." if success else "Reading unavailable.")

    with patch.object(engine, "select_tools", return_value=route if route is not None else ["getWeather", "stop"]), \
         patch.object(engine, "plan_query", return_value=[]), \
         patch.object(engine, "split_compound_query", return_value=parts or []), \
         patch.object(engine, "extract_search_params_for_memory", return_value={}), \
         patch.object(engine, "run_tool_with_retries", side_effect=execute), \
         patch.object(engine, "chat_with_messages", side_effect=chat):
        reply = engine.run_reply_engine(db, mock_config, None, "Check the weather.", dialogue_memory)
        if followup:
            reply = engine.run_reply_engine(db, mock_config, None, "Use that same recorded reading.", dialogue_memory)
    return reply, snapshots, executed


@pytest.mark.parametrize("model", ["gemma4:e2b", "gpt-oss:20b"])
@pytest.mark.parametrize("shape", ["literal", "fence", "function"])
def test_textual_request_and_result_survive_the_next_reply(mock_config, db, dialogue_memory, model, shape):
    mock_config.llm_chat_model = model
    arguments = {"location": "London"}
    reply, snapshots, executed = _run(mock_config, db, dialogue_memory, [
        _response(_text_call(arguments, shape)), _response("The reading is 14 C."),
        _response("The same recorded reading is 14 C."),
    ], followup=True)
    assert reply == "The same recorded reading is 14 C."
    assert executed == [("getWeather", arguments)]
    outcomes = _assert_paired_results(snapshots[-1])
    assert outcomes, "Follow-up lost the prior tool exchange"
    request = snapshots[-1][outcomes[0][0] - 1]
    assert request["tool_calls"][0]["function"]["arguments"] == arguments


@pytest.mark.parametrize("model", ["gemma4:e2b", "gpt-oss:20b"])
def test_invalid_textual_arguments_remain_failed_history_without_execution(mock_config, db, dialogue_memory, model):
    mock_config.llm_chat_model = model
    reply, snapshots, executed = _run(mock_config, db, dialogue_memory, [
        _response(_text_call("{broken", "literal")), _response("Please correct the supplied arguments."),
    ])
    assert reply == "Please correct the supplied arguments."
    assert executed == []
    outcomes = _assert_paired_results(snapshots[-1])
    assert outcomes and all(result["tool_failed"] for _, result in outcomes)


def test_textual_non_string_id_is_normalised_for_native_result_pairing(mock_config, db, dialogue_memory):
    mock_config.llm_chat_model = "gpt-oss:20b"
    reply, snapshots, _ = _run(mock_config, db, dialogue_memory, [
        _response(_text_call({"location": "London"}, "literal", call_id=7)),
        _response("The reading is 14 C."),
    ])
    assert reply == "The reading is 14 C."
    assert _assert_paired_results(snapshots[-1])


def test_ordinary_prose_is_not_recorded_as_a_tool_request(mock_config, db, dialogue_memory):
    reply, snapshots, executed = _run(mock_config, db, dialogue_memory, [_response("A normal reply.")])
    assert reply == "A normal reply."
    assert executed == []
    assert not any(message.get("tool_calls") for message in snapshots[-1])


def test_existing_native_request_is_preserved(mock_config, db, dialogue_memory):
    mock_config.llm_chat_model = "gpt-oss:20b"
    call = {"id": "native-call", "type": "function", "function": {
        "name": "getWeather", "arguments": {"location": "London"},
    }}
    reply, snapshots, _ = _run(mock_config, db, dialogue_memory, [
        _response("", tool_calls=[call]), _response("The reading is 14 C."),
    ])
    assert reply == "The reading is 14 C."
    outcomes = _assert_paired_results(snapshots[-1])
    assert snapshots[-1][outcomes[0][0] - 1]["tool_calls"] == [call]


@pytest.mark.parametrize("model", ["gemma4:e2b", "gpt-oss:20b"])
@pytest.mark.parametrize("success", [False, True])
def test_duplicate_guidance_retains_pairing_and_the_recorded_outcome(mock_config, db, dialogue_memory, model, success):
    mock_config.llm_chat_model = model
    request = _response(_text_call({"location": "London"}, "literal"))
    reply, snapshots, executed = _run(mock_config, db, dialogue_memory, [
        request, request, _response("The recorded outcome is available."),
    ], success=success)
    assert reply == "The recorded outcome is available."
    assert executed == [("getWeather", {"location": "London"})]
    outcomes = _assert_paired_results(snapshots[-1])
    assert [result["tool_failed"] for _, result in outcomes] == [not success, not success]


@pytest.mark.parametrize("model", ["gemma4:e2b", "gpt-oss:20b"])
@pytest.mark.parametrize("blocked", ["not_allowed", "search_cap"])
def test_blocked_calls_keep_a_failed_request_outcome(mock_config, db, dialogue_memory, model, blocked):
    mock_config.llm_chat_model = model
    mock_config.tool_search_max_calls = 0
    tool = "toolSearchTool" if blocked == "search_cap" else "getWeather"
    arguments = {"query": "a local lookup"} if blocked == "search_cap" else {"location": "London"}
    reply, snapshots, executed = _run(mock_config, db, dialogue_memory, [
        _response(_text_call(arguments, "literal", tool=tool)), _response("This request was not executed."),
    ], route=["stop"] if blocked == "not_allowed" else None)
    assert reply == "This request was not executed."
    assert executed == []
    outcomes = _assert_paired_results(snapshots[-1])
    assert outcomes and all(result["tool_failed"] for _, result in outcomes)


@pytest.mark.parametrize("model", ["gemma4:e2b", "gpt-oss:20b"])
def test_reused_textual_ids_do_not_alias_distinct_requests(mock_config, db, dialogue_memory, model):
    mock_config.llm_chat_model = model
    first, second = {"location": "London"}, {"location": "Paris"}
    reply, snapshots, executed = _run(mock_config, db, dialogue_memory, [
        _response(_text_call(first, "literal")),
        _response(_text_call(second, "literal")),
        _response("Both readings are recorded."),
    ])
    assert reply == "Both readings are recorded."
    assert executed == [("getWeather", first), ("getWeather", second)]
    _assert_paired_results(snapshots[-1])
    identities = [call["id"] for message in snapshots[-1] if message.get("role") == "assistant"
                  for call in message.get("tool_calls", [])]
    assert len(set(identities)) == len(identities), "Distinct requests share a result identity"


def test_duplicate_feedback_keeps_all_requested_outcomes_visible(mock_config, db, dialogue_memory):
    mock_config.llm_chat_model = "gemma4:e2b"
    first, second = {"location": "London"}, {"location": "Paris"}
    repeated = _response(_text_call(first, "literal"))
    reply, snapshots, executed = _run(mock_config, db, dialogue_memory, [
        repeated, repeated, _response(_text_call(second, "literal")),
        _response("Two readings are recorded."),
    ], parts=["London reading", "Paris reading", "Ankara reading"])
    assert reply == "Two readings are recorded."
    assert executed == [("getWeather", first), ("getWeather", second)]
    assert _assert_paired_results(snapshots[-1])
    feedback = snapshots[-1][-1]["content"]
    for part in ("London reading", "Paris reading", "Ankara reading"):
        assert part in feedback
    assert "You have answered" not in feedback
    assert "Still unanswered:" not in feedback


def test_textual_outcome_identity_matches_the_request(mock_config, db, dialogue_memory):
    mock_config.llm_chat_model = 'gemma4:e2b'
    _, snapshots, _ = _run(mock_config, db, dialogue_memory, [
        _response(_text_call({'location': 'London'}, 'literal')),
        _response('The recorded reading is 14 C.'),
    ])
    index, outcome = _assert_paired_results(snapshots[-1])[0]
    assert outcome.get('tool_call_id') == snapshots[-1][index - 1]['tool_calls'][0]['id']
    diagnostics = []
    with patch.object(engine, 'debug_log', side_effect=lambda message, *args: diagnostics.append(message)):
        engine._previous_turn_failed_tool_names(snapshots[-1])
    assert not any('have no matching' in message for message in diagnostics)
