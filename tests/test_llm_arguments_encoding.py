"""Provider requests encode tool arguments without rewriting caller history."""

from copy import deepcopy
import json
from unittest.mock import MagicMock

import pytest
import requests

from jarvis.llm.openai_compatible import OpenAICompatibleBackend

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("no_network")]


def _response(message):
    response = MagicMock(status_code=200)
    response.json.return_value = {"choices": [{"message": message}]}
    response.__enter__.return_value = response
    return response


@pytest.fixture
def post(monkeypatch):
    transport = MagicMock(return_value=_response({"role": "assistant", "content": "Done"}))
    monkeypatch.setattr("jarvis.llm.http.post", transport)
    return transport


def _call(arguments, ident="fixture"):
    return {"id": ident, "type": "function", "function": {
        "name": "fixtureTool", "arguments": arguments,
    }}


@pytest.mark.parametrize("argument_values", [
    pytest.param([{}], id="empty-object"),
    pytest.param([{"items": [{"enabled": True, "label": "été"}]}], id="nested-object"),
    pytest.param([None], id="null"),
    pytest.param([["fixture", 2]], id="array"),
    pytest.param([42], id="number"),
    pytest.param([True], id="boolean"),
    pytest.param([{"city": "Paris"}, {"city": "Tokyo"}], id="multiple-calls"),
])
def test_response_roundtrip_preserves_decoded_history_and_encodes_all_calls(post, argument_values):
    upstream = {"role": "assistant", "content": None, "tool_calls": [
        _call(json.dumps(arguments), f"fixture-{index}")
        for index, arguments in enumerate(argument_values)
    ]}
    saved_upstream = deepcopy(upstream)
    post.side_effect = [_response(upstream), _response({"role": "assistant", "content": "Done"})]
    backend = OpenAICompatibleBackend("http://localhost:1234/v1")
    decoded = backend.chat("fixture-model", [{"role": "user", "content": "Fixture request"}])
    saved_response = deepcopy(decoded)
    history = [decoded["message"], *[
        {"role": "tool", "tool_call_id": f"fixture-{index}", "content": "Synthetic result"}
        for index in range(len(argument_values))
    ]]
    saved_history = deepcopy(history)
    result = backend.chat("fixture-model", history)
    sent_calls = post.call_args.kwargs["json"]["messages"][0]["tool_calls"]
    sent_values = [json.loads(call["function"]["arguments"]) for call in sent_calls]
    assert sent_values == argument_values
    assert [type(value) for value in sent_values] == [type(value) for value in argument_values]
    assert result["message"]["content"] == "Done"
    assert decoded == saved_response
    assert history == saved_history
    assert upstream == saved_upstream


def test_encoded_string_keeps_its_literal_wire_value(post):
    arguments = '{"query": "été", "limit": 2}'
    history = [{"role": "assistant", "content": "", "tool_calls": [_call(arguments)]}]
    saved = deepcopy(history)
    OpenAICompatibleBackend("http://localhost:1234/v1").chat("fixture-model", history)
    sent = post.call_args.kwargs["json"]["messages"][0]["tool_calls"][0]
    assert sent["function"]["arguments"] == arguments
    assert history == saved


def test_messages_without_tool_calls_keep_their_values(post):
    history = [{"role": "user", "content": "Fixture request"}, {"role": "assistant", "content": "Fixture reply"}]
    saved = deepcopy(history)
    OpenAICompatibleBackend("http://localhost:1234/v1").chat("fixture-model", history)
    assert post.call_args.kwargs["json"]["messages"] == saved
    assert history == saved


@pytest.mark.parametrize("arguments", [{"items": [1, 2]}, None, ["fixture"]])
def test_connection_failure_preserves_caller_history(post, arguments):
    history = [{"role": "assistant", "content": "", "tool_calls": [_call(arguments)]}]
    saved = deepcopy(history)
    post.side_effect = requests.ConnectionError("Synthetic unavailable server")
    with pytest.raises(requests.ConnectionError, match="Synthetic unavailable server"):
        OpenAICompatibleBackend("http://localhost:1234/v1").chat("fixture-model", history)
    assert history == saved
