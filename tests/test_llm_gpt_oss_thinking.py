"""Ollama GPT-OSS requests use supported thinking levels on the wire."""

import json
from unittest.mock import MagicMock, patch

import pytest

from jarvis.llm import OllamaBackend


pytestmark = pytest.mark.unit


def _response(*, content="Answer", stream=False):
    response = MagicMock()
    response.__enter__.return_value = response
    response.__exit__.return_value = None
    response.json.return_value = {"message": {"content": content}}
    if stream:
        response.iter_lines.return_value = [
            json.dumps({"message": {"thinking": "Reasoning", "content": ""}}).encode(),
            json.dumps({"message": {"content": content}}).encode(),
        ]
    return response


@pytest.mark.parametrize(
    ("model", "thinking", "expected"),
    [
        ("gpt-oss", False, "low"),
        ("gpt-oss:20b", True, "high"),
        ("local/gpt-oss:120b", False, "low"),
        ("registry.example/models/gpt-oss:20b", True, "high"),
        ("gpt-oss-safeguard:20b", False, False),
        ("my-gpt-oss:20b", True, True),
        ("gemma4:e2b", False, False),
    ],
)
def test_direct_serialises_thinking_and_delivers_answer(model, thinking, expected):
    with patch("jarvis.llm.requests.post", return_value=_response()) as post:
        result = OllamaBackend("http://localhost:11434").direct(
            model, "system", "question", thinking=thinking,
        )

    assert post.call_args.kwargs["json"]["think"] == expected
    assert result == "Answer"


@pytest.mark.parametrize(("thinking", "expected"), [(False, "low"), (True, "high")])
def test_streaming_serialises_thinking_and_delivers_only_answer(thinking, expected):
    seen = []
    with patch("jarvis.llm.requests.post", return_value=_response(stream=True)) as post:
        result = OllamaBackend("http://localhost:11434").streaming(
            "gpt-oss:20b", "system", "question", on_token=seen.append,
            thinking=thinking,
        )

    assert post.call_args.kwargs["json"]["think"] == expected
    assert result == "Answer"
    assert seen == ["Answer"]


@pytest.mark.parametrize(("thinking", "expected"), [(False, "low"), (True, "high")])
def test_chat_serialises_thinking_and_delivers_response(thinking, expected):
    with patch("jarvis.llm.requests.post", return_value=_response()) as post:
        result = OllamaBackend("http://localhost:11434").chat(
            "gpt-oss:20b", [{"role": "user", "content": "question"}],
            thinking=thinking,
        )

    assert post.call_args.kwargs["json"]["think"] == expected
    assert result == {"message": {"content": "Answer"}}


@pytest.mark.parametrize(
    ("model", "override", "expected"),
    [
        ("gpt-oss:20b", False, "low"),
        ("gpt-oss:20b", True, "high"),
        ("gpt-oss:20b", "medium", "medium"),
        ("gemma4:e2b", False, False),
    ],
)
def test_chat_normalises_extra_options_think_after_override(model, override, expected):
    with patch("jarvis.llm.requests.post", return_value=_response()) as post:
        result = OllamaBackend("http://localhost:11434").chat(
            model, [{"role": "user", "content": "question"}],
            thinking=True, extra_options={"think": override},
        )

    assert post.call_args.kwargs["json"]["think"] == expected
    assert result == {"message": {"content": "Answer"}}
