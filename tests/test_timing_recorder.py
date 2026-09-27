"""The timing harness observes backend calls on either local provider."""

import threading
from types import SimpleNamespace

import pytest

from jarvis.llm.ollama import OllamaBackend
from jarvis.llm.openai_compatible import OpenAICompatibleBackend
from jarvis.reply import engine
from jarvis.reply.execution import ExecutionControl
from tests.performance.timing_recorder import TimingRecorder, _Call


@pytest.mark.parametrize("backend_type", [OllamaBackend, OpenAICompatibleBackend])
def test_records_backend_call_and_failure(monkeypatch, backend_type):
    def fake_direct(self, chat_model, system_prompt, user_content, **kwargs):
        if user_content == "fail":
            raise RuntimeError("test failure")
        return "answer"

    monkeypatch.setattr(backend_type, "direct", fake_direct)
    backend = backend_type("http://127.0.0.1:1")

    with TimingRecorder() as recorder:
        assert backend.direct("local-model", "system", "ok") == "answer"
        with pytest.raises(RuntimeError, match="test failure"):
            backend.direct("local-model", "system", "fail")

    assert len(recorder.calls) == 2
    assert [call.outcome for call in recorder.calls] == ["success", "error"]
    assert all(call.model == "local-model" for call in recorder.calls)
    assert all(call.provider == backend_type.__name__ for call in recorder.calls)


def test_p95_uses_observed_upper_tail():
    recorder = TimingRecorder(calls=[
        _Call("main_chat_turn", value, "local-model", 1, 1)
        for value in (1.0, 2.0, 3.0, 4.0)
    ])

    assert recorder.p50("main_chat_turn") == 2.5
    assert recorder.p95("main_chat_turn") == 4.0


@pytest.mark.parametrize("message,outcome", [
    ({"content": ""}, "empty"),
    ({"content": "   "}, "empty"),
    ({"content": "answer"}, "success"),
    ({"content": "", "tool_calls": [{"function": {"name": "getTime", "arguments": {}}}]}, "success"),
])
def test_empty_chat_is_not_a_successful_latency_sample(monkeypatch, message, outcome):
    monkeypatch.setattr(OllamaBackend, "chat", lambda self, *args, **kwargs: {"message": message})
    with TimingRecorder() as recorder:
        result = OllamaBackend("http://127.0.0.1:1").chat("model", [])
    assert result == {"message": message}
    assert recorder.calls[0].outcome == outcome


def test_worker_thread_main_chat_is_named_and_counts_message_content(monkeypatch):
    def fake_chat(self, chat_model, messages, **kwargs):
        return {"message": {"role": "assistant", "content": "Hello."}}

    monkeypatch.setattr(OllamaBackend, "chat", fake_chat)
    backend = OllamaBackend("http://127.0.0.1:1")
    monkeypatch.setattr(engine, "get_llm_backend", lambda cfg: backend)
    control = ExecutionControl(threading.Event(), timeout_sec=2)
    cfg = SimpleNamespace(llm_chat_model="local-model")

    with TimingRecorder() as recorder:
        result = control.call(engine.chat_with_messages, cfg, [
            {"role": "user", "content": "hi"},
        ])

    assert result["message"]["content"] == "Hello."
    assert [(call.context, call.response_chars) for call in recorder.calls] == [
        ("main_chat_turn", len("Hello.")),
    ]


def test_new_llm_contexts_are_reported_by_name(monkeypatch):
    monkeypatch.setattr(OllamaBackend, "direct", lambda self, *args, **kwargs: "ok")
    backend = OllamaBackend("http://127.0.0.1:1")

    def prepare_turn():
        return backend.direct("model", "system", "user")

    def plan_query():
        return backend.direct("model", "system", "user")

    def resolve_next_tool_call():
        return backend.direct("model", "system", "user")

    def ingest_dialogue_facts():
        return backend.direct("model", "system", "user")

    def merge_node_data():
        return backend.direct("model", "system", "user")

    def _select_llm():
        return backend.direct("model", "system", "user")

    with TimingRecorder() as recorder:
        for call in (prepare_turn, plan_query, resolve_next_tool_call, ingest_dialogue_facts, merge_node_data, _select_llm):
            assert call() == "ok"

    assert [call.context for call in recorder.calls] == [
        "turn_preparation", "planner", "plan_step_resolver", "fact_extraction", "graph_node_merge", "tool_router",
    ]


def test_delegating_direct_call_records_one_backend_request(monkeypatch):
    def fake_chat(self, chat_model, messages, **kwargs):
        return {"message": {"content": "answer"}}

    def delegated_direct(self, chat_model, system_prompt, user_content, **kwargs):
        result = self.chat(chat_model, [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ])
        return result["message"]["content"]

    monkeypatch.setattr(OllamaBackend, "chat", fake_chat)
    monkeypatch.setattr(OllamaBackend, "direct", delegated_direct)
    backend = OllamaBackend("http://127.0.0.1:1")

    with TimingRecorder() as recorder:
        assert backend.direct("model", "system", "user") == "answer"

    assert len(recorder.calls) == 1
    assert recorder.calls[0].response_chars == len("answer")
