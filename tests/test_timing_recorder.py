"""The timing harness observes backend calls on either local provider."""

import pytest

from jarvis.llm.ollama import OllamaBackend
from jarvis.llm.openai_compatible import OpenAICompatibleBackend
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
