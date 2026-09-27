"""Performance reports state what the harness actually observed."""

import json
from types import SimpleNamespace

from jarvis.reply import engine as reply_engine

from tests.performance import test_pipeline_timings as harness
from tests.performance.timing_recorder import TimingRecorder


def test_report_keeps_unobserved_output_latency_unknown(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "PERF_REPORT_DIR", tmp_path)
    recorder = TimingRecorder()

    path = harness._write_report(
        recorder,
        "pipeline",
        end_to_end_sec=[0.3, 0.7],
        unnecessary_tool_calls=2,
        preparation_mode="staged",
        warm_condition="one measured tiny call before pipeline; prior server state unknown",
    )
    report = json.loads(path.read_text())

    assert report["end_to_end_sec"]["p95"] == 0.7
    assert report["first_useful_text_sec"] is None
    assert report["first_useful_spoken_sec"] is None
    assert report["unnecessary_tool_calls"] == 2
    assert report["preparation_mode"] == "staged"
    assert "measured tiny call" in report["warm_condition"]


def test_pipeline_config_pins_model_and_preparation_mode(monkeypatch):
    monkeypatch.setattr(harness, "PERF_MODEL", "local-model")
    monkeypatch.setattr(harness, "PERF_PREPARATION", "combined")
    monkeypatch.setattr(harness, "PROVIDER_STATUS", SimpleNamespace(provider="ollama"))

    cfg = harness._make_cfg()

    assert cfg.ollama_chat_model == "local-model"
    assert cfg.llm_chat_model == "local-model"
    assert cfg.agentic_preparation == "combined"
    assert cfg.tool_selection_strategy == "llm"


def test_pipeline_uses_disposable_fresh_state_and_never_dispatches_real_tools(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "PERF_RUNS", 2)
    monkeypatch.setattr(harness, "PIPELINE_QUERIES", ["hello", "what time is it in Tokyo?"])
    monkeypatch.setattr(harness, "PERF_PREPARATION", "combined")
    observed = []
    warmups = []
    reports = []

    class Backend:
        def direct(self, **kwargs):
            warmups.append(kwargs)
            return "OK"

    class Recorder:
        def __init__(self):
            self.calls = [SimpleNamespace(context="main_chat_turn")]

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def print_report(self, **kwargs):
            pass

    def fake_tool(*args, **kwargs):
        raise AssertionError("real tool dispatch must not occur")

    def fake_reply(db, cfg, tts, query, dialogue):
        result = reply_engine.run_tool_with_retries(
            db, cfg, "webSearch", {}, "system", query, query,
        )
        assert not result.success
        observed.append((db.db_path, cfg.db_path, dialogue, query))
        return "fixture reply"

    monkeypatch.setattr("jarvis.llm.factory.get_llm_backend", lambda cfg: Backend())
    monkeypatch.setattr(reply_engine, "run_reply_engine", fake_reply)
    monkeypatch.setattr(reply_engine, "run_tool_with_retries", fake_tool)
    monkeypatch.setattr(harness, "TimingRecorder", Recorder)
    monkeypatch.setattr(harness, "_write_report", lambda rec, name, **kwargs: reports.append((name, kwargs)) or tmp_path / "report.json")

    harness.test_pipeline_timings_by_context(tmp_path)

    assert len(warmups) == 1
    assert len(observed) == len(harness.PIPELINE_QUERIES) * harness.PERF_RUNS
    assert all(db_path == cfg_path for db_path, cfg_path, _, _ in observed)
    assert all(db_path.startswith(str(tmp_path)) for db_path, _, _, _ in observed)
    assert len({db_path for db_path, _, _, _ in observed}) == len(observed)
    assert len({id(dialogue) for _, _, dialogue, _ in observed}) == len(observed)
    assert reports[0][0] == "pipeline-combined"
    assert reports[0][1]["unnecessary_tool_calls"] == harness.PERF_RUNS
    assert reports[0][1]["initial_call_sec"] is not None
