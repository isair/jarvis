"""Behaviour checks for benchmark outcomes and eval reporting."""

from types import SimpleNamespace

from evals import conftest as eval_hooks
from evals import helpers as eval_helpers
from evals.benchmark_report import Attempt, Scenario, format_markdown, summarise_scenarios
from evals.helpers import MockConfig


def test_first_attempt_and_recovery_are_separate_outcomes():
    scenarios = [
        Scenario("direct", "offline", [Attempt(True)]),
        Scenario("recovered", "offline", [Attempt(False), Attempt(True)]),
        Scenario("failed", "offline", [Attempt(False), Attempt(False)]),
        Scenario("unavailable", "live", [], availability="unavailable"),
    ]

    result = summarise_scenarios(scenarios)

    assert result["offline"]["eligible"] == 3
    assert result["offline"]["first_attempt_success"] == 1
    assert result["offline"]["recovered_success"] == 1
    assert result["offline"]["final_success"] == 2
    assert result["live"]["availability"] == "unavailable"
    assert result["live"]["eligible"] == 0
    markdown = format_markdown(result)
    assert "First attempt" in markdown
    assert "Recovered" in markdown
    assert "unavailable" in markdown


def test_scenario_metrics_keep_failure_types_and_latency_distinct():
    scenario = Scenario(
        "grounding",
        "live",
        [Attempt(False, unnecessary_tools=2, stale_retrieval=True), Attempt(True)],
        stage_durations={"preparation": [0.1, 0.2, 0.3]},
        end_to_end_sec=0.5,
        first_useful_text_sec=0.4,
    )

    report = summarise_scenarios([scenario])["live"]

    assert report["unnecessary_tool_calls"] == 2
    assert report["stale_retrieval_errors"] == 1
    assert report["stage_latency_sec"]["preparation"]["p50"] == 0.2
    assert report["end_to_end_sec"]["p50"] == 0.5
    assert report["first_useful_text_sec"]["p50"] == 0.4
    assert report["first_useful_spoken_sec"] is None


def test_setup_skip_is_recorded_as_skip_not_success(monkeypatch):
    report = eval_hooks.EvalReport()
    monkeypatch.setattr(eval_hooks, "_eval_report", report)
    event = SimpleNamespace(
        when="setup",
        outcome="skipped",
        nodeid="evals/test_example.py::test_needs_model",
        duration=0.0,
        longrepr=("example.py", 1, "Model unavailable"),
        sections=[],
        capstdout="",
    )

    eval_hooks.pytest_runtest_logreport(event)

    assert len(report.results) == 1
    assert report.results[0].outcome == "skipped"
    assert "Model unavailable" in report.results[0].reason


def test_unavailable_live_model_is_not_reported_as_accuracy():
    report = eval_hooks.EvalReport(model_availability="unavailable")
    markdown = report.generate_markdown()
    assert "**Live model accuracy:** unavailable" in markdown


def test_eval_config_accepts_versioned_openai_base_url(monkeypatch):
    monkeypatch.setenv("EVAL_JUDGE_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setenv("EVAL_JUDGE_MODEL", "local-model")

    config = MockConfig()

    assert config.llm_provider == "openai_compatible"
    assert config.llm_base_url == "http://127.0.0.1:8000/v1"
    assert config.llm_chat_model == "local-model"


def test_eval_config_uses_explicit_provider_on_custom_port(monkeypatch):
    monkeypatch.setenv("EVAL_PROVIDER", "openai_compatible")
    monkeypatch.setenv("EVAL_JUDGE_BASE_URL", "http://127.0.0.1:11434/v1")

    config = MockConfig()

    assert config.llm_provider == "openai_compatible"
    assert config.llm_base_url == "http://127.0.0.1:11434/v1"


def test_eval_config_uses_custom_ollama_endpoint(monkeypatch):
    monkeypatch.setenv("EVAL_PROVIDER", "ollama")
    monkeypatch.setenv("EVAL_JUDGE_BASE_URL", "http://127.0.0.1:14000")
    monkeypatch.setenv("EVAL_JUDGE_MODEL", "local-small")

    config = MockConfig()

    assert config.ollama_base_url == "http://127.0.0.1:14000"
    assert config.ollama_chat_model == "local-small"


def test_judge_call_uses_versioned_openai_url_once(monkeypatch):
    from unittest.mock import Mock
    import requests

    monkeypatch.setattr(eval_helpers, "JUDGE_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setattr(eval_helpers, "JUDGE_MODEL", "local-model")
    monkeypatch.setattr(requests, "get", Mock(side_effect=ConnectionError("not Ollama")))
    response = Mock()
    response.json.return_value = {"choices": [{"message": {"content": "yes"}}]}
    post = Mock(return_value=response)
    monkeypatch.setattr(requests, "post", post)

    assert eval_helpers.call_judge_llm("system", "user") == "yes"
    assert post.call_args.args[0] == "http://127.0.0.1:8000/v1/chat/completions"


def test_judge_availability_uses_versioned_openai_url_once(monkeypatch):
    from unittest.mock import Mock
    import requests

    monkeypatch.setattr(eval_helpers, "JUDGE_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setattr(eval_helpers, "JUDGE_MODEL", "local-model")
    not_ollama = Mock(status_code=404)
    openai = Mock(status_code=200)
    openai.json.return_value = {"data": [{"id": "local-model"}]}
    get = Mock(side_effect=[not_ollama, openai])
    monkeypatch.setattr(requests, "get", get)

    assert eval_helpers.is_judge_llm_available() is True
    assert get.call_args.args[0] == "http://127.0.0.1:8000/v1/models"


def test_scenario_recorder_writes_structured_outcomes(tmp_path, monkeypatch):
    import json

    path = tmp_path / "scenarios.json"
    monkeypatch.setenv("EVAL_SCENARIO_REPORT_PATH", str(path))
    monkeypatch.setattr(eval_hooks, "_eval_report", None)
    monkeypatch.setattr(eval_hooks, "_scenario_observations", [])
    record = eval_hooks.scenario_recorder.__wrapped__()
    record(Scenario("recovery", "offline", [Attempt(False), Attempt(True)]))

    eval_hooks.pytest_sessionfinish(None, 0)

    data = json.loads(path.read_text())
    assert data["summary"]["offline"]["first_attempt_success"] == 0
    assert data["summary"]["offline"]["recovered_success"] == 1


def test_live_preparation_cases_report_unavailable_when_model_is_absent(tmp_path):
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    report_path = tmp_path / "scenarios.json"
    root = Path(__file__).resolve().parents[1]
    env = {
        **os.environ,
        "EVAL_JUDGE_BASE_URL": "http://127.0.0.1:1",
        "EVAL_JUDGE_MODEL": "not-installed",
        "EVAL_SCENARIO_REPORT_PATH": str(report_path),
    }
    run = subprocess.run(
        [sys.executable, "-m", "pytest", "evals/test_turn_preparation.py", "-q"],
        cwd=root, env=env, capture_output=True, text=True, timeout=30,
    )

    assert run.returncode == 0, run.stdout + run.stderr
    payload = json.loads(report_path.read_text())
    categories = payload["summary"].values()
    assert sum(category["scenarios"] for category in categories) == len(payload["scenarios"]) > 0
    for category in categories:
        assert category["availability"] == "unavailable"
        assert category["eligible"] == 0
        assert category["first_attempt_success"] == 0
        assert category["recovered_success"] == 0


def test_live_preparation_case_records_observed_first_attempt(monkeypatch):
    from evals import test_turn_preparation as live
    from jarvis.reply.preparation import PreparedTurn

    observed = []
    monkeypatch.setattr(live, "_JUDGE_LLM_AVAILABLE", True)
    monkeypatch.setattr(
        "jarvis.reply.preparation.prepare_turn",
        lambda **kwargs: PreparedTurn([], [], False, {}),
    )

    live.test_combined_preparation_quality("Hello", set(), False, observed.append)

    assert len(observed) == 1
    assert observed[0].category == "live_preparation"
    assert observed[0].attempts == [Attempt(True)]
    assert observed[0].stage_durations["turn_preparation"][0] >= 0
