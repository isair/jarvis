"""Benchmarks reject unusable samples and retain private failure metadata."""

import importlib
import json

import pytest

from jarvis.llm.ollama import OllamaBackend

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures('no_network')]


@pytest.fixture
def benchmark(monkeypatch, tmp_path):
    module = importlib.import_module('tests.performance.test_pipeline_timings')
    monkeypatch.setattr(module, 'PERF_REPORT_DIR', tmp_path)
    monkeypatch.setattr(module, 'PERF_RUNS', 1)
    monkeypatch.setattr(module, 'PERF_MODEL', 'owned-sample-model')
    monkeypatch.setattr(module, 'OLLAMA_URL', 'http://owned.invalid')
    return module


@pytest.mark.parametrize('sample', [None, '', '   '])
def test_empty_micro_samples_fail_and_keep_report(benchmark, monkeypatch, sample):
    monkeypatch.setattr(OllamaBackend, 'direct', lambda *args, **kwargs: sample)
    with pytest.raises(AssertionError):
        benchmark.test_micro_benchmark_tiny_prompt()
    reports = list(benchmark.PERF_REPORT_DIR.glob('micro-*.json'))
    assert len(reports) == 1
    payload = json.loads(reports[0].read_text())
    assert payload['status'] == 'failed'
    assert payload['failure_type'] == 'AssertionError'
    assert {row['outcome'] for row in payload['raw']} == {'empty'}


def test_failed_micro_request_keeps_error_evidence(benchmark, monkeypatch):
    failure = RuntimeError('owned transport failure')
    def direct(*args, **kwargs):
        raise failure
    monkeypatch.setattr(OllamaBackend, 'direct', direct)
    with pytest.raises(RuntimeError) as raised:
        benchmark.test_micro_benchmark_tiny_prompt()
    assert raised.value is failure
    reports = list(benchmark.PERF_REPORT_DIR.glob('micro-*.json'))
    assert len(reports) == 1
    payload = json.loads(reports[0].read_text())
    assert [row['outcome'] for row in payload['raw']] == ['error']


def test_valid_micro_samples_keep_provider_and_outcome(benchmark, monkeypatch):
    monkeypatch.setattr(OllamaBackend, 'direct', lambda *args, **kwargs: 'OK')
    benchmark.test_micro_benchmark_tiny_prompt()
    payload = json.loads(next(benchmark.PERF_REPORT_DIR.glob('micro-*.json')).read_text())
    assert payload['status'] == 'completed'
    assert payload['failure_type'] is None
    assert {row['outcome'] for row in payload['raw']} == {'response'}
    assert {row['provider'] for row in payload['raw']} == {OllamaBackend.__name__}


def test_empty_pipeline_samples_fail_and_keep_report(benchmark, monkeypatch):
    from jarvis.reply import engine
    monkeypatch.setattr(OllamaBackend, 'direct', lambda *args, **kwargs: None)
    def run(db, cfg, location, query, dialogue):
        return OllamaBackend('http://owned.invalid').direct('owned model', 'system', 'user')
    monkeypatch.setattr(engine, 'run_reply_engine', run)
    with pytest.raises(AssertionError):
        benchmark.test_pipeline_timings_by_context()
    assert len(list(benchmark.PERF_REPORT_DIR.glob('pipeline-*.json'))) == 1


def test_empty_pipeline_answer_fails_even_with_returned_model_text(benchmark, monkeypatch):
    from jarvis.reply import engine
    monkeypatch.setattr(OllamaBackend, 'direct', lambda *args, **kwargs: 'intermediate response')
    def run(db, cfg, location, query, dialogue):
        OllamaBackend('http://owned.invalid').direct('owned model', 'system', 'user')
        return ''
    monkeypatch.setattr(engine, 'run_reply_engine', run)
    with pytest.raises(AssertionError):
        benchmark.test_pipeline_timings_by_context()
    reports = list(benchmark.PERF_REPORT_DIR.glob('pipeline-*.json'))
    assert len(reports) == 1
    assert json.loads(reports[0].read_text())['status'] == 'failed'


def test_failed_pipeline_keeps_error_evidence_and_closes_store(benchmark, monkeypatch):
    from jarvis.reply import engine
    stores = []
    failure = RuntimeError('owned pipeline failure')
    def direct(*args, **kwargs):
        raise failure
    def run(db, cfg, location, query, dialogue):
        stores.append(db)
        return OllamaBackend('http://owned.invalid').direct('owned model', 'system', 'user')
    monkeypatch.setattr(OllamaBackend, 'direct', direct)
    monkeypatch.setattr(engine, 'run_reply_engine', run)
    with pytest.raises(RuntimeError) as raised:
        benchmark.test_pipeline_timings_by_context()
    assert raised.value is failure
    assert len(list(benchmark.PERF_REPORT_DIR.glob('pipeline-*.json'))) == 1
    # The owned connection is closed after a failed query.
    import sqlite3
    with pytest.raises(sqlite3.ProgrammingError):
        stores[0].conn.execute('SELECT 1')


def test_valid_pipeline_remains_usable_without_retaining_answer_text(benchmark, monkeypatch):
    from jarvis.reply import engine
    answer = 'owned private answer'
    monkeypatch.setattr(OllamaBackend, 'direct', lambda *args, **kwargs: answer)
    def run(db, cfg, location, query, dialogue):
        return OllamaBackend('http://owned.invalid').direct('owned model', 'system', query)
    monkeypatch.setattr(engine, 'run_reply_engine', run)
    benchmark.test_pipeline_timings_by_context()
    report = next(benchmark.PERF_REPORT_DIR.glob('pipeline-*.json')).read_text()
    assert json.loads(report)['summary']['total_calls'] == len(benchmark.PIPELINE_QUERIES) * benchmark.PERF_RUNS
    assert answer not in report


def test_pipeline_failure_is_explicit_without_retaining_exception_text(benchmark, monkeypatch):
    from jarvis.reply import engine
    failure = RuntimeError('owned private failure detail')
    def run(*args, **kwargs):
        raise failure
    monkeypatch.setattr(engine, 'run_reply_engine', run)
    with pytest.raises(RuntimeError) as raised:
        benchmark.test_pipeline_timings_by_context()
    assert raised.value is failure
    report = next(benchmark.PERF_REPORT_DIR.glob('pipeline-*.json')).read_text()
    payload = json.loads(report)
    assert payload['status'] == 'failed'
    assert payload['failure_type'] == type(failure).__name__
    assert str(failure) not in report


@pytest.mark.parametrize('request_fails', [True, False])
def test_report_failure_does_not_hide_original_error(benchmark, monkeypatch, tmp_path, capsys, request_fails):
    target = tmp_path / 'owned-file'
    target.write_text('owned marker')
    monkeypatch.setattr(benchmark, 'PERF_REPORT_DIR', target)
    failure = RuntimeError('owned private backend detail')
    def direct(*args, **kwargs):
        if request_fails:
            raise failure
        return 'OK'
    monkeypatch.setattr(OllamaBackend, 'direct', direct)
    with pytest.raises(RuntimeError if request_fails else FileExistsError) as raised:
        benchmark.test_micro_benchmark_tiny_prompt()
    if request_fails:
        assert raised.value is failure
        output = capsys.readouterr().out
        assert 'FileExistsError' in output and 'report' in output
        assert str(failure) not in output
        assert str(target) not in output
    assert target.read_text() == 'owned marker'
