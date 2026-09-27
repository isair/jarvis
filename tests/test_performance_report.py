"""Performance reports state what the harness actually observed."""

import json

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
    )
    report = json.loads(path.read_text())

    assert report["end_to_end_sec"]["p95"] == 0.7
    assert report["first_useful_text_sec"] is None
    assert report["first_useful_spoken_sec"] is None
    assert report["unnecessary_tool_calls"] == 2
