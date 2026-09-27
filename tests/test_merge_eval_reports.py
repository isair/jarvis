"""The combined report reads the current per-model report format."""

from evals.conftest import EvalReport, TestResult as EvalTestResult
from scripts.merge_eval_reports import parse_report


def test_merge_parser_keeps_current_summary_counts(tmp_path):
    report = EvalReport(judge_model="local", model_availability="available")
    report.add_result(EvalTestResult(
        name="evals/test_sample.py::TestSample::test_case",
        outcome="passed",
        duration=0.1,
        class_name="TestSample",
        test_name="test_case",
        description="A case",
    ))
    path = tmp_path / "report.md"
    path.write_text(report.generate_markdown())

    parsed = parse_report(str(path), "local")

    assert parsed.passed == 1
    assert parsed.total == 1
    assert parsed.results["A case"].outcome == "passed"
