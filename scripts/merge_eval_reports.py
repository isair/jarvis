#!/usr/bin/env python3
"""
Merge multiple eval reports into a single combined EVALS.md.

This script takes pairs of (report_path, model_name) arguments and generates
a combined report showing results from all models side by side.

Usage:
    python merge_eval_reports.py report1.md model1 report2.md model2 > EVALS.md
"""

import sys
import re
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class TestResult:
    """Result for a single test case (aggregated across multiple runs)."""
    name: str
    outcome: str  # passed, failed, skipped, xfailed, xpassed, partial
    duration: float
    pass_rate: str = ""  # e.g., "3/3 (100%)" or "2/3 (67%)"
    class_name: str = ""  # The test class this result belongs to


@dataclass
class ModelReport:
    """Parsed report for a single model."""
    model_name: str
    results: Dict[str, TestResult] = field(default_factory=dict)
    total: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    duration: float = 0.0


def parse_report(report_path: str, model_name: str) -> Optional[ModelReport]:
    """Parse a markdown eval report into a ModelReport."""
    path = Path(report_path)
    if not path.exists():
        print(f"Warning: Report not found: {report_path}", file=sys.stderr)
        return None

    content = path.read_text(encoding="utf-8")
    report = ModelReport(model_name=model_name)

    # Parse summary stats
    for line in content.split("\n"):
        if "| ✅ Passed |" in line:
            match = re.search(r"\|\s*(\d+)\s*\|", line.split("Passed")[1])
            if match:
                report.passed = int(match.group(1))
        elif "| ❌ Failed |" in line:
            match = re.search(r"\|\s*(\d+)\s*\|", line.split("Failed")[1])
            if match:
                report.failed = int(match.group(1))
        elif "| ⏭️ Skipped |" in line:
            match = re.search(r"\|\s*(\d+)\s*\|", line.split("Skipped")[1])
            if match:
                report.skipped = int(match.group(1))
        elif "| **Total** |" in line:
            match = re.search(r"\|\s*\*\*(\d+)\*\*\s*\|", line)
            if match:
                report.total = int(match.group(1))
        elif "**Duration:**" in line:
            match = re.search(r"([\d.]+)s", line)
            if match:
                report.duration = float(match.group(1))

    # Parse individual test results from:
    # 1. Table format: | Test Case | Pass Rate | Status | Avg Duration |
    # 2. Detailed format: #### ✅ test_name (used for judge tests with notes)
    # Track current class name from section headers like "### ✅ TestClassName"
    in_table = False
    table_format = "old"  # "old" or "new"
    current_class = ""
    current_detailed_test = None  # Track test name for detailed format parsing
    lines = content.split("\n")

    for i, line in enumerate(lines):
        # Detect class section headers (e.g., "### ✅ TestIntentJudgeAccuracy")
        # Use a more lenient pattern that handles multi-byte emoji characters
        class_header_match = re.match(r'^###\s+\S+\s+(Test\w+)', line)
        if class_header_match:
            current_class = class_header_match.group(1)
            in_table = False  # Reset table state for new section
            current_detailed_test = None
            continue

        # Detect detailed test headers (e.g., "#### ✅ wake_word_simple_question")
        # Use a more lenient pattern that handles multi-byte emoji characters
        detailed_test_match = re.match(r'^####\s+(\S+)\s+(.+)$', line)
        if detailed_test_match:
            in_table = False
            emoji_str = detailed_test_match.group(1)
            test_name = detailed_test_match.group(2).strip()

            # Determine outcome from emoji (check for emoji presence)
            outcome = "unknown"
            if "✅" in emoji_str:
                outcome = "passed"
            elif "❌" in emoji_str:
                outcome = "failed"
            elif "⏭" in emoji_str:  # May be ⏭️ or just ⏭
                outcome = "skipped"
            elif "🔸" in emoji_str:
                outcome = "xfailed"
            elif "🎉" in emoji_str:
                outcome = "xpassed"
            elif "⚠" in emoji_str:  # May be ⚠️ or just ⚠
                outcome = "partial"

            current_detailed_test = test_name
            # Initialize with placeholder values, will be updated below
            report.results[test_name] = TestResult(
                name=test_name,
                outcome=outcome,
                duration=0.0,
                pass_rate="",
                class_name=current_class
            )
            continue

        # Parse pass rate and duration for detailed format
        if current_detailed_test and current_detailed_test in report.results:
            # Parse pass rate line: "**Pass Rate:** 1/1 (100%)" or "**Pass Rate:** 1/1 XFAIL"
            if line.startswith("**Pass Rate:**"):
                pass_rate_match = re.search(r'\*\*Pass Rate:\*\*\s*(.+)', line)
                if pass_rate_match:
                    report.results[current_detailed_test].pass_rate = pass_rate_match.group(1).strip()
            # Parse duration line: "*Avg Duration: 1.23s*"
            elif line.startswith("*Avg Duration:"):
                duration_match = re.search(r'([\d.]+)s', line)
                if duration_match:
                    report.results[current_detailed_test].duration = float(duration_match.group(1))
                current_detailed_test = None  # Done parsing this test

        # Table format parsing
        if "| Test Case | Pass Rate | Status | Avg Duration |" in line:
            in_table = True
            table_format = "new"
            current_detailed_test = None
            continue
        if "| Test Case | Status | Duration |" in line:
            in_table = True
            table_format = "old"
            current_detailed_test = None
            continue
        if in_table and line.startswith("|") and "---" not in line:
            parts = [p.strip() for p in line.split("|")[1:-1]]

            if table_format == "new" and len(parts) >= 4:
                # Parse new format: | Test Case | Pass Rate | Status | Avg Duration |
                test_name = parts[0]
                pass_rate = parts[1]
                status_cell = parts[2]
                duration_cell = parts[3]
            elif len(parts) >= 3:
                # Parse old format: | Test Case | Status | Duration |
                test_name = parts[0]
                pass_rate = ""
                status_cell = parts[1]
                duration_cell = parts[2]
            else:
                continue

            # Extract outcome from status cell
            outcome = "unknown"
            if "✅" in status_cell:
                outcome = "passed"
            elif "❌" in status_cell:
                outcome = "failed"
            elif "⏭️" in status_cell:
                outcome = "skipped"
            elif "🔸" in status_cell:
                outcome = "xfailed"
            elif "🎉" in status_cell:
                outcome = "xpassed"
            elif "⚠️" in status_cell:
                outcome = "partial"

            # Extract duration
            duration_match = re.search(r"([\d.]+)s", duration_cell)
            duration = float(duration_match.group(1)) if duration_match else 0.0

            report.results[test_name] = TestResult(
                name=test_name,
                outcome=outcome,
                duration=duration,
                pass_rate=pass_rate,
                class_name=current_class
            )
        elif in_table and not line.startswith("|"):
            in_table = False

    return report


def is_fixed_model_test(result: TestResult) -> bool:
    """Check if a test uses a fixed model, independent of the judge model.

    Some tests are pinned to specific models regardless of EVAL_JUDGE_MODEL:
    - Intent judge tests use gemma4 (the intent classification model)
    - Tool selection tests use nomic-embed-text (the embedding model)

    These shouldn't be compared across judge models since they always use the
    same model — they belong in their own section.

    NOTE: This list is kept in sync manually. When you add a new test class or
    file whose model is pinned (not controlled by EVAL_JUDGE_MODEL), add its
    class-name substring below or its test-name pattern to the fallback list.
    """
    fixed_model_classes = [
        "IntentJudge",  # TestIntentJudgeAccuracy, TestIntentJudgeMultiSegment, etc.
        "ProcessedSegmentFiltering",  # Intent judge processed segment filtering
        "ToolSelectionFiltering",  # Uses fixed nomic-embed-text
    ]

    if result.class_name:
        for class_pattern in fixed_model_classes:
            if class_pattern in result.class_name:
                return True

    fixed_model_name_patterns = [
        "test_hot_window_mode_indicated_in_prompt",
        "test_tts_text_included_for_echo_detection",
        "test_system_prompt_has_echo_guidance",
        "test_returns_none_when_ollama_unavailable",
    ]
    return any(pattern in result.name for pattern in fixed_model_name_patterns)


# Backwards-compatible alias
is_intent_judge_test = is_fixed_model_test


def _parse_pass_rate_fraction(pass_rate: str) -> Optional[Tuple[int, int]]:
    """Parse a pass rate string like '2/3 (67%)' into (passes, total).

    Returns None for non-standard formats (SKIPPED, XFAIL, N/A, etc.).
    """
    match = re.match(r'(\d+)/(\d+)', pass_rate)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def _calc_run_level_pass_rate(
    report: ModelReport, main_llm_tests: set
) -> Tuple[int, int]:
    """Calculate pass rate from individual run results across all main LLM tests.

    Returns (total_passes, total_runs) by parsing each test's pass_rate string.
    Falls back to counting fully-passed/failed tests when pass_rate data is missing.
    """
    total_passes = 0
    total_runs = 0

    for test_name in main_llm_tests:
        result = report.results.get(test_name)
        if not result:
            continue

        # Skip xfailed/skipped — not countable
        if result.outcome in ("xfailed", "skipped"):
            continue

        fraction = _parse_pass_rate_fraction(result.pass_rate) if result.pass_rate else None
        if fraction:
            total_passes += fraction[0]
            total_runs += fraction[1]
        else:
            # Fallback: treat passed as 1/1, failed as 0/1
            if result.outcome == "passed":
                total_passes += 1
                total_runs += 1
            elif result.outcome == "failed":
                total_runs += 1

    return total_passes, total_runs


STATUS_EMOJI = {
    "passed": "✅",
    "failed": "❌",
    "skipped": "⏭️",
    "xfailed": "🔸",
    "xpassed": "🎉",
    "partial": "⚠️",
    "unknown": "❓",
}


def _classify_fixed_model(result: TestResult) -> Optional[Tuple[str, str]]:
    """Return (category_key, pinned_model) for fixed-model tests, else None."""
    cls = result.class_name or ""
    name = result.name or ""
    if "IntentJudge" in cls or "ProcessedSegmentFiltering" in cls or any(
        p in name
        for p in (
            "test_hot_window_mode_indicated_in_prompt",
            "test_tts_text_included_for_echo_detection",
            "test_system_prompt_has_echo_guidance",
            "test_returns_none_when_ollama_unavailable",
        )
    ):
        return ("intent_judge", "gemma4:e2b")
    if "ToolSelectionFiltering" in cls:
        return ("tool_selection", "nomic-embed-text")
    return None


def _rate_emoji(rate: float) -> str:
    return "🟢" if rate >= 80 else "🟡" if rate >= 50 else "🔴"


def _count_outcomes(results) -> Dict[str, int]:
    """Count outcome buckets (run-level: uses pass_rate fractions where available)."""
    passed = failed = skipped = xfailed = partial = 0
    total_passes = total_runs = 0
    for r in results:
        if r.outcome == "passed":
            passed += 1
        elif r.outcome == "failed":
            failed += 1
        elif r.outcome == "skipped":
            skipped += 1
        elif r.outcome == "xfailed":
            xfailed += 1
        elif r.outcome == "partial":
            partial += 1
        if r.outcome in ("xfailed", "skipped"):
            continue
        fraction = _parse_pass_rate_fraction(r.pass_rate) if r.pass_rate else None
        if fraction:
            total_passes += fraction[0]
            total_runs += fraction[1]
        elif r.outcome == "passed":
            total_passes += 1
            total_runs += 1
        elif r.outcome == "failed":
            total_runs += 1
    rate = (total_passes / total_runs * 100) if total_runs > 0 else 0.0
    return {
        "passed": passed, "failed": failed, "skipped": skipped,
        "xfailed": xfailed, "partial": partial,
        "total": passed + failed + skipped + xfailed + partial,
        "run_passes": total_passes, "run_total": total_runs, "rate": rate,
    }


def generate_combined_report(reports: List[ModelReport]) -> str:
    """Generate a combined markdown report grouped by test category."""
    lines: List[str] = []
    now = datetime.now()

    # Bucket results into three categories:
    #   judge_compared: run once per judge model, compared side-by-side
    #   intent_judge:   pinned to gemma4:e2b, shown once
    #   tool_selection: pinned to nomic-embed-text, shown once
    judge_compared: set[str] = set()
    intent_judge_results: Dict[str, TestResult] = {}
    tool_selection_results: Dict[str, TestResult] = {}

    for report in reports:
        for test_name, result in report.results.items():
            fm = _classify_fixed_model(result)
            if fm is None:
                judge_compared.add(test_name)
                continue
            bucket = intent_judge_results if fm[0] == "intent_judge" else tool_selection_results
            existing = bucket.get(test_name)
            if existing is None or (existing.outcome == "skipped" and result.outcome != "skipped"):
                bucket[test_name] = result

    # Per-model stats for the judge-compared bucket
    per_model_stats: Dict[str, Dict[str, int]] = {}
    for report in reports:
        results = [r for n, r in report.results.items() if n in judge_compared]
        per_model_stats[report.model_name] = _count_outcomes(results)

    intent_stats = _count_outcomes(list(intent_judge_results.values()))
    tool_stats = _count_outcomes(list(tool_selection_results.values()))

    # Overall aggregate (sum of runs across all categories)
    overall_passes = sum(s["run_passes"] for s in per_model_stats.values()) + intent_stats["run_passes"] + tool_stats["run_passes"]
    overall_runs = sum(s["run_total"] for s in per_model_stats.values()) + intent_stats["run_total"] + tool_stats["run_total"]
    overall_rate = (overall_passes / overall_runs * 100) if overall_runs > 0 else 0.0

    # Header
    lines.append("# 🧪 Jarvis Evaluation Report")
    lines.append("")
    lines.append(f"**Generated:** {now.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # TL;DR
    lines.append("## 📊 TL;DR")
    lines.append("")
    lines.append(f"**Overall:** {_rate_emoji(overall_rate)} **{overall_passes}/{overall_runs} passed ({overall_rate:.1f}%)** across all categories")
    lines.append("")
    lines.append("| Category | Model | Passed | Failed | Skipped | Pass Rate |")
    lines.append("|----------|-------|-------:|-------:|--------:|----------:|")

    def _fmt_row(label: str, model_note: str, stats: Dict[str, int]) -> str:
        emoji = _rate_emoji(stats["rate"]) if stats["run_total"] else "➖"
        rate_str = f"{emoji} {stats['rate']:.1f}%" if stats["run_total"] else "➖"
        return (
            f"| {label} | {model_note} | {stats['passed']} | {stats['failed']} | "
            f"{stats['skipped']} | {rate_str} |"
        )

    for report in reports:
        lines.append(_fmt_row("🤖 Agent behaviour", f"`{report.model_name}`", per_model_stats[report.model_name]))
    if intent_judge_results:
        lines.append(_fmt_row("🎤 Intent judge", "`gemma4:e2b` (fixed)", intent_stats))
    if tool_selection_results:
        lines.append(_fmt_row("🔍 Tool selection", "`nomic-embed-text` (fixed)", tool_stats))
    lines.append("")

    # Model selection guide (only when comparing judges)
    if len(reports) > 1:
        lines.append("### 💡 Model Selection Guide")
        lines.append("")
        lines.append("| Model | Best For | Trade-offs |")
        lines.append("|-------|----------|------------|")
        lines.append("| `gemma4:e2b` | Quick responses, lower RAM usage | May struggle with complex reasoning |")
        lines.append("| `gpt-oss:20b` | Best accuracy, complex tasks | Slower, requires more RAM |")
        lines.append("")

    # Agent behaviour: per-test comparison across judge models
    lines.append("---")
    lines.append("")
    lines.append("## 🤖 Agent behaviour")
    lines.append("")
    lines.append("> Runs the full agent pipeline against each judge model. Tests are compared side-by-side.")
    lines.append("")
    header = "| Test Case |"
    separator = "|-----------|"
    for report in reports:
        header += f" {report.model_name} |"
        separator += "----------:|"
    lines.append(header)
    lines.append(separator)
    for test_name in sorted(judge_compared):
        row = f"| {test_name} |"
        for report in reports:
            result = report.results.get(test_name)
            if result:
                emoji = STATUS_EMOJI.get(result.outcome, "❓")
                row += f" {emoji} {result.pass_rate} |" if result.pass_rate else f" {emoji} |"
            else:
                row += " ➖ |"
        lines.append(row)
    lines.append("")

    def _render_fixed_section(title: str, blurb: str, results: Dict[str, TestResult]) -> None:
        if not results:
            return
        lines.append("---")
        lines.append("")
        lines.append(f"## {title}")
        lines.append("")
        lines.append(f"> {blurb}")
        lines.append("")
        lines.append("| Test Case | Pass Rate | Status |")
        lines.append("|-----------|-----------|:------:|")
        for test_name in sorted(results.keys()):
            result = results[test_name]
            emoji = STATUS_EMOJI.get(result.outcome, "❓")
            pass_rate_str = result.pass_rate if result.pass_rate else "N/A"
            lines.append(f"| {test_name} | {pass_rate_str} | {emoji} |")
        lines.append("")

    _render_fixed_section(
        "🎤 Intent judge",
        "Pinned to `gemma4:e2b` (the voice intent classifier). Not affected by the judge model.",
        intent_judge_results,
    )
    _render_fixed_section(
        "🔍 Tool selection",
        "Pinned to `nomic-embed-text` (embedding-based filter). Not affected by the judge model.",
        tool_selection_results,
    )

    # Legend
    lines.append("---")
    lines.append("")
    lines.append("### 📖 Legend")
    lines.append("")
    lines.append("| Symbol | Meaning |")
    lines.append("|--------|---------|")
    lines.append("| ✅ | Fully passed (100% pass rate) |")
    lines.append("| ⚠️ | Partial pass (some runs failed) |")
    lines.append("| ❌ | Fully failed (0% pass rate) |")
    lines.append("| ⏭️ | Skipped (missing dependencies) |")
    lines.append("| 🔸 | Expected failure (known limitation) |")
    lines.append("| 🎉 | Unexpectedly passed (bug fixed!) |")
    lines.append("| ➖ | Not run for this model |")
    lines.append("")
    lines.append("*Report generated by Jarvis eval suite*")

    return "\n".join(lines)


def main():
    if len(sys.argv) < 5 or len(sys.argv) % 2 != 1:
        print("Usage: merge_eval_reports.py report1.md model1 report2.md model2 ...", file=sys.stderr)
        sys.exit(1)

    # Parse arguments into pairs
    reports = []
    args = sys.argv[1:]
    for i in range(0, len(args), 2):
        report_path = args[i]
        model_name = args[i + 1]
        report = parse_report(report_path, model_name)
        if report:
            reports.append(report)

    if not reports:
        print("Error: No valid reports found", file=sys.stderr)
        sys.exit(1)

    # Generate combined report
    combined = generate_combined_report(reports)
    sys.stdout.buffer.write(combined.encode("utf-8"))


if __name__ == "__main__":
    main()
