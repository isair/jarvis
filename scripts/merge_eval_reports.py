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

    # Parse individual test results from tables
    # Support both old format (| Test Case | Status | Duration |)
    # and new format (| Test Case | Pass Rate | Status | Avg Duration |)
    in_table = False
    table_format = "old"  # "old" or "new"
    for line in content.split("\n"):
        if "| Test Case | Pass Rate | Status | Avg Duration |" in line:
            in_table = True
            table_format = "new"
            continue
        if "| Test Case | Status | Duration |" in line:
            in_table = True
            table_format = "old"
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
                pass_rate=pass_rate
            )
        elif in_table and not line.startswith("|"):
            in_table = False

    return report


def is_intent_judge_test(test_name: str) -> bool:
    """Check if a test is an intent judge test (uses fixed model, not configurable)."""
    intent_judge_patterns = [
        "test_hot_window_mode_indicated_in_prompt",
        "test_tts_text_included_for_echo_detection",
        "test_system_prompt_has_echo_guidance",
        "test_returns_none_when_ollama_unavailable",
        "test_intent_judge_case",
        "test_multi_segment_case",
    ]
    return any(pattern in test_name for pattern in intent_judge_patterns)


def generate_combined_report(reports: List[ModelReport]) -> str:
    """Generate a combined markdown report comparing multiple models."""
    lines = []
    now = datetime.now()

    # Separate intent judge tests from main LLM tests
    # Intent judge tests use fixed model (llama3.2:3b), so only show once
    intent_judge_results = {}
    main_llm_tests = set()

    for report in reports:
        for test_name, result in report.results.items():
            if is_intent_judge_test(test_name):
                # Only keep one result for intent judge tests (they're identical across runs)
                if test_name not in intent_judge_results:
                    intent_judge_results[test_name] = result
            else:
                main_llm_tests.add(test_name)

    # Header
    lines.append("# 🧪 Jarvis Evaluation Report")
    lines.append("")
    lines.append(f"**Generated:** {now.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # Model comparison table
    lines.append("## 📊 Model Comparison")
    lines.append("")
    lines.append("This report compares eval results across officially supported models.")
    lines.append("Use this to understand the performance tradeoffs when choosing a model.")
    lines.append("")

    # Calculate stats for main LLM tests only (excluding intent judge)
    def calc_main_llm_stats(report: ModelReport) -> dict:
        passed = failed = skipped = 0
        duration = 0.0
        for test_name, result in report.results.items():
            if not is_intent_judge_test(test_name):
                if result.outcome == "passed":
                    passed += 1
                elif result.outcome == "failed":
                    failed += 1
                elif result.outcome == "skipped":
                    skipped += 1
                duration += result.duration
        return {"passed": passed, "failed": failed, "skipped": skipped,
                "total": passed + failed + skipped, "duration": duration}

    # Summary comparison table (main LLM tests only)
    header = "| Metric |"
    separator = "|--------|"
    for report in reports:
        header += f" {report.model_name} |"
        separator += "--------|"

    lines.append(header)
    lines.append(separator)

    # Rows using main LLM stats
    stats_by_model = {r.model_name: calc_main_llm_stats(r) for r in reports}

    metrics = [
        ("✅ Passed", "passed"),
        ("❌ Failed", "failed"),
        ("⏭️ Skipped", "skipped"),
        ("📊 Total", "total"),
        ("⏱️ Duration", "duration"),
    ]

    for label, attr in metrics:
        row = f"| {label} |"
        for report in reports:
            val = stats_by_model[report.model_name][attr]
            if attr == "duration":
                row += f" {val:.1f}s |"
            else:
                row += f" {val} |"
        lines.append(row)

    # Pass rate row
    row = "| 📈 Pass Rate |"
    for report in reports:
        stats = stats_by_model[report.model_name]
        countable = stats["passed"] + stats["failed"]
        rate = (stats["passed"] / countable * 100) if countable > 0 else 0.0
        emoji = "🟢" if rate >= 80 else "🟡" if rate >= 50 else "🔴"
        row += f" {emoji} {rate:.1f}% |"
    lines.append(row)

    lines.append("")

    # Pass rate bars (main LLM tests only)
    lines.append("### Pass Rate Visualization")
    lines.append("")
    for report in reports:
        stats = stats_by_model[report.model_name]
        countable = stats["passed"] + stats["failed"]
        rate = (stats["passed"] / countable * 100) if countable > 0 else 0.0
        bar_filled = int(rate / 5)  # 20 chars max
        bar_empty = 20 - bar_filled
        bar = "█" * bar_filled + "░" * bar_empty
        emoji = "🟢" if rate >= 80 else "🟡" if rate >= 50 else "🔴"
        lines.append(f"**{report.model_name}:** {emoji} `{bar}` **{rate:.1f}%**")
    lines.append("")

    # Model recommendations
    lines.append("### 💡 Model Selection Guide")
    lines.append("")
    lines.append("| Model | Best For | Trade-offs |")
    lines.append("|-------|----------|------------|")
    lines.append("| `llama3.2:3b` | Quick responses, lower RAM usage | May struggle with complex reasoning |")
    lines.append("| `gpt-oss:20b` | Best accuracy, complex tasks | Slower, requires more RAM |")
    lines.append("")

    # Detailed comparison per test (main LLM tests only)
    lines.append("---")
    lines.append("")
    lines.append("## 📋 Detailed Test Results")
    lines.append("")

    # Build comparison table for main LLM tests only
    header = "| Test Case |"
    separator = "|-----------|"
    for report in reports:
        header += f" {report.model_name} |"
        separator += "----------|"

    lines.append(header)
    lines.append(separator)

    status_emoji = {
        "passed": "✅",
        "failed": "❌",
        "skipped": "⏭️",
        "xfailed": "🔸",
        "xpassed": "🎉",
        "partial": "⚠️",
        "unknown": "❓",
    }

    for test_name in sorted(main_llm_tests):
        row = f"| {test_name} |"
        for report in reports:
            result = report.results.get(test_name)
            if result:
                emoji = status_emoji.get(result.outcome, "❓")
                # Include pass rate if available
                if result.pass_rate:
                    row += f" {emoji} {result.pass_rate} |"
                else:
                    row += f" {emoji} |"
            else:
                row += " ➖ |"
        lines.append(row)

    lines.append("")

    # Intent Judge Tests Section (separate, single model)
    if intent_judge_results:
        lines.append("---")
        lines.append("")
        lines.append("## 🎤 Intent Judge Tests")
        lines.append("")
        lines.append("> These tests evaluate the voice intent classification system.")
        lines.append("> They use a fixed model (`llama3.2:3b`) and are not part of the model comparison.")
        lines.append("")

        # Calculate intent judge stats
        ij_passed = sum(1 for r in intent_judge_results.values() if r.outcome == "passed")
        ij_failed = sum(1 for r in intent_judge_results.values() if r.outcome == "failed")
        ij_xfailed = sum(1 for r in intent_judge_results.values() if r.outcome == "xfailed")
        ij_total = len(intent_judge_results)

        lines.append(f"**Model:** `llama3.2:3b` (fixed)")
        lines.append(f"**Results:** {ij_passed} passed, {ij_failed} failed, {ij_xfailed} expected failures")
        lines.append("")

        lines.append("| Test Case | Pass Rate | Status |")
        lines.append("|-----------|-----------|--------|")

        for test_name in sorted(intent_judge_results.keys()):
            result = intent_judge_results[test_name]
            emoji = status_emoji.get(result.outcome, "❓")
            pass_rate_str = result.pass_rate if result.pass_rate else "N/A"
            lines.append(f"| {test_name} | {pass_rate_str} | {emoji} |")

        lines.append("")

    # Legend
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

    # Footer
    lines.append("---")
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
    print(combined)


if __name__ == "__main__":
    main()
