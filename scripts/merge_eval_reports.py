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
    """Result for a single test case."""
    name: str
    outcome: str  # passed, failed, skipped, xfailed, xpassed
    duration: float


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
    in_table = False
    for line in content.split("\n"):
        if "| Test Case | Status | Duration |" in line:
            in_table = True
            continue
        if in_table and line.startswith("|") and "---" not in line:
            # Parse table row: | Test Case | ✅ PASSED | 1.23s |
            parts = [p.strip() for p in line.split("|")[1:-1]]
            if len(parts) >= 3:
                test_name = parts[0]
                status_cell = parts[1]
                duration_cell = parts[2]

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

                # Extract duration
                duration_match = re.search(r"([\d.]+)s", duration_cell)
                duration = float(duration_match.group(1)) if duration_match else 0.0

                report.results[test_name] = TestResult(
                    name=test_name,
                    outcome=outcome,
                    duration=duration
                )
        elif in_table and not line.startswith("|"):
            in_table = False

    return report


def generate_combined_report(reports: List[ModelReport]) -> str:
    """Generate a combined markdown report comparing multiple models."""
    lines = []
    now = datetime.now()

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

    # Summary comparison table
    header = "| Metric |"
    separator = "|--------|"
    for report in reports:
        header += f" {report.model_name} |"
        separator += "--------|"

    lines.append(header)
    lines.append(separator)

    # Rows
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
            val = getattr(report, attr)
            if attr == "duration":
                row += f" {val:.1f}s |"
            else:
                row += f" {val} |"
        lines.append(row)

    # Pass rate row
    row = "| 📈 Pass Rate |"
    for report in reports:
        countable = report.passed + report.failed
        rate = (report.passed / countable * 100) if countable > 0 else 0.0
        emoji = "🟢" if rate >= 80 else "🟡" if rate >= 50 else "🔴"
        row += f" {emoji} {rate:.1f}% |"
    lines.append(row)

    lines.append("")

    # Pass rate bars
    lines.append("### Pass Rate Visualization")
    lines.append("")
    for report in reports:
        countable = report.passed + report.failed
        rate = (report.passed / countable * 100) if countable > 0 else 0.0
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

    # Detailed comparison per test
    lines.append("---")
    lines.append("")
    lines.append("## 📋 Detailed Test Results")
    lines.append("")

    # Collect all unique test names
    all_tests = set()
    for report in reports:
        all_tests.update(report.results.keys())

    # Group by test class (extract from test name pattern)
    tests_by_class: Dict[str, List[str]] = {}
    for test_name in sorted(all_tests):
        # Try to infer class from test name patterns
        tests_by_class.setdefault("All Tests", []).append(test_name)

    # Build comparison table
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
        "unknown": "❓",
    }

    for test_name in sorted(all_tests):
        row = f"| {test_name} |"
        for report in reports:
            result = report.results.get(test_name)
            if result:
                emoji = status_emoji.get(result.outcome, "❓")
                row += f" {emoji} |"
            else:
                row += " ➖ |"
        lines.append(row)

    lines.append("")

    # Legend
    lines.append("### 📖 Legend")
    lines.append("")
    lines.append("| Symbol | Meaning |")
    lines.append("|--------|---------|")
    lines.append("| ✅ | Passed |")
    lines.append("| ❌ | Failed |")
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
