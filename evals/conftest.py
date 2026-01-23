"""
Shared fixtures and configuration for evals.

Evals test end-to-end quality of the reply engine with real or mock LLM responses.
"""

import sys
import os
import re
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import pytest

# Robustly locate repository root
_this_file = Path(__file__).resolve()
ROOT = None
for parent in _this_file.parents:
    if (parent / "src" / "jarvis").exists():
        ROOT = parent
        break
if ROOT is None:
    ROOT = _this_file.parent.parent

SRC = ROOT / "src"
EVALS = ROOT / "evals"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(EVALS) not in sys.path:
    sys.path.insert(0, str(EVALS))

from helpers import MockConfig, JUDGE_MODEL


# =============================================================================
# Test Case Descriptions
# =============================================================================

# Human-readable descriptions for test classes
CLASS_DESCRIPTIONS = {
    "TestResponseQuality": "LLM-as-judge evaluations for response quality",
    "TestContextUtilization": "Tests that agent uses location/time/memory context",
    "TestToolUsage": "Validates tool selection and argument quality",
    "TestMultiStepReasoning": "Complex scenarios requiring tool chaining and synthesis",
    "TestMemoryEnrichment": "Tests automatic memory enrichment keyword extraction",
    "TestLiveEndToEnd": "Live tests with real LLM inference",
    "TestNutritionExtraction": "Tests LLM nutrition extraction accuracy for meal logging",
    "TestNutritionToolIntegration": "Tests full meal logging tool with macro extraction",
    "TestNutritionModelComparison": "Baseline tests for comparing nutrition extraction across models",
}

# Descriptions for non-parametrized tests
TEST_DESCRIPTIONS = {
    "test_weather_response_quality": "Judge evaluates weather response quality",
    "test_location_context_in_search": "Location context flows to search queries",
    "test_simple_search_flow": "Agent calls webSearch for info queries",
    "test_tool_chaining_search_then_fetch": "Agent chains search → fetch for details",
    "test_nutrition_advice_uses_memory_and_data": "Agent uses memory + nutrition data",
    "test_personalized_news_uses_memory_for_interests": "Agent recalls interests before personalized search (mocked)",
    "test_enrichment_extracts_correct_keywords": "Enrichment extracts personalization keywords",
    "test_enrichment_provides_context_to_llm": "Enrichment results appear in system message",
    "test_llm_uses_enrichment_without_redundant_tool_call": "LLM uses enrichment, skips redundant recallConversation",
    "test_weather_query_live": "Live weather query with real LLM",
    "test_personalized_query_recalls_memory_live": "Live: LLM checks memory before asking about interests",
    # Nutrition extraction tests
    "test_meal_extraction_accuracy": "Extracts accurate macros for common meals",
    "test_extraction_returns_valid_json_structure": "Returns valid JSON with all required fields",
    "test_extraction_handles_ambiguous_portions": "Handles ambiguous portion descriptions",
    "test_extraction_rejects_non_food": "Returns NONE for non-food inputs",
    "test_log_meal_tool_extracts_macros": "LogMealTool stores meals with macros",
    "test_simple_meal_extraction": "Simple meal baseline (2 boiled eggs)",
    "test_extraction_with_quantities": "Extraction with explicit quantities",
}


def _parse_parametrize_id(node_id: str) -> Optional[str]:
    """Extract the parametrize case ID from a node_id like 'test_foo[case-name]'."""
    match = re.search(r'\[(.+)\]$', node_id)
    return match.group(1) if match else None


def _extract_judge_notes(stdout: Optional[str]) -> Optional[Dict[str, str]]:
    """Parse judge evaluation output from stdout."""
    if not stdout:
        return None

    notes = {}

    # Extract score
    score_match = re.search(r'Score:\s*([\d.]+)', stdout)
    if score_match:
        notes["score"] = score_match.group(1)

    # Extract reasoning
    reasoning_match = re.search(r'Reasoning:\s*(.+?)(?:\n|$)', stdout)
    if reasoning_match:
        notes["reasoning"] = reasoning_match.group(1).strip()

    # Extract response being evaluated
    response_match = re.search(r'Response:\s*(.+?)(?:\.\.\.|$)', stdout)
    if response_match:
        notes["response"] = response_match.group(1).strip()

    return notes if notes else None


def _get_test_description(test_name: str, case_id: Optional[str]) -> str:
    """
    Get the description for a test case.

    For parametrized tests, the case_id IS the description (set via pytest.param id=).
    For non-parametrized tests, use the TEST_DESCRIPTIONS lookup.
    """
    if case_id:
        # Parametrized test: the ID is the description (defined in the test file)
        return case_id

    # Non-parametrized test: use lookup or fall back to test name
    return TEST_DESCRIPTIONS.get(test_name, test_name)


# =============================================================================
# Markdown Report Generation
# =============================================================================

@dataclass
class TestResult:
    """Captured result from a single test."""
    name: str
    outcome: str  # passed, failed, skipped, xfailed, xpassed
    duration: float
    class_name: str
    test_name: str
    case_id: Optional[str] = None
    description: str = ""
    reason: Optional[str] = None
    stdout: Optional[str] = None
    judge_notes: Optional[Dict[str, str]] = None


@dataclass
class EvalReport:
    """Aggregated eval results for markdown generation."""
    results: List[TestResult] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    judge_model: str = ""

    def add_result(self, result: TestResult):
        self.results.append(result)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.outcome == "passed")

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.outcome == "failed")

    @property
    def skipped(self) -> int:
        return sum(1 for r in self.results if r.outcome == "skipped")

    @property
    def xfailed(self) -> int:
        return sum(1 for r in self.results if r.outcome == "xfailed")

    @property
    def xpassed(self) -> int:
        return sum(1 for r in self.results if r.outcome == "xpassed")

    @property
    def pass_rate(self) -> float:
        countable = self.passed + self.failed + self.xpassed
        return (self.passed + self.xpassed) / countable * 100 if countable > 0 else 0.0

    @property
    def duration(self) -> float:
        return sum(r.duration for r in self.results)

    def generate_markdown(self) -> str:
        """Generate a pretty markdown report."""
        lines = []

        # Header
        lines.append("# 🧪 Jarvis Evaluation Report")
        lines.append("")
        lines.append(f"**Generated:** {self.end_time.strftime('%Y-%m-%d %H:%M:%S') if self.end_time else 'N/A'}")
        lines.append(f"**Judge Model:** `{self.judge_model}`")
        lines.append(f"**Duration:** {self.duration:.2f}s")
        lines.append("")

        # Summary stats
        lines.append("## 📊 Summary")
        lines.append("")
        lines.append("| Metric | Count |")
        lines.append("|--------|-------|")
        lines.append(f"| ✅ Passed | {self.passed} |")
        lines.append(f"| ❌ Failed | {self.failed} |")
        lines.append(f"| ⏭️ Skipped | {self.skipped} |")
        lines.append(f"| 🔸 Expected Fail | {self.xfailed} |")
        lines.append(f"| 🎉 Unexpectedly Passed | {self.xpassed} |")
        lines.append(f"| **Total** | **{self.total}** |")
        lines.append("")

        # Pass rate bar
        pass_rate = self.pass_rate
        bar_filled = int(pass_rate / 5)  # 20 chars max
        bar_empty = 20 - bar_filled
        bar = "█" * bar_filled + "░" * bar_empty
        emoji = "🟢" if pass_rate >= 80 else "🟡" if pass_rate >= 50 else "🔴"
        lines.append(f"**Pass Rate:** {emoji} `{bar}` **{pass_rate:.1f}%**")
        lines.append("")

        # Group results by class
        by_class: Dict[str, List[TestResult]] = {}
        for result in self.results:
            if result.class_name not in by_class:
                by_class[result.class_name] = []
            by_class[result.class_name].append(result)

        # Detailed results
        lines.append("---")
        lines.append("")
        lines.append("## 📋 Detailed Results")
        lines.append("")

        for class_name, class_results in by_class.items():
            class_passed = sum(1 for r in class_results if r.outcome in ("passed", "xpassed"))
            class_total = len([r for r in class_results if r.outcome not in ("skipped",)])
            class_emoji = "✅" if class_passed == class_total and class_total > 0 else "⚠️" if class_passed > 0 else "❌"

            # Class header with description
            lines.append(f"### {class_emoji} {class_name}")
            if class_name in CLASS_DESCRIPTIONS:
                lines.append(f"> {CLASS_DESCRIPTIONS[class_name]}")
            lines.append("")

            # Check if this class has judge notes (only for LLMAsJudge class)
            is_judge_class = "Judge" in class_name
            has_judge_notes = is_judge_class and any(r.judge_notes for r in class_results)

            if has_judge_notes:
                # Detailed format for judge tests
                for result in class_results:
                    status_emoji = {
                        "passed": "✅",
                        "failed": "❌",
                        "skipped": "⏭️",
                        "xfailed": "🔸",
                        "xpassed": "🎉",
                    }.get(result.outcome, "❓")

                    lines.append(f"#### {status_emoji} {result.description}")
                    lines.append("")

                    if result.judge_notes:
                        notes = result.judge_notes
                        if "response" in notes:
                            lines.append(f"**Input:** `{notes['response']}`")
                        if "score" in notes:
                            score = float(notes['score'])
                            score_bar = "●" * int(score * 10) + "○" * (10 - int(score * 10))
                            lines.append(f"**Score:** {score_bar} ({notes['score']})")
                        if "reasoning" in notes:
                            lines.append(f"**Judge notes:** {notes['reasoning']}")
                        lines.append("")

                    lines.append(f"*Duration: {result.duration:.2f}s*")
                    lines.append("")
            else:
                # Table format for non-judge tests
                lines.append("| Test Case | Status | Duration |")
                lines.append("|-----------|--------|----------|")

                for result in class_results:
                    status_emoji = {
                        "passed": "✅",
                        "failed": "❌",
                        "skipped": "⏭️",
                        "xfailed": "🔸",
                        "xpassed": "🎉",
                    }.get(result.outcome, "❓")

                    status_text = result.outcome.upper()
                    if result.reason:
                        reason_short = result.reason[:30] + "..." if len(result.reason) > 30 else result.reason
                        status_text += f" ({reason_short})"

                    lines.append(f"| {result.description} | {status_emoji} {status_text} | {result.duration:.2f}s |")

                lines.append("")

        # Footer
        lines.append("---")
        lines.append("")
        lines.append("*Report generated by Jarvis eval suite*")

        return "\n".join(lines)


# Global report instance
_eval_report: Optional[EvalReport] = None


def pytest_configure(config):
    """Initialize the eval report at test session start."""
    global _eval_report
    if os.environ.get("EVAL_GENERATE_REPORT") == "1":
        _eval_report = EvalReport(
            start_time=datetime.now(),
            judge_model=JUDGE_MODEL
        )


def pytest_runtest_logreport(report):
    """Capture each test result."""
    global _eval_report
    if _eval_report is None:
        return

    # Only capture the final result (call phase for passed/failed, setup/teardown for errors)
    if report.when != "call" and not (report.when in ("setup", "teardown") and report.outcome == "failed"):
        return

    # Parse the node ID to extract class and test name
    node_id = report.nodeid
    parts = node_id.split("::")
    class_name = parts[1] if len(parts) > 1 else "Unknown"
    full_test_name = parts[-1] if parts else node_id

    # Extract parametrize case ID (which is the description for parametrized tests)
    case_id = _parse_parametrize_id(full_test_name)
    test_name = full_test_name.split("[")[0]

    # Get description: for parametrized tests, it's the case_id; otherwise from lookup
    description = _get_test_description(test_name, case_id)

    # Determine outcome
    outcome = report.outcome
    if hasattr(report, "wasxfail"):
        outcome = "xpassed" if report.passed else "xfailed"

    # Get skip reason if applicable
    reason = None
    if outcome == "skipped" and hasattr(report, "longrepr"):
        if isinstance(report.longrepr, tuple) and len(report.longrepr) >= 3:
            reason = str(report.longrepr[2])

    # Capture stdout and parse judge notes
    stdout = None
    judge_notes = None
    if hasattr(report, "capstdout") and report.capstdout:
        stdout = report.capstdout
        judge_notes = _extract_judge_notes(stdout)

    # Also check sections for captured stdout
    if not stdout:
        for section_name, section_content in report.sections:
            if "stdout" in section_name.lower():
                stdout = section_content
                judge_notes = _extract_judge_notes(stdout)
                break

    _eval_report.add_result(TestResult(
        name=node_id,
        outcome=outcome,
        duration=report.duration,
        class_name=class_name,
        test_name=test_name,
        case_id=case_id,
        description=description,
        reason=reason,
        stdout=stdout,
        judge_notes=judge_notes,
    ))


def pytest_sessionfinish(session, exitstatus):
    """Generate the markdown report at session end."""
    global _eval_report
    if _eval_report is None:
        return

    _eval_report.end_time = datetime.now()

    # Write the markdown report (ensure UTF-8 encoding for emojis/unicode)
    report_path = ROOT / "EVALS.md"
    markdown = _eval_report.generate_markdown()
    report_path.write_text(markdown, encoding="utf-8")
    print(f"\n📄 Eval report saved to: {report_path}")


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_config():
    """Provide a mock configuration for eval tests."""
    return MockConfig()


@pytest.fixture
def eval_db():
    """Provide an in-memory database for eval tests."""
    from jarvis.memory.db import Database
    db = Database(":memory:", sqlite_vss_path=None)
    yield db
    db.close()


@pytest.fixture
def eval_dialogue_memory():
    """Provide a dialogue memory instance for eval tests."""
    from jarvis.memory.conversation import DialogueMemory
    return DialogueMemory(inactivity_timeout=300, max_interactions=20)

