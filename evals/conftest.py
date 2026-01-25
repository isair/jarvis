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
    "TestIntentJudgeAccuracy": "Intent judge accuracy for voice command classification",
    "TestIntentJudgePromptQuality": "Intent judge prompt construction quality",
    "TestIntentJudgeFallback": "Intent judge fallback behavior when unavailable",
    "TestIntentJudgeMultiSegment": "Intent judge with multi-segment buffers and multi-person conversations",
    "TestTopicSwitching": "Tests correct tool selection when conversation topic changes",
    "TestFollowUpContext": "Tests context retention for follow-up questions",
    "TestMultiTurnExtended": "Extended multi-turn scenarios with longer conversations",
    "TestModelSizeDetection": "Tests model size detection from model names",
    "TestGreetingNoTools": "Tests that greetings don't trigger tool calls (mocked)",
    "TestGreetingNoToolsLive": "Live tests that greetings don't trigger tool calls",
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
    # Multi-turn context tests
    "test_weather_then_store_hours": "Topic switch: weather → store hours uses webSearch",
    "test_weather_then_restaurant_search": "Topic switch: weather → restaurant uses webSearch",
    "test_search_then_weather": "Topic switch: search → weather uses getWeather",
    "test_follow_up_references_previous_context": "Follow-up references previous turn context",
    "test_three_turn_topic_changes": "3-turn conversation with topic changes",
    "test_rapid_topic_switching": "Rapid back-and-forth topic switching",
    # Greeting no-tools tests
    "test_model_size_detection": "Model size detection from model name",
    "test_greeting_no_tool_calls": "Greeting should not trigger tool calls",
    "test_tool_queries_still_work": "Tool-requiring queries still trigger tools",
    "test_greeting_no_tools_live": "Live: greeting should not trigger tool calls",
    "test_weather_still_triggers_tools_live": "Live: weather query triggers tools",
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
    """Captured result from a single test run."""
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
class AggregatedTestResult:
    """Aggregated results from multiple runs of the same test."""
    name: str
    class_name: str
    test_name: str
    description: str
    runs: List[TestResult] = field(default_factory=list)

    @property
    def pass_count(self) -> int:
        return sum(1 for r in self.runs if r.outcome in ("passed", "xpassed"))

    @property
    def fail_count(self) -> int:
        return sum(1 for r in self.runs if r.outcome == "failed")

    @property
    def skip_count(self) -> int:
        return sum(1 for r in self.runs if r.outcome == "skipped")

    @property
    def xfail_count(self) -> int:
        return sum(1 for r in self.runs if r.outcome == "xfailed")

    @property
    def total_runs(self) -> int:
        return len(self.runs)

    @property
    def pass_rate(self) -> float:
        countable = self.pass_count + self.fail_count
        return (self.pass_count / countable * 100) if countable > 0 else 0.0

    @property
    def total_duration(self) -> float:
        return sum(r.duration for r in self.runs)

    @property
    def avg_duration(self) -> float:
        return self.total_duration / len(self.runs) if self.runs else 0.0

    @property
    def overall_outcome(self) -> str:
        """Determine overall outcome based on pass rate."""
        if self.skip_count == self.total_runs:
            return "skipped"
        if self.xfail_count == self.total_runs:
            return "xfailed"
        if self.pass_count == self.total_runs:
            return "passed"
        if self.fail_count == self.total_runs:
            return "failed"
        return "partial"

    @property
    def pass_rate_str(self) -> str:
        """Format pass rate as 'X/Y (Z%)'."""
        countable = self.pass_count + self.fail_count
        if countable == 0:
            if self.skip_count > 0:
                return "SKIPPED"
            if self.xfail_count > 0:
                return f"{self.xfail_count}/{self.total_runs} XFAIL"
            return "N/A"
        return f"{self.pass_count}/{countable} ({self.pass_rate:.0f}%)"

    @property
    def judge_notes(self) -> Optional[Dict[str, str]]:
        """Return judge notes from first run that has them."""
        for run in self.runs:
            if run.judge_notes:
                return run.judge_notes
        return None

    @property
    def reason(self) -> Optional[str]:
        """Return reason from first run that has it."""
        for run in self.runs:
            if run.reason:
                return run.reason
        return None


def _strip_repeat_suffix(node_id: str) -> str:
    """
    Strip pytest-repeat iteration suffix from node ID.

    pytest-repeat adds suffixes like [1-3], [2-3], [3-3] to repeated tests.
    This strips those suffixes to get the base test identifier for aggregation.
    """
    # Match patterns like [1-3], [2-3], [3-3] at the end of node ID
    # But preserve parametrize IDs like [greeting-en], [weather-query], etc.
    return re.sub(r'\[(\d+)-(\d+)\]$', '', node_id)


def _get_aggregation_key(result: TestResult) -> str:
    """Get a unique key for aggregating repeated test runs."""
    # Use class_name + test_name + case_id (if any) as the aggregation key
    key_parts = [result.class_name, result.test_name]
    if result.case_id:
        # Strip repeat suffix from case_id too
        case_id = re.sub(r'-\d+-\d+$', '', result.case_id)
        key_parts.append(case_id)
    return "::".join(key_parts)


@dataclass
class EvalReport:
    """Aggregated eval results for markdown generation."""
    results: List[TestResult] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    judge_model: str = ""

    def add_result(self, result: TestResult):
        self.results.append(result)

    def get_aggregated_results(self) -> List[AggregatedTestResult]:
        """Aggregate results from multiple runs of the same test."""
        aggregated: Dict[str, AggregatedTestResult] = {}

        for result in self.results:
            key = _get_aggregation_key(result)
            if key not in aggregated:
                # Strip repeat suffix from description too
                desc = re.sub(r'-\d+-\d+$', '', result.description)
                aggregated[key] = AggregatedTestResult(
                    name=_strip_repeat_suffix(result.name),
                    class_name=result.class_name,
                    test_name=result.test_name,
                    description=desc,
                )
            aggregated[key].runs.append(result)

        return list(aggregated.values())

    @property
    def total_unique_tests(self) -> int:
        return len(self.get_aggregated_results())

    @property
    def total_runs(self) -> int:
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
        """Generate a pretty markdown report with pass rates from multiple runs."""
        lines = []
        aggregated_results = self.get_aggregated_results()

        # Calculate overall stats from aggregated results
        total_tests = len(aggregated_results)
        fully_passed = sum(1 for r in aggregated_results if r.overall_outcome == "passed")
        fully_failed = sum(1 for r in aggregated_results if r.overall_outcome == "failed")
        partial = sum(1 for r in aggregated_results if r.overall_outcome == "partial")
        skipped = sum(1 for r in aggregated_results if r.overall_outcome == "skipped")
        xfailed = sum(1 for r in aggregated_results if r.overall_outcome == "xfailed")

        # Header
        lines.append("# 🧪 Jarvis Evaluation Report")
        lines.append("")
        lines.append(f"**Generated:** {self.end_time.strftime('%Y-%m-%d %H:%M:%S') if self.end_time else 'N/A'}")
        lines.append(f"**Judge Model:** `{self.judge_model}`")
        lines.append(f"**Duration:** {self.duration:.2f}s")
        lines.append(f"**Runs per test:** {self.total_runs // total_tests if total_tests > 0 else 0}")
        lines.append("")

        # Summary stats
        lines.append("## 📊 Summary")
        lines.append("")
        lines.append("| Metric | Count |")
        lines.append("|--------|-------|")
        lines.append(f"| ✅ Fully Passed (100%) | {fully_passed} |")
        lines.append(f"| ⚠️ Partial Pass | {partial} |")
        lines.append(f"| ❌ Fully Failed (0%) | {fully_failed} |")
        lines.append(f"| ⏭️ Skipped | {skipped} |")
        lines.append(f"| 🔸 Expected Fail | {xfailed} |")
        lines.append(f"| **Unique Tests** | **{total_tests}** |")
        lines.append(f"| **Total Runs** | **{self.total_runs}** |")
        lines.append("")

        # Pass rate bar (based on individual runs)
        pass_rate = self.pass_rate
        bar_filled = int(pass_rate / 5)  # 20 chars max
        bar_empty = 20 - bar_filled
        bar = "█" * bar_filled + "░" * bar_empty
        emoji = "🟢" if pass_rate >= 80 else "🟡" if pass_rate >= 50 else "🔴"
        lines.append(f"**Overall Pass Rate:** {emoji} `{bar}` **{pass_rate:.1f}%** ({self.passed}/{self.passed + self.failed} runs)")
        lines.append("")

        # Group aggregated results by class
        by_class: Dict[str, List[AggregatedTestResult]] = {}
        for result in aggregated_results:
            if result.class_name not in by_class:
                by_class[result.class_name] = []
            by_class[result.class_name].append(result)

        # Detailed results
        lines.append("---")
        lines.append("")
        lines.append("## 📋 Detailed Results")
        lines.append("")

        for class_name, class_results in by_class.items():
            class_fully_passed = sum(1 for r in class_results if r.overall_outcome == "passed")
            class_total = len([r for r in class_results if r.overall_outcome not in ("skipped",)])
            class_emoji = "✅" if class_fully_passed == class_total and class_total > 0 else "⚠️" if class_fully_passed > 0 else "❌"

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
                        "partial": "⚠️",
                    }.get(result.overall_outcome, "❓")

                    lines.append(f"#### {status_emoji} {result.description}")
                    lines.append("")
                    lines.append(f"**Pass Rate:** {result.pass_rate_str}")

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

                    lines.append(f"*Avg Duration: {result.avg_duration:.2f}s*")
                    lines.append("")
            else:
                # Table format for non-judge tests with pass rates
                lines.append("| Test Case | Pass Rate | Status | Avg Duration |")
                lines.append("|-----------|-----------|--------|--------------|")

                for result in class_results:
                    status_emoji = {
                        "passed": "✅",
                        "failed": "❌",
                        "skipped": "⏭️",
                        "xfailed": "🔸",
                        "partial": "⚠️",
                    }.get(result.overall_outcome, "❓")

                    status_text = result.overall_outcome.upper()
                    if result.reason:
                        reason_short = result.reason[:30] + "..." if len(result.reason) > 30 else result.reason
                        status_text += f" ({reason_short})"

                    lines.append(f"| {result.description} | {result.pass_rate_str} | {status_emoji} {status_text} | {result.avg_duration:.2f}s |")

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
    # Support custom report path via environment variable
    report_path_str = os.environ.get("EVAL_REPORT_PATH")
    if report_path_str:
        report_path = Path(report_path_str)
    else:
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

