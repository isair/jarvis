"""Structured, provider-neutral measurements for agent quality benchmarks."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil
from statistics import median


@dataclass(frozen=True)
class Attempt:
    """One attempt to satisfy a scenario, in the order it occurred."""

    success: bool
    unnecessary_tools: int = 0
    stale_retrieval: bool = False
    unsupported_claim: bool = False


@dataclass
class Scenario:
    """A single request and its observed attempts.

    ``availability`` distinguishes an unrun model case from a failed request.
    Stage durations contain only spans that the harness actually measured.
    First useful spoken output stays absent unless audio is measured.
    """

    name: str
    category: str
    attempts: list[Attempt]
    availability: str = "available"
    stage_durations: dict[str, list[float]] = field(default_factory=dict)
    end_to_end_sec: float | None = None
    first_useful_text_sec: float | None = None
    first_useful_spoken_sec: float | None = None


def latency_summary(values: list[float]) -> dict[str, float | int] | None:
    """Return observed p50 and nearest-rank p95, or unknown when unmeasured."""
    if not values:
        return None
    if any(value < 0 for value in values):
        raise ValueError("latencies must be non-negative")
    ordered = sorted(values)
    return {
        "n": len(ordered),
        "p50": median(ordered),
        "p95": ordered[ceil(0.95 * len(ordered)) - 1],
    }


def summarise_scenarios(scenarios: list[Scenario]) -> dict[str, dict]:
    """Report quality and latency separately for each measurement category."""
    grouped: dict[str, list[Scenario]] = {}
    for scenario in scenarios:
        grouped.setdefault(scenario.category, []).append(scenario)

    result: dict[str, dict] = {}
    for category, members in grouped.items():
        available = [item for item in members if item.availability == "available"]
        eligible = [item for item in available if item.attempts]
        stage_names = {
            stage for item in eligible for stage in item.stage_durations
        }
        result[category] = {
            "availability": "available" if available else "unavailable",
            "scenarios": len(members),
            "eligible": len(eligible),
            "first_attempt_success": sum(item.attempts[0].success for item in eligible),
            "recovered_success": sum(
                not item.attempts[0].success and any(a.success for a in item.attempts[1:])
                for item in eligible
            ),
            "final_success": sum(any(a.success for a in item.attempts) for item in eligible),
            "unnecessary_tool_calls": sum(
                a.unnecessary_tools for item in eligible for a in item.attempts
            ),
            "stale_retrieval_errors": sum(
                a.stale_retrieval for item in eligible for a in item.attempts
            ),
            "unsupported_claim_errors": sum(
                a.unsupported_claim for item in eligible for a in item.attempts
            ),
            "stage_latency_sec": {
                name: latency_summary([
                    value
                    for item in eligible
                    for value in item.stage_durations.get(name, [])
                ])
                for name in sorted(stage_names)
            },
            "end_to_end_sec": latency_summary([
                item.end_to_end_sec for item in eligible if item.end_to_end_sec is not None
            ]),
            "first_useful_text_sec": latency_summary([
                item.first_useful_text_sec
                for item in eligible if item.first_useful_text_sec is not None
            ]),
            "first_useful_spoken_sec": latency_summary([
                item.first_useful_spoken_sec
                for item in eligible if item.first_useful_spoken_sec is not None
            ]),
        }
    return result


def format_markdown(summary: dict[str, dict]) -> str:
    """Render distinct quality outcomes and observed latency as Markdown."""
    lines = [
        "## 🎯 Scenario outcomes",
        "",
        "| Category | Availability | Eligible | First attempt | Recovered | Final success | Unnecessary tools | Stale retrieval | Unsupported claims |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for category, data in sorted(summary.items()):
        lines.append(
            f"| {category} | {data['availability']} | {data['eligible']} "
            f"| {data['first_attempt_success']} | {data['recovered_success']} "
            f"| {data['final_success']} | {data['unnecessary_tool_calls']} "
            f"| {data['stale_retrieval_errors']} | {data['unsupported_claim_errors']} |"
        )
    lines.extend([
        "",
        "## ⏱️ Observed latency (seconds)",
        "",
        "| Category | Stage | Samples | p50 | p95 |",
        "|---|---|---:|---:|---:|",
    ])
    for category, data in sorted(summary.items()):
        durations = {
            **data["stage_latency_sec"],
            "end to end": data["end_to_end_sec"],
            "first useful text": data["first_useful_text_sec"],
            "first useful spoken output": data["first_useful_spoken_sec"],
        }
        for stage, values in durations.items():
            if values is None:
                lines.append(f"| {category} | {stage} | unavailable | unavailable | unavailable |")
            else:
                lines.append(
                    f"| {category} | {stage} | {values['n']} "
                    f"| {values['p50']:.3f} | {values['p95']:.3f} |"
                )
    return "\n".join(lines) + "\n"
