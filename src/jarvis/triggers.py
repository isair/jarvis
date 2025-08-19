from __future__ import annotations
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TriggerResult:
    should_fire: bool
    reasons: list[str]


_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("test_failure", re.compile(r"\b(FAIL|FAILED|AssertionError|expect\(.+\)\.to|\d+ failing)\b", re.IGNORECASE)),
    ("stack_trace", re.compile(r"\b(Traceback|at\s+.+\(.+\)|File \".+\", line \d+)\b", re.IGNORECASE)),
    # Make the npm error pattern robust to optional colon/spacing
    ("npm_error", re.compile(r"\bnpm\s+ERR!?\b", re.IGNORECASE)),
    ("eslint", re.compile(r"\bESLint\s+\d+\s+problems\b|\bParsing error:\b", re.IGNORECASE)),
    ("tsc", re.compile(r"\bTS\d{3,5}:\b|\berror TS\d+\b", re.IGNORECASE)),
    ("merge_conflict", re.compile(r"<<<<<<<\s+HEAD|=======|>>>>>>>\s+", re.MULTILINE)),
    ("ci_red", re.compile(r"\b(\[x\]|failed\s+checks|red\s+badge)\b", re.IGNORECASE)),
]


def evaluate_triggers(text: str) -> TriggerResult:
    reasons: list[str] = []
    for name, pattern in _PATTERNS:
        if pattern.search(text):
            reasons.append(name)
    return TriggerResult(should_fire=len(reasons) > 0, reasons=reasons)
