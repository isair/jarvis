"""Rule-based Latvian text quality checks (ported from Nimbus, config-driven)."""

from __future__ import annotations

import re
from typing import Any

_ENGLISH_LEAKS = (
    "settings",
    "configuration",
    "performance",
    "dashboard",
    "error",
    "warning",
    "please",
    "hello",
    "thanks",
)

_BAD_PATTERNS: list[tuple[str, str]] = [
    (r"\bjums ir\b", "calque — prefer natural Latvian phrasing"),
    (r"\bBizness\b", "anglicism — consider uzņēmējdarbība"),
    (r"<[^>]+>", "remove HTML/XML tags"),
]

_VOWELS_LV = set("aāeēiīouū")


def check_latvian(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if not raw:
        return {"score": 1.0, "issues": [], "ok": True}

    issues: list[dict[str, str]] = []
    low = raw.lower()
    for word in _ENGLISH_LEAKS:
        if re.search(rf"\b{re.escape(word)}\b", low, re.IGNORECASE):
            issues.append({"kind": "english", "message": f"English leak: «{word}»"})

    for pattern, msg in _BAD_PATTERNS:
        if re.search(pattern, raw, re.IGNORECASE):
            issues.append({"kind": "style", "message": msg})

    letters = [c for c in low if c.isalpha()]
    if len(letters) > 20:
        lvish = sum(1 for c in letters if c in _VOWELS_LV or c in "bcčdfģhjķlļmnņprsštvxz")
        if lvish / len(letters) < 0.85:
            issues.append(
                {"kind": "language", "message": "Text may not be Latvian (low LV character ratio)"}
            )

    score = max(0.0, 1.0 - min(len(issues) * 0.15, 0.85))
    return {"score": round(score, 2), "issues": issues[:12], "ok": len(issues) == 0}


def is_weak_latvian_model(model_id: str) -> bool:
    name = (model_id or "").lower()
    if "latvian" in name or "latvie" in name or "openeuro" in name:
        return False
    return any(m in name for m in ("llama3.2", "llama3.1", "mistral", "phi3", "gemma2:2b"))
