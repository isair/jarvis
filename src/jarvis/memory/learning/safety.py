"""Extra redaction / rejection gates before lesson persistence."""

from __future__ import annotations

import re
from typing import Optional, Tuple

from ...utils.redact import scrub_secrets

# Values that look like secrets / credentials → reject entirely.
_SECRETISH = re.compile(
    r"(?i)("
    r"\b(password|passwd|parola|api[_-]?key|secret|token|cookie|credential|"
    r"bearer|authorization)\b"
    r"|sk-[A-Za-z0-9]{16,}"
    r"|gh[pousr]_[A-Za-z0-9]{20,}"
    r"|eyJ[A-Za-z0-9._\-]{20,}"
    r"|\.env\b"
    r")"
)

# Full street-ish addresses / IBAN-ish — reject.
_PII_HEAVY = re.compile(
    r"(?i)("
    r"\bIBAN\b|\bRO\d{2}[A-Z0-9]{10,}\b"
    r"|\b\d{1,5}\s+[A-Za-zăâîșț][A-Za-zăâîșț\s\-]{3,40}\s+(nr\.?|strada|str\.)"
    r"|\b(card|cvv|cvc)\b\s*[:=]?\s*\d"
    r")"
)

# Transient facts — clock / weather / arithmetic leftovers.
_TRANSIENT = re.compile(
    r"(?i)\b("
    r"ora\s+\d|ceasul|azi\s+este|vremea|grade\s+celsius|temperatur"
    r"|plus\s+\d|minus\s+\d|\d+\s*\+\s*\d+"
    r")\b"
)

# Prompt-injection shaped content.
_INJECTION = re.compile(
    r"(?i)("
    r"ignore\s+(all\s+)?(previous|prior)\s+instructions"
    r"|system\s*prompt"
    r"|you\s+are\s+now"
    r"|jailbreak"
    r"|<<<\s*BEGIN"
    r")"
)

# One-shot constraint phrasing — not a permanent preference.
_ONESHOT_CONSTRAINT = re.compile(
    r"(?i)\b("
    r"nu\s+instala|nu\s+modifica|doar\s+aceast[aă]|doar\s+o\s+dat[aă]"
    r"|for\s+this\s+(one\s+)?(command|request|task)"
    r"|don'?t\s+install|don'?t\s+modify"
    r")\b"
)


def scrub_lesson_text(text: str) -> str:
    return scrub_secrets(text or "").strip()


def should_reject_lesson_value(value: str, lesson_type: str = "") -> Tuple[bool, str]:
    """Return (reject, reason). Never persist rejected values."""
    if not value or not value.strip():
        return True, "empty"
    v = value.strip()
    if _SECRETISH.search(v):
        return True, "secret_pattern"
    if _PII_HEAVY.search(v):
        return True, "pii_heavy"
    if _INJECTION.search(v):
        return True, "injection"
    if _ONESHOT_CONSTRAINT.search(v) and lesson_type in (
        "user_preference",
        "user_fact",
        "project_context",
    ):
        return True, "oneshot_constraint"
    if _TRANSIENT.search(v) and lesson_type in ("user_fact", "user_preference"):
        return True, "transient"
    # Fully redacted leftovers are useless.
    if re.fullmatch(r"\[REDACTED[^\]]*\]", v):
        return True, "fully_redacted"
    return False, ""


def looks_sensitive_for_ui(value: str, quote: str = "") -> bool:
    blob = f"{value}\n{quote}"
    if _SECRETISH.search(blob) or _PII_HEAVY.search(blob):
        return True
    if "[REDACTED" in blob:
        return True
    return False
