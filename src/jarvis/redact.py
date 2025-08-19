from __future__ import annotations
import re

# Deterministic structural scrub patterns
_REDACTION_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", re.IGNORECASE), "[REDACTED_EMAIL]"),
    (re.compile(r"\b(?:\d[ -]*?){13,19}\b"), "[REDACTED_CARD]"),
    (re.compile(r"\b(AWS|GH|GCP|AZURE|xox[abpcr]-)[A-Za-z0-9_\-]{10,}\b", re.IGNORECASE), "[REDACTED_TOKEN]"),
    (re.compile(r"\b(?:eyJ[0-9A-Za-z._\-]+)\b"), "[REDACTED_JWT]"),
    (re.compile(r"\b(pass(word)?|secret|token|apikey|api_key)\s*[:=]\s*\S+\b", re.IGNORECASE), r"\1=[REDACTED]"),
    (re.compile(r"\b[0-9A-Fa-f]{32,}\b"), "[REDACTED_HEX]"),
    (re.compile(r"\b\d{6}\b(?=.*(otp|2fa|code))", re.IGNORECASE), "[REDACTED_OTP]"),
]


def redact(text: str, max_len: int = 8000) -> str:
    scrubbed = text
    for pattern, repl in _REDACTION_RULES:
        scrubbed = pattern.sub(repl, scrubbed)
    scrubbed = " ".join(scrubbed.split())
    if len(scrubbed) > max_len:
        scrubbed = scrubbed[:max_len]
    return scrubbed
