from __future__ import annotations
import re

# Deterministic structural scrub patterns. Order matters: specific
# vendor-shaped tokens are matched before generic catches so the more
# informative label wins (e.g. "[REDACTED_AWS_KEY]" beats "[REDACTED_HEX]").
_REDACTION_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", re.IGNORECASE), "[REDACTED_EMAIL]"),
    (re.compile(r"\b(?:\d[ -]*?){13,19}\b"), "[REDACTED_CARD]"),
    # Vendor-specific access keys (bare, no surrounding keyword required).
    (re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), "[REDACTED_AWS_KEY]"),
    (re.compile(r"\b(?:sk|pk|rk)_(?:live|test)_[A-Za-z0-9]{16,}\b"), "[REDACTED_STRIPE_KEY]"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"), "[REDACTED_GH_TOKEN]"),
    (re.compile(r"\bsk-[A-Za-z0-9]{32,}\b"), "[REDACTED_OPENAI_KEY]"),
    (re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"), "[REDACTED_GOOG_KEY]"),
    # Authorisation headers — Bearer/Basic carry credentials in line.
    (re.compile(r"Authorization:\s*Bearer\s+\S+", re.IGNORECASE), "Authorization: Bearer [REDACTED]"),
    (re.compile(r"Authorization:\s*Basic\s+[A-Za-z0-9+/=]+", re.IGNORECASE), "Authorization: Basic [REDACTED]"),
    # Generic prefix catch — left after the vendor-specific rules so
    # newer formats like gh[pousr]_ get a precise label first.
    (re.compile(r"\b(AWS|GH|GCP|AZURE|xox[abpcr]-)[A-Za-z0-9_\-]{10,}\b", re.IGNORECASE), "[REDACTED_TOKEN]"),
    (re.compile(r"\b(?:eyJ[0-9A-Za-z._\-]+)\b"), "[REDACTED_JWT]"),
    # Keyword-anchored credentials. Covers refresh/access/oauth/session
    # variants in addition to the original pass/secret/token/apikey set.
    (re.compile(
        r"\b(pass(?:word)?|secret|token|apikey|api_key|"
        r"(?:refresh|access|id|oauth)_?token|session(?:_?id)?|sid)"
        r"\s*[:=]\s*\S+\b",
        re.IGNORECASE,
    ), r"\1=[REDACTED]"),
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


def scrub_secrets(text: str) -> str:
    """Apply the structural scrub rules without whitespace collapse or length cap.

    Use for structured content (tool output, multi-line payloads) where
    preserving newlines matters but tokens/emails/etc. must still be masked.
    """
    scrubbed = text
    for pattern, repl in _REDACTION_RULES:
        scrubbed = pattern.sub(repl, scrubbed)
    return scrubbed
