"""Deterministic residence-claim checks for English-language evaluation fixtures."""

import re


def has_current_residence_claim(text: str, place: str) -> bool:
    """Require an affirmative link between the user and their current home."""
    value = text.casefold().replace("’", "'")
    location = re.escape(place.casefold())
    subject = r"(?:the\s+)?user|they"
    possessive = r"(?:the\s+)?user's|their"
    patterns = (
        rf"\b(?:{subject})\s+(?:(?:now|currently)\s+)?(?:lives?|resides?)\s+in\s+{location}\b",
        rf"\b(?:{subject})\s+(?:is|are)\s+(?:(?:now|currently)\s+)?based\s+in\s+{location}\b",
        rf"\b(?:{possessive})\s+(?:current\s+)?(?:home|residence|address|city|location)\s+(?:is|:)\s+{location}\b",
        rf"\b(?:current\s+)?(?:home|residence|location)\s*:\s*{location}\b",
        rf"\b{location}\s+is\s+where\s+(?:{subject})\s+(?:currently\s+)?(?:lives?|resides?)\b",
        rf"\b{location}\s+is\s+(?:{possessive})\s+(?:current\s+)?(?:home|residence|city|location)\b",
    )
    return any(re.search(pattern, value) for pattern in patterns)
