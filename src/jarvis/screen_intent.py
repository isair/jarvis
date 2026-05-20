"""Detect when the user is asking about what is on their display."""

from __future__ import annotations

import re

# Broad, language-mixed hints — not a full NLU stack; fail-open to normal tool use.
_SCREEN_HINT = re.compile(
    r"(?:"
    r"screen|ekran|ekrān|monitor|display|screenshot|"
    r"what(?:'s| is) on (?:my |the )?screen|see (?:my |the )?screen|"
    r"look at (?:my |the )?screen|on my screen|"
    r"ko (?:tu )?redz|ko redzu|skat(?:ī|i)t(?:ies)? (?:manu )?ekr|"
    r"man(?:ā|a) ekrān|redz(?:ēt|i) (?:manu )?ekr|"
    r"vad(?:ī|i)t (?:manu )?dator|kontrol(?:ē|e) dator|"
    r"control (?:my )?computer|automate (?:my )?desktop"
    r")",
    re.IGNORECASE | re.UNICODE,
)


def mentions_screen(query: str) -> bool:
    """True when the utterance likely refers to on-screen content."""
    text = (query or "").strip()
    if len(text) < 4:
        return False
    return bool(_SCREEN_HINT.search(text))
