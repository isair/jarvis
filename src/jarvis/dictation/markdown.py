"""Markdown voice-note formatter for the dictation engine.

When ``dictation_markdown_mode`` is enabled, spoken structural cues in the
transcribed text are converted into Markdown syntax before the text is pasted.

This lets a user dictate structured notes hands-free, e.g.:

    "new heading: Project Ideas"     ->  "# Project Ideas"
    "sub heading: this week"         ->  "## this week"
    "bullet point: buy milk"         ->  "- buy milk"
    "numbered: first step"           ->  "1. first step"
    "bold: important"                ->  "**important**"
    "italic: maybe later"            ->  "*maybe later*"
    "code: pip install jarvis"       ->  "`pip install jarvis`"
    "new paragraph"                  ->  (blank line separator)

The formatter is stateless per utterance: each dictated chunk is formatted
independently. This keeps the feature predictable and free of cross-session
state that could surprise the user.
"""

from __future__ import annotations

import re
from typing import List, Tuple

# (compiled pattern, replacement-builder)
# Patterns are matched case-insensitively against the WHOLE utterance first;
# if the entire utterance is a single structural cue, the whole thing is
# replaced. Otherwise inline cues are left alone (raw dictation stays natural).
_STRUCTURAL_CUES: List[Tuple[re.Pattern, str]] = [
    # new heading: <text>  ->  # <text>
    (re.compile(r"^\s*new\s+heading\s*:?\s*(.+)$", re.IGNORECASE), "# {}"),
    # heading: <text>
    (re.compile(r"^\s*heading\s*:?\s*(.+)$", re.IGNORECASE), "# {}"),
    # sub heading / subheading: <text>  ->  ## <text>
    (re.compile(r"^\s*sub[ -]?heading\s*:?\s*(.+)$", re.IGNORECASE), "## {}"),
    # bullet point / bullet: <text>  ->  - <text>
    (re.compile(r"^\s*bullet\s*(?:point)?\s*:?\s*(.+)$", re.IGNORECASE), "- {}"),
    # numbered / number: <text>  ->  1. <text> (caller handles auto-increment)
    (re.compile(r"^\s*numbered\s*:?\s*(.+)$", re.IGNORECASE), "1. {}"),
    (re.compile(r"^\s*number\s*:?\s*(.+)$", re.IGNORECASE), "1. {}"),
    # bold: <text>  ->  **<text>**
    (re.compile(r"^\s*bold\s*:?\s*(.+)$", re.IGNORECASE), "**{}**"),
    # italic / italics: <text>  ->  *<text>*
    (re.compile(r"^\s*italic(?:s)?\s*:?\s*(.+)$", re.IGNORECASE), "*{}*"),
    # code: <text>  ->  `<text>`
    (re.compile(r"^\s*code\s*:?\s*(.+)$", re.IGNORECASE), "`{}`"),
    # link: <text> <url> -> [text](url) (user says "link: label url")
    (re.compile(r"^\s*link\s*:?\s*(.+?)\s+(\S+)$", re.IGNORECASE), "[{}]({})"),
    # new paragraph / new line -> blank-line separator (handled specially)
    (re.compile(r"^\s*(?:new\s+paragraph|new\s+line|line\s+break)\s*$", re.IGNORECASE), "__PARAGRAPH__"),
    # horizontal rule
    (re.compile(r"^\s*(?:divider|horizontal\s+rule|hr)\s*$", re.IGNORECASE), "---"),
    # quote: <text> -> > <text>
    (re.compile(r"^\s*quote\s*:?\s*(.+)$", re.IGNORECASE), "> {}"),
]


def format_markdown(text: str) -> str:
    """Convert spoken Markdown cues in *text* into Markdown syntax.

    The whole utterance is tested against the structural patterns. If it
    matches, the entire utterance is replaced. Otherwise the text is returned
    unchanged (so natural-language dictation is never mangled).

    Returns the formatted Markdown string.
    """
    if not text or not text.strip():
        return text

    stripped = text.strip()

    # Whole-utterance structural cue?
    for pattern, template in _STRUCTURAL_CUES:
        m = pattern.match(stripped)
        if not m:
            continue

        if template == "__PARAGRAPH__":
            # A paragraph break has no text of its own; emit a marker the
            # caller turns into a blank line.
            return "\n\n"

        if template.count("{}") == 2:  # link template uses two groups
            label, url = m.group(1).strip(), m.group(2).strip()
            return template.format(label, url)

        if template.count("{}") == 0:  # no-arg templates (divider, etc.)
            return template

        inner = m.group(1).strip()
        return template.format(inner)

    # No structural cue matched: return the raw text unchanged.
    return text


def format_markdown_lines(lines: List[str]) -> str:
    """Format a list of separate utterances and join them into a Markdown block.

    Each line is formatted independently. Consecutive numbered items are
    auto-numbered so lists dictated across multiple utterances stay sequential.
    Blank ``\\n\\n`` markers (paragraph breaks) are honoured.

    Returns the joined Markdown text ready to paste.
    """
    out: List[str] = []
    numbered_counter = 0

    for raw in lines:
        if not raw or not raw.strip():
            continue
        formatted = format_markdown(raw)
        if formatted == "\n\n":
            # Paragraph break: ensure a blank separator.
            if out and out[-1] != "":
                out.append("")
            continue
        if formatted.startswith("1. "):
            numbered_counter += 1
            formatted = f"{numbered_counter}. " + formatted[3:]
        else:
            # Any non-numbered content resets the counter.
            numbered_counter = 0
        out.append(formatted.rstrip())

    return "\n".join(out).strip()


# Alias used by the dictation engine and the spec.
format_markdown_text = format_markdown
