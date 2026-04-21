"""
Compound-query decomposition helper.

Small models (text-based tool calling) struggle to multi-step when a user asks
two questions joined by a conjunction — they answer one side and stop. The
engine splits such queries upfront so it can inject a targeted "still
unanswered" nudge after each tool result.

Language-aware: conjunction shape varies wildly across languages (whitespace
boundaries for Latin/Cyrillic, character-level for CJK, enclitic particles
for Arabic/Hebrew that can't be split on safely). We keep a small per-
language rule table and fall back to "no decomposition" when the language
is unknown, rather than misapplying rules from a different family.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

# Minimum length of EACH sub-clause after the split. Empirical default tuned
# against ``evals/test_complex_flows.py::TestMultiStepEntityQuery`` — filters
# out short idiomatic phrases (English "rock and roll", French "va et vient",
# German "hin und her") without dropping typical multi-part entity queries
# whose clauses usually exceed 15 characters each. CJK languages use a
# smaller threshold (see ``_RULES``) because each character carries far more
# semantic weight than a Latin letter.
DEFAULT_MIN_CLAUSE_CHARS = 9
CJK_MIN_CLAUSE_CHARS = 4
# Back-compat alias kept for existing tests that imported the original constant.
MIN_CLAUSE_CHARS = DEFAULT_MIN_CLAUSE_CHARS


@dataclass(frozen=True)
class _LangRule:
    """Splitting policy for one language.

    ``pattern`` matches the conjunction boundary. For languages that use
    whitespace between words the pattern includes ``\\s+`` padding; for CJK
    it matches the conjunction character(s) directly so "电影和音乐" splits
    cleanly without requiring authors to insert spaces.
    """
    pattern: re.Pattern[str]
    min_clause_chars: int = DEFAULT_MIN_CLAUSE_CHARS


def _ws(words: str) -> re.Pattern[str]:
    """Whitespace-bounded conjunction pattern, case-insensitive."""
    return re.compile(rf"\s+(?:{words})\s+", flags=re.IGNORECASE)


# Per-language rules. Only languages we can reasonably vouch for — either
# structurally (whitespace-separated families where the pattern is
# mechanical) or with explicit testing (see ``tests/test_compound_query.py``).
# Languages outside this table fall through to "no decomposition" rather
# than risk mis-splitting with borrowed rules.
_RULES: dict[str, _LangRule] = {
    # ── Germanic / Romance (whitespace-separated) ─────────────────────────
    "en": _LangRule(_ws("and")),
    "es": _LangRule(_ws("y|e")),                 # "e" before i-/hi- words
    "fr": _LangRule(_ws("et")),
    "de": _LangRule(_ws("und")),
    "pt": _LangRule(_ws("e")),
    "it": _LangRule(_ws("e|ed")),                # "ed" before vowel
    "nl": _LangRule(_ws("en")),
    "sv": _LangRule(_ws("och")),
    "no": _LangRule(_ws("og")),                  # Norwegian (Bokmål)
    "da": _LangRule(_ws("og")),                  # Danish
    "fi": _LangRule(_ws("ja|sekä")),             # Finnish
    # ── Slavic (Cyrillic + Latin) ─────────────────────────────────────────
    "ru": _LangRule(_ws("и|а также")),
    "uk": _LangRule(_ws("і|та|й")),              # Ukrainian — і / та / й
    "be": _LangRule(_ws("і|ды")),                # Belarusian
    "pl": _LangRule(_ws("i|oraz")),
    "cs": _LangRule(_ws("a|i")),                 # Czech
    "sk": _LangRule(_ws("a|i")),                 # Slovak
    "bg": _LangRule(_ws("и")),                   # Bulgarian
    "sr": _LangRule(_ws("и|i")),                 # Serbian (both scripts)
    "hr": _LangRule(_ws("i")),                   # Croatian
    "sl": _LangRule(_ws("in")),                  # Slovenian
    # ── Other European ────────────────────────────────────────────────────
    "el": _LangRule(_ws("και|κι")),              # Greek
    "tr": _LangRule(_ws("ve")),
    "hu": _LangRule(_ws("és|meg")),              # Hungarian
    "ro": _LangRule(_ws("și|şi")),               # Romanian (both diacritics)
    # ── Asian (whitespace-separated) ──────────────────────────────────────
    "vi": _LangRule(_ws("và")),                  # Vietnamese
    "id": _LangRule(_ws("dan")),                 # Indonesian
    "ms": _LangRule(_ws("dan")),                 # Malay
    "hi": _LangRule(_ws("और|तथा")),              # Hindi (Devanagari)
    # ── CJK (no whitespace around conjunctions) ───────────────────────────
    # Chinese: 和 / 与 / 以及 / 并且 — common coordinating conjunctions.
    # Pattern matches either a character-level conjunction OR the two-char
    # forms. Clause-length threshold is lowered to CJK_MIN_CLAUSE_CHARS
    # because each Han character carries word-level meaning.
    "zh": _LangRule(
        re.compile(r"以及|并且|以及|和|与"),
        min_clause_chars=CJK_MIN_CLAUSE_CHARS,
    ),
    # Japanese: そして / および / また are freestanding sentence-level
    # connectors. We intentionally avoid the enclitic particles と/や —
    # they attach to nouns and splitting on them produces nonsense. Users
    # who write multi-part questions typically use the freestanding forms.
    "ja": _LangRule(
        re.compile(r"そして|および|また|かつ"),
        min_clause_chars=CJK_MIN_CLAUSE_CHARS,
    ),
    # Korean: 그리고 / 및 are freestanding; 와/과 are postpositional
    # particles attached to the preceding noun, so we avoid those for the
    # same reason as Japanese. Allow optional whitespace around the
    # freestanding forms since Korean usage varies.
    "ko": _LangRule(
        re.compile(r"\s*(?:그리고|및)\s*"),
        min_clause_chars=CJK_MIN_CLAUSE_CHARS,
    ),
}
# Languages NOT included on purpose:
# - Arabic (ar) / Hebrew (he): the conjunction "و" / "ו" is an enclitic
#   prefix attached directly to the following word (e.g. "وكتاب" = "and a
#   book"). A safe split would need a morphological tokenizer; a regex
#   produces silent false positives on every word starting with "و"/"ו".
# - Thai (th), Khmer (km), Lao (lo): no inter-word whitespace and the
#   conjunctions overlap common syllables; same tokenizer requirement as
#   above, without a cheap workaround.


def _normalise_language(language: Optional[str]) -> Optional[str]:
    """Return a lowercase ISO-639-1 code or None for unknown input.

    Accepts locale-style codes like "en-US" or "zh-CN" and returns the
    primary subtag. Returns None for empty strings, non-strings, or
    tags whose primary subtag is not a valid ISO-639-1 alpha-2 code.
    """
    if not language or not isinstance(language, str):
        return None
    code = language.strip().lower().split("-")[0][:2]
    return code if code.isalpha() and len(code) == 2 else None


def split_compound_query(text: str, language: Optional[str] = None) -> list[str]:
    """Split a compound question into ordered sub-questions.

    Returns an empty list when the query is not compound, the language is
    unknown/unsupported, or either clause is shorter than the language's
    minimum clause length. Callers should treat an empty list as "run the
    query as a single unit" — we never guess across languages we don't
    explicitly support.
    """
    if not text or not isinstance(text, str):
        return []

    # Default to English when language is not provided (non-voice entrypoints
    # like evals and text chat carry no ISO code). Voice flows always pass a
    # Whisper-detected language; if that language isn't in our table, we
    # return no decomposition rather than fall back to English and mis-split.
    code = _normalise_language(language) or "en"
    rule = _RULES.get(code)
    if rule is None:
        return []

    parts = rule.pattern.split(text, maxsplit=1)
    if len(parts) != 2:
        return []

    left, right = parts[0].strip(), parts[1].strip()
    if len(left) < rule.min_clause_chars or len(right) < rule.min_clause_chars:
        return []
    return [left, right]
