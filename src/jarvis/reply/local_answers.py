"""Deterministic answers for questions that need no model at all.

Date, clock time and simple arithmetic are facts the process already holds.
Routing them through a 2B model turns a certainty into a guess: asked in
Romanian, with an English context line ("Monday, July 20, 2026"), gemma4:e2b
was observed inventing both the weekday and the word "vigileoptitudine".

Answering these locally is not an optimisation — it removes a class of
hallucination, and it costs no web call, no memory lookup and no LLM turn.

Anything not matched here returns None and falls through to the normal
pipeline. The matchers are deliberately narrow: a false positive would hijack
a question the model should have answered.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from typing import Optional

# Romanian calendar names — the whole point is not to make the model translate.
_ZILE = ["luni", "marți", "miercuri", "joi", "vineri", "sâmbătă", "duminică"]
_LUNI = [
    "ianuarie", "februarie", "martie", "aprilie", "mai", "iunie",
    "iulie", "august", "septembrie", "octombrie", "noiembrie", "decembrie",
]

_NUMERALE = {
    "zero": 0, "unu": 1, "una": 1, "doi": 2, "două": 2, "trei": 3, "patru": 4,
    "cinci": 5, "șase": 6, "sase": 6, "șapte": 7, "sapte": 7, "opt": 8,
    "nouă": 9, "noua": 9, "zece": 10, "unsprezece": 11, "doisprezece": 12,
    "douăsprezece": 12, "treisprezece": 13, "paisprezece": 14,
    "patrusprezece": 14, "cincisprezece": 15, "șaisprezece": 16,
    "saisprezece": 16, "șaptesprezece": 17, "saptesprezece": 17,
    "optsprezece": 18, "nouăsprezece": 19, "nouasprezece": 19,
    "douăzeci": 20, "douazeci": 20, "treizeci": 30, "patruzeci": 40,
    "cincizeci": 50, "șaizeci": 60, "saizeci": 60, "sută": 100, "suta": 100,
}

_OPERATII = {
    "plus": "+", "adunat cu": "+", "și cu": "+", "si cu": "+", "adaug": "+",
    "minus": "-", "fără": "-", "fara": "-", "scăzut": "-", "scazut": "-",
    "ori": "*", "înmulțit cu": "*", "inmultit cu": "*", "înmulțit": "*",
    "împărțit la": "/", "impartit la": "/", "supra": "/",
}


def _fold(text: str) -> str:
    """Lowercase and strip diacritics — Whisper drops them unpredictably."""
    t = unicodedata.normalize("NFD", (text or "").lower())
    return "".join(c for c in t if unicodedata.category(c) != "Mn")


def _minute_phrase(n: int) -> str:
    """Romanian minute agreement — not the English one/many split.

    Romanian inserts "de" from 20 upward: 19 minute, but 20 de minute. Using
    `n == 1` as the pivot (correct for English) produced "9 de minute".
    """
    if n == 1:
        return "un minut"
    if n < 20:
        return f"{n} minute"
    return f"{n} de minute"


# --------------------------------------------------------------------- date

_DATE_RE = re.compile(
    r"\b(ce (zi|data) (este|e|avem)|in ce zi (suntem|sintem)|"
    r"data de (azi|astazi)|ce data e|cata zi e|"
    r"ce zi (e|este) (azi|astazi)|azi ce zi (e|este))\b"
)


def _answer_date(folded: str, now: datetime) -> Optional[str]:
    if not _DATE_RE.search(folded):
        return None
    return (
        f"Astăzi este {_ZILE[now.weekday()]}, "
        f"{now.day} {_LUNI[now.month - 1]} {now.year}."
    )


# --------------------------------------------------------------------- time

# After _fold, "cât" → "cat" and "câte" → "cate". Whisper often emits the
# latter for the clock question ("Cora, câte este ceasul?"), so both stems
# must match. Keep the pattern narrow: require (e|este) + (ceasul|ora) so
# "câte ceasuri sunt" / "ceasul biologic" still fall through to the LLM.
_TIME_RE = re.compile(
    r"\b((cat|cate) (e|este) (ceasul|ora)|ce ora (e|este)|"
    r"spune-?mi ora|ora exacta|(cat|cate) arata ceasul)\b"
)


def _answer_time(folded: str, now: datetime) -> Optional[str]:
    if not _TIME_RE.search(folded):
        return None
    h, m = now.hour, now.minute
    if m == 0:
        return f"Este ora {h} fix."
    return f"Este ora {h} și {_minute_phrase(m)}."


# --------------------------------------------------------------- arithmetic

def _to_number(token: str) -> Optional[int]:
    token = token.strip()
    if token.isdigit():
        return int(token)
    return _NUMERALE.get(_fold(token))


_ARITH_RE = re.compile(
    r"\b(?:cat (?:fac|face|e|este))\b(?P<expr>.{1,60}?)\s*[?.!]*$"
)


def _answer_arithmetic(folded: str, original: str) -> Optional[str]:
    m = _ARITH_RE.search(folded)
    if not m:
        return None
    expr = m.group("expr")

    op = None
    for word, symbol in _OPERATII.items():
        if re.search(rf"\b{re.escape(_fold(word))}\b", expr):
            op = symbol
            parts = re.split(rf"\b{re.escape(_fold(word))}\b", expr, maxsplit=1)
            break
    if op is None or len(parts) != 2:
        return None

    left, right = (_to_number(p) for p in parts)
    if left is None or right is None:
        return None
    if op == "/" and right == 0:
        return "Nu se poate împărți la zero."

    result = {"+": left + right, "-": left - right,
              "*": left * right, "/": left / right}[op]
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    if isinstance(result, float):
        result = round(result, 4)
    verb = {"+": "plus", "-": "minus", "*": "ori", "/": "împărțit la"}[op]
    return f"{left} {verb} {right} fac {result}."


# ------------------------------------------------------------------- public

def try_local_answer(text: str, now: Optional[datetime] = None) -> Optional[str]:
    """Return a deterministic Romanian answer, or None to use the full pipeline.

    Args:
        text: the user's query, already stripped of the wake word.
        now: injected for testing; defaults to the real local clock.
    """
    if not text or not text.strip():
        return None
    now = now or datetime.now()
    folded = _fold(text)

    for handler in (
        lambda: _answer_date(folded, now),
        lambda: _answer_time(folded, now),
        lambda: _answer_arithmetic(folded, text),
    ):
        answer = handler()
        if answer:
            return answer
    return None
