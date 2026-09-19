"""Local Hunspell post-processing for the final Whisper transcript.

Pipeline position::

    audio -> Whisper -> final transcript -> this module -> wake/command/intent/LLM

Only the *final* transcript is touched; in-progress partial events keep their
original text. Both values always survive: :attr:`TranscriptCorrection.raw` and
:attr:`TranscriptCorrection.corrected`.

The layer is a conservative typo/diacritics repair, not a grammar corrector:
Hunspell works word by word, so a dictionary-valid-but-contextually-wrong word
is left alone. Dictionaries are plain ``.aff`` + ``.dic`` pairs vendored under
``jarvis/resources/hunspell`` (see ``SOURCES.md``) and read by ``spylls``, so a
production install needs no LibreOffice, no system Hunspell and no network.
"""

from __future__ import annotations

import re
import sys
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import as_file, files
from pathlib import Path

from spylls.hunspell import Dictionary

# Whisper's ISO-639-1 label (and a couple of common script-tag spellings) to the
# vendored dictionary id.
WHISPER_TO_DICTIONARY = {
    "en": "en_US",
    "en-US": "en_US",
    "en_US": "en_US",
    "cs": "cs_CZ",
    "cs-CZ": "cs_CZ",
    "cs_CZ": "cs_CZ",
    "vi": "vi_VN",
    "vi-VN": "vi_VN",
    "vi_VN": "vi_VN",
    "sk": "sk_SK",
    "sk-SK": "sk_SK",
    "sk_SK": "sk_SK",
}

# Word tokens: letter runs, allowing the in-word separators used by the four
# supported languages. Digits and underscores are excluded by construction, so
# technical tokens (``HTTP2``, ``py_token``) never become word tokens.
WORD_RE = re.compile(r"(?u)[^\W\d_]+(?:['’\-][^\W\d_]+)*")

# Work bound for the voice loop: at most this many word tokens per transcript,
# and at most this many Hunspell candidates per token.
_MAX_WORDS = 256
_MAX_SUGGESTIONS = 5
# Long tokens get the wider repair budget.
_LONG_TOKEN_LEN = 8

# Languages whose word segmentation is preserved exactly (complex-word spaces).
_TOKEN_PRESERVING_LANGUAGES = frozenset({"vi"})

_DAMERAU_CACHE: dict[tuple[str, str], int] = {}


@dataclass(frozen=True, slots=True)
class TranscriptCorrection:
    """Outcome of one post-processing pass.

    ``raw`` is Whisper's text untouched, ``corrected`` is what downstream
    consumers see, ``replacements`` pairs each swapped token with its
    replacement in document order.
    """

    raw: str
    corrected: str
    language: str
    replacements: tuple[tuple[str, str], ...]
    #: True only when the requested dictionary was loaded and every eligible
    #: token passed through the Hunspell decision loop.
    checked: bool = False


@lru_cache(maxsize=4)
def _load_dictionary(dictionary_id: str) -> Dictionary:
    """Load one ``.aff``/``.dic`` pair lazily, once per dictionary id."""
    if getattr(sys, "frozen", False):
        base = getattr(sys, "_MEIPASS", None)
        if base:
            stem = Path(base) / "jarvis" / "resources" / "hunspell" / dictionary_id / dictionary_id
            if stem.with_suffix(".aff").exists():
                return Dictionary.from_files(str(stem))
    root = files("jarvis.resources.hunspell").joinpath(dictionary_id)
    stem = root.joinpath(dictionary_id)
    with as_file(stem) as local_stem:
        return Dictionary.from_files(str(local_stem))


def _damerau(a: str, b: str) -> int:
    """Damerau-Levenshtein distance (adjacent transpositions counted once)."""
    key = (a, b)
    cached = _DAMERAU_CACHE.get(key)
    if cached is not None:
        return cached
    la, lb = len(a), len(b)
    prev2: list[int] = []
    prev: list[int] = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(
                cur[j - 1] + 1,          # insertion
                prev[j] + 1,             # deletion
                prev[j - 1] + cost,      # substitution
            )
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                cur[j] = min(cur[j], prev2[j - 2] + 1)  # transposition
        prev2, prev = prev, cur
    distance = prev[lb]
    if len(_DAMERAU_CACHE) < 4096:
        _DAMERAU_CACHE[key] = distance
    return distance


def _resolve_dictionary_id(language: str | None) -> str | None:
    """Map a Whisper language label to a dictionary id, else ``None``."""
    if not language:
        return None
    label = str(language).strip()
    if not label:
        return None
    for candidate in (label, label.lower(), label.replace("_", "-").lower()):
        dictionary_id = WHISPER_TO_DICTIONARY.get(candidate)
        if dictionary_id:
            return dictionary_id
    return None


def _is_protected(token: str, protected_folded: frozenset[str]) -> bool:
    """True for tokens the layer must not rewrite."""
    if len(token) < 2:                                  # single letter
        return True
    if token.casefold() in protected_folded:
        return True
    upper_flags = [ch.isupper() for ch in token if ch.isalpha()]
    if "@" in token:                                    # e-mail
        return True
    if "://" in token or token.lower().startswith("www."):
        return True
    if "." in token and not token.endswith("."):        # dotted technical token / TLD
        return True
    caps = {i for i, ch in enumerate(token) if ch.isupper()}
    if not any(upper_flags):
        return False                                    # all lower
    if caps == {0}:
        return False                                    # Capitalised
    if caps == set(range(len(token))) and token.isupper():
        return True                                     # CAPS abbreviation
    return True                                         # CamelCase / mixed forms


def _identity_fold(value: str) -> str:
    """Case/diacritic-insensitive identity used only for named aliases."""
    return "".join(
        ch
        for ch in unicodedata.normalize("NFKD", str(value)).casefold()
        if not unicodedata.combining(ch)
    )


def _match_case(token: str, candidate: str) -> str:
    """Apply the token's casing shape to a candidate."""
    if token.islower():
        return candidate.lower()
    if token.istitle():
        return candidate[:1].upper() + candidate[1:].lower() if len(candidate) > 1 else candidate.upper()
    if token.isupper():
        return candidate.upper()
    return candidate


def _protected_prefix_match(token: str, folded_protected: frozenset[str]) -> str | None:
    """Complete an unfinished token from the protected vocabulary.

    A protected term is a name the layer must keep verbatim, so it is the
    authoritative spelling of its own prefix: ``toastova`` is not a Czech word,
    while ``toastovač`` is the wake word. Returns the unique *shortest* protected
    term that strictly extends ``token``; ``None`` when there is no such term or
    when two same-length terms compete. Multi-word terms cannot complete a
    single token and are skipped.
    """
    if not folded_protected:
        return None
    folded_token = unicodedata.normalize("NFC", token).casefold()
    if not folded_token:
        return None

    extensions: list[str] = []
    for term in folded_protected:
        if not term or " " in term:
            continue
        if term.startswith(folded_token) and term != folded_token:
            extensions.append(term)
    if not extensions:
        return None

    shortest_len = min(len(term) for term in extensions)
    shortest = {term for term in extensions if len(term) == shortest_len}
    if len(shortest) != 1:
        return None
    return next(iter(shortest))


def _warn_once(warnings: list[str], message: str) -> None:
    if message not in warnings:
        warnings.append(message)


def correct_transcript(
    text: str,
    language: str | None,
    *,
    enabled: bool = True,
    protected_terms: frozenset[str] = frozenset(),
    canonical_terms: Mapping[str, str] | None = None,
) -> TranscriptCorrection:
    """Spell-correct a final transcript for ``language``, bypassing otherwise.

    Bypass cases: the layer is disabled, the text is empty, the language has no
    vendored dictionary, the dictionary cannot be read, or the token is
    protected. Every bypass returns the original text with an empty
    ``replacements`` tuple.
    """
    raw = text if isinstance(text, str) else str(text or "")
    lang = (str(language).strip().lower() if language else "")

    if not enabled or not raw.strip():
        return TranscriptCorrection(raw=raw, corrected=raw, language=lang, replacements=())

    dictionary_id = _resolve_dictionary_id(language)
    if dictionary_id is None:
        return TranscriptCorrection(raw=raw, corrected=raw, language=lang, replacements=())

    # NFC only: no NFKD, no diacritic stripping, no case changes.
    folded_protected = frozenset(
        str(term).casefold() for term in protected_terms if str(term)
    )
    # Casing-shape comparison happens on the original token; the folded set is
    # matched against the casefolded token, so `Toustovac` and `toustovač` both
    # resolve to their protected entry while the output keeps its own spelling.
    normalized = unicodedata.normalize("NFC", raw)
    folded_protected = frozenset(folded_protected) | frozenset(
        unicodedata.normalize("NFC", term).casefold() for term in protected_terms if str(term)
    )
    canonical_by_fold = {
        _identity_fold(alias): unicodedata.normalize("NFC", str(canonical))
        for alias, canonical in (canonical_terms or {}).items()
        if str(alias).strip()
        and str(canonical).strip()
        and " " not in str(alias).strip()
        and " " not in str(canonical).strip()
    }

    try:
        dictionary = _load_dictionary(dictionary_id)
    except Exception as exc:  # missing or broken dictionary: transparent bypass
        from ..debug import debug_log

        debug_log(
            f"event=speech_spellcheck_dictionary_unavailable language={lang} "
            f"dictionary={dictionary_id} error={type(exc).__name__}",
            "voice",
        )
        return TranscriptCorrection(
            raw=raw, corrected=normalized, language=lang, replacements=()
        )

    is_strict_segmentation = dictionary_id in {
        WHISPER_TO_DICTIONARY[code] for code in _TOKEN_PRESERVING_LANGUAGES
    }
    allowed_distance_by_len = _LONG_TOKEN_LEN

    replacements: list[tuple[str, str]] = []
    pieces: list[str] = []
    cursor = 0
    processed = 0

    for match in WORD_RE.finditer(normalized):
        token = match.group()
        pieces.append(normalized[cursor:match.start()])
        cursor = match.end()

        # Brand/wake aliases are identities, not ordinary spelling guesses.
        # Canonicalise them before Hunspell can turn e.g. ``toastováč`` into
        # the valid Czech adjective ``toastová``. NFKD folding deliberately
        # accepts Whisper's unstable diacritics while preserving the configured
        # canonical spelling in downstream wake detection.
        canonical = canonical_by_fold.get(_identity_fold(token))
        if canonical is not None:
            replacement = _match_case(token, canonical)
            if replacement != token:
                replacements.append((token, replacement))
            pieces.append(replacement)
            continue

        if folded_protected and token.casefold() in folded_protected:
            pieces.append(token)
            continue
        if _is_protected(token, folded_protected):
            pieces.append(token)
            continue
        if processed >= _MAX_WORDS:
            # Remaining words ride verbatim; the voice loop stays unblocked.
            processed += 1
            pieces.append(token)
            continue
        processed += 1

        try:
            known = bool(dictionary.lookup(token))
        except Exception:
            known = True
        if known:
            pieces.append(token)
            continue

        # Protected names are the authoritative spelling of their own prefix, so
        # an unfinished token is completed from that vocabulary before the
        # dictionary ranking runs. This is what turns the truncated wake word
        # `toastova` into `toastovač` even though the Czech dictionary offers
        # several equally-close inflections of `toastov` (`toastová`,
        # `toastově`, `toastové`, `toastový`, `toastoví`) and the plain
        # uniqueness gate would leave the token untouched.
        prefix_term = _protected_prefix_match(token, folded_protected)
        if prefix_term is not None:
            prefix_replacement = _match_case(token, prefix_term)
            if prefix_replacement and prefix_replacement != token:
                replacements.append((token, prefix_replacement))
                pieces.append(prefix_replacement)
                continue
            pieces.append(token)
            continue

        try:
            suggestions = list(dictionary.suggest(token))[:_MAX_SUGGESTIONS]
        except Exception:
            suggestions = []

        if is_strict_segmentation:
            # vi: Hunspell cannot decide compound-word spaces, so only
            # one-token-to-one-token suggestions are eligible.
            suggestions = [s for s in suggestions if " " not in s]

        if not suggestions:
            pieces.append(token)
            continue

        ranked: list[tuple[int, int, str]] = []
        for order, candidate in enumerate(suggestions):
            if not candidate or candidate == token:
                continue
            ranked.append((_damerau(token, candidate), order, candidate))
        if not ranked:
            pieces.append(token)
            continue
        ranked.sort(key=lambda item: (item[0], item[1]))

        best_distance = ranked[0][0]
        best = {candidate for distance, _order, candidate in ranked if distance == best_distance}
        max_allowed = 2 if len(token) >= allowed_distance_by_len else 1
        if len(best) != 1 or best_distance > max_allowed:
            pieces.append(token)
            continue

        replacement = _match_case(token, next(iter(best)))
        if not replacement or replacement == token:
            pieces.append(token)
            continue
        replacements.append((token, replacement))
        pieces.append(replacement)

    pieces.append(normalized[cursor:])
    corrected = "".join(pieces)

    if normalized != raw and not replacements:
        # NFC alone counts as a change of the transcript text for the log.
        replacements.append(("", normalized))

    return TranscriptCorrection(
        raw=raw,
        corrected=corrected,
        language=lang,
        replacements=tuple(replacements),
        checked=True,
    )


def format_correction_event(
    correction: TranscriptCorrection,
    *,
    include_text: bool,
    latency_ms: float,
) -> str:
    """One structured log line for a real change.

    ``include_text`` follows the active privacy/telemetry setting: with it off,
    only the language, the change count and the latency are emitted.
    """
    parts = [
        "event=speech_transcript_corrected",
        f"language={correction.language or 'unknown'}",
        f"replacement_count={len(correction.replacements)}",
        f"latency_ms={latency_ms:.1f}",
    ]
    if include_text:
        parts.append(f"raw_text={correction.raw}")
        parts.append(f"corrected_text={correction.corrected}")
    return " ".join(parts)
