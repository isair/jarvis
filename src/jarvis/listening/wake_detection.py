"""Wake word and stop command detection logic.

The wake pipeline is Whisper-driven (no Porcupine): a multilingual faster-whisper
/ MLX pass feeds these text matchers. All three functions operate on lowercase
text and use a shared diacritic-folding normaliser so Czech vocative variants
(toustovač / toustovači) and accent-free transcriptions match consistently.
"""

from typing import List, Optional
import difflib
import unicodedata

from ..debug import debug_log


def _fold(text: str) -> str:
    """Normalise for wake matching: lowercase, and remove Czech diacritics
    (NFKD combining marks) so 'toustovač', 'toustovači' and 'toustovac' fold
    onto one stem. NFKD is applied after lowercasing and the combining-mark
    filter keeps every remaining character at its original index (one fold
    char per source char), so folded offsets map 1:1 back onto the source.
    """
    if not text:
        return ""
    folded = unicodedata.normalize("NFKD", text.strip().lower())
    return "".join(ch for ch in folded if not unicodedata.combining(ch))


def _matchable(text: str) -> str:
    """Fold text but keep interior punctuation as spaces so phrase aliases
    (e.g. 'hej toustovač') still line up with the spoken token stream."""
    folded = _fold(text)
    for ch in ".,;:!?":
        folded = folded.replace(ch, " ")
    return " ".join(folded.split())


def _all_aliases(wake_word: str, aliases: List[str]) -> List[str]:
    """Unique alias list, primary word first, longest-first afterwards."""
    seen: list[str] = []
    for candidate in [wake_word, *aliases]:
        folded = _matchable(candidate)
        if folded and folded not in seen:
            seen.append(folded)
    # Longest-first so 'hej toustovač' wins over 'toustovač'.
    return sorted(seen, key=len, reverse=True)


def is_wake_word_detected(text_lower: str, wake_word: str, aliases: List[str], fuzzy_ratio: float = 0.78) -> bool:
    """
    Check if text contains wake word using exact and fuzzy matching.

    Matching is phrase-aware: multi-word aliases such as "hej toustovač" are
    folded and matched as substrings first (longest alias wins), then single
    tokens fall back to fuzzy matching for common Whisper variants.

    Args:
        text_lower: Lowercase text to check
        wake_word: Primary wake word
        aliases: List of wake word aliases
        fuzzy_ratio: Threshold for fuzzy matching (0.0-1.0)

    Returns:
        True if wake word detected
    """
    if not text_lower or not text_lower.strip():
        return False

    folded_text = _matchable(text_lower)
    if not folded_text:
        return False

    # Exact substring match, longest alias first (phrase or bare name).
    for alias in _all_aliases(wake_word, aliases):
        if alias in folded_text:
            return True

    # Fuzzy matching for close per-token variations.
    try:
        heard_tokens = [t.strip(".,!?;:()[]{}\"'`).-_/") for t in folded_text.split() if t.strip()]
        for token in heard_tokens:
            for alias in _all_aliases(wake_word, aliases):
                for alias_token in alias.split(" "):
                    ratio = difflib.SequenceMatcher(a=alias_token, b=token).ratio()
                    if ratio >= fuzzy_ratio:
                        debug_log(f"wake word fuzzy match: '{alias_token}' ~ '{token}' (ratio: {ratio:.3f})", "wake")
                        return True
    except Exception:
        pass

    return False


def extract_query_after_wake(text_lower: str, wake_word: str, aliases: List[str]) -> str:
    """
    Extract the query portion after removing the wake phrase.

    Only the ONE matched phrase (the longest alias present) is removed from
    the normalised request, so a one-shot "Hej toustovač, jaké je počasí v
    Praze?" yields "jaké je počasí v praze?" without leaving "hej" behind.

    Args:
        text_lower: Lowercase text containing wake word
        wake_word: Primary wake word
        aliases: List of wake word aliases

    Returns:
        Query text with the wake phrase removed (diacritics preserved via the
        original string; only the fold is used for locating the phrase).
    """
    if not text_lower:
        return ""

    folded_text = _matchable(text_lower)
    if not folded_text:
        return ""

    # Match on the collapsed fold, but track a position map back into the
    # original (diacritic-preserved) string so the removed span can be cut
    # from the source without mangling "Počasí" → "Pocasi".
    raw = text_lower.lower()
    fold_chars: list[str] = []
    orig_idx: list[int] = []
    for i, ch in enumerate(raw):
        if ch in ".,;:!?()[]{}\"'`/":
            fold_chars.append(" ")
            orig_idx.append(i)
            continue
        for dec in unicodedata.normalize("NFKD", ch):
            if unicodedata.combining(dec):
                continue
            fold_chars.append(dec.lower())
            orig_idx.append(i)

    # Collapse runs of spaces while preserving the index map.
    collapsed: list[str] = []
    collapsed_idx: list[int] = []
    for ch, oi in zip(fold_chars, orig_idx):
        if ch == " " and collapsed and collapsed[-1] == " ":
            continue
        collapsed.append(ch)
        collapsed_idx.append(oi)
    mapped = "".join(collapsed)

    # Locate the longest alias occurrence, then cut that span from the
    # original string.
    matched: Optional[tuple[int, int]] = None
    for alias in _all_aliases(wake_word, aliases):
        start = mapped.find(alias)
        if start != -1:
            matched = (start, start + len(alias))
            break
    if matched is None:
        fragment = folded_text
    else:
        start, end = matched
        start_orig = collapsed_idx[start]
        end_orig = collapsed_idx[end - 1] + 1 if end - 1 < len(collapsed_idx) else len(raw)
        fragment = raw[:start_orig] + " " + raw[end_orig:]

    # Clean dangling punctuation left where the phrase was removed.
    fragment = fragment.strip().lstrip(",.!?;:")
    fragment = " ".join(fragment.split())

    return fragment if fragment else ""


def is_stop_command(text_lower: str, stop_commands: List[str], fuzzy_ratio: float = 0.8) -> bool:
    """
    Check if text contains a stop command.
    
    Args:
        text_lower: Lowercase text to check
        stop_commands: List of stop command phrases
        fuzzy_ratio: Threshold for fuzzy matching short inputs
    
    Returns:
        True if stop command detected
    """
    if not text_lower or not text_lower.strip():
        return False
    
    folded = _matchable(text_lower)
    if not folded:
        return False

    # Check for exact matches
    detected_commands = []
    for cmd in stop_commands:
        if _matchable(cmd) in folded:
            detected_commands.append(cmd)
    
    # Check fuzzy matches for short inputs (2 words or less)
    if len(folded.split()) <= 2:
        try:
            for token in folded.split():
                token_clean = token.strip(".,!?;:")
                for cmd in stop_commands:
                    ratio = difflib.SequenceMatcher(a=_matchable(cmd), b=token_clean).ratio()
                    if ratio >= fuzzy_ratio:
                        detected_commands.append(f"{cmd}~{token_clean}")
        except Exception:
            pass
    
    if detected_commands:
        debug_log(f"stop command detected: {detected_commands[0]} in '{text_lower}'", "voice")
        return True
    
    return False
