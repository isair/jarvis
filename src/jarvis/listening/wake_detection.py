"""Wake word and stop command detection logic."""

from typing import List, Optional
import difflib
import re
import unicodedata

from ..debug import debug_log


def _wake_name_pattern(wake_word: str, aliases: List[str]) -> Optional[re.Pattern[str]]:
    """Match literal configured names at Unicode word boundaries."""
    names = {name.strip() for name in [wake_word, *aliases] if name.strip()}
    if not names:
        return None
    alternatives = "|".join(re.escape(name) for name in sorted(names, key=len, reverse=True))
    return re.compile(r"(?<!\w)(?:" + alternatives + r")(?!\w)", re.IGNORECASE)


def is_wake_word_detected(text_lower: str, wake_word: str, aliases: List[str], fuzzy_ratio: float = 0.78) -> bool:
    """Accept whole configured names or a close primary-name token."""
    if not text_lower or not text_lower.strip():
        return False

    pattern = _wake_name_pattern(wake_word, aliases)
    if pattern and pattern.search(text_lower):
        return True

    # Aliases already represent known mishearings; do not approximate them again.
    primary = wake_word.strip().casefold()
    if not primary or any(char.isspace() for char in primary):
        return False
    for token in re.findall(r"\w+", text_lower.casefold(), re.UNICODE):
        # Longer tokens stay near the name; shorter pronunciations use the threshold.
        if len(token) > len(primary) + 1:
            continue
        ratio = difflib.SequenceMatcher(a=primary, b=token).ratio()
        if ratio >= fuzzy_ratio:
            debug_log(f"primary wake name fuzzy match (ratio={ratio:.3f})", "wake")
            return True
    return False


def extract_query_after_wake(text_lower: str, wake_word: str, aliases: List[str]) -> str:
    """
    Extract the query portion after removing wake word.
    
    Args:
        text_lower: Lowercase text containing wake word
        wake_word: Primary wake word
        aliases: List of wake word aliases
    
    Returns:
        Query text with wake word removed
    """
    if not text_lower:
        return ""
    
    pattern = _wake_name_pattern(wake_word, aliases)
    fragment = pattern.sub(" ", text_lower) if pattern else text_lower
    
    # Clean up punctuation that might be left after wake word removal
    fragment = fragment.strip().lstrip(",.!?;:")
    fragment = fragment.strip()
    
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
    
    heard = unicodedata.normalize("NFC", text_lower).casefold()
    commands = [unicodedata.normalize("NFC", command.strip()).casefold()
                for command in stop_commands if command.strip()]
    for command in commands:
        if command in heard:
            debug_log(f"stop command detected: {command} in '{heard}'", "voice")
            return True

    # Short inputs permit a close transcription of a configured phrase.
    words = heard.split()
    if len(words) <= 2:
        for word in words:
            for command in commands:
                if difflib.SequenceMatcher(a=command, b=word).ratio() >= fuzzy_ratio:
                    debug_log(f"stop command detected: {command}~{word} in '{heard}'", "voice")
                    return True
    return False


def is_stop_command_echo(text: str, tts_text: str, stop_commands: List[str], fuzzy_ratio: float = 0.8) -> bool:
    """Recognise literal TTS echo while preserving standalone control priority."""
    def normalise(value: str) -> str:
        value = unicodedata.normalize("NFC", value).casefold()
        value = "".join(" " if unicodedata.category(char).startswith("P") else char
                        for char in value)
        return " ".join(value.split())

    heard = normalise(text)
    spoken = normalise(tts_text)
    if not heard or not spoken:
        return False
    for command in stop_commands:
        control = normalise(command)
        if control and difflib.SequenceMatcher(a=heard, b=control).ratio() >= fuzzy_ratio:
            return False
    return heard in spoken
