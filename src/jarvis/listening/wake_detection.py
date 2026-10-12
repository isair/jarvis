"""Wake word and stop command detection logic."""

from typing import List, Optional
import difflib
import re
import unicodedata

from ..debug import debug_log


def _wake_name_pattern(text: str, wake_word: str, aliases: List[str]) -> Optional[re.Pattern[str]]:
    """Match literal configured names at Unicode word boundaries."""
    names = {unicodedata.normalize("NFC", name.strip())
             for name in [wake_word, *aliases] if name.strip()}
    if not names:
        return None
    alternatives = "|".join(re.escape(name) for name in sorted(names, key=len, reverse=True))
    marks = {char for char in text if unicodedata.category(char).startswith("M")}
    word = r"[\w" + re.escape("".join(sorted(marks))) + "]"
    return re.compile(r"(?<!" + word + r")(?:" + alternatives + r")(?!" + word + ")", re.IGNORECASE)


def _is_word_character(char: str) -> bool:
    return char.isalnum() or char == "_" or unicodedata.category(char).startswith("M")


def normalise_wake_aliases(text: str, aliases: List[str], assistant_name: str) -> str:
    """Canonicalise the judge's text copy and replace complete literal aliases."""
    if not text or not aliases:
        return text
    canonical = unicodedata.normalize("NFC", text)
    pattern = _wake_name_pattern(canonical, "", aliases)
    return pattern.sub(lambda match: assistant_name, canonical) if pattern else text


def is_wake_word_detected(text_lower: str, wake_word: str, aliases: List[str], fuzzy_ratio: float = 0.78) -> bool:
    """Accept whole configured names or a close primary-name token."""
    if not text_lower or not text_lower.strip():
        return False

    text_lower = unicodedata.normalize("NFC", text_lower)
    pattern = _wake_name_pattern(text_lower, wake_word, aliases)
    if pattern and pattern.search(text_lower):
        return True

    # Aliases already represent known mishearings; do not approximate them again.
    primary = unicodedata.normalize("NFC", wake_word.strip()).casefold()
    if not primary or any(char.isspace() for char in primary):
        return False
    tokens = "".join(char if _is_word_character(char) else " "
                     for char in text_lower.casefold()).split()
    for token in tokens:
        # Longer tokens stay near the name; shorter pronunciations use the threshold.
        if len(token) > len(primary) + 1:
            continue
        ratio = difflib.SequenceMatcher(a=primary, b=token).ratio()
        if ratio >= fuzzy_ratio:
            debug_log(f"primary wake name fuzzy match (ratio={ratio:.3f})", "wake")
            return True
    return False


def _normalise_control_text(value: str) -> str:
    """Keep composed word content while separating Unicode punctuation."""
    value = unicodedata.normalize("NFC", value).casefold()
    value = "".join(" " if unicodedata.category(char).startswith("P") else char
                    for char in value)
    return " ".join(value.split())


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
    
    heard = _normalise_control_text(text_lower)
    commands = [control for command in stop_commands
                if (control := _normalise_control_text(command))]
    if not heard:
        return False
    for command in commands:
        if f" {command} " in f" {heard} ":
            debug_log(f"stop command detected: {command} in '{heard}'", "voice")
            return True

    # Approximate a complete short control, rather than a word in other speech.
    if len(heard.split()) <= 2:
        for command in commands:
            if difflib.SequenceMatcher(a=command, b=heard).ratio() >= fuzzy_ratio:
                debug_log(f"stop command detected: {command}~{heard}", "voice")
                return True
    return False


def is_stop_command_echo(text: str, tts_text: str, stop_commands: List[str], fuzzy_ratio: float = 0.8) -> bool:
    """Recognise literal TTS echo while preserving standalone control priority."""
    heard = _normalise_control_text(text)
    spoken = _normalise_control_text(tts_text)
    if not heard or not spoken:
        return False
    for command in stop_commands:
        control = _normalise_control_text(command)
        if control and difflib.SequenceMatcher(a=heard, b=control).ratio() >= fuzzy_ratio:
            return False
    return heard in spoken
