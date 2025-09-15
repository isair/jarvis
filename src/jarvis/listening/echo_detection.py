"""Echo detection and suppression logic for preventing TTS feedback."""

import time
import difflib
from typing import Optional, List
import re

from ..debug import debug_log

try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False


class EchoDetector:
    """Handles echo detection to prevent TTS feedback loops."""
    
    def __init__(self, echo_tolerance: float = 0.3, energy_spike_threshold: float = 2.0):
        """
        Initialize echo detector.
        
        Args:
            echo_tolerance: Time window after TTS for echo detection (seconds)
            energy_spike_threshold: Energy multiplier to distinguish real input from echo
        """
        self.echo_tolerance = echo_tolerance
        self.energy_spike_threshold = energy_spike_threshold
        
        # TTS tracking
        self._tts_start_time: float = 0.0
        self._last_tts_finish_time: float = 0.0
        self._last_tts_text: str = ""
        self._tts_energy_baseline: float = 0.0
        # Acceptance policy
        self._min_overlap_accept_words: int = 3  # require at least this many words to overlap
        
        # Utterance timing
        self._utterance_start_time: float = 0.0
        self._utterance_end_time: float = 0.0
    
    def track_tts_start(self, tts_text: str, baseline_energy: float = 0.0045) -> None:
        """
        Track when TTS starts speaking.
        
        Args:
            tts_text: Text being spoken by TTS
            baseline_energy: Current audio energy baseline
        """
        self._tts_start_time = time.time()
        self._last_tts_text = tts_text.lower().strip()
        self._tts_energy_baseline = baseline_energy
        
        debug_log(f"TTS started, text_len={len(tts_text)}, baseline_energy={baseline_energy:.4f}", "echo")
    
    def track_tts_finish(self) -> None:
        """Track when TTS finishes speaking."""
        self._last_tts_finish_time = time.time()
        debug_log("TTS finished", "echo")
    
    def track_utterance_timing(self, start_time: float, end_time: float) -> None:
        """
        Track timing of user utterance.
        
        Args:
            start_time: When user started speaking
            end_time: When user finished speaking
        """
        self._utterance_start_time = start_time
        self._utterance_end_time = end_time
    
    def _check_text_similarity(self, heard_text: str, tts_text: str) -> bool:
        """
        Check if heard text is similar to TTS text using fuzzy matching.
        
        Args:
            heard_text: Text heard from audio
            tts_text: Text that was spoken by TTS
            
        Returns:
            True if texts are similar (likely echo)
        """
        if not heard_text or not tts_text:
            return False
        
        heard_lower = heard_text.lower().strip()
        tts_lower = tts_text.lower().strip()

        # Fallback to difflib if rapidfuzz is not available
        if not RAPIDFUZZ_AVAILABLE:
            if heard_lower in tts_lower:
                return True
            similarity = difflib.SequenceMatcher(a=tts_lower, b=heard_lower).ratio()
            return similarity >= 0.85

        # Use rapidfuzz for more robust matching.
        # partial_ratio is excellent for finding echoes which are often substrings.
        # token_set_ratio is good at handling ASR errors where some words might be wrong.
        partial_score = fuzz.partial_ratio(heard_lower, tts_lower)
        token_set_score = fuzz.token_set_ratio(heard_lower, tts_lower)

        # We take the higher of the two scores. A score of 85 is a reasonably high similarity threshold.
        best_score = max(partial_score, token_set_score)
        
        is_similar = best_score >= 85

        if is_similar:
            debug_log(f"text similarity match: score={best_score:.1f}, heard='{heard_lower}', tts='{tts_lower[:100]}...'", "echo")

        return is_similar
    
    def _matches_tts_segment(self, heard_text: str, tts_rate: float, utterance_start_time: float) -> bool:
        """Checks if heard text matches the likely TTS segment playing at a given time."""
        if not (self._tts_start_time > 0 and utterance_start_time > 0):
            return False

        time_offset = utterance_start_time - self._tts_start_time
        time_offset_with_tolerance = max(0, time_offset - self.echo_tolerance)
        
        estimated_words_per_sec = tts_rate / 60.0
        tts_words = self._last_tts_text.split()
        
        if not tts_words:
            return False
            
        estimated_word_index = int(time_offset_with_tolerance * estimated_words_per_sec)
        
        # The window for checking the echo must be large enough to account for transcription errors
        # and the length of the heard text itself.
        heard_word_count = len(heard_text.split())
        # Use round() instead of int() for better accuracy and add a base tolerance.
        tolerance_words = round(self.echo_tolerance * estimated_words_per_sec) + 5
        
        start_idx = max(0, estimated_word_index - tolerance_words)
        # The end of the window should be far enough out to contain all the words we heard.
        end_idx = min(len(tts_words), estimated_word_index + heard_word_count + tolerance_words)
        
        relevant_tts_text = " ".join(tts_words[start_idx:end_idx])
        if relevant_tts_text:
            debug_log(f"checking TTS portion: time_offset={time_offset:.2f}s, '{relevant_tts_text[:50]}...'", "echo")
            return self._check_text_similarity(heard_text, relevant_tts_text)
            
        return False

    def cleanup_leading_echo_during_tts(self, heard_text: str, tts_rate: float, utterance_start_time: float) -> str:
        """Remove leading overlap against the current TTS segment to salvage user suffix during TTS.

        If the user starts speaking while TTS is active and their transcript begins with the
        current TTS segment, trim that segment and return the remainder so we can accept it.
        """
        if not heard_text or not self._last_tts_text or not (self._tts_start_time > 0 and utterance_start_time > 0):
            return heard_text

        # Reconstruct the relevant TTS segment similar to _matches_tts_segment
        time_offset = utterance_start_time - self._tts_start_time
        time_offset_with_tolerance = max(0, time_offset - self.echo_tolerance)
        estimated_words_per_sec = tts_rate / 60.0
        tts_words = self._last_tts_text.lower().strip().split()
        if not tts_words:
            return heard_text

        estimated_word_index = int(time_offset_with_tolerance * estimated_words_per_sec)
        heard_words = heard_text.lower().strip().split()
        tolerance_words = round(self.echo_tolerance * estimated_words_per_sec) + 5
        start_idx = max(0, estimated_word_index - tolerance_words)
        end_idx = min(len(tts_words), estimated_word_index + len(heard_words) + tolerance_words)
        segment_words = tts_words[start_idx:end_idx]

        if not segment_words or not heard_words:
            return heard_text

        # Normalize tokens to ignore punctuation and curly quotes while comparing
        def _clean_token(token: str) -> str:
            t = token.replace("’", "'")
            # drop all non-alphanumeric except apostrophe
            return re.sub(r"[^a-z0-9']+", "", t)

        seg_clean = [_clean_token(w) for w in segment_words]
        heard_clean = [_clean_token(w) for w in heard_words]

        # Find the longest prefix of heard_clean matching a suffix of seg_clean
        max_overlap = 0
        limit = min(len(seg_clean), len(heard_clean))
        for i in range(limit, 0, -1):
            if seg_clean[-i:] == heard_clean[:i]:
                max_overlap = i
                break

        if 0 < max_overlap < len(heard_words) and max_overlap >= self._min_overlap_accept_words:
            cleaned_text = " ".join(heard_words[max_overlap:])
            overlap_text = " ".join(heard_words[:max_overlap])
            debug_log(f"cleaned leading echo during TTS. Overlap: '{overlap_text}'. Cleaned: '{cleaned_text}'", "echo")
            return cleaned_text

        return heard_text
    
    def cleanup_leading_echo(self, heard_text: str) -> str:
        """Removes leading text from a query if it overlaps with the end of the last TTS."""
        if not heard_text or not self._last_tts_text:
            return heard_text

        heard_words = heard_text.lower().strip().split()
        tts_words = self._last_tts_text.lower().strip().split()

        if not heard_words or not tts_words:
            return heard_text

        max_overlap = 0
        for i in range(min(len(tts_words), len(heard_words)), 0, -1):
            if tts_words[-i:] == heard_words[:i]:
                max_overlap = i
                break
        
        # Only cleanup if there's a remainder and the overlap is at least 2 words.
        if 0 < max_overlap < len(heard_words) and max_overlap >= 2:
            cleaned_text = " ".join(heard_words[max_overlap:])
            overlap_text = " ".join(heard_words[:max_overlap])
            debug_log(f"cleaned leading echo. Overlap: '{overlap_text}'. Original: '{heard_text}', Cleaned: '{cleaned_text}'", "echo")
            return cleaned_text
            
        return heard_text
    
    def should_reject_as_echo(self, heard_text: str, current_energy: float, 
                            is_during_tts: bool = False, tts_rate: float = 200.0,
                            utterance_start_time: float = 0.0) -> bool:
        """Main entry point for echo detection decision."""
        if not self._last_tts_text:
            return False

        debug_log(f"echo check: heard='{heard_text[:50]}...', tts_available=True, is_during_tts={is_during_tts}, energy={current_energy:.4f}", "echo")

        # --- Case 1: During TTS Playback ---
        # Must use segment matching to allow for interruptions like "stop".
        if is_during_tts:
            if self._matches_tts_segment(heard_text, tts_rate, utterance_start_time):
                debug_log(f"rejected as echo during TTS (segment match): '{heard_text}'", "echo")
                return True
            debug_log("NOT echo during TTS - text does not match segment.", "echo")
            return False

        # --- Case 2: After TTS Playback ---
        # Decisions are based on when the utterance started.
        if self._last_tts_finish_time > 0 and utterance_start_time > 0:
            time_since_finish = utterance_start_time - self._last_tts_finish_time
            text_matches_full_tts = self._check_text_similarity(heard_text, self._last_tts_text)

            # Primary Cooldown Window (e.g., < 0.3s)
            if 0 <= time_since_finish < self.echo_tolerance:
                is_low_energy = current_energy < self._tts_energy_baseline * self.energy_spike_threshold
                if text_matches_full_tts and is_low_energy:
                    debug_log(f"rejected as echo in cooldown (text match + low energy): '{heard_text}'", "echo")
                    return True
                else:
                    debug_log(f"accepted in cooldown (high energy or no text match): '{heard_text}'", "voice")
            
            # Extended Delayed-Echo Window (e.g., < 1.5s)
            elif self.echo_tolerance <= time_since_finish < 1.5:
                if text_matches_full_tts:
                    debug_log(f"rejected as delayed echo in extended window (text match): '{heard_text}'", "echo")
                    return True

        # --- Default Case ---
        debug_log("NOT echo - outside of all detection windows.", "echo")
        return False
