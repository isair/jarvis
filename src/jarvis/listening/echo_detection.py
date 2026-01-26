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
    
    def _normalize_for_comparison(self, text: str) -> str:
        """
        Normalize text for echo comparison.

        Handles differences between TTS text and how Whisper transcribes it:
        - Degree symbols: 9°C → 9 degrees celsius
        - Common TTS pronunciation variations
        """
        normalized = text.lower().strip()

        # Normalize degree symbols - TTS says "9 degrees celsius" for "9°C"
        # Handle patterns like "9°c", "9°C", "9° C", etc.
        normalized = re.sub(r'(\d+)\s*°\s*c\b', r'\1 degrees celsius', normalized)
        normalized = re.sub(r'(\d+)\s*°\s*f\b', r'\1 degrees fahrenheit', normalized)
        normalized = re.sub(r'(\d+)\s*°', r'\1 degrees', normalized)  # Generic degree

        # Remove parentheses (TTS often reads "48°F (9°C)" as separate parts)
        normalized = re.sub(r'\(([^)]+)\)', r'\1', normalized)

        return normalized

    def _check_text_similarity(self, heard_text: str, tts_text: str, threshold: int = 85) -> bool:
        """
        Check if heard text is similar to TTS text using fuzzy matching.

        Args:
            heard_text: Text heard from audio
            tts_text: Text that was spoken by TTS
            threshold: Similarity threshold (0-100). Higher = stricter matching.
                      Use 85 for normal mode, 92 for hot window mode.

        Returns:
            True if texts are similar (likely echo)
        """
        if not heard_text or not tts_text:
            return False

        # Normalize both texts to handle TTS/Whisper differences
        heard_lower = self._normalize_for_comparison(heard_text)
        tts_lower = self._normalize_for_comparison(tts_text)

        # Fallback to difflib if rapidfuzz is not available
        if not RAPIDFUZZ_AVAILABLE:
            if heard_lower in tts_lower:
                return True
            similarity = difflib.SequenceMatcher(a=tts_lower, b=heard_lower).ratio()
            return similarity >= (threshold / 100.0)

        # Use rapidfuzz for more robust matching.
        # partial_ratio is excellent for finding echoes which are often substrings.
        # token_set_ratio is good at handling ASR errors where some words might be wrong.
        partial_score = fuzz.partial_ratio(heard_lower, tts_lower)
        token_set_score = fuzz.token_set_ratio(heard_lower, tts_lower)

        # We take the higher of the two scores.
        best_score = max(partial_score, token_set_score)

        is_similar = best_score >= threshold

        if is_similar:
            debug_log(f"text similarity match: score={best_score:.1f} (threshold={threshold}), heard='{heard_lower}', tts='{tts_lower[:100]}...'", "echo")

        return is_similar
    
    def _matches_tts_segment(self, heard_text: str, tts_rate: float, utterance_start_time: float) -> bool:
        """Checks if heard text matches the likely TTS segment playing at a given time.

        Uses two-phase approach:
        1. First check time-based segment (handles typical cases)
        2. If no match, search forward with extended window (handles TTS timing drift)

        TTS timing can drift significantly from calculated position due to:
        - Variable speech rate (pauses, emphasis)
        - System TTS buffering delays
        - Audio processing latency
        """
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

        # Phase 1: Check precise time-based segment
        relevant_tts_text = " ".join(tts_words[start_idx:end_idx])
        if relevant_tts_text:
            debug_log(f"checking TTS portion: time_offset={time_offset:.2f}s, '{relevant_tts_text[:50]}...'", "echo")
            if self._check_text_similarity(heard_text, relevant_tts_text):
                return True

        # Phase 2: Search forward for TTS timing drift
        # TTS often runs ahead of calculated position due to variable speech rate and buffering
        # Extend search forward by up to 8 seconds worth of text (conservative to avoid false positives)
        drift_seconds = 8.0
        drift_words = int(drift_seconds * estimated_words_per_sec)
        extended_start = end_idx  # Start where phase 1 ended
        extended_end = min(len(tts_words), end_idx + drift_words)

        if extended_end > extended_start:
            extended_segment = " ".join(tts_words[extended_start:extended_end])
            if extended_segment:
                debug_log(f"checking extended TTS portion (drift +{extended_end - extended_start} words): '{extended_segment[:50]}...'", "echo")
                # Use higher threshold (90) to reduce false positives in extended search
                if self._check_text_similarity(heard_text, extended_segment, threshold=90):
                    debug_log(f"matched in extended search (TTS timing drift)", "echo")
                    return True

        return False

    def cleanup_leading_echo_during_tts(self, heard_text: str, tts_rate: float, utterance_start_time: float) -> str:
        """Remove leading overlap against the TTS text to salvage user suffix during TTS.

        If the user starts speaking while TTS is active and their transcript begins with
        TTS content, trim that content and return the remainder so we can accept it.

        This uses a two-phase approach:
        1. First try a timing-based segment (fast, handles typical cases)
        2. If that fails, search the full TTS text (handles timing mismatches)
        """
        if not heard_text or not self._last_tts_text or not (self._tts_start_time > 0 and utterance_start_time > 0):
            return heard_text

        tts_words = self._last_tts_text.lower().strip().split()
        heard_words = heard_text.lower().strip().split()

        if not tts_words or not heard_words:
            return heard_text

        # Normalize tokens to ignore punctuation and curly quotes while comparing
        def _clean_token(token: str) -> str:
            t = token.replace("'", "'")
            # drop all non-alphanumeric except apostrophe
            return re.sub(r"[^a-z0-9']+", "", t)

        tts_clean = [_clean_token(w) for w in tts_words]
        heard_clean = [_clean_token(w) for w in heard_words]

        # Phase 1: Try timing-based segment first (faster for typical cases)
        time_offset = utterance_start_time - self._tts_start_time
        time_offset_with_tolerance = max(0, time_offset - self.echo_tolerance)
        estimated_words_per_sec = tts_rate / 60.0
        estimated_word_index = int(time_offset_with_tolerance * estimated_words_per_sec)
        tolerance_words = round(self.echo_tolerance * estimated_words_per_sec) + 5
        start_idx = max(0, estimated_word_index - tolerance_words)
        end_idx = min(len(tts_words), estimated_word_index + len(heard_words) + tolerance_words)
        segment_clean = tts_clean[start_idx:end_idx]

        max_overlap = 0
        if segment_clean:
            limit = min(len(segment_clean), len(heard_clean))
            for i in range(limit, 0, -1):
                if segment_clean[-i:] == heard_clean[:i]:
                    max_overlap = i
                    break

        # Phase 2: Search full TTS text for better match
        # Always try to find the longest overlap at TTS end, not just timing-based segment
        # This handles timing drift and finds cases where entire heard text is TTS
        limit = min(len(tts_clean), len(heard_clean))
        for i in range(limit, max(max_overlap, self._min_overlap_accept_words - 1), -1):
            if tts_clean[-i:] == heard_clean[:i]:
                if i > max_overlap:
                    debug_log(f"salvage: found longer match at TTS end ({i} vs {max_overlap} words)", "echo")
                    max_overlap = i
                break

        if 0 < max_overlap < len(heard_words) and max_overlap >= self._min_overlap_accept_words:
            cleaned_text = " ".join(heard_words[max_overlap:])
            overlap_text = " ".join(heard_words[:max_overlap])
            debug_log(f"cleaned leading echo during TTS. Overlap: '{overlap_text}'. Cleaned: '{cleaned_text}'", "echo")
            return cleaned_text

        # Phase 3: Fuzzy matching fallback for transcription differences
        # When exact word matching fails (e.g., "cuppa" vs "cup"), try fuzzy matching
        # on prefixes of heard text against the TTS TAIL (not full TTS)
        if RAPIDFUZZ_AVAILABLE and len(heard_words) > self._min_overlap_accept_words:
            # Get the tail of TTS (last ~50% of words) - this is what would be echoed
            # when mic picks up the end of TTS playback
            tts_words_list = self._last_tts_text.lower().strip().split()
            tts_tail_start = max(0, len(tts_words_list) // 2)
            tts_tail = " ".join(tts_words_list[tts_tail_start:])
            tts_tail_normalized = self._normalize_for_comparison(tts_tail)

            # Try different split points in the heard text
            # Start from around 70% of words (likely some echo) and work down to min overlap
            min_prefix_words = self._min_overlap_accept_words
            max_prefix_words = min(len(heard_words) - 2, int(len(heard_words) * 0.85))

            for prefix_len in range(max_prefix_words, min_prefix_words - 1, -1):
                heard_prefix = " ".join(heard_words[:prefix_len])
                heard_prefix_normalized = self._normalize_for_comparison(heard_prefix)

                # Check if this prefix matches the TTS TAIL using partial_ratio
                # This ensures we're matching the END of TTS (the echo) not middle content
                score = fuzz.partial_ratio(heard_prefix_normalized, tts_tail_normalized)

                if score >= 85:
                    suffix = " ".join(heard_words[prefix_len:])
                    # Make sure suffix is meaningful (not just a word or two)
                    # AND that the suffix doesn't also match TTS (would mean pure echo)
                    if len(suffix.split()) >= 2:
                        suffix_normalized = self._normalize_for_comparison(suffix)
                        suffix_match = fuzz.partial_ratio(suffix_normalized, tts_tail_normalized)
                        # Only salvage if suffix is sufficiently DIFFERENT from TTS
                        if suffix_match < 70:
                            debug_log(
                                f"salvage (fuzzy): prefix_score={score}, suffix_score={suffix_match}, "
                                f"prefix='{heard_prefix[:40]}...', suffix='{suffix}'", "echo"
                            )
                            return suffix

        return heard_text
    
    def _salvage_suffix_from_echo(self, heard_text: str, tts_rate: float, utterance_start_time: float) -> Optional[str]:
        """Check if heard text has user speech after a TTS echo prefix.

        This handles the case where the microphone picks up the end of TTS
        followed by user speech. For example:
        - TTS: "...temperature will be around 10°C. A great day to grab a cuppa."
        - Heard: "10 degrees. A great day to grab a cup. Tell me a random topic."
        - Salvaged: "Tell me a random topic."

        Returns:
            Salvaged user speech if found, None otherwise
        """
        if not heard_text or not self._last_tts_text:
            return None

        # Use cleanup_leading_echo_during_tts which already handles this
        salvaged = self.cleanup_leading_echo_during_tts(heard_text, tts_rate, utterance_start_time)

        # If salvage returned something different, there's user speech
        if salvaged and salvaged != heard_text:
            return salvaged

        # Also try the simpler cleanup_leading_echo for cases where timing info isn't helpful
        salvaged = self.cleanup_leading_echo(heard_text)
        if salvaged and salvaged != heard_text:
            return salvaged

        return None

    def cleanup_leading_echo(self, heard_text: str) -> str:
        """Removes leading text from a query if it overlaps with the end of the last TTS."""
        if not heard_text or not self._last_tts_text:
            return heard_text

        # Normalize to handle TTS/Whisper differences (e.g., "5.7°C" vs "5.7 degrees Celsius")
        heard_normalized = self._normalize_for_comparison(heard_text)
        tts_normalized = self._normalize_for_comparison(self._last_tts_text)

        heard_words = heard_normalized.split()
        tts_words = tts_normalized.split()
        original_heard_words = heard_text.lower().strip().split()

        if not heard_words or not tts_words:
            return heard_text

        # Strip punctuation from words for comparison (handles "kensington," vs "kensington")
        def strip_punct(word: str) -> str:
            return re.sub(r"[^\w']", "", word)

        heard_clean = [strip_punct(w) for w in heard_words]
        tts_clean = [strip_punct(w) for w in tts_words]

        max_overlap = 0
        for i in range(min(len(tts_clean), len(heard_clean)), 0, -1):
            if tts_clean[-i:] == heard_clean[:i]:
                max_overlap = i
                break

        # Only cleanup if there's a remainder and the overlap is at least 2 words.
        if 0 < max_overlap < len(heard_words) and max_overlap >= 2:
            # Use original words for output (preserving capitalization etc.)
            # But we need to map normalized word count to original word count
            # This is approximate - normalized may have different word count
            original_word_count = len(original_heard_words)
            normalized_word_count = len(heard_words)
            if original_word_count == normalized_word_count:
                cleaned_text = " ".join(original_heard_words[max_overlap:])
            else:
                # Word count differs due to normalization - use normalized words
                cleaned_text = " ".join(heard_words[max_overlap:])
            overlap_text = " ".join(heard_words[:max_overlap])
            debug_log(f"cleaned leading echo. Overlap: '{overlap_text}'. Cleaned: '{cleaned_text}'", "echo")
            return cleaned_text

        return heard_text
    
    def should_reject_as_echo(self, heard_text: str, current_energy: float,
                            is_during_tts: bool = False, tts_rate: float = 200.0,
                            utterance_start_time: float = 0.0,
                            in_hot_window: bool = False) -> bool:
        """Main entry point for echo detection decision.

        Args:
            heard_text: Text heard from audio
            current_energy: Current audio energy level
            is_during_tts: Whether TTS is currently playing
            tts_rate: TTS speaking rate in words per minute
            utterance_start_time: When the utterance started
            in_hot_window: Whether we're in hot window mode (use higher threshold)
        """
        if not self._last_tts_text:
            return False

        # Use higher similarity threshold in hot window to reduce false rejections
        # of valid follow-up speech
        similarity_threshold = 92 if in_hot_window else 85

        debug_log(f"echo check: heard='{heard_text[:50]}...', tts_available=True, is_during_tts={is_during_tts}, energy={current_energy:.4f}, hot_window={in_hot_window}", "echo")

        # --- Case 1: During TTS Playback ---
        # Use segment matching first to allow for interruptions like "stop".
        # But also fallback to full-TTS check for long utterances with timing drift.
        if is_during_tts:
            if self._matches_tts_segment(heard_text, tts_rate, utterance_start_time):
                debug_log(f"rejected as echo during TTS (segment match): '{heard_text}'", "echo")
                return True

            # Fallback: For long utterances (>4 words), check against full TTS at lower threshold.
            # This catches echoes with significant timing drift that segment matching misses.
            # Short utterances skip this to avoid false rejections of "stop", "quiet" etc.
            word_count = len(heard_text.split())
            if word_count > 4:
                # Use threshold 70 for during-TTS fallback (same as hot window after-TTS check)
                if self._check_text_similarity(heard_text, self._last_tts_text, threshold=70):
                    # Before rejecting, check if the match is concentrated in a prefix
                    # If there's user speech in the suffix, we should salvage it, not reject
                    salvaged = self._salvage_suffix_from_echo(heard_text, tts_rate, utterance_start_time)
                    if salvaged and salvaged != heard_text:
                        debug_log(f"full-TTS fallback: salvaged suffix '{salvaged}' from mixed echo+speech", "echo")
                        # Don't reject - there's user speech to salvage
                        # The caller should use cleanup_leading_echo_during_tts to get the clean text
                        return False
                    debug_log(f"rejected as echo during TTS (full-TTS fallback, {word_count} words): '{heard_text}'", "echo")
                    return True

            debug_log("NOT echo during TTS - text does not match segment or full TTS.", "echo")
            return False

        # --- Case 2: After TTS Playback ---
        # Decisions are based on when the utterance started.
        if self._last_tts_finish_time > 0 and utterance_start_time > 0:
            time_since_finish = utterance_start_time - self._last_tts_finish_time
            text_matches_full_tts = self._check_text_similarity(heard_text, self._last_tts_text, similarity_threshold)

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
