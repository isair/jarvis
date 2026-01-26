"""Rolling transcript buffer for voice listening.

This module provides a timestamped buffer of transcribed speech segments,
aligned with short-term memory concepts. The buffer retains transcripts
for a configurable duration (default 5 minutes) and supports querying
by time ranges.
"""

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from ..debug import debug_log


@dataclass
class TranscriptSegment:
    """A single transcribed speech segment with metadata."""

    text: str                          # Transcribed text
    start_time: float                  # Unix timestamp when speech started
    end_time: float                    # Unix timestamp when speech ended
    energy: float = 0.0                # Audio energy level
    is_during_tts: bool = False        # Whether TTS was playing during this segment

    def __post_init__(self):
        """Normalize text on creation."""
        self.text = self.text.strip()

    @property
    def duration(self) -> float:
        """Duration of this segment in seconds."""
        return self.end_time - self.start_time

    def format_timestamp(self) -> str:
        """Format start time as HH:MM:SS for display."""
        return datetime.fromtimestamp(self.start_time).strftime('%H:%M:%S')

    def __str__(self) -> str:
        """String representation for debugging."""
        tts_marker = " [TTS]" if self.is_during_tts else ""
        return f"[{self.format_timestamp()}]{tts_marker} \"{self.text}\""


class TranscriptBuffer:
    """Rolling buffer of transcribed speech with timestamps.

    This buffer serves as the "live" portion of short-term memory,
    storing raw speech transcripts before they're processed into
    conversation turns.

    Thread-safe for concurrent access from audio processing threads.
    """

    def __init__(self, max_duration_sec: float = 120.0):
        """Initialize the transcript buffer.

        Args:
            max_duration_sec: Maximum duration of transcripts to retain (default 2 minutes)
        """
        self.max_duration_sec = max_duration_sec
        self._segments: List[TranscriptSegment] = []
        self._lock = threading.Lock()

    def add(
        self,
        text: str,
        start_time: float,
        end_time: float,
        energy: float = 0.0,
        is_during_tts: bool = False,
    ) -> None:
        """Add a transcribed segment to the buffer.

        Args:
            text: Transcribed text
            start_time: Unix timestamp when speech started
            end_time: Unix timestamp when speech ended
            energy: Audio energy level of the segment
            is_during_tts: Whether TTS was playing during this segment
        """
        if not text or not text.strip():
            return

        segment = TranscriptSegment(
            text=text,
            start_time=start_time,
            end_time=end_time,
            energy=energy,
            is_during_tts=is_during_tts,
        )

        with self._lock:
            self._segments.append(segment)
            self._prune_locked()

        debug_log(f"transcript buffer: added {segment}", "voice")

    def get_all(self) -> List[TranscriptSegment]:
        """Get all segments in the buffer.

        Returns:
            List of all transcript segments, oldest first
        """
        with self._lock:
            return list(self._segments)

    def get_since(self, timestamp: float) -> List[TranscriptSegment]:
        """Get all segments since a timestamp.

        Args:
            timestamp: Unix timestamp to filter from

        Returns:
            List of segments with start_time >= timestamp
        """
        with self._lock:
            return [s for s in self._segments if s.start_time >= timestamp]

    def get_before(self, timestamp: float) -> List[TranscriptSegment]:
        """Get all segments before a timestamp.

        Args:
            timestamp: Unix timestamp to filter until

        Returns:
            List of segments with start_time < timestamp
        """
        with self._lock:
            return [s for s in self._segments if s.start_time < timestamp]

    def get_around(
        self,
        timestamp: float,
        before_sec: float = 5.0,
        after_sec: float = 2.0,
    ) -> List[TranscriptSegment]:
        """Get segments in a time window around a timestamp.

        Args:
            timestamp: Center timestamp
            before_sec: Seconds to include before timestamp
            after_sec: Seconds to include after timestamp

        Returns:
            List of segments within the time window
        """
        start = timestamp - before_sec
        end = timestamp + after_sec

        with self._lock:
            return [
                s for s in self._segments
                if s.start_time >= start and s.start_time <= end
            ]

    def get_last_n(self, n: int) -> List[TranscriptSegment]:
        """Get the last N segments.

        Args:
            n: Number of segments to return

        Returns:
            List of the most recent N segments
        """
        with self._lock:
            return list(self._segments[-n:]) if self._segments else []

    def get_last_seconds(self, seconds: float) -> List[TranscriptSegment]:
        """Get segments from the last N seconds.

        Args:
            seconds: Duration in seconds

        Returns:
            List of segments from the last N seconds
        """
        cutoff = time.time() - seconds
        return self.get_since(cutoff)

    def format_for_llm(
        self,
        segments: Optional[List[TranscriptSegment]] = None,
        include_tts_marker: bool = True,
        wake_timestamp: Optional[float] = None,
    ) -> str:
        """Format segments for LLM input.

        Args:
            segments: Segments to format (defaults to all)
            include_tts_marker: Whether to include [TTS] markers
            wake_timestamp: If provided, mark the segment containing wake word

        Returns:
            Formatted string for LLM consumption
        """
        if segments is None:
            segments = self.get_all()

        if not segments:
            return "(no recent speech)"

        lines = []
        for seg in segments:
            ts = seg.format_timestamp()
            text = seg.text

            markers = []
            if include_tts_marker and seg.is_during_tts:
                markers.append("during TTS")
            if wake_timestamp and seg.start_time <= wake_timestamp <= seg.end_time:
                markers.append("WAKE WORD")

            marker_str = f" ({', '.join(markers)})" if markers else ""
            lines.append(f"[{ts}]{marker_str} \"{text}\"")

        return "\n".join(lines)

    def update_last_segment_text(self, new_text: str) -> bool:
        """Update the text of the most recent segment after echo salvage.

        Used when echo salvage extracts clean user speech from a mixed
        echo+speech segment. This ensures the intent judge sees clean data.

        IMPORTANT: This also clears the is_during_tts flag because the
        salvaged text is real user speech, not echo. Without this, the
        intent judge would skip the segment as echo.

        Args:
            new_text: The cleaned/salvaged text to replace the original

        Returns:
            True if update succeeded, False if buffer is empty
        """
        if not new_text or not new_text.strip():
            return False

        with self._lock:
            if not self._segments:
                return False

            old_text = self._segments[-1].text
            self._segments[-1].text = new_text.strip()
            # Clear TTS flag - salvaged text is user speech, not echo
            self._segments[-1].is_during_tts = False

        debug_log(f"transcript buffer: updated last segment from '{old_text[:50]}...' to '{new_text[:50]}...'", "voice")
        return True

    def clear_last_segment_tts_flag(self) -> bool:
        """Clear the is_during_tts flag on the most recent segment.

        Used when echo detection confirms a segment is NOT echo, even though
        it started during TTS. This ensures the intent judge sees it as
        user speech rather than skipping it as potential echo.

        Returns:
            True if flag was cleared, False if buffer is empty
        """
        with self._lock:
            if not self._segments:
                return False

            if self._segments[-1].is_during_tts:
                self._segments[-1].is_during_tts = False
                debug_log("transcript buffer: cleared TTS flag on last segment (confirmed not echo)", "voice")

        return True

    def clear(self) -> None:
        """Clear all segments from the buffer."""
        with self._lock:
            self._segments.clear()
        debug_log("transcript buffer cleared", "voice")

    def prune(self) -> int:
        """Remove segments older than max_duration_sec.

        Returns:
            Number of segments removed
        """
        with self._lock:
            return self._prune_locked()

    def _prune_locked(self) -> int:
        """Remove old segments (must hold lock).

        Returns:
            Number of segments removed
        """
        if not self._segments:
            return 0

        cutoff = time.time() - self.max_duration_sec
        original_count = len(self._segments)

        self._segments = [s for s in self._segments if s.end_time >= cutoff]

        removed = original_count - len(self._segments)
        if removed > 0:
            debug_log(f"transcript buffer: pruned {removed} old segments", "voice")

        return removed

    def __len__(self) -> int:
        """Return number of segments in buffer."""
        with self._lock:
            return len(self._segments)

    def __bool__(self) -> bool:
        """Return True if buffer has any segments."""
        with self._lock:
            return bool(self._segments)

    @property
    def total_duration(self) -> float:
        """Total duration of all segments in seconds."""
        with self._lock:
            if not self._segments:
                return 0.0
            return self._segments[-1].end_time - self._segments[0].start_time

    @property
    def oldest_timestamp(self) -> Optional[float]:
        """Timestamp of oldest segment, or None if empty."""
        with self._lock:
            return self._segments[0].start_time if self._segments else None

    @property
    def newest_timestamp(self) -> Optional[float]:
        """Timestamp of newest segment, or None if empty."""
        with self._lock:
            return self._segments[-1].end_time if self._segments else None
