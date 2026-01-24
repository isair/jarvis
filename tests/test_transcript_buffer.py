"""Tests for the transcript buffer module."""

import time
import threading
import pytest

from jarvis.listening.transcript_buffer import TranscriptBuffer, TranscriptSegment


def _now():
    """Get current timestamp for tests."""
    return time.time()


class TestTranscriptSegment:
    """Tests for TranscriptSegment dataclass."""

    def test_basic_creation(self):
        """Can create a basic segment."""
        seg = TranscriptSegment(
            text="hello world",
            start_time=1000.0,
            end_time=1001.5,
        )
        assert seg.text == "hello world"
        assert seg.start_time == 1000.0
        assert seg.end_time == 1001.5
        assert seg.energy == 0.0
        assert seg.is_during_tts is False

    def test_text_is_stripped(self):
        """Text is stripped of whitespace on creation."""
        seg = TranscriptSegment(
            text="  hello world  ",
            start_time=1000.0,
            end_time=1001.0,
        )
        assert seg.text == "hello world"

    def test_duration_property(self):
        """Duration is correctly calculated."""
        seg = TranscriptSegment(
            text="test",
            start_time=1000.0,
            end_time=1002.5,
        )
        assert seg.duration == 2.5

    def test_str_representation(self):
        """String representation includes timestamp and text."""
        seg = TranscriptSegment(
            text="hello",
            start_time=1000.0,
            end_time=1001.0,
        )
        s = str(seg)
        assert '"hello"' in s

    def test_str_with_tts_marker(self):
        """String representation includes TTS marker when applicable."""
        seg = TranscriptSegment(
            text="hello",
            start_time=1000.0,
            end_time=1001.0,
            is_during_tts=True,
        )
        s = str(seg)
        assert "[TTS]" in s


class TestTranscriptBuffer:
    """Tests for TranscriptBuffer class."""

    def test_add_segment(self):
        """Can add segments to buffer."""
        buf = TranscriptBuffer()
        now = _now()
        buf.add("hello", now - 1, now)
        assert len(buf) == 1

    def test_add_empty_text_ignored(self):
        """Empty text is not added."""
        buf = TranscriptBuffer()
        now = _now()
        buf.add("", now - 1, now)
        buf.add("   ", now - 1, now)
        assert len(buf) == 0

    def test_get_all(self):
        """Can retrieve all segments."""
        buf = TranscriptBuffer()
        now = _now()
        buf.add("first", now - 2, now - 1)
        buf.add("second", now - 1, now)

        segments = buf.get_all()
        assert len(segments) == 2
        assert segments[0].text == "first"
        assert segments[1].text == "second"

    def test_get_since(self):
        """Can filter segments by start time."""
        buf = TranscriptBuffer()
        now = _now()
        buf.add("old", now - 10, now - 9)
        buf.add("new", now - 2, now - 1)

        segments = buf.get_since(now - 5)
        assert len(segments) == 1
        assert segments[0].text == "new"

    def test_get_before(self):
        """Can filter segments before a timestamp."""
        buf = TranscriptBuffer()
        now = _now()
        buf.add("old", now - 10, now - 9)
        buf.add("new", now - 2, now - 1)

        segments = buf.get_before(now - 5)
        assert len(segments) == 1
        assert segments[0].text == "old"

    def test_get_around(self):
        """Can get segments in a time window."""
        buf = TranscriptBuffer()
        now = _now()
        buf.add("before", now - 20, now - 19)
        buf.add("around", now - 3, now - 2)
        buf.add("after", now + 10, now + 11)

        segments = buf.get_around(now - 2.5, before_sec=5.0, after_sec=5.0)
        assert len(segments) == 1
        assert segments[0].text == "around"

    def test_get_last_n(self):
        """Can get last N segments."""
        buf = TranscriptBuffer()
        now = _now()
        for i in range(5):
            buf.add(f"seg{i}", now - 10 + i, now - 9 + i)

        segments = buf.get_last_n(2)
        assert len(segments) == 2
        assert segments[0].text == "seg3"
        assert segments[1].text == "seg4"

    def test_get_last_seconds(self):
        """Can get segments from last N seconds."""
        buf = TranscriptBuffer()
        now = time.time()
        buf.add("old", now - 100, now - 99)
        buf.add("recent", now - 2, now - 1)

        segments = buf.get_last_seconds(10)
        assert len(segments) == 1
        assert segments[0].text == "recent"

    def test_prune_old_segments(self):
        """Old segments are pruned."""
        buf = TranscriptBuffer(max_duration_sec=60.0)
        now = time.time()

        # Add old segment
        buf.add("old", now - 120, now - 119)
        # Add recent segment
        buf.add("recent", now - 10, now - 9)

        # Prune should remove old segment
        buf.prune()

        segments = buf.get_all()
        assert len(segments) == 1
        assert segments[0].text == "recent"

    def test_auto_prune_on_add(self):
        """Old segments are pruned automatically when adding."""
        buf = TranscriptBuffer(max_duration_sec=60.0)
        now = _now()

        # Add a segment that's within the buffer duration
        buf.add("will_be_old", now - 55, now - 54)
        assert len(buf) == 1

        # Simulate time passing by manipulating the segment's end_time
        # to make it appear old (older than max_duration)
        buf._segments[0] = TranscriptSegment(
            text="will_be_old",
            start_time=now - 120,
            end_time=now - 119,
        )

        # Add new segment - should trigger prune of the old one
        buf.add("new", now - 5, now - 4)

        # Old segment should be gone
        segments = buf.get_all()
        assert len(segments) == 1
        assert segments[0].text == "new"

    def test_clear(self):
        """Can clear all segments."""
        buf = TranscriptBuffer()
        now = _now()
        buf.add("test", now - 1, now)
        assert len(buf) == 1

        buf.clear()
        assert len(buf) == 0

    def test_format_for_llm_basic(self):
        """Can format segments for LLM."""
        buf = TranscriptBuffer()
        now = _now()
        buf.add("hello world", now - 2, now - 1)
        buf.add("how are you", now - 1, now)

        formatted = buf.format_for_llm()
        assert '"hello world"' in formatted
        assert '"how are you"' in formatted

    def test_format_for_llm_with_tts_marker(self):
        """Format includes TTS markers."""
        buf = TranscriptBuffer()
        now = _now()
        buf.add("echo text", now - 1, now, is_during_tts=True)

        formatted = buf.format_for_llm()
        assert "during TTS" in formatted

    def test_format_for_llm_with_wake_timestamp(self):
        """Format marks wake word segment."""
        buf = TranscriptBuffer()
        now = _now()
        buf.add("jarvis what time", now - 2, now)

        formatted = buf.format_for_llm(wake_timestamp=now - 1)
        assert "WAKE WORD" in formatted

    def test_format_for_llm_empty(self):
        """Format handles empty buffer."""
        buf = TranscriptBuffer()
        formatted = buf.format_for_llm()
        assert "no recent speech" in formatted

    def test_bool_empty(self):
        """Empty buffer is falsy."""
        buf = TranscriptBuffer()
        assert not buf

    def test_bool_with_content(self):
        """Buffer with content is truthy."""
        buf = TranscriptBuffer()
        now = _now()
        buf.add("test", now - 1, now)
        assert buf

    def test_total_duration(self):
        """Total duration is correctly calculated."""
        buf = TranscriptBuffer()
        now = _now()
        buf.add("first", now - 12, now - 11)
        buf.add("last", now - 2, now)

        assert buf.total_duration == 12.0  # now - (now - 12)

    def test_oldest_newest_timestamps(self):
        """Can get oldest and newest timestamps."""
        buf = TranscriptBuffer()
        assert buf.oldest_timestamp is None
        assert buf.newest_timestamp is None

        now = _now()
        buf.add("first", now - 12, now - 11)
        buf.add("last", now - 2, now)

        assert buf.oldest_timestamp == now - 12
        assert buf.newest_timestamp == now


class TestThreadSafety:
    """Tests for thread safety."""

    def test_concurrent_add_and_read(self):
        """Buffer is thread-safe for concurrent access."""
        buf = TranscriptBuffer()
        errors = []

        def writer():
            for i in range(50):
                try:
                    buf.add(f"segment{i}", 1000.0 + i, 1001.0 + i)
                except Exception as e:
                    errors.append(e)

        def reader():
            for _ in range(50):
                try:
                    _ = buf.get_all()
                    _ = len(buf)
                    _ = buf.format_for_llm()
                except Exception as e:
                    errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
