"""Tests for the intent judge module."""

import pytest
from unittest.mock import patch, MagicMock

from jarvis.listening.intent_judge import (
    IntentJudge,
    IntentJudgeConfig,
    IntentJudgment,
    create_intent_judge,
)
from jarvis.listening.transcript_buffer import TranscriptSegment


class TestIntentJudgeConfig:
    """Tests for IntentJudgeConfig."""

    def test_default_config(self):
        """Default config has reasonable values."""
        config = IntentJudgeConfig()
        assert config.assistant_name == "Jarvis"
        assert config.model == "gemma4:e2b"
        assert config.timeout_sec == 15.0
        assert config.aliases == []

    def test_custom_config(self):
        """Can customize config values."""
        config = IntentJudgeConfig(
            assistant_name="Friday",
            model="llama3.2:1b",
            aliases=["computer"],
        )
        assert config.assistant_name == "Friday"
        assert config.model == "llama3.2:1b"
        assert config.aliases == ["computer"]


class TestIntentJudgment:
    """Tests for IntentJudgment dataclass."""

    def test_basic_judgment(self):
        """Can create a basic judgment."""
        judgment = IntentJudgment(
            directed=True,
            query="what time is it",
            stop=False,
            confidence="high",
            reasoning="clear wake word",
        )
        assert judgment.directed is True
        assert judgment.query == "what time is it"
        assert judgment.stop is False
        assert judgment.confidence == "high"


class TestIntentJudge:
    """Tests for IntentJudge class."""

    def test_init(self):
        """Can initialize intent judge."""
        judge = IntentJudge()
        assert judge.config.assistant_name == "Jarvis"

    def test_init_with_config(self):
        """Can initialize with custom config."""
        config = IntentJudgeConfig(assistant_name="Friday")
        judge = IntentJudge(config)
        assert judge.config.assistant_name == "Friday"

    def test_available_when_requests_installed(self):
        """available is True when requests is installed."""
        judge = IntentJudge()
        judge._available = True
        judge._last_error_time = 0.0
        assert judge.available is True

    def test_unavailable_during_error_cooldown(self):
        """available is False during error cooldown."""
        import time
        judge = IntentJudge()
        judge._available = True
        judge._last_error_time = time.time()
        judge._error_cooldown = 30.0
        assert judge.available is False

    def test_build_system_prompt(self):
        """System prompt includes assistant name."""
        config = IntentJudgeConfig(assistant_name="Friday")
        judge = IntentJudge(config)
        prompt = judge._build_system_prompt()
        assert "Friday" in prompt

    def test_build_user_prompt_basic(self):
        """User prompt includes transcript."""
        judge = IntentJudge()
        segments = [
            TranscriptSegment("hello jarvis", 1000.0, 1001.0),
        ]
        prompt = judge._build_user_prompt(
            segments,
            wake_timestamp=1000.5,
            last_tts_text="",
            last_tts_finish_time=0.0,
            in_hot_window=False,
        )
        assert "hello jarvis" in prompt

    def test_build_user_prompt_hot_window(self):
        """User prompt indicates hot window mode."""
        judge = IntentJudge()
        segments = [
            TranscriptSegment("what time is it", 1000.0, 1001.0),
        ]
        prompt = judge._build_user_prompt(
            segments,
            wake_timestamp=None,
            last_tts_text="",
            last_tts_finish_time=0.0,
            in_hot_window=True,
        )
        assert "HOT WINDOW" in prompt

    def test_build_user_prompt_normalises_aliases(self):
        """Aliases (Whisper variants) are replaced with the assistant name in the prompt."""
        config = IntentJudgeConfig(
            assistant_name="Jarvis",
            aliases=["jervis", "jaivis", "jar is"],
        )
        judge = IntentJudge(config)
        segments = [
            TranscriptSegment("Jervis what time is it", 1000.0, 1001.0),
            TranscriptSegment("Jaivis tell me a joke", 1002.0, 1003.0),
            TranscriptSegment("hey Jar is, are you there", 1004.0, 1005.0),
        ]
        prompt = judge._build_user_prompt(
            segments,
            wake_timestamp=1000.5,
            last_tts_text="",
            last_tts_finish_time=0.0,
            in_hot_window=False,
        )
        assert "Jervis" not in prompt
        assert "Jaivis" not in prompt
        assert "Jar is" not in prompt
        # Each aliased segment is rewritten to use the primary wake word.
        assert prompt.count("Jarvis") >= 3

    def test_build_user_prompt_alias_word_boundary(self):
        """Alias normalisation respects word boundaries (won't eat substrings)."""
        config = IntentJudgeConfig(assistant_name="Jarvis", aliases=["jar"])
        judge = IntentJudge(config)
        segments = [
            TranscriptSegment("put the jar on the table", 1000.0, 1001.0),
        ]
        prompt = judge._build_user_prompt(
            segments,
            wake_timestamp=None,
            last_tts_text="",
            last_tts_finish_time=0.0,
            in_hot_window=False,
        )
        # "jar" as a standalone word still gets normalised — that's expected
        # given the user configured it as an alias.
        assert "Jarvis" in prompt
        # But "jarring" would NOT be replaced if it appeared.
        segments2 = [TranscriptSegment("the noise was jarring", 1000.0, 1001.0)]
        prompt2 = judge._build_user_prompt(
            segments2,
            wake_timestamp=None,
            last_tts_text="",
            last_tts_finish_time=0.0,
            in_hot_window=False,
        )
        assert "jarring" in prompt2
        assert "Jarvisring" not in prompt2

    def test_build_user_prompt_no_aliases_unchanged(self):
        """With no aliases configured, segment text is passed through unchanged."""
        config = IntentJudgeConfig(assistant_name="Jarvis", aliases=[])
        judge = IntentJudge(config)
        segments = [TranscriptSegment("Jervis what time", 1000.0, 1001.0)]
        prompt = judge._build_user_prompt(
            segments,
            wake_timestamp=None,
            last_tts_text="",
            last_tts_finish_time=0.0,
            in_hot_window=False,
        )
        assert "Jervis" in prompt

    def test_build_user_prompt_with_tts(self):
        """User prompt includes TTS info."""
        judge = IntentJudge()
        segments = [
            TranscriptSegment("the weather is nice", 1000.0, 1001.0, is_during_tts=True),
        ]
        prompt = judge._build_user_prompt(
            segments,
            wake_timestamp=None,
            last_tts_text="The weather is nice and sunny",
            last_tts_finish_time=999.0,
            in_hot_window=True,
        )
        assert "TTS" in prompt
        assert "weather is nice and sunny" in prompt

    def test_parse_response_valid_json(self):
        """Parses valid JSON response."""
        judge = IntentJudge()
        response = '{"directed": true, "query": "what time", "stop": false, "confidence": "high", "reasoning": "clear"}'
        result = judge._parse_response(response)

        assert result is not None
        assert result.directed is True
        assert result.query == "what time"
        assert result.stop is False
        assert result.confidence == "high"

    def test_parse_response_with_extra_text(self):
        """Parses response with extra text around JSON."""
        judge = IntentJudge()
        response = 'Here is my analysis: {"directed": true, "query": "test", "stop": false, "confidence": "medium", "reasoning": "test"}'
        result = judge._parse_response(response)

        assert result is not None
        assert result.directed is True

    def test_parse_response_invalid_json(self):
        """Returns None for invalid JSON."""
        judge = IntentJudge()
        response = "This is not valid JSON at all"
        result = judge._parse_response(response)

        assert result is None

    def test_parse_response_missing_fields(self):
        """Handles missing fields with defaults."""
        judge = IntentJudge()
        response = '{"directed": true}'
        result = judge._parse_response(response)

        assert result is not None
        assert result.directed is True
        assert result.query == ""
        assert result.stop is False
        assert result.confidence == "low"

    def test_judge_returns_none_when_unavailable(self):
        """judge() returns None when unavailable."""
        judge = IntentJudge()
        judge._available = False

        segments = [TranscriptSegment("test", 1000.0, 1001.0)]
        result = judge.judge(segments)

        assert result is None

    def test_judge_returns_none_for_empty_segments(self):
        """judge() returns None for empty segments."""
        judge = IntentJudge()
        result = judge.judge([])
        assert result is None

    def test_judge_with_mock_api(self):
        """judge() calls API and parses response."""
        judge = IntentJudge()
        judge._available = True

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": '{"directed": true, "query": "what time is it", "stop": false, "confidence": "high", "reasoning": "wake word detected"}'
        }

        segments = [
            TranscriptSegment("jarvis what time is it", 1000.0, 1002.0),
        ]

        with patch('jarvis.listening.intent_judge.requests.post', return_value=mock_response):
            result = judge.judge(
                segments,
                wake_timestamp=1000.5,
                last_tts_text="",
                last_tts_finish_time=0.0,
                in_hot_window=False,
            )

        assert result is not None
        assert result.directed is True
        assert result.query == "what time is it"

    def test_judge_handles_api_error(self):
        """judge() handles API errors gracefully."""
        judge = IntentJudge()
        judge._available = True

        mock_response = MagicMock()
        mock_response.status_code = 500

        segments = [TranscriptSegment("test", 1000.0, 1001.0)]

        with patch('jarvis.listening.intent_judge.requests.post', return_value=mock_response):
            result = judge.judge(segments)

        assert result is None

    def test_judge_handles_timeout(self):
        """judge() handles timeout gracefully."""
        import requests as real_requests
        judge = IntentJudge()
        judge._available = True

        segments = [TranscriptSegment("test", 1000.0, 1001.0)]

        with patch('jarvis.listening.intent_judge.requests.post', side_effect=real_requests.Timeout()):
            result = judge.judge(segments)

        assert result is None

    def test_timeout_does_not_trigger_backoff(self):
        """Timeouts must NOT trigger the 30s cooldown.

        Voice is a high-turn environment: a single slow call must not lock out
        intent judging for the next half-minute of conversation. The upstream
        engagement-signal gate (wake word / hot window / TTS) already prevents
        hammering Ollama on ambient speech, so individual timeouts are safe to
        retry immediately on the next real engagement.
        """
        import requests as real_requests
        judge = IntentJudge()
        judge._available = True
        judge._last_error_time = 0.0

        segments = [TranscriptSegment("test", 1000.0, 1001.0)]

        with patch('jarvis.listening.intent_judge.requests.post', side_effect=real_requests.Timeout()):
            judge.judge(segments)

        assert judge._last_error_time == 0.0, "timeout must NOT lock out future calls"
        assert judge.available is True, "judge must remain available after a single timeout"

    def test_http_error_does_not_trigger_backoff(self):
        """Transient HTTP errors (503 etc.) must NOT trigger the 30s cooldown.

        Same reasoning as timeouts — we want to retry on the next engagement
        signal, not lock out intent judging.
        """
        judge = IntentJudge()
        judge._available = True
        judge._last_error_time = 0.0

        mock_response = MagicMock()
        mock_response.status_code = 503
        segments = [TranscriptSegment("test", 1000.0, 1001.0)]

        with patch('jarvis.listening.intent_judge.requests.post', return_value=mock_response):
            judge.judge(segments)

        assert judge._last_error_time == 0.0
        assert judge.available is True

    def test_connection_error_does_trigger_backoff(self):
        """Connection errors (Ollama actually down) DO trigger the 30s cooldown.

        If the server is unreachable, retrying on every engagement just wastes
        time. This is the one case where backoff is appropriate — it gives
        Ollama a chance to come back up.
        """
        import requests as real_requests
        judge = IntentJudge()
        judge._available = True
        judge._last_error_time = 0.0

        segments = [TranscriptSegment("test", 1000.0, 1001.0)]

        with patch(
            'jarvis.listening.intent_judge.requests.post',
            side_effect=real_requests.ConnectionError("refused"),
        ):
            judge.judge(segments)

        assert judge._last_error_time > 0.0
        assert judge.available is False

    def test_last_failure_reason_recorded_on_timeout(self):
        """Judge should remember why the last call failed so the listener can surface it."""
        import requests as real_requests
        judge = IntentJudge()
        judge._available = True

        segments = [TranscriptSegment("test", 1000.0, 1001.0)]

        with patch('jarvis.listening.intent_judge.requests.post', side_effect=real_requests.Timeout()):
            judge.judge(segments)

        assert "timeout" in judge.last_failure_reason.lower()

    def test_last_failure_reason_recorded_on_http_error(self):
        """HTTP non-200 responses should be recorded as a failure reason."""
        judge = IntentJudge()
        judge._available = True
        # Clear any stray _last_error_time from earlier test setup
        judge._last_error_time = 0.0

        mock_response = MagicMock()
        mock_response.status_code = 503
        segments = [TranscriptSegment("test", 1000.0, 1001.0)]

        with patch('jarvis.listening.intent_judge.requests.post', return_value=mock_response):
            judge.judge(segments)

        assert "503" in judge.last_failure_reason

    def test_last_failure_reason_cleared_on_success(self):
        """Successful judgments clear the last failure reason."""
        judge = IntentJudge()
        judge._available = True
        judge._last_failure_reason = "timeout"

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": '{"directed": false, "query": "", "stop": false, "confidence": "high", "reasoning": "ok"}'
        }
        segments = [TranscriptSegment("test", 1000.0, 1001.0)]

        with patch('jarvis.listening.intent_judge.requests.post', return_value=mock_response):
            result = judge.judge(segments)

        assert result is not None
        assert judge.last_failure_reason == ""


class TestResponseParserRobustness:
    """Tests for response parser edge cases seen in the wild."""

    def test_parse_response_with_nested_braces(self):
        """Parser handles JSON where a string value contains braces.

        The old regex `\\{[^{}]*\\}` failed on any nested brace, producing
        spurious "unavailable" errors when the model quoted code in reasoning.
        """
        judge = IntentJudge()
        response = '{"directed": true, "query": "format as {json}", "stop": false, "confidence": "high", "reasoning": "user asked about {formatting}"}'
        result = judge._parse_response(response)

        assert result is not None
        assert result.directed is True
        assert "json" in result.query

    def test_parse_response_with_markdown_code_fence(self):
        """Parser handles JSON wrapped in ```json ... ``` fences."""
        judge = IntentJudge()
        response = '```json\n{"directed": true, "query": "hi", "stop": false, "confidence": "high", "reasoning": "ok"}\n```'
        result = judge._parse_response(response)

        assert result is not None
        assert result.directed is True
        assert result.query == "hi"


class TestCreateIntentJudge:
    """Tests for create_intent_judge factory function."""

    def test_creates_judge_with_defaults(self):
        """Creates judge from config with defaults."""
        mock_cfg = MagicMock()
        mock_cfg.intent_judge_enabled = True
        mock_cfg.intent_judge_model = "gemma4:e2b"
        mock_cfg.ollama_base_url = "http://localhost:11434"
        mock_cfg.intent_judge_timeout_sec = 3.0
        mock_cfg.wake_word = "jarvis"
        mock_cfg.wake_aliases = []

        judge = create_intent_judge(mock_cfg)

        assert judge is not None
        assert judge.config.model == "gemma4:e2b"

    def test_always_returns_judge_when_requests_available(self):
        """Always returns judge when requests library is available (per spec)."""
        mock_cfg = MagicMock()
        mock_cfg.intent_judge_model = "gemma4:e2b"
        mock_cfg.ollama_base_url = "http://localhost:11434"
        mock_cfg.intent_judge_timeout_sec = 3.0
        mock_cfg.wake_word = "jarvis"
        mock_cfg.wake_aliases = []

        judge = create_intent_judge(mock_cfg)
        # Judge should always be created (per spec - falls back only when unavailable)
        assert judge is not None


class TestWarmUp:
    """Tests for IntentJudge.warm_up()."""

    def test_warmup_posts_to_generate_with_keep_alive(self):
        """Warmup issues a /api/generate request that pins the model in memory."""
        judge = IntentJudge(IntentJudgeConfig(model="gemma4:e2b"))
        with patch("jarvis.listening.intent_judge.requests") as mock_requests:
            mock_requests.post.return_value = MagicMock(status_code=200)
            ok = judge.warm_up()

        assert ok is True
        args, kwargs = mock_requests.post.call_args
        assert args[0].endswith("/api/generate")
        assert kwargs["json"]["model"] == "gemma4:e2b"
        assert kwargs["json"]["keep_alive"] == "30m"
        assert kwargs["json"]["stream"] is False

    def test_warmup_returns_false_on_http_error(self):
        """Warmup reports failure when Ollama returns a non-200 status."""
        judge = IntentJudge()
        with patch("jarvis.listening.intent_judge.requests") as mock_requests:
            mock_requests.post.return_value = MagicMock(status_code=500)
            assert judge.warm_up() is False

    def test_warmup_swallows_exceptions(self):
        """Warmup never raises — transport errors return False."""
        judge = IntentJudge()
        with patch("jarvis.listening.intent_judge.requests") as mock_requests:
            mock_requests.post.side_effect = RuntimeError("boom")
            assert judge.warm_up() is False

    def test_warmup_skipped_when_unavailable(self):
        """Warmup is a no-op when requests isn't installed."""
        judge = IntentJudge()
        judge._available = False
        assert judge.warm_up() is False


class TestEchoFollowUpPattern:
    """Tests for echo + follow-up pattern handling."""

    def test_system_prompt_includes_echo_followup_guidance(self):
        """System prompt includes guidance for echo + follow-up pattern."""
        judge = IntentJudge()
        prompt = judge._build_system_prompt()

        # Check that the prompt mentions echo handling
        assert "(during TTS)" in prompt  # Should explain during TTS marker
        assert "echo" in prompt.lower()  # Should mention echo

    def test_user_prompt_with_echo_and_followup(self):
        """User prompt correctly formats transcript with potential echo + follow-up."""
        judge = IntentJudge()
        segments = [
            TranscriptSegment(
                "London has 8 hours of daylight. That's cool tell me more",
                1000.0, 1003.0
            ),
        ]
        prompt = judge._build_user_prompt(
            segments,
            wake_timestamp=None,
            last_tts_text="London has around 8 hours of daylight",
            last_tts_finish_time=999.0,
            in_hot_window=True,
        )

        # Prompt should show hot window mode and include TTS text
        assert "HOT WINDOW" in prompt
        assert "8 hours of daylight" in prompt  # TTS text included

    def test_judge_extracts_followup_from_echo_mixed_transcript(self):
        """Judge correctly extracts follow-up from transcript containing echo."""
        judge = IntentJudge()
        judge._available = True

        # Simulate response where LLM correctly identifies echo + follow-up
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": '{"directed": true, "query": "that\'s cool tell me more", "stop": false, "confidence": "high", "reasoning": "first part matches TTS (echo), second part is user follow-up"}'
        }

        segments = [
            TranscriptSegment(
                "London has 8 hours of daylight. That's cool tell me more",
                1000.0, 1003.0
            ),
        ]

        with patch('jarvis.listening.intent_judge.requests.post', return_value=mock_response):
            result = judge.judge(
                segments,
                wake_timestamp=None,
                last_tts_text="London has around 8 hours of daylight",
                last_tts_finish_time=999.0,
                in_hot_window=True,
            )

        assert result is not None
        assert result.directed is True
        # The extracted query should be the follow-up, not the echo
        assert "tell me more" in result.query.lower()


class TestCurrentSegmentMarker:
    """Tests for CURRENT - JUDGE THIS marker functionality."""

    def test_current_segment_marked_in_prompt(self):
        """Prompt marks the current segment being judged."""
        judge = IntentJudge()
        segments = [
            TranscriptSegment("old query from before", 1000.0, 1001.0),
            TranscriptSegment("hello jarvis", 1002.0, 1003.0),  # New segment
        ]
        prompt = judge._build_user_prompt(
            segments,
            wake_timestamp=None,
            last_tts_text="",
            last_tts_finish_time=0.0,
            in_hot_window=True,
            current_text="hello jarvis",  # Mark this as current
        )

        # The current segment should be marked
        assert "CURRENT - JUDGE THIS" in prompt
        # Verify it's associated with the right segment
        assert '"hello jarvis"' in prompt

    def test_current_segment_not_marked_when_no_match(self):
        """Prompt doesn't mark segments when current_text doesn't match."""
        judge = IntentJudge()
        segments = [
            TranscriptSegment("hello jarvis", 1000.0, 1001.0),
        ]
        prompt = judge._build_user_prompt(
            segments,
            wake_timestamp=None,
            last_tts_text="",
            last_tts_finish_time=0.0,
            in_hot_window=True,
            current_text="something else",  # Doesn't match any segment
        )

        # No segment should be marked as current
        assert "CURRENT - JUDGE THIS" not in prompt

    def test_current_segment_case_insensitive_match(self):
        """Current segment matching is case insensitive."""
        judge = IntentJudge()
        segments = [
            TranscriptSegment("Hello Jarvis", 1000.0, 1001.0),
        ]
        prompt = judge._build_user_prompt(
            segments,
            wake_timestamp=None,
            last_tts_text="",
            last_tts_finish_time=0.0,
            in_hot_window=True,
            current_text="hello jarvis",  # Different case
        )

        # Should still mark the segment
        assert "CURRENT - JUDGE THIS" in prompt

    def test_judge_passes_current_text_to_prompt(self):
        """judge() method passes current_text parameter correctly."""
        judge = IntentJudge()
        judge._available = True

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": '{"directed": true, "query": "no thank you", "stop": false, "confidence": "high", "reasoning": "user response"}'
        }

        segments = [
            TranscriptSegment("old processed query", 1000.0, 1001.0),
            TranscriptSegment("no thank you", 1002.0, 1003.0),
        ]

        with patch('jarvis.listening.intent_judge.requests.post', return_value=mock_response) as mock_post:
            judge.judge(
                segments,
                wake_timestamp=None,
                last_tts_text="Would you like more info?",
                last_tts_finish_time=1001.5,
                in_hot_window=True,
                current_text="no thank you",
            )

            # Verify the prompt sent to the API contains the marker
            call_args = mock_post.call_args
            prompt = call_args[1]["json"]["prompt"]
            assert "CURRENT - JUDGE THIS" in prompt

    def test_system_prompt_includes_current_segment_guidance(self):
        """System prompt explains the CURRENT - JUDGE THIS marker."""
        judge = IntentJudge()
        prompt = judge._build_system_prompt()

        # System prompt should explain the marker
        assert "CURRENT - JUDGE THIS" in prompt
        assert "segment to judge" in prompt.lower()


class TestCrossSegmentContextInPrompt:
    """Tests that the system prompt guides cross-segment reference resolution.

    When the CURRENT segment contains vague references like "that", "it", "this",
    the intent judge should use PREVIOUS segments to resolve them into a complete query.
    """

    def test_system_prompt_encourages_cross_segment_resolution(self):
        """System prompt should explicitly tell the LLM to resolve references from other segments."""
        judge = IntentJudge()
        prompt = judge._build_system_prompt()

        # The prompt must mention resolving references from other/previous/background segments
        prompt_lower = prompt.lower()
        assert "previous" in prompt_lower or "other segment" in prompt_lower or "background" in prompt_lower, (
            "System prompt should mention using previous/background segments to resolve references"
        )

    def test_system_prompt_has_cross_segment_example(self):
        """System prompt should include an example of cross-segment reference resolution."""
        judge = IntentJudge()
        prompt = judge._build_system_prompt()

        # Should have an example where context comes from a DIFFERENT segment than the wake word
        # The key indicator is showing a multi-segment scenario in the prompt examples
        assert "previous segment" in prompt.lower() or "background context" in prompt.lower() or "earlier segment" in prompt.lower(), (
            "System prompt should have guidance about using earlier/background segments for context"
        )

    def test_context_segments_included_in_user_prompt(self):
        """Background context segments (unprocessed, no wake word) appear in the user prompt."""
        judge = IntentJudge()
        segments = [
            TranscriptSegment("I think dinosaurs are cool", 1000.0, 1001.0),
            TranscriptSegment("What do you think about that Jarvis", 1002.0, 1003.0),
        ]
        prompt = judge._build_user_prompt(
            segments,
            wake_timestamp=1002.5,
            last_tts_text="",
            last_tts_finish_time=0.0,
            in_hot_window=False,
            current_text="What do you think about that Jarvis",
        )

        # Both segments should be in the prompt — the first provides context
        assert "dinosaurs are cool" in prompt
        assert "What do you think about that Jarvis" in prompt
        assert "CURRENT - JUDGE THIS" in prompt


class TestProcessedSegmentFiltering:
    """Tests for processed segment filtering functionality.

    When segments have had queries extracted, they should be filtered out
    from the intent judge prompt to prevent re-extraction of old queries.
    """

    def test_processed_segments_filtered_from_prompt(self):
        """Processed segments are not included in the prompt."""
        judge = IntentJudge()
        segments = [
            TranscriptSegment("jarvis whats the weather", 1000.0, 1001.0, processed=True),
            TranscriptSegment("jarvis tell me a joke", 1002.0, 1003.0),
        ]
        prompt = judge._build_user_prompt(
            segments,
            wake_timestamp=None,
            last_tts_text="",
            last_tts_finish_time=0.0,
            in_hot_window=True,
            current_text="jarvis tell me a joke",
        )

        # The processed segment should NOT appear in the prompt
        assert "whats the weather" not in prompt
        # The current segment should appear
        assert "tell me a joke" in prompt

    def test_current_segment_shown_even_if_processed(self):
        """Current segment is shown even if marked as processed (edge case)."""
        judge = IntentJudge()
        # This edge case shouldn't happen in practice, but handle it gracefully
        segments = [
            TranscriptSegment("jarvis tell me a joke", 1000.0, 1001.0, processed=True),
        ]
        prompt = judge._build_user_prompt(
            segments,
            wake_timestamp=None,
            last_tts_text="",
            last_tts_finish_time=0.0,
            in_hot_window=True,
            current_text="jarvis tell me a joke",  # Same as processed segment
        )

        # Current segment should still be shown (it's what we're judging)
        assert "tell me a joke" in prompt
        assert "CURRENT - JUDGE THIS" in prompt

    def test_multiple_processed_segments_all_filtered(self):
        """Multiple processed segments are all filtered."""
        judge = IntentJudge()
        segments = [
            TranscriptSegment("first old query", 1000.0, 1001.0, processed=True),
            TranscriptSegment("second old query", 1001.0, 1002.0, processed=True),
            TranscriptSegment("new query", 1002.0, 1003.0),
        ]
        prompt = judge._build_user_prompt(
            segments,
            wake_timestamp=None,
            last_tts_text="",
            last_tts_finish_time=0.0,
            in_hot_window=True,
            current_text="new query",
        )

        # Both processed segments should be filtered
        assert "first old query" not in prompt
        assert "second old query" not in prompt
        # Current segment should be present
        assert "new query" in prompt

    def test_unprocessed_context_segments_preserved(self):
        """Non-wake-word context segments (unprocessed) are preserved."""
        judge = IntentJudge()
        segments = [
            TranscriptSegment("I wonder about the weather", 1000.0, 1001.0),  # Context
            TranscriptSegment("jarvis old query", 1001.0, 1002.0, processed=True),  # Processed
            TranscriptSegment("Yeah me too", 1002.0, 1003.0),  # Context
            TranscriptSegment("jarvis what do you think", 1003.0, 1004.0),  # Current
        ]
        prompt = judge._build_user_prompt(
            segments,
            wake_timestamp=None,
            last_tts_text="",
            last_tts_finish_time=0.0,
            in_hot_window=True,
            current_text="jarvis what do you think",
        )

        # Context segments (not processed, not wake word) should be preserved
        assert "I wonder about the weather" in prompt
        assert "Yeah me too" in prompt
        # Processed segment should be filtered
        assert "old query" not in prompt
        # Current segment should be present
        assert "what do you think" in prompt
