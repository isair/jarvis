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
        assert config.model == "llama3.2:3b"
        assert config.timeout_sec == 3.0
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


class TestCreateIntentJudge:
    """Tests for create_intent_judge factory function."""

    def test_creates_judge_with_defaults(self):
        """Creates judge from config with defaults."""
        mock_cfg = MagicMock()
        mock_cfg.intent_judge_enabled = True
        mock_cfg.intent_judge_model = "llama3.2:3b"
        mock_cfg.ollama_base_url = "http://localhost:11434"
        mock_cfg.intent_judge_timeout_sec = 3.0
        mock_cfg.wake_word = "jarvis"
        mock_cfg.wake_aliases = []

        judge = create_intent_judge(mock_cfg)

        assert judge is not None
        assert judge.config.model == "llama3.2:3b"

    def test_always_returns_judge_when_requests_available(self):
        """Always returns judge when requests library is available (per spec)."""
        mock_cfg = MagicMock()
        mock_cfg.intent_judge_model = "llama3.2:3b"
        mock_cfg.ollama_base_url = "http://localhost:11434"
        mock_cfg.intent_judge_timeout_sec = 3.0
        mock_cfg.wake_word = "jarvis"
        mock_cfg.wake_aliases = []

        judge = create_intent_judge(mock_cfg)
        # Judge should always be created (per spec - falls back only when unavailable)
        assert judge is not None


class TestEchoFollowUpPattern:
    """Tests for echo + follow-up pattern handling."""

    def test_system_prompt_includes_echo_followup_guidance(self):
        """System prompt includes guidance for echo + follow-up pattern."""
        judge = IntentJudge()
        prompt = judge._build_system_prompt()

        # Check that the prompt mentions handling mixed echo + real speech
        assert "BOTH echo AND real speech" in prompt or "echo + follow-up" in prompt.lower()

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
