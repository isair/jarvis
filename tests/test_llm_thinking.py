"""
Tests for the LLM thinking mode feature.

Verifies that the ``llm_thinking_enabled`` config option correctly controls
the ``think`` parameter sent to Ollama across all call sites.
"""

import json
import threading
from unittest.mock import patch, MagicMock

import pytest

from jarvis.config import get_default_config


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------

class TestThinkingConfig:
    """Config layer tests for llm_thinking_enabled."""

    def test_default_config_has_thinking_disabled(self):
        """llm_thinking_enabled should default to False."""
        config = get_default_config()
        assert "llm_thinking_enabled" in config
        assert config["llm_thinking_enabled"] is False


# ---------------------------------------------------------------------------
# llm.py — payload construction
# ---------------------------------------------------------------------------

class TestLlmThinkingPayload:
    """Verify the ``think`` key appears in Ollama request payloads."""

    def _capture_payload(self, mock_post, *, expect_stream=False):
        """Extract the JSON payload from the first call to requests.post."""
        assert mock_post.called, "requests.post was never called"
        _, kwargs = mock_post.call_args
        payload = kwargs.get("json") or {}
        return payload

    @patch("jarvis.llm.requests.post")
    def test_call_llm_direct_thinking_false(self, mock_post):
        from jarvis.llm import call_llm_direct

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"message": {"content": "ok"}}
        mock_post.return_value = mock_resp

        call_llm_direct("http://localhost:11434", "gemma4:e2b", "sys", "hi", thinking=False)
        payload = self._capture_payload(mock_post)
        assert payload["think"] is False

    @patch("jarvis.llm.requests.post")
    def test_call_llm_direct_thinking_true(self, mock_post):
        from jarvis.llm import call_llm_direct

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"message": {"content": "ok"}}
        mock_post.return_value = mock_resp

        call_llm_direct("http://localhost:11434", "gemma4:e2b", "sys", "hi", thinking=True)
        payload = self._capture_payload(mock_post)
        assert payload["think"] is True

    @patch("jarvis.llm.requests.post")
    def test_call_llm_direct_thinking_defaults_false(self, mock_post):
        from jarvis.llm import call_llm_direct

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"message": {"content": "ok"}}
        mock_post.return_value = mock_resp

        call_llm_direct("http://localhost:11434", "gemma4:e2b", "sys", "hi")
        payload = self._capture_payload(mock_post)
        assert payload["think"] is False

    @patch("jarvis.llm.requests.post")
    def test_call_llm_streaming_thinking(self, mock_post):
        from jarvis.llm import call_llm_streaming

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_lines.return_value = [
            json.dumps({"message": {"content": "hi"}}).encode()
        ]
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        call_llm_streaming("http://localhost:11434", "gemma4:e2b", "sys", "hi", thinking=True)
        payload = self._capture_payload(mock_post)
        assert payload["think"] is True

    @patch("jarvis.llm.requests.post")
    def test_chat_with_messages_thinking(self, mock_post):
        from jarvis.llm import chat_with_messages

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"message": {"content": "ok"}}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        msgs = [{"role": "user", "content": "hi"}]
        chat_with_messages("http://localhost:11434", "gemma4:e2b", msgs, thinking=True)
        payload = self._capture_payload(mock_post)
        assert payload["think"] is True

    @patch("jarvis.llm.requests.post")
    def test_chat_with_messages_thinking_defaults_false(self, mock_post):
        from jarvis.llm import chat_with_messages

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"message": {"content": "ok"}}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        msgs = [{"role": "user", "content": "hi"}]
        chat_with_messages("http://localhost:11434", "gemma4:e2b", msgs)
        payload = self._capture_payload(mock_post)
        assert payload["think"] is False


# ---------------------------------------------------------------------------
# Intent judge
# ---------------------------------------------------------------------------

class TestIntentJudgeThinking:
    """Intent judge respects the thinking config."""

    def test_config_default_thinking_false(self):
        from jarvis.listening.intent_judge import IntentJudgeConfig
        config = IntentJudgeConfig()
        assert config.thinking is False

    def test_config_accepts_thinking_true(self):
        from jarvis.listening.intent_judge import IntentJudgeConfig
        config = IntentJudgeConfig(thinking=True)
        assert config.thinking is True

    def test_create_intent_judge_passes_thinking(self):
        """create_intent_judge should read llm_thinking_enabled from cfg."""
        from jarvis.listening.intent_judge import create_intent_judge

        cfg = MagicMock()
        cfg.wake_word = "jarvis"
        cfg.wake_aliases = []
        cfg.intent_judge_model = "gemma4:e2b"
        cfg.ollama_base_url = "http://localhost:11434"
        cfg.intent_judge_timeout_sec = 10.0
        cfg.llm_thinking_enabled = True

        judge = create_intent_judge(cfg)
        assert judge is not None
        assert judge.config.thinking is True


# ---------------------------------------------------------------------------
# Dictation engine
# ---------------------------------------------------------------------------

class TestDictationThinking:
    """Dictation engine respects the thinking config."""

    def test_llm_clean_dictation_sends_think_false(self):
        from src.jarvis.dictation.dictation_engine import _llm_clean_dictation

        with patch("requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"response": "cleaned"}
            mock_post.return_value = mock_resp

            _llm_clean_dictation("um hello", "http://localhost:11434", thinking=False)
            payload = mock_post.call_args[1].get("json") or mock_post.call_args[0][1] if len(mock_post.call_args[0]) > 1 else mock_post.call_args[1]["json"]
            assert payload["think"] is False

    def test_llm_clean_dictation_sends_think_true(self):
        from src.jarvis.dictation.dictation_engine import _llm_clean_dictation

        with patch("requests.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"response": "cleaned"}
            mock_post.return_value = mock_resp

            _llm_clean_dictation("um hello", "http://localhost:11434", thinking=True)
            payload = mock_post.call_args[1].get("json") or mock_post.call_args[0][1] if len(mock_post.call_args[0]) > 1 else mock_post.call_args[1]["json"]
            assert payload["think"] is True

    def test_engine_stores_thinking(self):
        from src.jarvis.dictation.dictation_engine import DictationEngine

        engine = DictationEngine(
            whisper_model_ref=lambda: MagicMock(),
            whisper_backend_ref=lambda: "faster-whisper",
            mlx_repo_ref=lambda: None,
            hotkey="ctrl+shift+d",
            sample_rate=16000,
            transcribe_lock=threading.Lock(),
            thinking=True,
        )
        assert engine._thinking is True


# ---------------------------------------------------------------------------
# Settings window metadata
# ---------------------------------------------------------------------------

class TestSettingsWindowThinking:
    """Settings window includes the thinking field."""

    def test_field_metadata_includes_thinking(self):
        from desktop_app.settings_window import FIELD_METADATA
        keys = [fm.key for fm in FIELD_METADATA]
        assert "llm_thinking_enabled" in keys

    def test_thinking_field_is_bool_type(self):
        from desktop_app.settings_window import FIELD_METADATA
        field = next(fm for fm in FIELD_METADATA if fm.key == "llm_thinking_enabled")
        assert field.field_type == "bool"

    def test_thinking_field_in_llm_category(self):
        from desktop_app.settings_window import FIELD_METADATA
        field = next(fm for fm in FIELD_METADATA if fm.key == "llm_thinking_enabled")
        assert field.category == "llm"
