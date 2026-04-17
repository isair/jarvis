"""LLM-based intent judge for voice assistant.

This module provides intelligent intent classification and query extraction
using a larger LLM model. It receives full context (transcript buffer,
TTS history, state) and makes informed decisions about whether speech
is directed at the assistant and what the actual query is.
"""

import json
import re
import time
from dataclasses import dataclass
from typing import Optional, List

from ..debug import debug_log
from .transcript_buffer import TranscriptSegment

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    requests = None
    REQUESTS_AVAILABLE = False


@dataclass
class IntentJudgment:
    """Result of intent judgment."""

    directed: bool           # Is this speech directed at the assistant?
    query: str               # Extracted query (cleaned of filler, echo, pre-wake-word)
    stop: bool               # Is this a stop command?
    confidence: str          # "high", "medium", or "low"
    reasoning: str           # Brief explanation for debugging
    raw_response: str = ""   # Raw LLM response for debugging


@dataclass
class IntentJudgeConfig:
    """Configuration for the intent judge."""

    assistant_name: str = "Jarvis"
    aliases: list = None
    model: str = "gemma4:e2b"
    ollama_base_url: str = "http://127.0.0.1:11434"
    timeout_sec: float = 10.0
    thinking: bool = False

    def __post_init__(self):
        if self.aliases is None:
            self.aliases = []


class IntentJudge:
    """LLM-based intent classification and query extraction.

    This judge receives full context about the conversation and makes
    intelligent decisions about:
    1. Whether speech is directed at the assistant
    2. What the actual query is (excluding echo, pre-wake-word chatter, filler)
    3. Whether this is a stop command

    Uses a small model (gemma4) for better accuracy compared to
    the simpler intent_validator.
    """

    SYSTEM_PROMPT_TEMPLATE = '''You are the intent judge for voice assistant "{name}".

Two modes:

WAKE WORD MODE:
- Extract complete query from segment containing "{name}" — may be a question, statement, or command/imperative addressed to the assistant (e.g. "set a timer", "remind me to...", "play music"). All are valid queries.
- If current segment contains a vague ref ("that", "it", "this") OR a topic-less open question ("what do you think", "how much does it cost", "is it worth it") — NAME the topic from earlier segments inside the query string. Do NOT output the vague/open form literally.
- Example: "I made carbonara" + "Jarvis find recipe for that" -> "find recipe for carbonara"
- Example: "the weather will be nice tomorrow" + "Jarvis what do you think" -> "what do you think about the weather tomorrow"
- Example: "the new iPhone is cool" + "Jarvis how much does it cost" -> "how much does the iPhone cost"
- If standalone imperative command ("answer that", "respond to that", "reply to that", "address that", "answer my question", "go ahead and answer") NOT a question -> re-issue prior question
  Variants: "answered that", "answers that", "answering that" = same imperative (Whisper tense errors)
  Exception: If segment has BOTH imperative + new question -> new question wins
- Query must be answerable alone (without the transcript)

HOT WINDOW MODE (no wake word needed):
- User IS DIRECTED (directed=true)
- Extract from segments WITHOUT "(during TTS)" marker
- Question or statement both valid

ECHO / MARKER RULES:
- "(during TTS)" = echo of assistant -> skip, never extract
- "(CURRENT - JUDGE THIS)" = segment to judge now
- Use earlier segments to resolve references only, not as query source

STOP DETECTION:
- "stop", "quiet" (standalone or short command) -> directed=true, stop=true, query=""

NOT DIRECTED:
- No wake word AND not hot window -> directed=false
- Wake word used only as a narrative mention ("I told my friend about {name}") -> directed=false

Output JSON only:
{{"directed": true/false, "query": "...", "stop": true/false, "confidence": "high/medium/low", "reasoning": "brief"}}

Examples:
- "Jarvis what time is it" -> {{"directed": true, "query": "what time is it", "stop": false, "confidence": "high", "reasoning": "wake word + question"}}
- Previous "dinosaurs are cool" + Current "Jarvis what do you think about that" -> {{"directed": true, "query": "what do you think about dinosaurs being cool", "stop": false, "confidence": "high", "reasoning": "resolved 'that' to dinosaurs"}}
- Previous "How's the weather?" + Current "Jarvis answer that" -> {{"directed": true, "query": "how is the weather", "stop": false, "confidence": "high", "reasoning": "imperative -> re-issue prior question"}}
- Hot window, user says "I think absurdism is better" -> {{"directed": true, "query": "I think absurdism is better", "stop": false, "confidence": "high", "reasoning": "user statement in hot window"}}
- "(during TTS)" segments only -> {{"directed": false, "query": "", "stop": false, "confidence": "high", "reasoning": "only echo"}}
- "stop" -> {{"directed": true, "query": "", "stop": true, "confidence": "high", "reasoning": "stop command"}}
- No wake word, not hot window -> {{"directed": false, "query": "", "stop": false, "confidence": "high", "reasoning": "no wake word"}}'''

    def __init__(self, config: Optional[IntentJudgeConfig] = None):
        """Initialize the intent judge.

        Args:
            config: Configuration for the judge
        """
        self.config = config or IntentJudgeConfig()
        self._available = REQUESTS_AVAILABLE
        self._last_error_time: float = 0.0
        self._error_cooldown: float = 30.0

        if not self._available:
            debug_log("intent judge disabled: requests not available", "voice")

    @property
    def available(self) -> bool:
        """Check if intent judge is available."""
        if not self._available:
            return False
        if time.time() - self._last_error_time < self._error_cooldown:
            return False
        return True

    def _build_system_prompt(self) -> str:
        """Build the system prompt with configuration."""
        return self.SYSTEM_PROMPT_TEMPLATE.format(name=self.config.assistant_name)

    def _build_user_prompt(
        self,
        segments: List[TranscriptSegment],
        wake_timestamp: Optional[float],
        last_tts_text: str,
        last_tts_finish_time: float,
        in_hot_window: bool,
        current_text: str = "",
    ) -> str:
        """Build the user prompt with full context.

        Args:
            segments: Recent transcript segments
            wake_timestamp: When wake word was detected (None if hot window)
            last_tts_text: What TTS last said
            last_tts_finish_time: When TTS finished
            in_hot_window: Whether we're in hot window mode
            current_text: The text that triggered this intent judgment (for marking)

        Returns:
            Formatted prompt for the LLM
        """
        lines = ["Transcript:"]

        # Find the segment matching current_text (normalize for comparison)
        current_text_lower = current_text.lower().strip() if current_text else ""

        for seg in segments:
            # Skip processed segments entirely - they already had queries extracted
            # The dialogue memory has context from those processed turns
            is_current_segment = current_text_lower and seg.text.lower().strip() == current_text_lower
            if seg.processed and not is_current_segment:
                continue

            ts = seg.format_timestamp()
            markers = []

            if seg.is_during_tts:
                markers.append("during TTS")
            if wake_timestamp and seg.start_time <= wake_timestamp <= seg.end_time:
                markers.append("WAKE WORD DETECTED")
            # Mark the current segment being judged (match by text content)
            if is_current_segment:
                markers.append("CURRENT - JUDGE THIS")

            marker_str = f" ({', '.join(markers)})" if markers else ""
            lines.append(f'[{ts}]{marker_str} "{seg.text}"')

        if not segments:
            lines.append("(no speech)")

        lines.append("")

        # Wake word info
        if in_hot_window:
            lines.append("Mode: HOT WINDOW (listening for follow-up, no wake word needed)")
        elif wake_timestamp:
            from datetime import datetime
            wake_ts_str = datetime.fromtimestamp(wake_timestamp).strftime('%H:%M:%S.%f')[:-3]
            lines.append(f"Wake word detected at: {wake_ts_str}")
        else:
            lines.append("Mode: WAKE WORD (waiting for wake word)")

        # TTS info
        lines.append("")
        if last_tts_text:
            from datetime import datetime
            tts_ts_str = datetime.fromtimestamp(last_tts_finish_time).strftime('%H:%M:%S') if last_tts_finish_time > 0 else "unknown"
            lines.append(f'Last TTS output: "{last_tts_text[:200]}{"..." if len(last_tts_text) > 200 else ""}"')
            lines.append(f"TTS finished at: {tts_ts_str}")
        else:
            lines.append("Last TTS: None")

        return "\n".join(lines)

    def _parse_response(self, response_text: str) -> Optional[IntentJudgment]:
        """Parse the LLM response into a judgment.

        Args:
            response_text: Raw response from the LLM

        Returns:
            IntentJudgment or None if parsing failed
        """
        # Try to extract JSON from response
        json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
        if not json_match:
            debug_log(f"intent judge: no JSON found in response: {response_text[:100]}", "voice")
            return None

        try:
            data = json.loads(json_match.group())

            return IntentJudgment(
                directed=bool(data.get("directed", False)),
                query=str(data.get("query", "")).strip(),
                stop=bool(data.get("stop", False)),
                confidence=str(data.get("confidence", "low")).lower(),
                reasoning=str(data.get("reasoning", "")),
                raw_response=response_text,
            )
        except (json.JSONDecodeError, KeyError) as e:
            debug_log(f"intent judge: failed to parse response: {e}", "voice")
            return None

    def judge(
        self,
        segments: List[TranscriptSegment],
        wake_timestamp: Optional[float] = None,
        last_tts_text: str = "",
        last_tts_finish_time: float = 0.0,
        in_hot_window: bool = False,
        current_text: str = "",
    ) -> Optional[IntentJudgment]:
        """Judge whether speech is directed at assistant and extract query.

        Args:
            segments: Recent transcript segments
            wake_timestamp: When wake word was detected (None if hot window/text-based)
            last_tts_text: What TTS last said (for echo detection)
            last_tts_finish_time: When TTS finished
            in_hot_window: Whether we're in hot window mode
            current_text: The text that triggered this judgment (for marking current segment)

        Returns:
            IntentJudgment or None if judgment failed
        """
        if not self.available:
            return None

        if not segments:
            return None

        try:
            system_prompt = self._build_system_prompt()
            user_prompt = self._build_user_prompt(
                segments,
                wake_timestamp,
                last_tts_text,
                last_tts_finish_time,
                in_hot_window,
                current_text,
            )

            # Log input
            mode = "hot_window" if in_hot_window else "wake_word"
            transcript_preview = "; ".join(s.text[:30] for s in segments[-3:])
            debug_log(f"🧠 Intent judge [{mode}]: \"{transcript_preview}...\"", "voice")

            # Call Ollama API
            response = requests.post(
                f"{self.config.ollama_base_url}/api/generate",
                json={
                    "model": self.config.model,
                    "prompt": user_prompt,
                    "system": system_prompt,
                    "stream": False,
                    "think": self.config.thinking,
                    "options": {
                        "temperature": 0.0,
                        "num_predict": 800,
                    },
                },
                timeout=self.config.timeout_sec,
            )

            if response.status_code != 200:
                debug_log(f"intent judge: Ollama error {response.status_code}", "voice")
                self._last_error_time = time.time()
                return None

            result = response.json()
            response_text = result.get("response", "")

            judgment = self._parse_response(response_text)

            if judgment:
                direction = "✅ DIRECTED" if judgment.directed else "❌ NOT DIRECTED"
                stop_str = " [STOP]" if judgment.stop else ""
                query_str = f" → \"{judgment.query}\"" if judgment.query else ""
                debug_log(
                    f"🧠 Intent judge: {direction} ({judgment.confidence}){stop_str}{query_str}",
                    "voice"
                )
                debug_log(f"   Reasoning: {judgment.reasoning}", "voice")
            else:
                debug_log(f"🧠 Intent judge: failed to parse: {response_text[:100]}", "voice")

            return judgment

        except requests.Timeout:
            debug_log(f"intent judge: timeout after {self.config.timeout_sec}s", "voice")
            return None
        except requests.RequestException as e:
            debug_log(f"intent judge: request error: {e}", "voice")
            self._last_error_time = time.time()
            return None
        except Exception as e:
            debug_log(f"intent judge: error: {e}", "voice")
            return None


def create_intent_judge(cfg) -> Optional[IntentJudge]:
    """Create an intent judge from Jarvis configuration.

    The intent judge is always used when available (per spec). Falls back to
    simple wake word detection only when Ollama is unavailable.

    Args:
        cfg: Jarvis Settings object

    Returns:
        IntentJudge instance or None if requests library unavailable
    """
    model = str(getattr(cfg, "intent_judge_model", "gemma4:e2b"))
    ollama_base_url = str(getattr(cfg, "ollama_base_url", "http://127.0.0.1:11434"))

    config = IntentJudgeConfig(
        assistant_name=str(getattr(cfg, "wake_word", "jarvis")).capitalize(),
        aliases=list(getattr(cfg, "wake_aliases", [])),
        model=model,
        ollama_base_url=ollama_base_url,
        timeout_sec=float(getattr(cfg, "intent_judge_timeout_sec", 10.0)),
        thinking=bool(getattr(cfg, "intent_judge_thinking_enabled", False)),
    )

    return IntentJudge(config)
