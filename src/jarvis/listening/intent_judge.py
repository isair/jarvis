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
    model: str = "llama3.2:3b"
    ollama_base_url: str = "http://127.0.0.1:11434"
    timeout_sec: float = 3.0

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

    Uses a larger model (llama3.2:3b) for better accuracy compared to
    the simpler intent_validator.
    """

    SYSTEM_PROMPT_TEMPLATE = '''You are the intent judge for voice assistant "{name}".

You receive:
1. Recent transcript with timestamps (may include multi-person conversation)
2. Wake word detection info (timestamp or "hot window" mode)
3. Last TTS output (what the assistant just said)
4. Current state

Your job:
1. Determine if speech is directed at the assistant
2. Synthesize a COMPLETE QUERY using conversation context (not just post-wake-word text)
3. Detect stop commands

CRITICAL - Query synthesis:
- The transcript may contain conversation CONTEXT that's relevant to the query
- Synthesize a complete, actionable query that includes necessary context
- Example: "I wonder what the weather will be tomorrow" ... "Jarvis what do you think"
  → query is "what do you think about the weather tomorrow" (includes context)
- Example: "The new iPhone looks cool" ... "Jarvis how much does that cost"
  → query is "how much does the new iPhone cost" (resolves "that")
- For simple direct questions, just extract the question: "Jarvis what time is it" → "what time is it"
- Ignore irrelevant chatter that's not related to the final question

CRITICAL - Echo handling:
- If transcript text matches or closely resembles the last TTS output, it's likely ECHO
- Echo is the assistant's own speech being picked up by the microphone
- Echo typically appears within 1-2 seconds after TTS finishes
- Segments marked "(during TTS)" are almost certainly echo
- Real user speech has DIFFERENT content than what TTS said
- IMPORTANT: A single transcript may contain BOTH echo AND real speech!
  Example: "London has 8 hours of daylight. That's cool, tell me more."
  If TTS said "London has 8 hours of daylight" → extract "That's cool, tell me more" as the query

Output JSON only, no other text:
{{"directed": true/false, "query": "synthesized query with context", "stop": true/false, "confidence": "high/medium/low", "reasoning": "brief explanation"}}

Examples:
- Wake word + question → {{"directed": true, "query": "what time is it", "stop": false, "confidence": "high", "reasoning": "clear wake word with direct question"}}
- Multi-person conversation → {{"directed": true, "query": "what do you think about the weather tomorrow for the picnic", "stop": false, "confidence": "high", "reasoning": "synthesized context from conversation about weather and picnic"}}
- Just echo of TTS → {{"directed": false, "query": "", "stop": false, "confidence": "high", "reasoning": "transcript matches TTS output, likely echo"}}
- Echo + follow-up in hot window → {{"directed": true, "query": "that's interesting tell me more", "stop": false, "confidence": "high", "reasoning": "first part matches TTS (echo), second part is user follow-up"}}
- "stop" during TTS → {{"directed": true, "query": "", "stop": true, "confidence": "high", "reasoning": "stop command during TTS"}}
- Mentioned in narrative → {{"directed": false, "query": "", "stop": false, "confidence": "high", "reasoning": "assistant mentioned in past tense, not addressing"}}'''

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
    ) -> str:
        """Build the user prompt with full context.

        Args:
            segments: Recent transcript segments
            wake_timestamp: When wake word was detected (None if hot window)
            last_tts_text: What TTS last said
            last_tts_finish_time: When TTS finished
            in_hot_window: Whether we're in hot window mode

        Returns:
            Formatted prompt for the LLM
        """
        lines = ["Transcript:"]

        for seg in segments:
            ts = seg.format_timestamp()
            markers = []
            if seg.is_during_tts:
                markers.append("during TTS")
            if wake_timestamp and seg.start_time <= wake_timestamp <= seg.end_time:
                markers.append("WAKE WORD DETECTED")

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
    ) -> Optional[IntentJudgment]:
        """Judge whether speech is directed at assistant and extract query.

        Args:
            segments: Recent transcript segments
            wake_timestamp: When wake word was detected (None if hot window/text-based)
            last_tts_text: What TTS last said (for echo detection)
            last_tts_finish_time: When TTS finished
            in_hot_window: Whether we're in hot window mode

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
                    "options": {
                        "temperature": 0.0,
                        "num_predict": 200,
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
    model = str(getattr(cfg, "intent_judge_model", "llama3.2:3b"))
    ollama_base_url = str(getattr(cfg, "ollama_base_url", "http://127.0.0.1:11434"))

    config = IntentJudgeConfig(
        assistant_name=str(getattr(cfg, "wake_word", "jarvis")).capitalize(),
        aliases=list(getattr(cfg, "wake_aliases", [])),
        model=model,
        ollama_base_url=ollama_base_url,
        timeout_sec=float(getattr(cfg, "intent_judge_timeout_sec", 3.0)),
    )

    return IntentJudge(config)
