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


def warm_up_ollama_model(base_url: str, model: str, timeout: float) -> bool:
    """Ask Ollama to load ``model`` into memory with a 30m keep_alive.

    Issues a minimal ``/api/generate`` request so the weights are resident
    before the first real request. Best-effort — errors are logged and
    swallowed so callers never crash on warmup failure.
    """
    if not REQUESTS_AVAILABLE or not base_url or not model:
        return False
    try:
        response = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": "",
                "stream": False,
                "keep_alive": "30m",
                "options": {"num_predict": 1},
            },
            timeout=timeout,
        )
        ok = response.status_code == 200
        debug_log(
            f"ollama warmup {'ok' if ok else f'failed HTTP {response.status_code}'} "
            f"(model={model})",
            "voice",
        )
        return ok
    except Exception as e:
        debug_log(f"ollama warmup error (model={model}): {e}", "voice")
        return False


def _extract_json_object(text: str) -> str:
    """Return the first balanced `{...}` object in `text`, or "" if none.

    Walks character-by-character tracking brace depth while respecting string
    literals and escapes. Handles markdown code fences and values containing
    braces — cases a simple regex cannot.
    """
    start = text.find("{")
    if start == -1:
        return ""

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return ""


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
    timeout_sec: float = 15.0
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
- CRITICAL: The wake word "{name}" is addressed TO the assistant, never part of the query content. Remove every occurrence of "{name}" from the extracted query, whether it appears at the start, end, or middle of the sentence — including when it sits next to a named entity (e.g. "movie called Possessor Jarvis" → the film is "Possessor", not "Possessor Jarvis"). Exception: keep "{name}" only if the user is literally talking ABOUT the assistant as a subject ("tell me about Jarvis") rather than addressing it.
- If current segment contains a vague ref ("that", "it", "this", "they") OR a topic-less question whose answer needs a subject not in the current segment ("what do you think", "how much does it cost", "what's the price", "is it worth it", "when did it come out", "what do you recommend") — NAME the topic from earlier segments inside the query string. Do NOT output the vague/open form literally.
- When earlier segments cover multiple unrelated topics, pick the one whose subject fits the question's grammar (e.g. "what's the price" -> a purchasable thing, not a sports game). Ignore unrelated threads.
- Example: "I made carbonara" + "Jarvis find recipe for that" -> "find recipe for carbonara"
- Example: "the weather will be nice tomorrow" + "Jarvis what do you think" -> "what do you think about the weather tomorrow"
- Example: "the new iPhone is cool" + "Jarvis how much does it cost" -> "how much does the iPhone cost"
- Example: "did you catch the ball game" + "the new iPhone is out" + "I want the pro model" + "Jarvis what's the price" -> "what's the price of the iPhone pro model". NOT "what's the price of the pro model" (which pro model? ambiguous) — always prepend the brand/parent from earlier segments.
- If standalone imperative command ("answer that", "respond to that", "reply to that", "address that", "answer my question", "go ahead and answer") NOT a question -> re-issue prior question
  Variants: "answered that", "answers that", "answering that" = same imperative (Whisper tense errors)
  Exception: If segment has BOTH imperative + new question -> new question wins
  This rule ONLY applies to imperatives that explicitly reference a prior thing ("that", "my question", "answer"). Self-contained imperatives with open subjects ("say something", "tell me a joke", "tell me anything", "give me advice", "surprise me") are valid queries — pass them through literally, do NOT treat them as vague or as needing a prior question.
- Query must be answerable alone (without the transcript). When resolving to a sub-item ("pro model", "the red one"), also include the parent noun/brand from earlier segments — "pro model" alone is not self-contained; "iPhone pro model" is.

HOT WINDOW MODE (no wake word needed):
- User IS DIRECTED (directed=true) — always. This overrides any "topic-less question" heuristic above; follow-ups like "tell me more" are directed in hot window.
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
- "what do you know about the movie called Possessor Jarvis" -> {{"directed": true, "query": "what do you know about the movie called Possessor", "stop": false, "confidence": "high", "reasoning": "wake word at end; entity is Possessor, not Possessor Jarvis"}}
- "hey Jarvis what's the weather in London" -> {{"directed": true, "query": "what's the weather in London", "stop": false, "confidence": "high", "reasoning": "wake word removed from mid-sentence position"}}
- "Jarvis say something please" -> {{"directed": true, "query": "say something please", "stop": false, "confidence": "high", "reasoning": "self-contained imperative"}}
- "Jarvis tell me a joke" -> {{"directed": true, "query": "tell me a joke", "stop": false, "confidence": "high", "reasoning": "self-contained imperative"}}
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
        self._last_failure_reason: str = ""

        if not self._available:
            debug_log("intent judge disabled: requests not available", "voice")

    @property
    def last_failure_reason(self) -> str:
        """Human-readable reason the most recent judge() call failed, if any."""
        return self._last_failure_reason

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

    def _normalize_aliases(self, text: str) -> str:
        """Replace wake-word aliases with the primary assistant name.

        Aliases are Whisper mishearings of the wake word (e.g. "Jervis",
        "Jaivis"). Without normalisation the small judge model sees "Jervis"
        in the transcript, doesn't know it refers to {name}, and may decide
        the user is addressing a different person.
        """
        if not text or not self.config.aliases:
            return text
        # Longest-first avoids a shorter alias matching inside a longer one.
        for alias in sorted(self.config.aliases, key=len, reverse=True):
            if not alias:
                continue
            pattern = r"\b" + re.escape(alias) + r"\b"
            text = re.sub(pattern, self.config.assistant_name, text, flags=re.IGNORECASE)
        return text

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
            display_text = self._normalize_aliases(seg.text)
            lines.append(f'[{ts}]{marker_str} "{display_text}"')

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
        # Locate the outermost JSON object by brace-matching. This handles
        # markdown code fences and JSON whose string values contain braces
        # — cases the old `\{[^{}]*\}` regex missed.
        json_text = _extract_json_object(response_text)
        if not json_text:
            debug_log(f"intent judge: no JSON found in response: {response_text[:100]}", "voice")
            return None

        try:
            data = json.loads(json_text)

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

    def warm_up(self) -> bool:
        """Trigger Ollama to load the model into memory ahead of first use."""
        if not self._available:
            return False
        return warm_up_ollama_model(
            self.config.ollama_base_url,
            self.config.model,
            timeout=max(self.config.timeout_sec, 60.0),
        )

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

            # Call Ollama API. keep_alive keeps the model resident between
            # calls so we don't pay the ~5s cold-reload on each engagement
            # (which was the original timeout culprit). Ollama's default is
            # 5m; we pin to 30m because voice sessions can have long quiet
            # stretches and reloading mid-conversation is a bad experience.
            response = requests.post(
                f"{self.config.ollama_base_url}/api/generate",
                json={
                    "model": self.config.model,
                    "prompt": user_prompt,
                    "system": system_prompt,
                    "stream": False,
                    "think": self.config.thinking,
                    "keep_alive": "30m",
                    "options": {
                        "temperature": 0.0,
                        "num_predict": 200,
                    },
                },
                timeout=self.config.timeout_sec,
            )

            if response.status_code != 200:
                # Don't back off on transient HTTP errors — voice is high-turn
                # and a 503 from an overloaded Ollama shouldn't kill the next
                # 30s of intent judging. Retry on the next engagement signal.
                reason = f"HTTP {response.status_code} from Ollama"
                debug_log(f"intent judge: {reason}", "voice")
                self._last_failure_reason = reason
                return None

            result = response.json()
            response_text = result.get("response", "")

            judgment = self._parse_response(response_text)

            if judgment:
                self._last_failure_reason = ""
                direction = "✅ DIRECTED" if judgment.directed else "❌ NOT DIRECTED"
                stop_str = " [STOP]" if judgment.stop else ""
                query_str = f" → \"{judgment.query}\"" if judgment.query else ""
                debug_log(
                    f"🧠 Intent judge: {direction} ({judgment.confidence}){stop_str}{query_str}",
                    "voice"
                )
                debug_log(f"   Reasoning: {judgment.reasoning}", "voice")
            else:
                self._last_failure_reason = f"unparseable response: {response_text[:80]}"
                debug_log(f"🧠 Intent judge: failed to parse: {response_text[:100]}", "voice")

            return judgment

        except requests.Timeout:
            # Do NOT back off on timeout. Voice is high-turn: a single slow
            # call must not lock out intent judging for the next 30s. The
            # engagement-signal gate upstream already prevents calling the
            # judge on ambient speech, so timeouts don't hammer Ollama.
            self._last_failure_reason = f"timeout after {self.config.timeout_sec}s"
            debug_log(f"intent judge: {self._last_failure_reason}", "voice")
            return None
        except requests.RequestException as e:
            self._last_failure_reason = f"request error: {e}"
            debug_log(f"intent judge: {self._last_failure_reason}", "voice")
            self._last_error_time = time.time()
            return None
        except Exception as e:
            self._last_failure_reason = f"error: {e}"
            debug_log(f"intent judge: {self._last_failure_reason}", "voice")
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
