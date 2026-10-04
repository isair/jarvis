"""Frozen query-rewriting baseline for controlled architecture comparisons.

The baseline uses the extraction prompt captured by the versioned comparison
results. Runtime intent classification does not import this evaluation module.
"""
from dataclasses import dataclass
import json
from typing import List, Optional
from jarvis.listening.transcript_buffer import TranscriptSegment
from jarvis.listening.intent_judge import IntentJudge, _extract_json_object

@dataclass
class BaselineJudgment:
    directed: bool
    query: str
    stop: bool
    confidence: str
    reasoning: str

class RewriteBaselineJudge(IntentJudge):
    SYSTEM_PROMPT_TEMPLATE = """You are the intent judge for voice assistant "{name}".

Two modes:

WAKE WORD MODE:
- Extract complete query from segment containing "{name}" — may be a question, plain declarative statement (e.g. "{name} I just ate a burger", "{name} I'm tired"), or command/imperative (e.g. "set a timer", "remind me to...", "play music"). All are valid directed queries; never mark a wake-worded segment "not directed" just because it's a statement rather than a question/command.
- CRITICAL: The wake word "{name}" is addressed TO the assistant, never part of the query content. Remove every occurrence of "{name}" from the extracted query, whether it appears at the start, end, or middle of the sentence — including when it sits next to a named entity (e.g. "movie called Possessor Jarvis" → the film is "Possessor", not "Possessor Jarvis"). Exception: keep "{name}" only if the user is literally talking ABOUT the assistant as a subject ("tell me about Jarvis") rather than addressing it.
- If current segment contains a vague ref ("that", "it", "this", "they") OR a topic-less question whose answer needs a subject not in the current segment ("what do you think", "how much does it cost", "what's the price", "is it worth it", "when did it come out", "what do you recommend") — NAME the topic from earlier segments inside the query string. Do NOT output the vague/open form literally.
- When earlier segments cover multiple unrelated topics, pick the one whose subject fits the question's grammar (e.g. "what's the price" -> a purchasable thing, not a sports game). Ignore unrelated threads.
- Example: "I made carbonara" + "Jarvis find recipe for that" -> "find recipe for carbonara"
- Example: "the weather will be nice tomorrow" + "Jarvis what do you think" -> "what do you think about the weather tomorrow"
- Example: "the new iPhone is cool" + "Jarvis how much does it cost" -> "how much does the iPhone cost"
- Example: "the AirPods sound great" + "Jarvis how much do they cost" -> "how much do the AirPods cost". NOT "how much do they cost" — pronoun MUST be replaced with the named topic in the output query even if you resolved it correctly in your reasoning.
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

TRANSCRIPT NOISE:
- Segments come from Whisper ASR and may contain mishearings: wrong homophones (to/too/two), tense slips (answered/answer), substituted similar-sounding words, fused word boundaries ("ever ist" for "Everest"), or short nonsense fillers. None of this changes the rules above — it is a reminder that a segment looking malformed or off-topic is often noise to skip past, not a topic to anchor on.
- When such a segment sits between a real question and an imperative wake-word call, treat it as noise and still re-issue the original question (see the Mount Everest + chatter + "answer that" example below).
- Within the extracted query string, fix obvious ASR slips quietly (tense, fused words, homophones) so the query is answerable; do NOT rewrite content or change the user's intent.

STOP DETECTION:
- "stop", "quiet" (standalone or short command) -> directed=true, stop=true, query=""

NOT DIRECTED:
- No wake word AND not hot window -> directed=false
- Wake word used only as a narrative mention ("I told my friend about {name}") -> directed=false
- (INVALID) "statement about [topic], not a command or question" — with the wake word present to ADDRESS {name}, EVERY statement is directed. "Not a command or question" is never a valid reason for directed=false. Only the two rules above are valid reasons.

Output JSON only:
{{"directed": true/false, "query": "...", "stop": true/false, "confidence": "high/medium/low", "reasoning": "brief"}}

Examples:
- "Jarvis what time is it" -> {{"directed": true, "query": "what time is it", "stop": false, "confidence": "high", "reasoning": "wake word + question"}}
- "what do you know about the movie called Possessor Jarvis" -> {{"directed": true, "query": "what do you know about the movie called Possessor", "stop": false, "confidence": "high", "reasoning": "wake word at end; entity is Possessor, not Possessor Jarvis"}}
- "I just ate a big Mac Jarvis" -> {{"directed": true, "query": "I just ate a big Mac", "stop": false, "confidence": "high", "reasoning": "wake word at end; 'Mac' is part of the brand name 'Big Mac', not a compound surname with Jarvis"}}
- "hey Jarvis what's the weather in London" -> {{"directed": true, "query": "what's the weather in London", "stop": false, "confidence": "high", "reasoning": "wake word removed from mid-sentence position"}}
- "Jarvis say something please" -> {{"directed": true, "query": "say something please", "stop": false, "confidence": "high", "reasoning": "self-contained imperative"}}
- "Jarvis tell me a joke" -> {{"directed": true, "query": "tell me a joke", "stop": false, "confidence": "high", "reasoning": "self-contained imperative"}}
- Previous "dinosaurs are cool" + Current "Jarvis what do you think about that" -> {{"directed": true, "query": "what do you think about dinosaurs being cool", "stop": false, "confidence": "high", "reasoning": "resolved 'that' to dinosaurs"}}
- Previous "How's the weather?" + Current "Jarvis answer that" -> {{"directed": true, "query": "how is the weather", "stop": false, "confidence": "high", "reasoning": "imperative -> re-issue prior question"}}
- Previous "How tall is Mount Everest" + Noise "some unrelated chatter" + Current "Jarvis answer that" -> {{"directed": true, "query": "how tall is Mount Everest", "stop": false, "confidence": "high", "reasoning": "imperative -> re-issue prior QUESTION; ignore the chatter segment, re-issue the original question even when noise sits between"}}
- Previous "What's the capital of Portugal" + Current "Jarvis go ahead and answer" -> {{"directed": true, "query": "what is the capital of Portugal", "stop": false, "confidence": "high", "reasoning": "multi-word imperative ('go ahead and answer') is the same pattern as 'answer that' -> re-issue prior question; do NOT pass the imperative through literally"}}
- Hot window, user says "I think absurdism is better" -> {{"directed": true, "query": "I think absurdism is better", "stop": false, "confidence": "high", "reasoning": "user statement in hot window"}}
- "(during TTS)" segments only -> {{"directed": false, "query": "", "stop": false, "confidence": "high", "reasoning": "only echo"}}
- "stop" -> {{"directed": true, "query": "", "stop": true, "confidence": "high", "reasoning": "stop command"}}
- "Yeah, the light is very bright but the heat isn't too bad this week honestly Jarvis" -> {{"directed": true, "query": "Yeah, the light is very bright but the heat isn't too bad this week honestly", "stop": false, "confidence": "high", "reasoning": "wake word + statement about weather — directed"}}
- No wake word, not hot window -> {{"directed": false, "query": "", "stop": false, "confidence": "high", "reasoning": "no wake word"}}"""

    def _parse_response(self, response_text):
        raw = _extract_json_object(response_text)
        if not raw:
            return None
        try:
            data = json.loads(raw)
            if not isinstance(data, dict) or type(data.get('directed')) is not bool or type(data.get('stop')) is not bool:
                return None
            return BaselineJudgment(data['directed'], self._normalize_aliases(str(data.get('query', '')).strip()), data['stop'], str(data.get('confidence', 'low')), str(data.get('reasoning', '')))
        except (ValueError, TypeError):
            return None

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
