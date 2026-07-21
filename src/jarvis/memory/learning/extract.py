"""Hybrid lesson extraction: deterministic RO/EN patterns + optional LLM proposals."""

from __future__ import annotations

import json
import re
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ...debug import debug_log
from .safety import scrub_lesson_text, should_reject_lesson_value
from .types import (
    CONFIDENCE_CAPS,
    LESSON_TYPES,
    NAMESPACE_FOR_TYPE,
    Lesson,
    LessonCandidate,
    LessonType,
)

# Trivial local-answer conversations — no lessons.
_TRIVIAL_RE = re.compile(
    r"(?i)^\s*("
    r"c[aâ]t[e]?\s+(e|este)\s+(ceasul|ora)"
    r"|ce\s+ora\s+(e|este)"
    r"|ce\s+zi\s+(e|este)"
    r"|data\s+de\s+ast[aă]zi"
    r"|c[aâ]t\s+fac\s+.+"
    r"|what\s+time\s+is\s+it"
    r"|what\s+day\s+is\s+it"
    r")\s*[?.!]?\s*$"
)

_DET_PATTERNS: List[Tuple[re.Pattern[str], str, str, float]] = [
    # (regex with named groups subject/value or value only, type, provenance, confidence)
    (
        re.compile(
            r"(?i)\b(?:memoreaz[aă]|ține\s+minte|tine\s+minte|remember(?:\s+that)?)\s+(?:că\s+|ca\s+|that\s+)?(?P<value>.+)$"
        ),
        LessonType.USER_FACT.value,
        "user_direct",
        0.95,
    ),
    (
        re.compile(
            r"(?i)\b(?:prefer|îmi\s+place|imi\s+place)\s+(?P<value>.+)$"
        ),
        LessonType.USER_PREFERENCE.value,
        "user_direct",
        0.95,
    ),
    (
        re.compile(
            r"(?i)\b(?:răspunde-?mi|raspunde-?mi|always\s+reply|răspunde\s+mereu)\s+(?P<value>.+)$"
        ),
        LessonType.USER_PREFERENCE.value,
        "user_direct",
        0.95,
    ),
    (
        re.compile(
            r"(?i)\b(?:ai\s+greșit|ai\s+gresit|wrong[,:]?\s*|corect\s+(?:este|e)|correct\s+(?:is|one\s+is))\s*(?P<value>.+)$"
        ),
        LessonType.USER_CORRECTION.value,
        "user_explicit_correction",
        1.0,
    ),
    (
        re.compile(
            r"(?i)\b(?:nu\s+mai|stop|don't\s+anymore)\s+(?P<value>.+)$"
        ),
        LessonType.USER_PREFERENCE.value,
        "user_direct",
        0.95,
    ),
    (
        re.compile(
            r"(?i)\b(?:urm[aă]torul\s+pas\s+(?:este|e)|next\s+step\s+is)\s+(?P<value>.+)$"
        ),
        LessonType.PENDING_STEP.value,
        "user_direct",
        0.95,
    ),
]


def is_trivial_conversation(turns: Sequence[Dict[str, str]]) -> bool:
    """True when every user turn is date/clock/arithmetic — skip learning."""
    users = [t.get("content", "") for t in turns if t.get("role") == "user"]
    if not users:
        return True
    return all(_TRIVIAL_RE.match((u or "").strip()) is not None for u in users)


def _subject_from_value(value: str) -> str:
    words = re.findall(r"[A-Za-zăâîșțĂÂÎȘȚ0-9]+", value.lower())
    return " ".join(words[:6]) or "general"


def _quote_anchored(user_text: str, quote: str) -> bool:
    if not quote or not user_text:
        return False
    # Fold whitespace; require quote substring in user text.
    u = " ".join(user_text.split()).lower()
    q = " ".join(quote.split()).lower()
    if len(q) < 4:
        return False
    return q in u


def extract_deterministic(
    user_text: str,
    turn_id: str,
) -> List[LessonCandidate]:
    text = (user_text or "").strip()
    if not text:
        return []
    out: List[LessonCandidate] = []
    for pat, ltype, prov, conf in _DET_PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        value = scrub_lesson_text(m.group("value").strip(" .,\"'"))
        reject, _ = should_reject_lesson_value(value, ltype)
        if reject:
            continue
        quote = text[m.start(): m.end()][:240]
        out.append(
            LessonCandidate(
                lesson_type=ltype,
                subject_key=_subject_from_value(value),
                value=value,
                source_quote=quote,
                confidence=conf,
                provenance=prov,
                turn_id=turn_id,
            )
        )
    return out


_LLM_SYSTEM = """You propose structured memory lessons from a USER utterance only.
Return ONLY a JSON array. Each object keys: lesson_type, subject_key, value, source_quote, confidence.
lesson_type must be one of: user_preference, user_correction, user_fact, project_context, pending_step, asr_correction, recurring_failure.
source_quote MUST be an exact substring of the user utterance.
confidence for inferences must be <= 0.40.
Do not invent facts from assistant replies. Do not include secrets.
If nothing useful, return []."""


def propose_with_llm(
    user_text: str,
    turn_id: str,
    *,
    ollama_base_url: str,
    ollama_chat_model: str,
    timeout_sec: float,
) -> List[LessonCandidate]:
    """Model may only propose; all candidates are validated by the caller."""
    if not user_text or not ollama_chat_model:
        return []
    try:
        from ...llm import call_llm_direct

        raw = call_llm_direct(
            ollama_base_url,
            ollama_chat_model,
            _LLM_SYSTEM,
            f"User utterance:\n{user_text}",
            timeout_sec=timeout_sec,
            temperature=0.0,
        )
    except Exception as e:
        debug_log(f"learning LLM propose failed: {type(e).__name__}", "learning")
        return []
    if not raw:
        return []
    return _parse_llm_candidates(raw, user_text, turn_id)


def _parse_llm_candidates(raw: str, user_text: str, turn_id: str) -> List[LessonCandidate]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if "\n" in text:
            text = text.split("\n", 1)[1]
    try:
        data = json.loads(text)
    except Exception:
        # Try to find array
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except Exception:
            return []
    if not isinstance(data, list):
        return []
    out: List[LessonCandidate] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        ltype = str(item.get("lesson_type") or "").strip()
        if ltype not in LESSON_TYPES or ltype == LessonType.IMPROVEMENT_CANDIDATE.value:
            continue  # improvement_candidate only via recurrence path
        value = scrub_lesson_text(str(item.get("value") or ""))
        quote = str(item.get("source_quote") or "").strip()
        subject = scrub_lesson_text(str(item.get("subject_key") or _subject_from_value(value)))
        try:
            conf = float(item.get("confidence", 0.4))
        except Exception:
            conf = 0.4
        conf = min(conf, CONFIDENCE_CAPS["model_inference"])
        if not _quote_anchored(user_text, quote):
            continue  # invented quote
        reject, _ = should_reject_lesson_value(value, ltype)
        if reject:
            continue
        out.append(
            LessonCandidate(
                lesson_type=ltype,
                subject_key=subject or "general",
                value=value,
                source_quote=quote[:240],
                confidence=conf,
                provenance="model_inference",
                turn_id=turn_id,
            )
        )
    return out


def validate_candidate(c: LessonCandidate, user_texts: Sequence[str]) -> Optional[LessonCandidate]:
    if c.lesson_type not in LESSON_TYPES:
        return None
    if c.provenance == "model_inference" and c.confidence > CONFIDENCE_CAPS["model_inference"]:
        return None
    # Assistant-only provenance forbidden
    if c.provenance == "assistant_alone":
        return None
    blob = "\n".join(user_texts)
    if not _quote_anchored(blob, c.source_quote):
        return None
    reject, _ = should_reject_lesson_value(c.value, c.lesson_type)
    if reject:
        return None
    # Model inferences are never auto-promoted to durable user_fact preference
    # without user_direct provenance — still allowed to store at low confidence
    # but not as user_fact from web (handled separately).
    if c.provenance == "web_result":
        return None
    return c


def candidate_to_lesson(
    c: LessonCandidate,
    conversation_id: str,
) -> Lesson:
    from .types import NAMESPACE_FOR_TYPE, LessonType

    try:
        ns = NAMESPACE_FOR_TYPE[LessonType(c.lesson_type)].value
    except Exception:
        ns = "profile"
    return Lesson(
        id=str(uuid.uuid4()),
        lesson_type=c.lesson_type,
        subject_key=c.subject_key,
        value=c.value,
        source_quote=c.source_quote,
        conversation_id=conversation_id,
        turn_id=c.turn_id or "0",
        created_at="",
        updated_at="",
        confidence=c.confidence,
        sensitivity=c.sensitivity,
        namespace=ns,
        provenance=c.provenance,
    )


def extract_lessons_from_turns(
    turns: Sequence[Dict[str, str]],
    conversation_id: str,
    *,
    allow_llm: bool = False,
    ollama_base_url: str = "",
    ollama_chat_model: str = "",
    timeout_sec: float = 8.0,
    promote_inferences: bool = False,
) -> List[Lesson]:
    """Extract validated lessons. Assistant turns never create lessons alone."""
    if is_trivial_conversation(turns):
        return []

    user_turns = [
        (i, t.get("content", ""))
        for i, t in enumerate(turns)
        if t.get("role") == "user" and (t.get("content") or "").strip()
    ]
    if not user_turns:
        return []

    user_texts = [u for _, u in user_turns]
    candidates: List[LessonCandidate] = []
    for idx, text in user_turns:
        turn_id = str(idx)
        candidates.extend(extract_deterministic(text, turn_id))
        if allow_llm:
            proposed = propose_with_llm(
                text,
                turn_id,
                ollama_base_url=ollama_base_url,
                ollama_chat_model=ollama_chat_model,
                timeout_sec=timeout_sec,
            )
            for p in proposed:
                if not promote_inferences and p.provenance == "model_inference":
                    # Keep for logging only — do not auto-promote.
                    # Spec: max 0.40 and NOT auto-promoted → skip persist.
                    continue
                candidates.append(p)

    lessons: List[Lesson] = []
    seen = set()
    for c in candidates:
        v = validate_candidate(c, user_texts)
        if not v:
            continue
        key = (v.lesson_type, v.subject_key.lower(), v.value.lower())
        if key in seen:
            continue
        seen.add(key)
        lessons.append(candidate_to_lesson(v, conversation_id))
    return lessons
