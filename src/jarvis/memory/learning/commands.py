"""Voice / text commands for explicit learning control."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from .store import LearningStore
from .types import Lesson, LessonType, Namespace


@dataclass
class CommandResult:
    handled: bool
    reply: Optional[str] = None
    defer_learning: bool = False  # e.g. opt-out


_WHAT_LEARNED = re.compile(
    r"(?i)\b(ce\s+ai\s+[iî]nv[aă][țţ]at\s+despre\s+mine|what\s+have\s+you\s+learned\s+about\s+me)\b"
)
_MEMORIZE = re.compile(
    r"(?i)\b(?:memoreaz[aă]|ține\s+minte|tine\s+minte|remember)\s+(?:că\s+|ca\s+|that\s+)?(?P<value>.+)$"
)
_DONT_MEMORIZE = re.compile(
    r"(?i)\b(nu\s+memora\s+(conversa[țţ]ia\s+asta|asta)|don'?t\s+remember\s+(this|that)\s+conversation)\b"
)
_FORGET = re.compile(
    r"(?i)\b(?:uit[aă]|forget)\s+(?:că\s+|ca\s+|that\s+)?(?P<target>.+)$"
)
_CONFIRM_YES = re.compile(r"(?i)^\s*(da|yes|confirm|confirma|confirmă)\s*[!.]?\s*$")
_CONFIRM_NO = re.compile(r"(?i)^\s*(nu|no|anuleaz[aă]|cancel)\s*[!.]?\s*$")


def try_learning_command(
    text: str,
    *,
    store: Optional[LearningStore],
    dialogue_memory,
    conversation_id: str,
) -> CommandResult:
    """Handle learning voice commands. Returns handled=False to fall through."""
    raw = (text or "").strip()
    if not raw:
        return CommandResult(handled=False)

    # Pending forget confirmation
    pending = getattr(dialogue_memory, "_pending_forget_subject", None)
    if pending:
        if _CONFIRM_YES.match(raw):
            ids = store.soft_delete_by_subject(pending) if store else []
            dialogue_memory._pending_forget_subject = None
            if not ids:
                return CommandResult(
                    handled=True,
                    reply="Nu am găsit o lecție activă cu acel subiect.",
                )
            return CommandResult(
                handled=True,
                reply=f"Am uitat logic {len(ids)} lecții despre „{pending}”.",
            )
        if _CONFIRM_NO.match(raw):
            dialogue_memory._pending_forget_subject = None
            return CommandResult(handled=True, reply="Am anulat uitarea.")
        # Ambiguous — ask again, don't delete
        return CommandResult(
            handled=True,
            reply=f"Confirmă uitarea pentru „{pending}”: spune da sau nu.",
        )

    if _DONT_MEMORIZE.search(raw):
        setattr(dialogue_memory, "_learning_opt_out", True)
        return CommandResult(
            handled=True,
            reply="În regulă — nu voi memora această conversație.",
            defer_learning=True,
        )

    if _WHAT_LEARNED.search(raw):
        if store is None:
            return CommandResult(handled=True, reply="Nu am lecții salvate încă.")
        lessons = store.list_active(
            namespaces=[Namespace.PROFILE.value, Namespace.CORRECTIONS.value,
                        Namespace.PROJECTS.value],
            limit=12,
        )
        if not lessons:
            return CommandResult(
                handled=True,
                reply="Nu am învățat încă preferințe sau fapte despre tine.",
            )
        lines = []
        for L in lessons[:8]:
            lines.append(f"- {L.lesson_type}: {L.subject_key} ({L.confidence:.0%})")
        return CommandResult(
            handled=True,
            reply="Iată ce am învățat (fără detalii sensibile):\n" + "\n".join(lines),
        )

    m_forget = _FORGET.search(raw)
    if m_forget:
        target = m_forget.group("target").strip(" .,\"'")
        # Ambiguous if too short or multiple matches with different subjects
        if len(target) < 3:
            return CommandResult(
                handled=True,
                reply="Nu pot uita ceva atât de vag. Spune subiectul clar.",
            )
        if store is None:
            return CommandResult(handled=True, reply="Nu am nimic de uitat.")
        matches = [
            L for L in store.list_active(limit=50)
            if target.lower() in L.subject_key or target.lower() in L.value.lower()
        ]
        subjects = sorted({L.subject_key for L in matches})
        if not subjects:
            return CommandResult(
                handled=True,
                reply=f"Nu am găsit lecții despre „{target}”.",
            )
        if len(subjects) > 1:
            return CommandResult(
                handled=True,
                reply="Am găsit mai multe subiecte. Precizează pe care să-l uit: "
                + ", ".join(subjects[:5]),
            )
        dialogue_memory._pending_forget_subject = subjects[0]
        return CommandResult(
            handled=True,
            reply=f"Să uit lecțiile despre „{subjects[0]}”? Spune da pentru confirmare.",
        )

    m_mem = _MEMORIZE.search(raw)
    if m_mem and store is not None:
        from .extract import _subject_from_value
        from .safety import should_reject_lesson_value, scrub_lesson_text
        import uuid

        value = scrub_lesson_text(m_mem.group("value").strip(" .,\"'"))
        reject, reason = should_reject_lesson_value(value, LessonType.USER_FACT.value)
        if reject:
            return CommandResult(
                handled=True,
                reply="Nu pot memora asta — conține informații pe care nu le păstrez.",
            )
        lesson = Lesson(
            id=str(uuid.uuid4()),
            lesson_type=LessonType.USER_FACT.value,
            subject_key=_subject_from_value(value),
            value=value,
            source_quote=raw[:240],
            conversation_id=conversation_id or "interactive",
            turn_id="cmd",
            created_at="",
            updated_at="",
            confidence=0.95,
            namespace=Namespace.PROFILE.value,
            provenance="user_direct",
        )
        store.upsert_lesson(lesson)
        print("  🧠 Lecție salvată: tip=user_fact", flush=True)
        return CommandResult(
            handled=True,
            reply="Am memorat asta.",
        )

    return CommandResult(handled=False)
