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
# Phase 4 · Section E — explicit correction command (create new version, old
# superseded). Distinct from the async extractor's correction cue in extract.py.
# Deliberately NARROW: the ubiquitous Romanian filler "de fapt" ("actually") is
# NOT a trigger — it over-matched casual speech and, since corrections are
# high-trust, would have created confirmed memory from filler. Requires an
# explicit corrective verb.
_CORRECT = re.compile(
    r"(?i)\b(?:corecteaz[aă]|corect\s+este)\b[:,]?\s*(?P<value>.+)$"
)


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


def try_state_memory_command(
    text: str,
    *,
    state_store,
    dialogue_memory,
    conversation_id: str,
    owner: str = "user",
) -> CommandResult:
    """Phase 4 · Section E — explicit memory commands over the STATE store.

    The legacy ``_MEMORIZE`` branch above committed a lesson at confidence 0.95
    with NO confirmation — the exact path that persisted the ASR-corrupted
    "lessons" observed live. This safe variant instead:

      * memorize → ``candidate`` → ``pending_confirmation`` and READS THE VALUE
        BACK, requiring an explicit "da" before it becomes ``confirmed``. The
        readback is the ASR safety net — a garbled transcription is caught by
        ear and cancelled, never confirmed.
      * correct → supersede the old value + confirm the new (explicit, trusted).
      * forget → soft-delete confirmed items by subject (kept in audit).
      * "ce ai învățat" → lists ONLY confirmed items.

    Gated by the caller on ``state_memory_enabled``; returns handled=False to
    fall through. ``state_store`` is a jarvis.memory.state_store.StateStore.
    """
    raw = (text or "").strip()
    if not raw or state_store is None:
        return CommandResult(handled=False)

    # 1) A pending memorize confirmation takes priority over everything else.
    pend = getattr(dialogue_memory, "_pending_state_memorize", None)
    if pend:
        item_id, val, *_rest = pend
        kind = _rest[0] if _rest else "memorize"
        _noun = "corecția" if kind == "correct" else "memorarea"
        if _CONFIRM_YES.match(raw):
            ok = False
            try:
                ok = state_store.confirm(item_id, confirmed_by=owner)
            except Exception:
                ok = False
            setattr(dialogue_memory, "_pending_state_memorize", None)
            if not ok:
                return CommandResult(handled=True, reply="Nu am putut confirma.")
            done = "Am reținut corecția" if kind == "correct" else "Am memorat"
            return CommandResult(handled=True, reply=f"{done}: „{val}”.")
        if _CONFIRM_NO.match(raw):
            try:
                state_store.forget(item_id, reason="user_cancelled")
            except Exception:
                pass
            setattr(dialogue_memory, "_pending_state_memorize", None)
            return CommandResult(handled=True, reply="Am anulat — nu am reținut nimic.")
        return CommandResult(
            handled=True, reply=f"Confirmă {_noun} „{val}”: spune da sau nu.")

    if _DONT_MEMORIZE.search(raw):
        setattr(dialogue_memory, "_learning_opt_out", True)
        return CommandResult(
            handled=True,
            reply="În regulă — nu memorez conversația asta.",
            defer_learning=True,
        )

    if _WHAT_LEARNED.search(raw):
        try:
            items = state_store.retrieve_confirmed(limit=12)
        except Exception:
            items = []
        if not items:
            return CommandResult(
                handled=True, reply="Nu am încă informații confirmate despre tine.")
        lines = [f"- {(it.get('subject_key') or it.get('value') or '').strip()}"
                 for it in items[:8]]
        return CommandResult(
            handled=True, reply="Iată ce am confirmat:\n" + "\n".join(lines))

    # 2) Correction — explicit and high-trust: supersede + confirm.
    m_corr = _CORRECT.search(raw)
    if m_corr:
        from .safety import should_reject_lesson_value, scrub_lesson_text
        from .extract import _subject_from_value
        value = scrub_lesson_text(m_corr.group("value").strip(" .,\"'"))
        if not value:
            return CommandResult(handled=False)
        reject, _ = should_reject_lesson_value(value, "user_correction")
        if reject:
            return CommandResult(
                handled=True,
                reply="Nu pot păstra asta — conține informații pe care nu le rețin.")
        subj = _subject_from_value(value)
        try:
            item_id = state_store.add_candidate(
                "correction", subj, value,
                source=raw[:240], conversation_id=conversation_id,
                provenance="user_explicit_correction", confidence=1.0, actor=owner,
            )
            if item_id is None:
                return CommandResult(handled=True, reply="Nu pot păstra asta.")
            # Read back + require explicit "da" — same ASR safety net as memorize.
            # A mis-transcribed correction is caught by ear, never auto-confirmed.
            state_store.promote_to_pending(item_id)
        except Exception:
            return CommandResult(handled=True, reply="Nu am putut pregăti corecția.")
        setattr(dialogue_memory, "_pending_state_memorize", (item_id, value, "correct"))
        return CommandResult(
            handled=True,
            reply=f"Să rețin corecția „{value}”? Spune da pentru confirmare sau nu pentru anulare.")

    # 3) Forget — soft-delete confirmed items matching a subject.
    m_forget = _FORGET.search(raw)
    if m_forget:
        target = m_forget.group("target").strip(" .,\"'")
        if len(target) < 3:
            return CommandResult(handled=True, reply="Spune mai clar ce să uit.")
        try:
            confirmed = state_store.retrieve_confirmed(limit=100)
        except Exception:
            confirmed = []
        t = target.lower()
        matches = [it for it in confirmed
                   if t in ((it.get("subject_key") or "").lower()
                            + " " + (it.get("value") or "").lower())]
        if not matches:
            return CommandResult(
                handled=True, reply=f"Nu am găsit ceva confirmat despre „{target}”.")
        n = 0
        for it in matches:
            try:
                if state_store.forget(it["id"], reason="user_forget"):
                    n += 1
            except Exception:
                pass
        return CommandResult(handled=True, reply=f"Am uitat {n} lucru(ri) despre „{target}”.")

    # 4) Memorize — candidate → pending, read the value back, await confirm.
    m_mem = _MEMORIZE.search(raw)
    if m_mem:
        from .safety import should_reject_lesson_value, scrub_lesson_text
        from .extract import _subject_from_value
        value = scrub_lesson_text(m_mem.group("value").strip(" .,\"'"))
        if not value:
            return CommandResult(handled=False)
        reject, _ = should_reject_lesson_value(value, "user_fact")
        if reject:
            return CommandResult(
                handled=True,
                reply="Nu pot memora asta — conține informații pe care nu le păstrez.")
        subj = _subject_from_value(value)
        try:
            item_id = state_store.add_candidate(
                "user_fact", subj, value,
                source=raw[:240], conversation_id=conversation_id,
                provenance="user_direct", confidence=0.95, actor=owner,
            )
            if item_id is None:
                return CommandResult(handled=True, reply="Nu pot memora asta.")
            state_store.promote_to_pending(item_id)
        except Exception:
            return CommandResult(handled=True, reply="Nu am putut pregăti memorarea.")
        setattr(dialogue_memory, "_pending_state_memorize", (item_id, value, "memorize"))
        return CommandResult(
            handled=True,
            reply=f"Să memorez „{value}”? Spune da pentru confirmare sau nu pentru anulare.",
        )

    return CommandResult(handled=False)
