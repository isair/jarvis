"""Background single-flight learning worker — fail-open, no TTS, no voice delay."""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Callable, List, Optional

from ...debug import debug_log
from .extract import extract_lessons_from_turns, is_trivial_conversation
from .store import LearningStore


class LearningWorker:
    """At most one learning job at a time; errors never stop the listener."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._busy = False
        self._cancel_generation = 0
        self._pending: Optional[dict] = None
        self._thread: Optional[threading.Thread] = None

    @property
    def is_busy(self) -> bool:
        with self._lock:
            return self._busy

    def defer_for_new_command(self) -> None:
        """Bump generation so an in-flight/queued job is abandoned."""
        with self._lock:
            self._cancel_generation += 1
            self._pending = None
            debug_log("learning job deferred — new command", "learning")

    def schedule(
        self,
        *,
        db,
        cfg,
        dialogue_memory,
        conversation_id: Optional[str] = None,
        force: bool = False,
    ) -> bool:
        """Queue a background learn for the current dialogue snapshot.

        Returns False if skipped (disabled, busy, empty, already processed).
        Never blocks the caller for the LLM / DB work.
        """
        if not bool(getattr(cfg, "conversation_learning_enabled", False)):
            return False
        if dialogue_memory is None:
            return False
        if getattr(dialogue_memory, "_learning_opt_out", False) and not force:
            debug_log("learning skipped: conversation opt-out", "learning")
            return False

        try:
            turns = dialogue_memory.get_recent_messages()
        except Exception:
            turns = []
        if not turns:
            return False
        if is_trivial_conversation(turns):
            debug_log("learning skipped: trivial conversation", "learning")
            return False

        cid = conversation_id or getattr(dialogue_memory, "conversation_id", None)
        if not cid:
            cid = getattr(dialogue_memory, "_ensure_conversation_id", lambda: None)()
        if not cid:
            cid = str(uuid.uuid4())
            try:
                dialogue_memory.conversation_id = cid
            except Exception:
                pass

        store = LearningStore(db)
        if store.was_conversation_processed(cid) and not force:
            debug_log(f"learning skipped: already processed {cid[:8]}…", "learning")
            return False

        job = {
            "gen": None,
            "db": db,
            "cfg": cfg,
            "turns": list(turns),
            "conversation_id": cid,
            "force": force,
        }

        with self._lock:
            if self._busy:
                # Defer — do not run two jobs; keep latest snapshot as pending.
                self._pending = job
                debug_log("learning busy — queued latest snapshot", "learning")
                return False
            self._busy = True
            job["gen"] = self._cancel_generation
            self._thread = threading.Thread(
                target=self._run_job,
                args=(job,),
                name="cora-learning",
                daemon=True,
            )
            self._thread.start()
        return True

    def _run_job(self, job: dict) -> None:
        started = time.time()
        lesson_count = 0
        try:
            with self._lock:
                if job["gen"] != self._cancel_generation:
                    debug_log("learning abandoned (superseded)", "learning")
                    return
            cfg = job["cfg"]
            store = LearningStore(job["db"])
            cid = job["conversation_id"]
            if store.was_conversation_processed(cid) and not job.get("force"):
                return

            timeout = float(getattr(cfg, "conversation_learning_timeout_sec", 12.0))
            mode = str(getattr(cfg, "conversation_learning_mode", "safe_auto"))
            allow_llm = mode == "safe_auto"

            lessons = extract_lessons_from_turns(
                job["turns"],
                cid,
                allow_llm=allow_llm,
                ollama_base_url=getattr(cfg, "ollama_base_url", ""),
                ollama_chat_model=getattr(cfg, "ollama_chat_model", ""),
                timeout_sec=timeout,
                promote_inferences=False,
            )

            with self._lock:
                if job["gen"] != self._cancel_generation:
                    return

            for lesson in lessons:
                # Model inferences already filtered; never persist web/assistant alone.
                if lesson.provenance in ("web_result", "assistant_alone"):
                    continue
                store.upsert_lesson(lesson)
                lesson_count += 1

            store.mark_conversation_processed(cid, lesson_count)
            # Log type counts only — never full values.
            print(
                f"  🧠 Learning: saved {lesson_count} lesson(s) "
                f"for conversation {cid[:8]}…",
                flush=True,
            )
            debug_log(
                f"learning done count={lesson_count} elapsed={time.time()-started:.2f}s",
                "learning",
            )
        except Exception as e:
            # Fail-open: never raise into the listener.
            debug_log(f"learning worker error: {type(e).__name__}: {e}", "learning")
            try:
                print("  🧠 Learning: error (ignored, listener continues)", flush=True)
            except Exception:
                pass
        finally:
            next_job = None
            with self._lock:
                self._busy = False
                if self._pending and self._pending.get("gen") != self._cancel_generation:
                    # pending was set before cancel — still try if gen matches current
                    pass
                if self._pending is not None:
                    next_job = self._pending
                    self._pending = None
                    self._busy = True
            if next_job is not None:
                next_job["gen"] = self._cancel_generation
                t = threading.Thread(
                    target=self._run_job,
                    args=(next_job,),
                    name="cora-learning",
                    daemon=True,
                )
                with self._lock:
                    self._thread = t
                t.start()

    def flush_sync(self, *, db, cfg, dialogue_memory, timeout_sec: float = 15.0) -> None:
        """Shutdown path: run learning synchronously with a hard timeout."""
        if not bool(getattr(cfg, "conversation_learning_enabled", False)):
            return
        done = threading.Event()

        def _go():
            try:
                self.schedule(db=db, cfg=cfg, dialogue_memory=dialogue_memory, force=True)
                # Wait briefly for busy flag to clear
                deadline = time.time() + timeout_sec
                while self.is_busy and time.time() < deadline:
                    time.sleep(0.05)
            finally:
                done.set()

        t = threading.Thread(target=_go, daemon=True)
        t.start()
        done.wait(timeout=timeout_sec + 1.0)


_WORKER: Optional[LearningWorker] = None
_WORKER_LOCK = threading.Lock()


def get_learning_worker() -> LearningWorker:
    global _WORKER
    with _WORKER_LOCK:
        if _WORKER is None:
            _WORKER = LearningWorker()
        return _WORKER
