"""Module E — StateStore tests (Cora Brain Foundation, Phase 4).

Confirmation-gated state/provenance memory: candidate lifecycle, ASR safety,
recurrence-based auto-promotion guards, contradiction / secret filtering,
supersede-on-correct, soft-forget, per-transition audit trail, and durable
persistence across a close+reopen. All on an isolated temp Database.
"""

from __future__ import annotations

import pytest

from jarvis.memory.db import Database
from jarvis.memory.state_store import (
    STATE_MEMORY_SCHEMA_VERSION,
    StateStore,
    asr_quality_ok,
    STATUS_CANDIDATE,
    STATUS_PENDING,
    STATUS_CONFIRMED,
    STATUS_QUARANTINED,
    STATUS_SUPERSEDED,
    STATUS_FORGOTTEN,
    TYPE_USER_PREFERENCE,
    TYPE_USER_FACT,
    TYPE_OWNER_RULE,
    TYPE_WEB_FACT,
)


# --- fixtures ---------------------------------------------------------------

@pytest.fixture()
def db(tmp_path):
    database = Database(str(tmp_path / "jarvis.db"))
    yield database
    database.close()


@pytest.fixture()
def store(db):
    # require_confirmation=False so the auto-promotion path is exercisable.
    return StateStore(db, require_confirmation=False)


@pytest.fixture()
def strict_store(db):
    # Default policy: even eligible preferences must be explicitly confirmed.
    return StateStore(db, require_confirmation=True)


# --- schema -----------------------------------------------------------------

def test_schema_created_and_versioned(db):
    s = StateStore(db)
    s.list_by_state(STATUS_CANDIDATE)  # triggers lazy schema
    row = db.conn.execute(
        "SELECT version FROM schema_migrations WHERE component='memory_states'"
    ).fetchone()
    assert row is not None and int(row[0]) == STATE_MEMORY_SCHEMA_VERSION


def test_does_not_touch_learning_lessons(db):
    """Additive-only: creating the store leaves the pre-existing learning table
    intact and never creates a second connection/DB."""
    db.conn.execute(
        """INSERT INTO learning_lessons
             (id, lesson_type, subject_key, value, source_quote, conversation_id,
              turn_id, created_at, updated_at, confidence, namespace)
           VALUES ('L1','user_fact','k','v','q','c','t','now','now',0.9,'profile')"""
    )
    db.conn.commit()
    StateStore(db).list_by_state(STATUS_CANDIDATE)
    row = db.conn.execute("SELECT value FROM learning_lessons WHERE id='L1'").fetchone()
    assert row is not None and row["value"] == "v"


# --- lifecycle --------------------------------------------------------------

def test_candidate_to_pending_to_confirmed(strict_store):
    iid = strict_store.add_candidate(
        TYPE_USER_FACT, "owner name", "Razvan", provenance="user_direct",
    )
    assert iid is not None
    assert strict_store.get(iid)["status"] == STATUS_CANDIDATE
    assert strict_store.promote_to_pending(iid) is True
    assert strict_store.get(iid)["status"] == STATUS_PENDING
    assert strict_store.confirm(iid, confirmed_by="owner") is True
    item = strict_store.get(iid)
    assert item["status"] == STATUS_CONFIRMED
    assert item["confirmed_by"] == "owner"


def test_list_by_state(strict_store):
    a = strict_store.add_candidate(TYPE_USER_FACT, "a", "1")
    b = strict_store.add_candidate(TYPE_USER_FACT, "b", "2")
    strict_store.confirm(b, confirmed_by="owner")
    cand_ids = {i["id"] for i in strict_store.list_by_state(STATUS_CANDIDATE)}
    conf_ids = {i["id"] for i in strict_store.list_by_state(STATUS_CONFIRMED)}
    assert a in cand_ids and b not in cand_ids
    assert b in conf_ids and a not in conf_ids


# --- ASR safety -------------------------------------------------------------

def test_asr_quality_ok_helper():
    assert asr_quality_ok(0.92, {"avg_logprob": -0.2}) is True
    assert asr_quality_ok(0.3, {"avg_logprob": -0.2}) is False       # low conf
    assert asr_quality_ok(0.95, None) is False                       # no metrics
    assert asr_quality_ok(None, {"avg_logprob": -0.2}) is False      # no conf
    assert asr_quality_ok(0.95, {}) is False                         # empty metrics
    assert asr_quality_ok(0.95, {"no_speech_prob": 0.9}) is False    # silence
    assert asr_quality_ok(0.95, {"compression_ratio": 3.0}) is False # gibberish
    assert asr_quality_ok(0.95, {"avg_logprob": -2.0}) is False      # low logprob


def test_asr_low_confidence_not_auto_confirmed(store):
    # Even with require_confirmation=False, low-confidence ASR must NOT confirm;
    # it is quarantined instead.
    iid = store.add_from_asr(
        "brontosaurus", asr_confidence=0.25, asr_metrics={"avg_logprob": -0.3},
        item_type=TYPE_USER_FACT, subject_key="favourite dino",
    )
    assert iid is not None
    assert store.get(iid)["status"] == STATUS_QUARANTINED
    assert store.retrieve_confirmed() == []


def test_asr_missing_metrics_not_auto_confirmed(store):
    iid = store.add_from_asr(
        "the sky is green", asr_confidence=0.95, asr_metrics=None,
        item_type=TYPE_USER_FACT, subject_key="sky colour",
    )
    assert iid is not None
    assert store.get(iid)["status"] == STATUS_QUARANTINED
    assert store.retrieve_confirmed() == []


def test_asr_low_quality_does_not_unconfirm_existing(strict_store):
    """A noisy ASR echo of a value that already has a CONFIRMED item must not
    quarantine (un-confirm) that human-vetted memory."""
    iid = strict_store.add_from_asr(
        "espresso", asr_confidence=0.95, asr_metrics={"avg_logprob": -0.1},
        item_type=TYPE_USER_FACT, subject_key="coffee",
    )
    assert strict_store.confirm(iid, confirmed_by="owner") is True
    # Later, a low-quality re-hearing of the same word dedupes into the item.
    same = strict_store.add_from_asr(
        "espresso", asr_confidence=0.2, asr_metrics=None,
        item_type=TYPE_USER_FACT, subject_key="coffee",
    )
    assert same == iid
    assert strict_store.get(iid)["status"] == STATUS_CONFIRMED  # untouched


def test_asr_good_quality_stays_candidate(store):
    # Good ASR quality never auto-confirms either — it becomes a candidate.
    iid = store.add_from_asr(
        "espresso", asr_confidence=0.94, asr_metrics={"avg_logprob": -0.15},
        item_type=TYPE_USER_FACT, subject_key="coffee",
    )
    assert store.get(iid)["status"] == STATUS_CANDIDATE
    assert store.retrieve_confirmed() == []


# --- auto-promotion ---------------------------------------------------------

def test_three_independent_conversations_promote(store):
    """≥3 DISTINCT conversations promote; repeating the same conversation does not."""
    def add(conv):
        return store.add_candidate(
            TYPE_USER_PREFERENCE, "milk", "oat milk",
            provenance="user_direct", confidence=0.95, conversation_id=conv,
        )

    iid = add("c1")
    assert store.get(iid)["status"] == STATUS_CANDIDATE
    # Same conversation again → still one independent source, no promotion.
    assert add("c1") == iid
    assert store.get(iid)["status"] == STATUS_CANDIDATE
    assert store.get(iid)["recurrence_count"] == 1
    # Second distinct conversation → 2, still short of 3.
    add("c2")
    assert store.get(iid)["status"] == STATUS_CANDIDATE
    assert store.get(iid)["recurrence_count"] == 2
    # Third distinct conversation → promoted.
    add("c3")
    item = store.get(iid)
    assert item["status"] == STATUS_CONFIRMED
    assert item["recurrence_count"] == 3
    assert item["confirmed_by"] == "auto"


def test_require_confirmation_blocks_autopromote(strict_store):
    """With the default policy, even an eligible preference stays a candidate."""
    for conv in ("c1", "c2", "c3"):
        iid = strict_store.add_candidate(
            TYPE_USER_PREFERENCE, "milk", "oat milk",
            provenance="user_direct", confidence=0.95, conversation_id=conv,
        )
    assert strict_store.get(iid)["status"] == STATUS_CANDIDATE


def test_low_confidence_pref_not_promoted(store):
    for conv in ("c1", "c2", "c3"):
        iid = store.add_candidate(
            TYPE_USER_PREFERENCE, "milk", "oat milk",
            provenance="model_inference", conversation_id=conv,  # inference → capped 0.4
        )
    assert store.get(iid)["status"] == STATUS_CANDIDATE


def test_owner_rule_never_auto_promotes(store):
    """owner_rule always requires explicit confirmation, regardless of recurrence."""
    for conv in ("c1", "c2", "c3"):
        iid = store.add_candidate(
            TYPE_OWNER_RULE, "deploys", "never deploy on friday",
            provenance="user_direct", confidence=0.95, conversation_id=conv,
        )
    assert store.get(iid)["status"] == STATUS_CANDIDATE


def test_contradiction_with_owner_rule_blocks_autopromote(store):
    # Confirmed owner_rule on subject 'drink' = 'only water'.
    rule = store.add_candidate(
        TYPE_OWNER_RULE, "drink", "only water", provenance="user_direct",
    )
    assert store.confirm(rule, confirmed_by="owner") is True
    # A contradicting preference, recurring across 3 conversations, must NOT promote.
    for conv in ("c1", "c2", "c3"):
        pref = store.add_candidate(
            TYPE_USER_PREFERENCE, "drink", "beer",
            provenance="user_direct", confidence=0.95, conversation_id=conv,
        )
    assert store.get(pref)["status"] == STATUS_CANDIDATE


def test_web_fact_never_auto_promotes(store):
    for conv in ("c1", "c2", "c3"):
        iid = store.add_candidate(
            TYPE_WEB_FACT, "capital of france", "Paris",
            provenance="user_direct", confidence=0.95, conversation_id=conv,
            source="web", source_url="https://example.org",
        )
    assert store.get(iid)["status"] == STATUS_CANDIDATE


# --- filtering / dedupe -----------------------------------------------------

def test_secret_value_rejected(store):
    iid = store.add_candidate(
        TYPE_USER_FACT, "login", "my password is hunter2", provenance="user_direct",
    )
    assert iid is None
    assert store.list_by_state(STATUS_CANDIDATE) == []


def test_duplicate_dedupe(strict_store):
    a = strict_store.add_candidate(TYPE_USER_FACT, "city", "Bucharest")
    b = strict_store.add_candidate(TYPE_USER_FACT, "city", "bucharest")  # case-fold dup
    assert a == b
    assert len(strict_store.list_by_state(STATUS_CANDIDATE)) == 1


# --- retrieval boundaries ---------------------------------------------------

def test_retrieve_confirmed_only_confirmed(strict_store):
    confirmed = strict_store.add_candidate(TYPE_USER_FACT, "k1", "confirmed-val")
    strict_store.confirm(confirmed, confirmed_by="owner")
    strict_store.add_candidate(TYPE_USER_FACT, "k2", "candidate-val")  # stays candidate
    pending = strict_store.add_candidate(TYPE_USER_FACT, "k3", "pending-val")
    strict_store.promote_to_pending(pending)

    out = strict_store.retrieve_confirmed()
    values = {r["value"] for r in out}
    assert values == {"confirmed-val"}


def test_quarantined_excluded_from_retrieval(strict_store):
    iid = strict_store.add_candidate(TYPE_USER_FACT, "k", "bad-fact")
    strict_store.confirm(iid, confirmed_by="owner")
    assert len(strict_store.retrieve_confirmed()) == 1
    strict_store.quarantine(iid, reason="user_flagged")
    assert strict_store.retrieve_confirmed() == []
    assert strict_store.get(iid)["status"] == STATUS_QUARANTINED


def test_retrieve_confirmed_excludes_expired(strict_store):
    iid = strict_store.add_candidate(
        TYPE_WEB_FACT, "weather", "sunny", expires_at="2000-01-01T00:00:00+00:00",
    )
    strict_store.confirm(iid, confirmed_by="owner")
    assert strict_store.retrieve_confirmed(TYPE_WEB_FACT) == []


# --- correction / supersede -------------------------------------------------

def test_correct_supersede_chain(strict_store):
    old = strict_store.add_candidate(TYPE_USER_FACT, "car", "Dacia")
    strict_store.confirm(old, confirmed_by="owner")
    new = strict_store.correct(old, "Tesla", confirmed_by="owner")

    assert strict_store.get(old)["status"] == STATUS_SUPERSEDED
    new_item = strict_store.get(new)
    assert new_item["status"] == STATUS_CONFIRMED
    assert new_item["supersedes"] == old
    assert new_item["value"] == "Tesla"
    # retrieve_confirmed returns only the corrected value.
    out = strict_store.retrieve_confirmed(TYPE_USER_FACT, subject_key="car")
    assert [r["value"] for r in out] == ["Tesla"]


def test_correct_unknown_item_raises(strict_store):
    with pytest.raises(KeyError):
        strict_store.correct("does-not-exist", "whatever")


# --- forget -----------------------------------------------------------------

def test_forget_soft_delete_keeps_audit(strict_store):
    iid = strict_store.add_candidate(TYPE_USER_FACT, "secret-plan", "surprise party")
    strict_store.confirm(iid, confirmed_by="owner")
    assert strict_store.forget(iid) is True

    item = strict_store.get(iid)
    assert item is not None                       # row is kept (soft delete)
    assert item["status"] == STATUS_FORGOTTEN
    assert strict_store.retrieve_confirmed() == []  # excluded from retrieval
    actions = [a["action"] for a in strict_store.get_audit(iid)]
    assert "forget" in actions                    # audit retained


# --- audit ------------------------------------------------------------------

def test_audit_trail_recorded_per_transition(strict_store):
    iid = strict_store.add_candidate(TYPE_USER_FACT, "k", "v")
    strict_store.promote_to_pending(iid)
    strict_store.confirm(iid, confirmed_by="owner")
    strict_store.quarantine(iid, reason="oops")

    trail = strict_store.get_audit(iid)
    seq = [(a["action"], a["from_status"], a["to_status"]) for a in trail]
    assert seq == [
        ("add", None, STATUS_CANDIDATE),
        ("promote", STATUS_CANDIDATE, STATUS_PENDING),
        ("confirm", STATUS_PENDING, STATUS_CONFIRMED),
        ("quarantine", STATUS_CONFIRMED, STATUS_QUARANTINED),
    ]


# --- durability -------------------------------------------------------------

def test_restart_persistence(tmp_path):
    """Close and reopen a real Database on the SAME file: a pending candidate must
    survive and be confirmable after the restart."""
    path = str(tmp_path / "jarvis.db")

    db1 = Database(path)
    s1 = StateStore(db1, require_confirmation=True)
    iid = s1.add_candidate(TYPE_USER_FACT, "owner name", "Razvan")
    s1.promote_to_pending(iid)
    assert s1.get(iid)["status"] == STATUS_PENDING
    db1.close()

    db2 = Database(path)
    s2 = StateStore(db2, require_confirmation=True)
    assert s2.get(iid)["status"] == STATUS_PENDING       # survived restart
    assert s2.confirm(iid, confirmed_by="owner") is True
    assert s2.get(iid)["status"] == STATUS_CONFIRMED
    # Audit trail persisted across the restart too.
    actions = [a["action"] for a in s2.get_audit(iid)]
    assert actions == ["add", "promote", "confirm"]
    db2.close()


def test_no_authorization_method_exists():
    """Memory NEVER authorises tools or development — no such surface exists."""
    banned = ("authorize", "execute", "run_tool", "grant", "develop", "approve_tool")
    names = dir(StateStore)
    for token in banned:
        assert not any(token in n for n in names), f"unexpected {token!r} on StateStore"


# --- TOCTOU / race safety (adversarial-review fixes) ------------------------

def test_asr_quarantine_atomic_against_concurrent_confirm(store, monkeypatch):
    """Fix 1 — add_from_asr get()+quarantine() TOCTOU.

    A human ``confirm`` that lands AFTER add_from_asr re-reads the (still
    ``candidate``) status but BEFORE its quarantine step must NOT un-confirm the
    now-vetted memory. The guarded quarantine passes ``allowed_from=(candidate,)``
    and the status check happens atomically with the write, so it becomes a no-op
    the instant the item is confirmed.
    """
    # A plain candidate the low-quality ASR echo will dedupe into.
    iid = store.add_candidate(
        TYPE_USER_FACT, "coffee", "espresso", provenance="user_direct",
    )
    assert store.get(iid)["status"] == STATUS_CANDIDATE

    real_get = store.get
    fired = {"done": False}

    def racing_get(item_id):
        snap = real_get(item_id)
        # Simulate a concurrent human confirm() arriving in the TOCTOU window:
        # right after add_from_asr reads the candidate status, before it
        # quarantines. We still return the pre-confirm snapshot the caller saw.
        if not fired["done"] and snap is not None and snap["status"] == STATUS_CANDIDATE:
            fired["done"] = True
            store.confirm(item_id, confirmed_by="owner")
        return snap

    monkeypatch.setattr(store, "get", racing_get)
    same = store.add_from_asr(
        "espresso", asr_confidence=0.2, asr_metrics=None,  # low quality → quarantine path
        item_type=TYPE_USER_FACT, subject_key="coffee",
    )
    monkeypatch.undo()

    assert same == iid
    assert fired["done"] is True                          # the race actually happened
    assert store.get(iid)["status"] == STATUS_CONFIRMED   # NOT un-confirmed by the echo


def test_owner_rule_confirmed_in_window_blocks_autopromote(store, monkeypatch):
    """Fix 2 — _maybe_auto_promote guard/transition TOCTOU.

    A contradicting owner_rule that becomes ``confirmed`` AFTER the eligibility
    guard but BEFORE the promoting commit must still block promotion. The
    promoting transition re-checks the owner_rule invariant inside its own
    transaction (precommit_guard), so the preference cannot silently override the
    rule.
    """
    # Two explicit independent conversations so the 3rd will trip auto-promotion.
    for conv in ("c1", "c2"):
        store.add_candidate(
            TYPE_USER_PREFERENCE, "drink", "beer",
            provenance="user_direct", confidence=0.95, conversation_id=conv,
        )

    real_check = store._contradicts_owner_rule
    state = {"n": 0}

    def racing_check(subject_key, value):
        # First evaluation reflects the pre-insert world (no rule yet → False),
        # then a contradicting owner_rule lands, simulating a concurrent confirm
        # inside the promotion window. The atomic re-check (2nd call) sees it.
        result = real_check(subject_key, value)
        if state["n"] == 0:
            state["n"] += 1
            rule = store.add_candidate(
                TYPE_OWNER_RULE, "drink", "only water", provenance="user_direct",
            )
            store.confirm(rule, confirmed_by="owner")
        return result

    monkeypatch.setattr(store, "_contradicts_owner_rule", racing_check)
    pref = store.add_candidate(
        TYPE_USER_PREFERENCE, "drink", "beer",
        provenance="user_direct", confidence=0.95, conversation_id="c3",
    )
    monkeypatch.undo()

    assert state["n"] == 1                                 # the race was exercised
    assert store.get(pref)["status"] == STATUS_CANDIDATE   # NOT promoted over the rule


# --- explicit-only recurrence for auto-promotion (Fix 3) --------------------

def test_asr_mentions_do_not_pad_autopromote_count(store):
    """Fix 3 — only EXPLICIT independent conversations count toward promotion.

    1 explicit + 2 good-quality ASR mentions span 3 distinct conversations, yet
    the preference must NOT auto-promote: the ASR echoes add recurrence history
    but are ``model_inference`` / ``allow_auto_promote=False`` and so cannot pad
    the independent-conversation total to AUTO_PROMOTE_MIN_CONVERSATIONS.
    """
    # 1 explicit statement (creates the item; provenance frozen = user_direct).
    iid = store.add_candidate(
        TYPE_USER_PREFERENCE, "milk", "oat milk",
        provenance="user_direct", confidence=0.95, conversation_id="c1",
    )
    assert store.get(iid)["status"] == STATUS_CANDIDATE
    # 2 good-quality ASR echoes in two OTHER conversations dedupe into it.
    for conv in ("c2", "c3"):
        same = store.add_from_asr(
            "oat milk", asr_confidence=0.94, asr_metrics={"avg_logprob": -0.15},
            item_type=TYPE_USER_PREFERENCE, subject_key="milk", conversation_id=conv,
        )
        assert same == iid
        assert store.get(same)["status"] == STATUS_CANDIDATE
    # Distinct conversations = 3, but only 1 of them was explicit.
    assert store._distinct_conversations(iid) == 3
    assert store._distinct_explicit_conversations(iid) == 1
    # A further explicit mention in an existing conversation triggers the
    # promotion path; with only 1 explicit conversation it must stay candidate.
    again = store.add_candidate(
        TYPE_USER_PREFERENCE, "milk", "oat milk",
        provenance="user_direct", confidence=0.95, conversation_id="c1",
    )
    assert again == iid
    assert store.get(iid)["status"] == STATUS_CANDIDATE    # NOT auto-promoted


def test_three_explicit_conversations_still_promote(store):
    """Fix 3 sanity: 3 EXPLICIT independent conversations still auto-promote."""
    for conv in ("c1", "c2", "c3"):
        iid = store.add_candidate(
            TYPE_USER_PREFERENCE, "tea", "green tea",
            provenance="user_direct", confidence=0.95, conversation_id=conv,
        )
    assert store._distinct_explicit_conversations(iid) == 3
    item = store.get(iid)
    assert item["status"] == STATUS_CONFIRMED
    assert item["confirmed_by"] == "auto"
