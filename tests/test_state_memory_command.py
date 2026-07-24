"""Phase 4 · Section E — explicit state-memory command flow (integration).

Proves the fix for the live ASR-poisoning: "Cora, memorează …" now goes
candidate → pending → READBACK → confirm, and is confirmed ONLY on an explicit
"da". A garbled value is caught at the readback and cancelled — never confirmed
(unlike the legacy path that committed at 0.95 with no confirmation).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from jarvis.memory.db import Database
from jarvis.memory.state_store import StateStore
from jarvis.memory.learning.commands import try_state_memory_command


@pytest.fixture
def store(tmp_path):
    db = Database(str(tmp_path / "state.db"), None)
    return StateStore(db, require_confirmation=True)


@pytest.fixture
def dm():
    # Minimal dialogue-memory stand-in: holds conversation_id + arbitrary attrs.
    return SimpleNamespace(conversation_id="conv-1")


def _cmd(store, dm, text):
    return try_state_memory_command(
        text, state_store=store, dialogue_memory=dm, conversation_id="conv-1")


def test_memorize_asks_confirmation_not_confirmed_yet(store, dm):
    r = _cmd(store, dm, "memorează că prefer răspunsuri scurte")
    assert r.handled and r.reply
    assert "Să memorez" in r.reply
    # nothing confirmed yet
    assert store.retrieve_confirmed() == []
    # exactly one pending item
    assert len(store.list_by_state("pending_confirmation")) == 1
    assert getattr(dm, "_pending_state_memorize", None) is not None


def test_confirm_da_promotes_to_confirmed(store, dm):
    _cmd(store, dm, "memorează că prefer răspunsuri scurte")
    r = _cmd(store, dm, "da")
    assert r.handled and "Am memorat" in r.reply
    confirmed = store.retrieve_confirmed()
    assert len(confirmed) == 1
    assert "scurte" in confirmed[0]["value"]
    assert getattr(dm, "_pending_state_memorize", None) is None


def test_cancel_nu_discards_nothing_confirmed(store, dm):
    # This is the ASR safety net: a garbled readback is rejected by ear.
    _cmd(store, dm, "memorează căle răstunduri scure și direcțe")  # ASR-garbled
    r = _cmd(store, dm, "nu")
    assert r.handled and "anulat" in r.reply.lower()
    assert store.retrieve_confirmed() == []
    # not left dangling in pending either
    assert store.list_by_state("pending_confirmation") == []
    assert getattr(dm, "_pending_state_memorize", None) is None


def test_pending_blocks_other_commands_until_resolved(store, dm):
    _cmd(store, dm, "memorează că prefer cafea")
    r = _cmd(store, dm, "ceva nelegat")
    assert r.handled and "Confirmă" in r.reply  # still awaiting da/nu


def test_correct_reads_back_then_confirms(store, dm):
    # Correction now goes through the same readback→confirm ASR safety net.
    r = _cmd(store, dm, "corectează numele meu este Răzvan")
    assert r.handled and "corec" in r.reply.lower()  # readback, not done yet
    assert store.retrieve_confirmed() == []            # NOT confirmed before "da"
    r2 = _cmd(store, dm, "da")
    assert r2.handled and "corec" in r2.reply.lower()
    assert any("Răzvan" in it["value"] for it in store.retrieve_confirmed())


def test_correction_not_confirmed_without_da(store, dm):
    _cmd(store, dm, "corectează orașul meu este Cluj")
    # a "nu" discards it — nothing confirmed (ASR-corrupt correction is caught)
    r = _cmd(store, dm, "nu")
    assert r.handled and "anulat" in r.reply.lower()
    assert store.retrieve_confirmed() == []


def test_de_fapt_filler_is_not_a_correction(store, dm):
    # "de fapt" is common filler and must NOT create memory.
    r = _cmd(store, dm, "de fapt nu știu ce să zic")
    assert r.handled is False
    assert store.retrieve_confirmed() == []


def test_forget_soft_deletes_confirmed(store, dm):
    _cmd(store, dm, "memorează că prefer ceai verde")
    _cmd(store, dm, "da")
    assert len(store.retrieve_confirmed()) == 1
    r = _cmd(store, dm, "uită ceai verde")
    assert r.handled and "uitat" in r.reply.lower()
    assert store.retrieve_confirmed() == []


def test_what_learned_lists_only_confirmed(store, dm):
    # one confirmed via the command flow
    _cmd(store, dm, "memorează că prefer cafea")
    _cmd(store, dm, "da")  # confirms + clears the pending slot
    # one candidate added directly (never confirmed, no dm pending involved)
    store.add_candidate("user_fact", "whisky pref", "prefer whisky",
                        conversation_id="c2")
    r = _cmd(store, dm, "ce ai învățat despre mine")
    assert r.handled
    assert "cafea" in (r.reply or "").lower()          # confirmed → listed
    assert "whisky" not in (r.reply or "").lower()     # candidate → NOT listed


def test_non_command_falls_through(store, dm):
    r = _cmd(store, dm, "cât e ceasul?")
    assert r.handled is False


def test_none_store_falls_through(dm):
    r = try_state_memory_command(
        "memorează ceva", state_store=None, dialogue_memory=dm, conversation_id="c")
    assert r.handled is False


def test_secret_shaped_value_refused(store, dm):
    secret = "sk-" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
    r = _cmd(store, dm, f"memorează că cheia este {secret}")
    assert r.handled
    # either refused outright, or (if stored) never confirmed and value scrubbed
    for it in store.retrieve_confirmed():
        assert secret not in it["value"]
    assert secret not in (r.reply or "")
