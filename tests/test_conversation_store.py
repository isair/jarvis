"""Phase 3A: ConversationStore tests on a temp Database (additive, isolated)."""

from __future__ import annotations

import pytest

from jarvis.memory.db import Database
from jarvis.memory.conversation_store import ConversationStore, CONVERSATIONS_SCHEMA_VERSION


@pytest.fixture()
def store(tmp_path):
    db = Database(str(tmp_path / "jarvis.db"))
    yield ConversationStore(db)
    db.close()


# --- schema / idempotency ---------------------------------------------------

def test_schema_created_and_versioned(tmp_path):
    db = Database(str(tmp_path / "jarvis.db"))
    s = ConversationStore(db)
    s.list_conversations()  # triggers lazy schema
    row = db.conn.execute(
        "SELECT version FROM schema_migrations WHERE component='conversations'"
    ).fetchone()
    assert row is not None and int(row[0]) == CONVERSATIONS_SCHEMA_VERSION
    db.close()


def test_second_store_on_same_db_is_idempotent(tmp_path):
    db = Database(str(tmp_path / "jarvis.db"))
    ConversationStore(db).create_conversation("c1", title="A")
    s2 = ConversationStore(db)  # must not raise / duplicate tables
    assert any(c["id"] == "c1" for c in s2.list_conversations())
    db.close()


def test_does_not_touch_existing_tables(tmp_path):
    """Additive-only: creating the store must not disturb diary rows."""
    db = Database(str(tmp_path / "jarvis.db"))
    db.upsert_conversation_summary("2026-07-22", "a diary summary", topics="t")
    ConversationStore(db).list_conversations()
    got = db.get_conversation_summary("2026-07-22")
    assert got is not None and got["summary"] == "a diary summary"
    db.close()


# --- conversations ----------------------------------------------------------

def test_ensure_conversation_auto_titles_once(store):
    store.ensure_conversation("c1", title_seed="Cum îmi configurez serverul de joc azi?")
    assert store.get_conversation("c1")["title"].startswith("Cum îmi configurez")
    # a later turn must NOT overwrite the title
    store.ensure_conversation("c1", title_seed="alt seed complet diferit")
    assert store.get_conversation("c1")["title"].startswith("Cum îmi configurez")


def test_rename_updates_title(store):
    store.ensure_conversation("c1", title_seed="seed")
    assert store.rename("c1", "  Titlu   Nou  ") is True
    assert store.get_conversation("c1")["title"] == "Titlu Nou"


def test_rename_empty_rejected(store):
    store.ensure_conversation("c1", title_seed="seed")
    with pytest.raises(ValueError):
        store.rename("c1", "   ")


def test_soft_delete_hides_from_list(store):
    store.create_conversation("c1", title="A")
    assert store.delete("c1", confirm=True) is True
    assert all(c["id"] != "c1" for c in store.list_conversations())
    # still present when explicitly including deleted
    assert any(c["id"] == "c1" for c in store.list_conversations(include_deleted=True))


def test_delete_requires_confirm(store):
    store.create_conversation("c1", title="A")
    with pytest.raises(ValueError):
        store.delete("c1", confirm=False)


def test_hard_delete_removes_messages(store):
    store.ensure_conversation("c1", title_seed="s")
    store.append_user_message("c1", "m1", "hi", source="text", turn_id="t1")
    assert store.delete("c1", confirm=True, hard=True) is True
    assert store.get_messages("c1") == []
    assert store.get_conversation("c1") is None


def test_select_active_returns_most_recent(store):
    store.create_conversation("old", title="old")
    store.create_conversation("new", title="new")
    store.touch("new")
    assert store.select_active() == "new"


def test_list_ordered_by_updated_desc(store):
    store.create_conversation("a", title="a")
    store.create_conversation("b", title="b")
    store.touch("a")  # a is now most recent
    ids = [c["id"] for c in store.list_conversations()]
    assert ids[0] == "a"


# --- search (LIKE escaping) -------------------------------------------------

def test_search_escapes_like_wildcards(store):
    store.create_conversation("c1", title="100% sigur")
    store.create_conversation("c2", title="ceva random")
    hits = store.search_titles("100%")
    assert [h["id"] for h in hits] == ["c1"]  # % matched literally, not as wildcard


def test_search_underscore_literal(store):
    store.create_conversation("c1", title="a_b test")
    store.create_conversation("c2", title="axb test")
    hits = store.search_titles("a_b")
    assert [h["id"] for h in hits] == ["c1"]


# --- messages ---------------------------------------------------------------

def test_append_and_get_messages_in_order(store):
    store.ensure_conversation("c1", title_seed="s")
    store.append_user_message("c1", "m1", "u1", source="text", turn_id="t1")
    store.append_or_update_assistant_message("c1", "m2", "a1", status="completed", turn_id="t1")
    msgs = store.get_messages("c1")
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[1]["content"] == "a1"


def test_assistant_upsert_finalizes_same_row(store):
    store.ensure_conversation("c1", title_seed="s")
    store.append_or_update_assistant_message("c1", "m1", "", status="pending", turn_id="t1")
    store.append_or_update_assistant_message("c1", "m1", "final", status="completed", turn_id="t1")
    msgs = store.get_messages("c1")
    assert len(msgs) == 1 and msgs[0]["status"] == "completed" and msgs[0]["content"] == "final"


def test_clear_messages_keeps_conversation(store):
    store.ensure_conversation("c1", title_seed="s")
    store.append_user_message("c1", "m1", "u1", source="text", turn_id="t1")
    assert store.clear_messages("c1", confirm=True) == 1
    assert store.get_messages("c1") == []
    assert store.get_conversation("c1") is not None  # conversation row survives


def test_diacritics_roundtrip(store):
    store.ensure_conversation("c1", title_seed="s")
    txt = "Ăî âșț — răspuns cu diacritice complete"
    store.append_user_message("c1", "m1", txt, source="text", turn_id="t1")
    assert store.get_messages("c1")[0]["content"] == txt


def test_bad_metadata_rejected(store):
    with pytest.raises(ValueError):
        store.create_conversation("c1", metadata=["not", "a", "dict"])
