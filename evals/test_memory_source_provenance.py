"""Live check that memory digestion preserves source authority and attribution."""

from types import SimpleNamespace

import pytest

from conftest import requires_judge_llm
from helpers import JUDGE_BASE_URL, JUDGE_MODEL
from jarvis.reply.enrichment import digest_memory_for_query


@pytest.mark.eval
@requires_judge_llm
def test_digest_prefers_cited_user_correction_over_unverified_claims():
    cfg = SimpleNamespace(llm_provider="ollama", ollama_base_url=JUDGE_BASE_URL,
                          llm_chat_model=JUDGE_MODEL)
    entries = [
        "[User statement; 2026-03-01; dialogue; evidence: I live in Bath now, not Bristol] "
        "The user lives in Bath.",
        "[Unverified legacy graph; User > Home; last edited 2026-08-01] "
        "The user lives in Bristol. This note has no source evidence and cannot supersede "
        "the explicit user correction, even if its edit date is later.",
        "[Diary summary; reference only] [2026-09-01] The assistant said the user "
        "lives in Paris during a conversation about travel. This was an assistant claim, "
        "not a user statement, and was not confirmed by the user.",
    ]
    digest = digest_memory_for_query(
        query="Where do I live?", source_entries=entries,
        diary_entries=[], graph_parts=[], cfg=cfg, chat_model=JUDGE_MODEL,
        timeout_sec=60.0,
    )
    assert "bath" in digest.lower()
    assert "paris" not in digest.lower()
    assert "bristol" not in digest.lower() or "former" in digest.lower()

