"""Tests for reply enrichment helpers."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from jarvis.reply.engine import _build_enrichment_context_hint, _match_question
from jarvis.reply.enrichment import extract_search_params_for_memory


class TestMatchQuestion:
    """Verify question→node matching logic."""

    def test_returns_empty_when_no_questions(self):
        assert _match_question("some node data", []) == ""

    def test_matches_best_question_by_keyword_overlap(self):
        node_data = "The user enjoys Thai and Japanese cuisine and lives in London."
        questions = [
            "what cuisine does the user like?",
            "where is the user located?",
            "what are the user's hobbies?",
        ]
        result = _match_question(node_data, questions)
        assert result == "what cuisine does the user like?"

    def test_matches_location_question(self):
        node_data = "The user lives in Hackney, London."
        questions = [
            "what cuisine does the user like?",
            "where does the user live?",
        ]
        result = _match_question(node_data, questions)
        assert "live" in result

    def test_no_match_returns_empty(self):
        node_data = "The user has a cat named Mochi."
        questions = [
            "what programming languages does the user know?",
        ]
        result = _match_question(node_data, questions)
        assert result == ""

    def test_stop_words_excluded_from_matching(self):
        """Questions consisting only of stop words should not match."""
        node_data = "The user is an engineer."
        questions = ["what is the user?"]
        # All significant words are stop words, so no match
        result = _match_question(node_data, questions)
        assert result == ""

    def test_partial_overlap_still_matches(self):
        node_data = "The user boxes at Trenches gym three times a week."
        questions = [
            "what gym does the user go to?",
            "how often does the user exercise?",
        ]
        result = _match_question(node_data, questions)
        assert result == "what gym does the user go to?"


def _cfg(**over):
    base = dict(
        location_enabled=False,
        ollama_base_url="http://x",
        ollama_chat_model="m",
    )
    base.update(over)
    return SimpleNamespace(**base)


class TestBuildEnrichmentContextHint:
    """The hint is what lets the extractor skip questions already answerable."""

    def test_returns_none_when_nothing_to_say(self):
        # Location disabled and no recent messages → hint should still include the
        # "Location: Disabled" line (that IS useful context). Verify it isn't None.
        hint = _build_enrichment_context_hint(_cfg(), [])
        assert hint and "Location: Disabled" in hint
        assert "Recent dialogue" not in hint

    def test_includes_truncated_recent_dialogue(self):
        long_msg = "x" * 500
        msgs = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": long_msg},
        ]
        hint = _build_enrichment_context_hint(_cfg(), msgs)
        assert "- user: hello" in hint
        # Truncated to 200 chars, so the 500-char message must be shortened.
        assert ("x" * 500) not in hint
        assert ("x" * 200) in hint

    def test_caps_at_recent_messages_limit(self):
        msgs = [{"role": "user", "content": f"msg {i}"} for i in range(20)]
        hint = _build_enrichment_context_hint(_cfg(), msgs)
        # Only the last six should be mirrored.
        assert "msg 14" in hint
        assert "msg 19" in hint
        assert "msg 13" not in hint
        assert "msg 0" not in hint


class TestExtractorPromptRendering:
    """Prompt construction should not crash on tricky context_hint inputs."""

    def _run_and_capture_prompt(self, **kwargs) -> str:
        captured = {}

        def fake_call(**call_kwargs):
            captured["system_prompt"] = call_kwargs["system_prompt"]
            return '{"keywords": []}'

        with patch("jarvis.reply.enrichment.call_llm_direct", side_effect=fake_call):
            extract_search_params_for_memory(
                "dummy query", "http://x", "m", timeout_sec=1.0, **kwargs
            )
        return captured["system_prompt"]

    def test_no_hint_falls_back_to_utc_timestamp(self):
        # Behaviour: with no hint, the extractor still gets a current-time anchor
        # (UTC fallback) so it can resolve relative time phrases.
        prompt = self._run_and_capture_prompt()
        assert "UTC" in prompt

    def test_hint_is_injected_and_utc_fallback_dropped(self):
        # Use a value that can only have come from the hint, so the assertion
        # survives prompt rewording as long as the hint is actually threaded in.
        hint_marker = "Tbilisi, Georgia"
        fallback_marker = "fallback-sentinel-utc"
        hint_prompt = self._run_and_capture_prompt(
            context_hint=f"Current local time: ... . Location: {hint_marker}."
        )
        no_hint_prompt = self._run_and_capture_prompt()
        assert hint_marker in hint_prompt
        # The UTC fallback injects a marker that is present in the no-hint case;
        # that same marker must NOT appear when a hint is supplied (dedup).
        fallback_signature = "Current date/time:"
        assert fallback_signature in no_hint_prompt
        assert fallback_signature not in hint_prompt

    def test_extract_returns_empty_dict_when_no_usable_response(self):
        with patch("jarvis.reply.enrichment.call_llm_direct", return_value=""):
            result = extract_search_params_for_memory(
                "q", "http://x", "m", timeout_sec=0.1,
            )
        assert result == {}

    def test_braces_in_hint_do_not_break_format(self):
        # User dialogue could contain literal '{' or '}'. The outer .format must
        # treat the hint as a literal string, not re-interpret placeholders.
        hint = "Recent dialogue:\n- user: try running {env.HOME} or {{notathing}}"
        prompt = self._run_and_capture_prompt(context_hint=hint)
        assert "{env.HOME}" in prompt
        assert "{{notathing}}" in prompt


class TestGraphEnrichmentGating:
    """Graph enrichment is question-driven: no questions → no graph crawl."""

    def _run(self, extract_return: dict, enrichment_source: str = "all"):
        from jarvis.reply.engine import run_reply_engine

        class _DM:
            def has_recent_messages(self):
                return False

            def get_recent_messages(self):
                return []

            def add_message(self, role, content):
                pass

        class _TTS:
            enabled = False

        cfg = SimpleNamespace(
            ollama_base_url="http://x",
            ollama_chat_model="m",
            ollama_embed_model="e",
            llm_tools_timeout_sec=0.1,
            llm_embed_timeout_sec=0.1,
            llm_chat_timeout_sec=0.1,
            agentic_max_turns=0,
            active_profiles=["developer"],
            voice_debug=False,
            memory_enrichment_source=enrichment_source,
            memory_enrichment_max_results=0,
            mcps={},
            location_enabled=False,
            location_auto_detect=False,
            location_ip_address=None,
            location_cgnat_resolve_public_ip=True,
            db_path=":memory:",
        )

        store_calls: list[str] = []

        class _FakeStore:
            def __init__(self, *a, **kw):
                pass

            def search_nodes(self, query, limit=5):
                store_calls.append(query)
                return []

            def get_recent_nodes(self, limit=3):
                store_calls.append("get_recent_nodes")
                return []

            def get_ancestors(self, node_id):
                return []

        with patch("jarvis.reply.engine.extract_search_params_for_memory", return_value=extract_return), \
             patch("jarvis.memory.graph.GraphMemoryStore", _FakeStore):
            run_reply_engine(db=None, cfg=cfg, tts=None, text="q", dialogue_memory=_DM())

        return store_calls

    def test_skips_graph_when_no_questions(self):
        calls = self._run({"keywords": ["time"], "questions": []})
        assert calls == [], f"Graph should not be touched without questions, got {calls}"

    def test_crawls_graph_when_questions_present(self):
        calls = self._run({
            "keywords": ["food"],
            "questions": ["what cuisine does the user enjoy?"],
        })
        # search_nodes should have been called (with question-derived terms).
        assert any("cuisine" in c for c in calls), \
            f"Expected graph search using question words, got {calls}"
        # The removed recent-nodes fallback must stay removed.
        assert "get_recent_nodes" not in calls

    def test_skips_graph_when_source_is_diary_only(self):
        calls = self._run(
            {"keywords": ["food"], "questions": ["what cuisine?"]},
            enrichment_source="diary",
        )
        assert calls == []

    def test_skips_graph_when_questions_are_all_stopwords(self):
        # "what is the?" strips down to nothing meaningful — should not hit store.
        calls = self._run({
            "keywords": ["x"],
            "questions": ["what is the?"],
        })
        assert calls == []


class TestGraphContextReachesSystemMessage:
    """Regression: graph enrichment must reach the LLM system prompt.

    An earlier bug built a `context` list containing graph results but never
    threaded it into the system message, so the model was told "I know nothing
    about you" even though 🧠 Knowledge logs showed nodes surfaced.
    """

    def test_graph_context_appears_in_system_prompt(self):
        from jarvis.reply.engine import run_reply_engine

        class _Node:
            def __init__(self):
                self.id = "n1"
                self.name = "Food Preferences"
                self.data = "User loves sushi and spicy ramen."
                self.data_token_count = 10

        class _Ancestor:
            name = "Root"

        class _FakeStore:
            def __init__(self, *a, **kw):
                pass

            def search_nodes(self, query, limit=5):
                return [_Node()]

            def get_ancestors(self, node_id):
                return [_Ancestor()]

        class _DM:
            def has_recent_messages(self):
                return False

            def get_recent_messages(self):
                return []

            def add_message(self, role, content):
                pass

        cfg = SimpleNamespace(
            ollama_base_url="http://x",
            ollama_chat_model="m",
            ollama_embed_model="e",
            llm_tools_timeout_sec=0.1,
            llm_embed_timeout_sec=0.1,
            llm_chat_timeout_sec=0.1,
            agentic_max_turns=1,
            active_profiles=["developer"],
            voice_debug=False,
            memory_enrichment_source="all",
            memory_enrichment_max_results=0,
            mcps={},
            location_enabled=False,
            location_auto_detect=False,
            location_ip_address=None,
            location_cgnat_resolve_public_ip=True,
            db_path=":memory:",
            tts_engine="piper",
        )

        captured_messages: list = []

        def fake_chat(**kwargs):
            captured_messages.extend(kwargs.get("messages", []))
            return {"message": {"content": "ok", "role": "assistant"}}

        with patch(
            "jarvis.reply.engine.extract_search_params_for_memory",
            return_value={"keywords": ["food"], "questions": ["what cuisine does the user enjoy?"]},
        ), patch("jarvis.memory.graph.GraphMemoryStore", _FakeStore), \
             patch("jarvis.reply.engine.chat_with_messages", side_effect=fake_chat), \
             patch("jarvis.tools.selection.select_tools", return_value=[]):
            run_reply_engine(db=None, cfg=cfg, tts=None, text="what do you know about me?", dialogue_memory=_DM())

        system_msgs = [m for m in captured_messages if m.get("role") == "system"]
        assert system_msgs, "Expected a system message to be sent to the LLM"
        joined = "\n".join(m.get("content", "") for m in system_msgs)
        assert "Information the user has shared with you in prior conversations" in joined, \
            f"Graph context missing from system prompt. Got:\n{joined[:500]}"
        assert "sushi" in joined, \
            f"Graph node data missing from system prompt. Got:\n{joined[:500]}"


# ── Memory digest ──────────────────────────────────────────────────────


class TestDigestMemoryForQuery:
    """Behaviour of digest_memory_for_query — the cheap LLM pass that
    distils diary + graph dumps into a compact note before injecting into
    small-model system prompts.
    """

    def _base_kwargs(self):
        return dict(
            query="what did we discuss about cooking?",
            ollama_base_url="http://x",
            ollama_chat_model="gemma4",
            timeout_sec=1.0,
            thinking=False,
        )

    def test_empty_inputs_returns_empty(self):
        from jarvis.reply.enrichment import digest_memory_for_query

        result = digest_memory_for_query(
            diary_entries=[], graph_parts=[], **self._base_kwargs()
        )
        assert result == ""

    def test_short_input_passes_through_unchanged(self):
        """Below _DIGEST_MIN_CHARS, the raw block is already cheap; no LLM call."""
        from jarvis.reply.enrichment import digest_memory_for_query

        short_entry = "[2026-04-20] Brief chat about coffee."
        with patch("jarvis.reply.enrichment.call_llm_direct") as mock_llm:
            result = digest_memory_for_query(
                diary_entries=[short_entry], graph_parts=[], **self._base_kwargs()
            )
            # The raw block is short — we never call the distil LLM.
            mock_llm.assert_not_called()
        assert short_entry in result

    def test_none_sentinel_returns_empty(self):
        from jarvis.reply.enrichment import digest_memory_for_query

        big_entry = "[2026-04-20] " + ("x " * 300)
        with patch(
            "jarvis.reply.enrichment.call_llm_direct",
            return_value="NONE",
        ):
            result = digest_memory_for_query(
                diary_entries=[big_entry], graph_parts=[], **self._base_kwargs()
            )
        assert result == ""

    def test_bracketed_none_variants_return_empty(self):
        from jarvis.reply.enrichment import digest_memory_for_query

        big_entry = "[2026-04-20] " + ("x " * 300)
        for variant in ["(NONE)", "[NONE]", "none.", "N/A"]:
            with patch(
                "jarvis.reply.enrichment.call_llm_direct",
                return_value=variant,
            ):
                result = digest_memory_for_query(
                    diary_entries=[big_entry], graph_parts=[], **self._base_kwargs()
                )
            assert result == "", f"Variant {variant!r} should yield empty digest"

    def test_returns_digest_when_model_finds_relevance(self):
        from jarvis.reply.enrichment import digest_memory_for_query

        big_entry = "[2026-04-20] Long cooking chat. " + ("detail " * 100)
        with patch(
            "jarvis.reply.enrichment.call_llm_direct",
            return_value="User previously discussed cooking Thai curry on 2026-04-20.",
        ):
            result = digest_memory_for_query(
                diary_entries=[big_entry], graph_parts=[], **self._base_kwargs()
            )
        assert "cooking Thai curry" in result

    def test_truncates_oversized_digest(self):
        from jarvis.reply.enrichment import (
            _DIGEST_MAX_CHARS,
            digest_memory_for_query,
        )

        big_entry = "[2026-04-20] " + ("x " * 300)
        overflow = "A " * 600  # 1200 chars — well past _DIGEST_MAX_CHARS
        with patch(
            "jarvis.reply.enrichment.call_llm_direct",
            return_value=overflow,
        ):
            result = digest_memory_for_query(
                diary_entries=[big_entry], graph_parts=[], **self._base_kwargs()
            )
        assert len(result) <= _DIGEST_MAX_CHARS + 1  # +1 for the ellipsis
        assert result.endswith("…")

    def test_llm_failure_returns_empty(self):
        from jarvis.reply.enrichment import digest_memory_for_query

        big_entry = "[2026-04-20] " + ("x " * 300)
        with patch(
            "jarvis.reply.enrichment.call_llm_direct",
            side_effect=RuntimeError("boom"),
        ):
            result = digest_memory_for_query(
                diary_entries=[big_entry], graph_parts=[], **self._base_kwargs()
            )
        assert result == ""

    def test_batches_when_total_exceeds_cap(self):
        """Dumps larger than _DIGEST_BATCH_MAX_CHARS get split into batches."""
        from jarvis.reply.enrichment import (
            _DIGEST_BATCH_MAX_CHARS,
            digest_memory_for_query,
        )

        # Five entries each ~1000 chars → ~5 KB total, clearly multi-batch.
        entries = [
            f"[2026-04-{10 + i:02d}] " + ("detail " * 140)
            for i in range(5)
        ]
        assert sum(len(e) for e in entries) > _DIGEST_BATCH_MAX_CHARS

        call_count = {"n": 0}

        def fake_llm(**kwargs):
            call_count["n"] += 1
            # Alternate NONE / relevant so we also exercise the filter.
            return "NONE" if call_count["n"] % 2 == 0 else f"Note {call_count['n']}."

        with patch(
            "jarvis.reply.enrichment.call_llm_direct",
            side_effect=fake_llm,
        ):
            result = digest_memory_for_query(
                diary_entries=entries, graph_parts=[], **self._base_kwargs()
            )

        # Multiple batches triggered → multiple LLM calls.
        assert call_count["n"] >= 2
        # Surviving notes are joined; NONE batches drop out.
        assert "Note 1." in result

    def test_graph_parts_alone_produce_digest(self):
        """Graph is in alpha and optional — exercise the graph-only path."""
        from jarvis.reply.enrichment import digest_memory_for_query

        # Pad with enough chars to clear the MIN threshold.
        graph = ["[Preferences > Food] " + ("User loves ramen. " * 40)]
        with patch(
            "jarvis.reply.enrichment.call_llm_direct",
            return_value="User enjoys ramen.",
        ):
            result = digest_memory_for_query(
                diary_entries=[], graph_parts=graph, **self._base_kwargs()
            )
        assert "ramen" in result
