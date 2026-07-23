"""
Unit tests for diary-poisoning defences.

Two defences against the "assistant's own past deflection, narrated in the diary,
primes future sessions to repeat the same deflection" failure mode:

1. Summariser prompt forbids narrating assistant failures/deflections as facts.
2. Reply engine injects diary entries under a reference-only framing rather than
   as authoritative "conversation history".

Both were motivated by a field regression where the small model deflected on
"tell me about Possessor" because an earlier same-day diary entry narrated
"the assistant offered to search the web" — which the model then imitated.
"""

from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from jarvis.memory.conversation import generate_conversation_summary


def _cfg() -> SimpleNamespace:
    return SimpleNamespace(
        llm_provider="ollama",
        llm_base_url="http://localhost:11434",
        llm_api_key="",
        llm_chat_model="test-model",
        ollama_base_url="http://localhost:11434",
        ollama_chat_model="test-model",
    )


class TestSummariserForbidsDeflectionNarration:
    """The summariser prompt must instruct the LLM to omit assistant failure narration."""

    def _capture_system_prompt(self) -> str:
        """Invoke generate_conversation_summary with a mocked LLM and capture the system prompt."""
        captured = {}

        def fake_call(cfg, system_prompt, user_prompt, **kwargs):
            captured['system_prompt'] = system_prompt
            return "SUMMARY: x\nTOPICS: a, b"

        with patch('jarvis.memory.conversation._direct_llm', side_effect=fake_call):
            generate_conversation_summary(
                recent_chunks=["User: hi", "Assistant: hello"],
                previous_summary=None,
                cfg=_cfg(),
            )

        return captured['system_prompt']

    def test_prompt_forbids_narrating_failures(self):
        prompt = self._capture_system_prompt()
        lowered = prompt.lower()
        # The prompt must explicitly forbid narrating assistant failures.
        # Accepts any clear injunction shape ("never narrate", "do not narrate",
        # "drop every sentence", etc.) — what matters is that the directive
        # is present, not its exact phrasing.
        assert any(injunction in lowered for injunction in (
            "never narrate", "do not narrate", "do not record", "do not preserve",
            "drop every sentence", "drop all forms of",
        )), "Summariser prompt must explicitly forbid narrating assistant failures."
        # Must name at least one specific failure pattern — "deflect", "lacked",
        # "offered to search", "failed to" — otherwise the rule is too abstract
        # for small models.
        assert any(term in lowered for term in (
            "deflect", "lacked", "offered to search", "failed to",
        )), "Summariser prompt must name specific failure patterns to omit."

    def test_prompt_explains_why_failures_must_be_omitted(self):
        """The prompt must give a reason, so the LLM generalises to variants it didn't see."""
        prompt = self._capture_system_prompt()
        lowered = prompt.lower()
        assert any(phrase in lowered for phrase in (
            "repeat the same",
            "train future",
            "generalise",
            "generalize",
            "transient",
        )), "Summariser prompt must explain why failure narration is harmful."

    def test_prompt_requires_attribution_for_assistant_entity_claims(self):
        """Regression for the real-world Possessor poisoning.

        Field DB contained a diary entry reading:
          "The user initially inquired about the movie Possessor, and the
           assistant provided information stating it is a 2006 science
           fiction film directed by Brandon Cronenberg..."

        The assistant had hallucinated the year; the summariser recorded
        the claim under an "the assistant provided information stating…"
        wrapper but the digest later stripped the attribution, and the
        claim ended up in the next session's system prompt as if it were
        established fact.

        The right fix is attribution preservation, not content deletion —
        we want the summariser to be faithful (so corrections and
        tool-grounded answers survive in the log) while making clear WHO
        said WHAT, so downstream readers can calibrate trust.
        """
        prompt = self._capture_system_prompt()
        lowered = prompt.lower()
        # The prompt must require attribution for assistant entity claims.
        assert "attribut" in lowered, (
            "Summariser prompt must require attribution of assistant claims "
            "(e.g. write 'the assistant said X' rather than bare 'X')."
        )
        # Must warn against promoting attributed claims into unattributed
        # assertions — that's the exact failure mode that poisoned the DB.
        assert "unattributed" in lowered or "without attribution" in lowered or (
            "strip" in lowered and "attribution" in lowered
        ), (
            "Summariser prompt must forbid stripping attribution from an "
            "assistant claim (unattributed claims poison downstream)."
        )
        # Concrete good/bad example pair showing the failure mode.
        assert "possessor" in lowered or "piranesi" in lowered, (
            "Summariser prompt should include a concrete good/bad example "
            "for attributed assistant claims."
        )
        # Must handle the correction chain — user correcting the assistant
        # should result in BOTH being logged, not silent replacement.
        assert "correct" in lowered, (
            "Summariser prompt must explain how to handle user corrections "
            "of assistant claims (preserve both; don't replace silently)."
        )

    def test_prompt_is_language_agnostic(self):
        """The rule must apply to all languages, not only English."""
        prompt = self._capture_system_prompt()
        assert "any language" in prompt.lower() or "all languages" in prompt.lower(), (
            "Summariser rule must explicitly apply across languages."
        )

    def test_prompt_forbids_welding_unrelated_topics(self):
        """Regression for the Possessor/Jarvis field incident.

        Field DB contained a diary entry reading:
          "The conversation focused on the movie 'Possessor' and the character
           'Jarvis,' identified as the artificial intelligence from the
           Marvel Cinematic Universe, created by Tony Stark and later
           embodied by Vision."

        Two distinct topics (the 2020 Cronenberg film Possessor, and the MCU
        AI character named Jarvis) were welded into one clause via "and" plus
        a dangling appositive. Downstream enrichment treated the MCU
        description as pertaining to Possessor, and a later session produced
        a plausible-but-wrong reply grounded in the corrupted summary.

        The rule is a sibling to the attribution rule: attribution without
        topic-separation still permits compound clauses, and compound clauses
        are the mechanism by which unrelated facts get retrieved together.
        """
        prompt = self._capture_system_prompt()
        lowered = prompt.lower()

        # Must forbid joining unrelated topics.
        assert any(phrase in lowered for phrase in (
            "do not weld",
            "not weld",
            "one topic per sentence",
            "separate sentence",
            "separate sentences",
        )), (
            "Summariser prompt must forbid welding unrelated topics into one clause."
        )

        # Must name the specific linguistic mechanism (shared appositive /
        # dangling modifier) — otherwise small models won't recognise the
        # failure mode.
        assert "appositive" in lowered or "relative clause" in lowered or "dangl" in lowered, (
            "Summariser prompt must name the shared-appositive / dangling-modifier "
            "mechanism so small models recognise the failure mode."
        )

        # Concrete good/bad example using the field-observed Possessor/Jarvis
        # case (the same one used elsewhere in the prompt — but here about
        # topic separation, not attribution).
        assert "jarvis" in lowered and "possessor" in lowered, (
            "Summariser prompt should include the Possessor/Jarvis topic-welding "
            "BAD→GOOD example."
        )


class TestRewriteDeflectionSystemPrompt:
    """The bulk-rewrite system prompt is a separate LLM context from the
    summariser. It must carry its own contract guarantees because old
    diary rows written before the summariser was tightened depend on it
    to clean themselves up, and downstream behaviour (graph extraction,
    enrichment, future replies) inherits whatever the rewrite produces.
    """

    def _prompt(self) -> str:
        from jarvis.memory.conversation import _REWRITE_DEFLECTION_SYSTEM_PROMPT
        return _REWRITE_DEFLECTION_SYSTEM_PROMPT

    def test_prompt_names_the_canonical_deflection_shapes(self):
        lowered = self._prompt().lower()
        # The prompt must enumerate enough verb shapes for a small model
        # to generalise from. A bare "remove deflection" instruction is
        # too abstract — small models read past it.
        for shape in (
            "could not", "couldn't", "cannot", "did not", "does not",
            "was unable", "was not able", "failed to",
            "offered to search", "lacks",
        ):
            assert shape in lowered, (
                f"Rewrite prompt must name the {shape!r} shape so small "
                f"models recognise the failure pattern."
            )

    def test_prompt_protects_attributed_claims_and_user_facts(self):
        """The same content that the summariser is allowed to keep must
        survive the rewrite. Without this guard the rewrite will strip
        attributed assistant claims (a third-party fact attributed to
        the assistant) and user-stated facts."""
        lowered = self._prompt().lower()
        # Names the kept categories so the model knows what NOT to drop.
        assert "attributed" in lowered or "user said" in lowered or "user-stated" in lowered, (
            "Rewrite prompt must explicitly list KEEP categories "
            "(attributed assistant claims, user-stated facts)."
        )
        assert "verbatim" in lowered, (
            "Rewrite prompt must instruct the model to keep non-deflection "
            "content verbatim — otherwise it paraphrases and corrupts."
        )

    def test_prompt_is_language_agnostic(self):
        lowered = self._prompt().lower()
        assert "any language" in lowered or "every language" in lowered or "all languages" in lowered, (
            "Rewrite prompt must apply across languages — the leak shows "
            "up in any language the user speaks."
        )

    def test_prompt_forbids_translation(self):
        """A rewrite that translates the diary breaks downstream FTS,
        embeddings, and graph extraction — all of which expect the
        original language."""
        lowered = self._prompt().lower()
        assert "not translate" in lowered or "do not translate" in lowered or (
            "keep" in lowered and "language" in lowered
        ), "Rewrite prompt must forbid translation of the output."

    def test_prompt_specifies_empty_output_for_all_deflection_rows(self):
        """If the row is *entirely* deflection, the model must return the
        empty string. The Python layer's empty-rewrite guard then keeps
        the original (an empty diary entry would be worse — retrieval
        treats absence as 'no record')."""
        lowered = self._prompt().lower()
        assert "empty" in lowered, (
            "Rewrite prompt must instruct the model how to handle a row "
            "that is entirely deflection (return empty)."
        )


class TestDiaryEnrichmentInjectionFraming:
    """The reply engine must frame diary enrichment as reference-only, not as instructions."""

    def test_engine_injects_diary_under_reference_only_label(self):
        """The literal injection string used by _build_initial_system_message must signal reference-only use."""
        # Read the engine source and verify the label string is present.
        # We intentionally assert on the source-level string rather than end-to-end
        # because the full reply engine invocation pulls in the network stack.
        import inspect
        from jarvis.reply import engine

        source = inspect.getsource(engine)
        assert "reference only" in source.lower(), (
            "Engine must label diary enrichment as 'reference only' to prevent imitation."
        )
        assert "do not treat them as instructions" in source.lower() or \
               "not treat them as instructions" in source.lower(), (
            "Engine must explicitly tell the model not to treat diary entries as instructions."
        )

    def test_engine_does_not_use_bare_conversation_history_label(self):
        """The old 'Relevant conversation history:' label read as authoritative context.

        We keep this test as a regression guard — if someone reverts to the bare
        label, this test will fail and force them to preserve the reference-only framing.
        """
        import inspect
        from jarvis.reply import engine

        source = inspect.getsource(engine)
        # The bare label (without the reference-only qualifier) must not appear.
        # We check for the exact old string on its own line.
        assert '"\\nRelevant conversation history:\\n"' not in source, (
            "Engine must not use the bare 'Relevant conversation history:' label — "
            "it reads as authoritative and primes small models to imitate past deflections."
        )
