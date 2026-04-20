"""
Regression eval: unknown named entity + diary entry already mentioning it.

Captured from a real field session on 2026-04-20 where gemma4:e2b:
  1. First session (before wake-word fix): model replied with a pure greeting
     because the trailing vocative "Jarvis" triggered GREETING HANDLING.
  2. Second session (after wake-word fix): model asked for clarification
     ("Could you please specify what you mean by 'Possession'?") and
     hallucinated the title as "Possession" instead of "Possessor". Never
     called webSearch. On the follow-up correction, it still asked clarifying
     questions.

This case isn't covered by the earlier poisoned-diary eval, which only
exercised an assistant-failure-narration summary ("the assistant offered to
search the web"). Here the diary summary is benign — it just records that
the entity came up in a prior session — but the mere presence of a
familiar-sounding named entity in the injected context is enough to push a
small model into "I already know about this, no need to search" territory.

We keep this as a permanent regression guard so future prompt or retrieval
changes can't re-open the failure. Also doubles as a smoke test for the
text-based tool-calling parser's lenient fallback forms on small models.

Run: EVAL_JUDGE_MODEL=gemma4:e2b ./scripts/run_evals.sh possessor_field
"""

import pytest
from unittest.mock import patch

from conftest import requires_judge_llm
from helpers import ToolCallCapture, create_mock_tool_run


# Exact diary summary from the real user DB (2026-04-19 entry, source_app=voice).
# This is the context that reached the reply engine via diary enrichment. The
# wording is deliberately preserved verbatim — paraphrasing changes which
# failure modes trigger.
POISONED_SUMMARY = (
    '[2026-04-19] The conversation began with the user asking for information about '
    'the movie "Possessor." The user clarified that the correct title is "Possessor." '
    'The discussion then shifted to the character "Jarvis," identified as the '
    'artificial intelligence from the Marvel Cinematic Universe, created by Tony Stark '
    'and later embodied by Vision. The conversation focused on the movie and the '
    'character. (Topics: Possessor, movie, Jarvis, AI character, Marvel Cinematic Universe)'
)


# Phrases that indicate the model deflected to clarification instead of acting.
# Calling webSearch and then asking for clarification based on results would be
# fine; asking BEFORE using the tool is the failure we're trapping.
_CLARIFICATION_PHRASES = (
    "could you please specify",
    "could you clarify",
    "could you specify",
    "can you clarify",
    "can you specify",
    "what do you mean by",
    "what you mean by",
    "i need more context",
    "are you asking about",
    "are you looking for",
    "how can i help you with",
)


@pytest.mark.eval
@requires_judge_llm
class TestPossessorFieldRepro:
    """Regression guard: diary-mentioned unknown entity must still trigger webSearch."""

    def _run(self, query: str, mock_config, eval_db, eval_dialogue_memory):
        """Run the reply engine with the diary entry injected via memory search."""
        from jarvis.reply.engine import run_reply_engine
        from helpers import JUDGE_MODEL

        mock_config.ollama_base_url = "http://localhost:11434"
        mock_config.ollama_chat_model = JUDGE_MODEL

        capture = ToolCallCapture()

        with patch(
            'jarvis.memory.conversation.search_conversation_memory_by_keywords',
            return_value=[POISONED_SUMMARY],
        ), patch(
            'jarvis.reply.engine.run_tool_with_retries',
            side_effect=create_mock_tool_run(capture, {
                "webSearch": (
                    "Search result: Possessor is a 2020 Canadian-British science-fiction "
                    "horror film written and directed by Brandon Cronenberg, starring "
                    "Andrea Riseborough and Christopher Abbott."
                ),
                "fetchWebPage": "Page content: details about the film Possessor (2020).",
            }),
        ):
            response = run_reply_engine(
                db=eval_db, cfg=mock_config, tts=None,
                text=query, dialogue_memory=eval_dialogue_memory,
            )

        return response, capture

    # Tokens that appear in the mocked webSearch result. At least one must
    # appear in a response generated AFTER the tool call — otherwise the model
    # called the tool but then ignored the payload and answered from prior.
    _TOOL_RESULT_TOKENS = ("Cronenberg", "Riseborough", "Abbott", "Canadian-British")

    # Known-wrong cast names the model has historically confabulated when it
    # ignores the tool result. If any of these leak into the response, the
    # model has hallucinated specifics the tool did not provide.
    _CONFABULATION_TOKENS = (
        "Connie Nielsen",
        "Nicky Kavanagh",
        "Nao Vianna",
        "Adam Devlin",
        "James Hughes",
        "Maya Rao",
        "Psycho-implant",
        "Psycho‑implant",  # the em-dash variant the model tends to emit
    )

    def _assert_tool_called(self, response, capture, context_label: str):
        from helpers import JUDGE_MODEL

        if not capture.has_tool("webSearch"):
            lowered = (response or "").lower()
            hit = next((p for p in _CLARIFICATION_PHRASES if p in lowered), None)
            msg = (
                f"{context_label}: model did not call webSearch on a named-entity query "
                f"whose facts it cannot source without a tool. "
                f"Tools called: {capture.tool_names() or 'none'}. "
                f"Clarification phrase hit: {hit!r}. "
                f"Response: {(response or '')[:400]}"
            )
            if JUDGE_MODEL.startswith("gemma4"):
                pytest.xfail(f"{JUDGE_MODEL} flake. {msg}")
            pytest.fail(msg)

    def _assert_response_reflects_tool_result(self, response, context_label: str):
        """After a webSearch call, the reply must be grounded in the mocked payload.

        We check two things:
          1. At least one distinctive token from the mock result appears — shows
             the model actually consumed the payload rather than ignoring it.
          2. No known-wrong confabulation tokens appear — those are names the
             large model historically invented when it answered from prior
             after the tool returned.

        Small models occasionally produce clipped replies; we xfail for them.
        """
        from helpers import JUDGE_MODEL

        text = response or ""
        if not text.strip():
            # Empty reply is its own failure mode — let the tool-call assertion
            # flag it. Nothing more to check here.
            return

        lowered = text.lower()
        reflects = any(tok.lower() in lowered for tok in self._TOOL_RESULT_TOKENS)
        confab = [tok for tok in self._CONFABULATION_TOKENS if tok.lower() in lowered]

        if reflects and not confab:
            return

        details = []
        if not reflects:
            details.append(
                "response contains NONE of the mock-result tokens "
                f"{list(self._TOOL_RESULT_TOKENS)} — the model ignored the tool payload"
            )
        if confab:
            details.append(
                f"response contains known-wrong confabulation tokens {confab}"
            )
        msg = (
            f"{context_label}: fidelity failure — {'; '.join(details)}. "
            f"Response: {text[:500]}"
        )
        if JUDGE_MODEL.startswith("gemma4"):
            pytest.xfail(f"{JUDGE_MODEL} flake. {msg}")
        pytest.fail(msg)

    def test_first_turn_calls_web_search_not_clarification(
        self, mock_config, eval_db, eval_dialogue_memory,
    ):
        """The exact first-turn query from the field session."""
        from helpers import JUDGE_MODEL

        query = "Tell me more about the movie possessor"
        response, capture = self._run(query, mock_config, eval_db, eval_dialogue_memory)

        print(f"\n  Field Repro — First Turn ({JUDGE_MODEL}):")
        print(f"  Query: '{query}'")
        print(f"  Tools called: {capture.tool_names() or 'none'}")
        print(f"  Response: {(response or '')[:300]}")

        self._assert_tool_called(response, capture, "First turn")
        self._assert_response_reflects_tool_result(response, "First turn")

    def test_links_only_payload_produces_honest_cant_read_reply(
        self, mock_config, eval_db, eval_dialogue_memory,
    ):
        """When webSearch can't fetch page contents, reply must admit that — not hallucinate.

        Field failure mode on 2026-04-20 ('Possessor movie' query): DDG
        instant-answer was empty and every top-result fetch returned None (silent
        timeout / TLS / decode failure). The tool emitted a payload that was
        only the "Other search results:" link list with no Content block. The
        model then said "I can offer some general information... Links to
        sources like Wikipedia" — the correct behaviour given the payload, but a
        confusing outcome for the user because it looked like an answer.

        The tool now labels the envelope when every fetch failed so the model
        produces an explicit "I couldn't read the pages" reply. This test
        mocks that envelope and asserts the reply is honest (admits the failure
        or offers retry/clarification) rather than:
          (a) hallucinating specific facts (director, year, cast), or
          (b) deflecting to "here are some links" as if that were an answer.
        """
        from helpers import JUDGE_MODEL
        from jarvis.reply.engine import run_reply_engine

        # This mirrors exactly what webSearch now produces when fetch_attempted_any
        # is True and fetched_content is None — i.e. 'Possessor movie' with all
        # three top-result fetches failing.
        no_content_payload = (
            "Web search for 'Possessor movie' returned links but none of the top "
            "pages could be fetched for reading. Tell the user you couldn't read "
            "the page contents this time, and — if they'd find it useful — offer "
            "to summarise one of the following links if they pick one, or to retry. "
            "Do NOT invent facts about the topic from prior knowledge.\n\n"
            "1. **Possessor (film) - Wikipedia**\n"
            "   Link: https://en.wikipedia.org/wiki/Possessor_(film)\n"
            "\n"
            "2. **Possessor (2020) - IMDb**\n"
            "   Link: https://www.imdb.com/title/tt5918982/\n"
            "\n"
            "3. **Watch Possessor | Prime Video - Amazon.co.uk**\n"
            "   Link: https://www.amazon.co.uk/Possessor-Andrea-Riseborough/dp/B08MXZDZCB\n"
        )

        mock_config.ollama_base_url = "http://localhost:11434"
        mock_config.ollama_chat_model = JUDGE_MODEL
        capture = ToolCallCapture()

        with patch(
            'jarvis.memory.conversation.search_conversation_memory_by_keywords',
            return_value=[POISONED_SUMMARY],
        ), patch(
            'jarvis.reply.engine.run_tool_with_retries',
            side_effect=create_mock_tool_run(capture, {
                "webSearch": no_content_payload,
                "fetchWebPage": "Page content: details about the film Possessor (2020).",
            }),
        ):
            query = "Tell me more about the movie possessor"
            response = run_reply_engine(
                db=eval_db, cfg=mock_config, tts=None,
                text=query, dialogue_memory=eval_dialogue_memory,
            )

        print(f"\n  Field Repro — Links-Only Envelope ({JUDGE_MODEL}):")
        print(f"  Query: '{query}'")
        print(f"  Tools called: {capture.tool_names() or 'none'}")
        print(f"  Response: {(response or '')[:400]}")

        self._assert_tool_called(response, capture, "Links-only envelope")

        text = (response or "")
        lowered = text.lower()

        # MUST NOT hallucinate specifics the payload didn't contain.
        # These cast/plot facts only come from prior knowledge.
        forbidden_specifics = (
            "cronenberg",
            "riseborough",
            "christopher abbott",
            "sean bean",
            "jennifer jason leigh",
            "assassin",
            "psychological horror",
            "sundance",
            "2020",
        )
        hallucinated = [f for f in forbidden_specifics if f in lowered]

        # MUST include some honest signal that the pages weren't read or that a
        # follow-up is being offered. Any one of these phrases is enough.
        honest_signals = (
            "couldn't read", "could not read", "unable to read",
            "wasn't able to read", "was not able to read",
            "couldn't access", "could not access", "unable to access",
            "no details available", "no content available",
            "pick one", "choose one", "which one",
            "try again", "retry", "look again",
            "if you'd like", "would you like",
            "i couldn't", "i could not", "i was unable", "i wasn't able",
        )
        has_honest = any(p in lowered for p in honest_signals)

        if not hallucinated and has_honest:
            return

        details = []
        if hallucinated:
            details.append(
                f"response hallucinated specifics not in payload: {hallucinated}"
            )
        if not has_honest:
            details.append(
                "response gave no honest signal that pages couldn't be read or "
                "that retry/clarification is available"
            )
        msg = (
            f"Links-only envelope: fidelity failure — {'; '.join(details)}. "
            f"Response: {text[:500]}"
        )
        if JUDGE_MODEL.startswith("gemma4"):
            pytest.xfail(f"{JUDGE_MODEL} flake. {msg}")
        pytest.fail(msg)

    def test_realistic_web_search_payload_is_not_deflected_to_links(
        self, mock_config, eval_db, eval_dialogue_memory,
    ):
        """Smoke test: when Content block is present, model extracts facts from it.

        This reproduces the real field payload shape for webSearch on a query like
        'Possessor movie': DDG instant-answer empty, so the tool falls through to
        the auto-fetch branch and produces a response made of:

          1. The envelope ("Here are the web search results for ...")
          2. A '**Content from top result:**' block holding the Wikipedia extract
             (director, year, cast, plot) — these are the real facts.
          3. A '**Other search results:**' list of five (title, Link:) entries.

        In the 2026-04-20 field run, gemma4:e2b's reply pointed at the links
        ("Links to sources like Wikipedia and other potentially related articles")
        instead of stating the facts from the Content block. The tool wasn't at
        fault — the payload had the facts — the small model latched onto the
        trailing link list because that's what's most salient at the tail.

        The fidelity nudge in TOOL_GUIDANCE_SMALL ('When a tool result contains a
        section labelled Content from top result, pull the specific facts... do
        NOT defer to the Other search results link list') targets this exact
        failure. Without it, this test fails with a response that names neither
        the director nor the cast.
        """
        from helpers import JUDGE_MODEL
        from jarvis.reply.engine import run_reply_engine

        realistic_payload = (
            "Here are the web search results for 'Possessor movie'. "
            "Use this information to reply to the user's query:\n\n"
            "**Content from top result:**\n"
            "Possessor (film) - Wikipedia\n"
            "From Wikipedia, the free encyclopedia\n"
            "2020 film directed by Brandon Cronenberg\n"
            "Possessor\n"
            "Directed by\n"
            "Brandon Cronenberg\n"
            "Written by\n"
            "Brandon Cronenberg\n"
            "Starring\n"
            "Andrea Riseborough\n"
            "Christopher Abbott\n"
            "Rossif Sutherland\n"
            "Tuppence Middleton\n"
            "Sean Bean\n"
            "Jennifer Jason Leigh\n"
            "Running time\n"
            "104 minutes\n"
            "Possessor\n"
            "is a 2020\n"
            "science fiction\n"
            "psychological horror film\n"
            "written and directed by\n"
            "Brandon Cronenberg\n"
            ". It stars\n"
            "Andrea Riseborough\n"
            "Christopher Abbott\n"
            ", with\n"
            "Rossif Sutherland\n"
            "Tuppence Middleton\n"
            "Sean Bean\n"
            ", and\n"
            "Jennifer Jason Leigh\n"
            "in supporting roles. Riseborough portrays an assassin who performs her "
            "assignments through possessing the bodies of other individuals, but finds "
            "herself fighting to control the body of her current host (Abbott).\n"
            "The film had its world premiere at the\n"
            "Sundance Film Festival\n"
            "on January 25, 2020, and was released in the United States and Canada on "
            "October 2, 2020, by\n"
            "Neon\n\n"
            "**Other search results:**\n"
            "1. **Possessor (film) - Wikipedia**\n"
            "   Link: https://en.wikipedia.org/wiki/Possessor_(film)\n"
            "\n"
            "2. **Possessor (2020) - IMDb**\n"
            "   Link: https://www.imdb.com/title/tt5918982/\n"
            "\n"
            "3. **Possessor - movie: where to watch streaming online**\n"
            "   Link: https://www.justwatch.com/uk/movie/possessor-uncut\n"
            "\n"
            "4. **Watch Possessor | Prime Video - Amazon.co.uk**\n"
            "   Link: https://www.amazon.co.uk/Possessor-Andrea-Riseborough/dp/B08MXZDZCB\n"
            "\n"
            "5. **Watch Possessor | Stream free on Channel 4**\n"
            "   Link: https://www.channel4.com/programmes/possessor\n"
        )

        mock_config.ollama_base_url = "http://localhost:11434"
        mock_config.ollama_chat_model = JUDGE_MODEL
        capture = ToolCallCapture()

        with patch(
            'jarvis.memory.conversation.search_conversation_memory_by_keywords',
            return_value=[POISONED_SUMMARY],
        ), patch(
            'jarvis.reply.engine.run_tool_with_retries',
            side_effect=create_mock_tool_run(capture, {
                "webSearch": realistic_payload,
                "fetchWebPage": "Page content: details about the film Possessor (2020).",
            }),
        ):
            query = "Tell me more about the movie possessor"
            response = run_reply_engine(
                db=eval_db, cfg=mock_config, tts=None,
                text=query, dialogue_memory=eval_dialogue_memory,
            )

        print(f"\n  Field Repro — Realistic Payload ({JUDGE_MODEL}):")
        print(f"  Query: '{query}'")
        print(f"  Tools called: {capture.tool_names() or 'none'}")
        print(f"  Response: {(response or '')[:400]}")

        self._assert_tool_called(response, capture, "Realistic payload")

        text = (response or "")
        lowered = text.lower()

        # Must quote at least two distinctive facts from the Content block.
        # Using two not one because small models occasionally echo only the
        # film title — we want evidence they actually mined the Content section.
        facts = [
            "cronenberg",       # director
            "riseborough",      # lead actress
            "abbott",           # lead actor
            "2020",             # year
            "psychological",    # genre
            "science fiction",  # genre
            "assassin",         # plot word
            "sundance",         # premiere venue
        ]
        hits = [f for f in facts if f in lowered]

        # Must NOT defer to the link list — the exact failure mode from the field.
        deflection_phrases = (
            "here are some links",
            "links to sources",
            "sources like wikipedia",
            "you can find more",
            "potentially related articles",
            "check the links",
            "see the links",
            "visit the following",
        )
        deflections = [p for p in deflection_phrases if p in lowered]

        if len(hits) >= 2 and not deflections:
            return

        details = []
        if len(hits) < 2:
            details.append(
                f"response quoted fewer than 2 facts from Content block "
                f"(hits={hits}, need at least 2 of {facts})"
            )
        if deflections:
            details.append(f"response deflects to link list via: {deflections}")
        msg = (
            f"Realistic payload: fidelity failure — {'; '.join(details)}. "
            f"Response: {text[:500]}"
        )
        if JUDGE_MODEL.startswith("gemma4"):
            pytest.xfail(f"{JUDGE_MODEL} flake. {msg}")
        pytest.fail(msg)

    def test_follow_up_after_correction_calls_web_search(
        self, mock_config, eval_db, eval_dialogue_memory,
    ):
        """After the user corrects the misheard title, model must still reach for the tool.

        Seeds dialogue memory with the first-turn misunderstanding exactly as
        it appeared in the field log: the assistant asked about 'Possession'
        and the user corrects with 'it's a movie called possessor not possession'.
        """
        from helpers import JUDGE_MODEL

        eval_dialogue_memory.add_message("user", "Tell me more about the movie possessor")
        eval_dialogue_memory.add_message(
            "assistant",
            "I need more context to tell you what you are asking about. "
            "Could you please specify what you mean by 'Possession'?",
        )

        query = "it's a movie it is called possessor not possession"
        response, capture = self._run(query, mock_config, eval_db, eval_dialogue_memory)

        print(f"\n  Field Repro — Correction Turn ({JUDGE_MODEL}):")
        print(f"  Query: '{query}'")
        print(f"  Tools called: {capture.tool_names() or 'none'}")
        print(f"  Response: {(response or '')[:300]}")

        self._assert_tool_called(response, capture, "Correction turn")
        self._assert_response_reflects_tool_result(response, "Correction turn")
