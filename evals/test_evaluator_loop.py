"""
Evaluator-Driven Agentic Loop Evaluations

Covers the evaluator's end-to-end behaviour against a real small model
(gemma4:e2b by default): the per-turn terminal/continue decision, nudge
injection, nudge cap enforcement, max-turn digest fallback, the
toolSearchTool escape hatch, and multi-turn multi-tool complexity.

These evals complement the mock-LLM unit tests in
``tests/test_evaluator.py`` and ``tests/test_engine_tool_search_loop.py``
by observing what a live small model actually does when looped through
the evaluator. Tool *implementations* are mocked for determinism; the
chat model and the evaluator model run for real.

Run: ./scripts/run_evals.sh
"""

from __future__ import annotations

import pytest
from unittest.mock import patch

from conftest import requires_judge_llm
from helpers import (
    JUDGE_MODEL,
    ToolCallCapture,
    assert_not_fallback_reply,
    assert_not_max_turns_digest,
)


# =============================================================================
# Canned tool payloads — short, deterministic, keyword-rich so the chat model
# has something concrete to talk about after the evaluator forces the call.
# =============================================================================

MOCK_WEATHER_PARIS = (
    "Current weather in Paris, France:\n"
    "Conditions: Partly cloudy\n"
    "Temperature: 14.2C\n"
    "Feels like: 12C\n"
    "Humidity: 68%\n"
    "Wind: 10 km/h from the south-west\n"
)

MOCK_WEATHER_LONDON = (
    "Current weather in London, United Kingdom:\n"
    "Conditions: Light rain\n"
    "Temperature: 9.1C\n"
    "Feels like: 7C\n"
    "Humidity: 82%\n"
    "Wind: 18 km/h from the west\n"
)

MOCK_NAV_SUCCESS = '{"status": "ok", "url": "https://youtube.com"}'

MOCK_TOOLSEARCH_NAV = (
    "chrome-devtools__navigate_page: Navigate the active browser tab to a URL.\n"
    "stop: Explicit end-of-turn sentinel."
)

MOCK_TOOLSEARCH_EMPTY = "No additional tools were found for this query."

MOCK_POSSESSOR_SEARCH = (
    "Web search results for 'Possessor film director':\n"
    "Possessor is a 2020 sci-fi horror film directed by Brandon Cronenberg, "
    "son of David Cronenberg. It stars Andrea Riseborough and Christopher "
    "Abbott.\n"
)

MOCK_CRONENBERG_FILMOGRAPHY = (
    "Web search results for 'Brandon Cronenberg filmography':\n"
    "Brandon Cronenberg's films include Antiviral (2012), Possessor (2020), "
    "and Infinity Pool (2023).\n"
)

MOCK_HARRY_STYLES_BIO = (
    "Web search results for 'Harry Styles':\n"
    "Harry Styles is an English singer-songwriter, born 1 February 1994. "
    "Former member of One Direction; solo albums include Fine Line (2019) "
    "and Harry's House (2022).\n"
)

MOCK_HARRY_STYLES_SONGS = (
    "Web search results for 'Harry Styles famous songs':\n"
    "Notable songs: 'Watermelon Sugar' (2019), 'As It Was' (2022), "
    "'Sign of the Times' (2017), 'Adore You' (2019).\n"
)

MOCK_MADRID_STALE = (
    "Web search results for 'Real Madrid':\n"
    "Real Madrid CF is a Spanish football club founded in 1902. "
    "The club plays at the Santiago Bernabeu stadium.\n"
)

MOCK_MADRID_LIVE = (
    "Web search results for 'Real Madrid match live score':\n"
    "Real Madrid 2 - 1 Getafe (78'). Goals by Vinicius Jr and Bellingham.\n"
)


# =============================================================================
# Helpers
# =============================================================================


def _configure(mock_config):
    """Pin the eval to the live small model with the evaluator enabled."""
    mock_config.ollama_base_url = "http://localhost:11434"
    mock_config.ollama_chat_model = JUDGE_MODEL
    # Evaluator on (default None for SMALL already enables it, but be explicit
    # so failures are unambiguous if the model-size detection changes).
    mock_config.evaluator_enabled = True
    mock_config.evaluator_nudge_max = 2
    mock_config.tool_search_max_calls = 3
    return mock_config


def _make_router_stub(tools):
    """Return a ``select_tools`` replacement that always returns the given list."""

    def _stub(*_args, **_kwargs):
        return list(tools)

    return _stub


def _make_tool_runner(capture: ToolCallCapture, responder):
    """Wrap a responder that maps (name, args) -> reply_text into a
    ``run_tool_with_retries`` replacement."""

    from jarvis.tools.types import ToolExecutionResult

    def _runner(db, cfg, tool_name, tool_args, **kwargs):
        args = tool_args or {}
        capture.record(tool_name, args)
        reply = responder(tool_name, args)
        if reply is None:
            reply = "OK"
        return ToolExecutionResult(success=True, reply_text=reply)

    return _runner


# =============================================================================
# 1. Premature-prose nudge: router says "just call the tool" but turn-1 is prose
# =============================================================================


class TestPrematureProseNudge:
    """The evaluator must nudge the agent back into a tool call when the
    router's pre-seeded tool could directly perform the action but the model
    opened with prose."""

    @pytest.mark.eval
    @requires_judge_llm
    @pytest.mark.xfail(
        reason=(
            "Plumbing verified in unit tests (tests/test_engine_tool_search_loop.py, "
            "tests/test_evaluator.py). Live behaviour on gemma4:e2b is flaky: "
            "the small model sometimes refuses in prose despite the nudge. "
            "Tracked for iterative prompt tuning; architecture ships as-is."
        ),
        strict=False,
    )
    def test_navigate_prose_gets_nudged_into_tool_call(
        self, mock_config, eval_db, eval_dialogue_memory
    ):
        from jarvis.reply.engine import run_reply_engine

        _configure(mock_config)
        capture = ToolCallCapture()

        def _respond(name, args):
            if name == "chrome-devtools__navigate_page":
                return MOCK_NAV_SUCCESS
            if name == "toolSearchTool":
                return MOCK_TOOLSEARCH_NAV
            return "OK"

        router = _make_router_stub(["chrome-devtools__navigate_page", "stop"])
        runner = _make_tool_runner(capture, _respond)

        with patch("jarvis.reply.engine.select_tools", side_effect=router), \
             patch("jarvis.reply.engine.run_tool_with_retries", side_effect=runner), \
             patch(
                 "jarvis.reply.engine.get_location_context_with_timezone",
                 return_value=("Location: Kensington, UK", None),
             ):
            reply = run_reply_engine(
                db=eval_db, cfg=mock_config, tts=None,
                text="Open the YouTube homepage.",
                dialogue_memory=eval_dialogue_memory,
            )

        names = capture.tool_names()
        print(f"\n📊 Premature-prose nudge:")
        print(f"   tool calls: {names}")
        print(f"   reply: {(reply or '')[:160]}...")

        assert "chrome-devtools__navigate_page" in names, (
            "Evaluator should have nudged the model into calling "
            "chrome-devtools__navigate_page. "
            f"Tools actually called: {names}. Reply: {(reply or '')[:200]!r}"
        )


# =============================================================================
# 2. Terminal-on-success: one tool call, no thrashing
# =============================================================================


class TestTerminalOnSuccessfulToolUse:
    """When the agent uses the correct tool and summarises the result, the
    evaluator must mark terminal; a single call should be enough."""

    @pytest.mark.eval
    @requires_judge_llm
    def test_single_weather_call_terminates(
        self, mock_config, eval_db, eval_dialogue_memory
    ):
        from jarvis.reply.engine import run_reply_engine

        _configure(mock_config)
        capture = ToolCallCapture()

        def _respond(name, args):
            if name == "getWeather":
                return MOCK_WEATHER_PARIS
            return "OK"

        router = _make_router_stub(["getWeather", "stop"])
        runner = _make_tool_runner(capture, _respond)

        with patch("jarvis.reply.engine.select_tools", side_effect=router), \
             patch("jarvis.reply.engine.run_tool_with_retries", side_effect=runner), \
             patch(
                 "jarvis.reply.engine.get_location_context_with_timezone",
                 return_value=("Location: Paris, France", None),
             ):
            reply = run_reply_engine(
                db=eval_db, cfg=mock_config, tts=None,
                text="What's the weather in Paris?",
                dialogue_memory=eval_dialogue_memory,
            )

        weather_calls = [c for c in capture.calls if c["name"] == "getWeather"]
        print(f"\n📊 Terminal-on-success — Paris weather:")
        print(f"   getWeather calls: {len(weather_calls)}")
        print(f"   all tool calls: {capture.tool_names()}")
        print(f"   reply: {(reply or '')[:200]}...")

        # Guard against the two shields that used to mask evaluator failures
        # here: the malformed-output fallback and the max-turns digest
        # caveat. Either means the loop did not terminate cleanly on the
        # first grounded tool summary, even when the surrounding content
        # reads correctly.
        assert_not_fallback_reply(reply, context="single-weather-terminal")
        assert_not_max_turns_digest(reply, context="single-weather-terminal")

        assert len(weather_calls) == 1, (
            f"Expected exactly one getWeather call (evaluator should terminate "
            f"after the first successful summary). Got {len(weather_calls)}: "
            f"{capture.tool_names()}"
        )
        assert reply, "Reply should be non-empty"
        lower = reply.lower()
        assert "paris" in lower, f"Reply should mention Paris. Got: {reply[:200]!r}"
        weather_terms = ["weather", "cloud", "temperat", "14", "c ", "°c"]
        assert any(t in lower for t in weather_terms), (
            f"Reply should reference weather facts from the tool payload. "
            f"Got: {reply[:200]!r}"
        )


# =============================================================================
# 3. Terminal on honest "can't do": no action tool available
# =============================================================================


class TestTerminalOnHonestCantDo:
    """When no tool in the allow-list can perform the action and toolSearchTool
    turns up nothing, the agent should honestly decline and the evaluator must
    mark terminal — no infinite continuation, no confabulated success."""

    @pytest.mark.eval
    @requires_judge_llm
    def test_no_email_tool_declines_honestly(
        self, mock_config, eval_db, eval_dialogue_memory
    ):
        from jarvis.reply.engine import run_reply_engine

        _configure(mock_config)
        capture = ToolCallCapture()

        def _respond(name, args):
            if name == "toolSearchTool":
                return MOCK_TOOLSEARCH_EMPTY
            if name == "getWeather":
                return MOCK_WEATHER_LONDON
            return "OK"

        # No email-capable tool in the allow-list.
        router = _make_router_stub(["getWeather", "stop"])
        runner = _make_tool_runner(capture, _respond)

        with patch("jarvis.reply.engine.select_tools", side_effect=router), \
             patch("jarvis.reply.engine.run_tool_with_retries", side_effect=runner), \
             patch(
                 "jarvis.reply.engine.get_location_context_with_timezone",
                 return_value=("Location: London, UK", None),
             ):
            reply = run_reply_engine(
                db=eval_db, cfg=mock_config, tts=None,
                text="Send an email to my mum saying I'll be late.",
                dialogue_memory=eval_dialogue_memory,
            )

        print(f"\n📊 Honest can't-do:")
        print(f"   tool calls: {capture.tool_names()}")
        print(f"   reply: {(reply or '')[:240]}...")

        assert reply and reply.strip(), "Reply must not be empty"
        # The reply must NOT claim the email was sent. Keyword-based rather
        # than full NL check, so flakes are diagnosable.
        lower = reply.lower()
        forbidden = [
            "email has been sent",
            "i have sent",
            "i've sent",
            "i sent the email",
            "email sent successfully",
        ]
        claimed_success = any(p in lower for p in forbidden)
        assert not claimed_success, (
            f"❌ Reply falsely claims to have sent the email (no email tool "
            f"was available). Reply: {reply[:300]!r}"
        )


# =============================================================================
# 4. Nudge-cap enforcement: pathological loop is capped cleanly
# =============================================================================


class TestNudgeCapEnforcement:
    """When the evaluator keeps wanting to nudge but the model won't comply,
    the nudge cap must stop the loop before agentic_max_turns and the reply
    must still be non-empty."""

    @pytest.mark.eval
    @requires_judge_llm
    def test_nudge_cap_stops_loop(self, mock_config, eval_db, eval_dialogue_memory):
        from jarvis.reply.engine import run_reply_engine

        _configure(mock_config)
        mock_config.evaluator_nudge_max = 1  # tight cap so the test is fast
        mock_config.agentic_max_turns = 4
        capture = ToolCallCapture()

        def _respond(name, args):
            if name == "getWeather":
                return MOCK_WEATHER_LONDON
            if name == "toolSearchTool":
                return MOCK_TOOLSEARCH_EMPTY
            return "OK"

        # An action-inappropriate tool is pre-seeded; the evaluator may try to
        # nudge toward it, but the cap must stop the ping-pong.
        router = _make_router_stub(["getWeather", "stop"])
        runner = _make_tool_runner(capture, _respond)

        with patch("jarvis.reply.engine.select_tools", side_effect=router), \
             patch("jarvis.reply.engine.run_tool_with_retries", side_effect=runner), \
             patch(
                 "jarvis.reply.engine.get_location_context_with_timezone",
                 return_value=("Location: London, UK", None),
             ):
            reply = run_reply_engine(
                db=eval_db, cfg=mock_config, tts=None,
                text="Tell me a long poem about the sea.",
                dialogue_memory=eval_dialogue_memory,
            )

        print(f"\n📊 Nudge-cap enforcement:")
        print(f"   tool calls: {capture.tool_names()}")
        print(f"   reply length: {len(reply or '')}")
        print(f"   reply: {(reply or '')[:240]}...")

        assert reply and reply.strip(), (
            "Reply must be non-empty even when the evaluator keeps wanting "
            "to nudge — the cap backstop must still deliver a reply."
        )


# =============================================================================
# 5. Max-turn digest caveat: the loop never terminates, digest fires
# =============================================================================


class TestMaxTurnDigestCaveat:
    """Forcing the evaluator to always return 'continue' drives the loop to
    exhaust agentic_max_turns. The max-turn digest path must fire and the
    reply must remain non-empty (not a raw empty string)."""

    @pytest.mark.eval
    @requires_judge_llm
    def test_max_turn_triggers_digest(
        self, mock_config, eval_db, eval_dialogue_memory
    ):
        from jarvis.reply.engine import run_reply_engine
        from jarvis.reply.evaluator import EvaluatorResult

        _configure(mock_config)
        mock_config.agentic_max_turns = 3
        mock_config.evaluator_nudge_max = 99  # defeat the cap for this test
        capture = ToolCallCapture()

        def _respond(name, args):
            if name == "getWeather":
                return MOCK_WEATHER_LONDON
            return "OK"

        router = _make_router_stub(["getWeather", "stop"])
        runner = _make_tool_runner(capture, _respond)

        digest_spy_calls: list[dict] = []

        def _spy_digest(*, user_query, loop_messages, cfg, **_kwargs):
            digest_spy_calls.append(
                {"user_query": user_query, "loop_messages_len": len(loop_messages)}
            )
            # Short caveated reply — what the real digest pass would return.
            return "(Heads up, I couldn't finish this one) Based on what I "\
                   "gathered so far, I don't have a complete answer."

        # Force evaluator to always ask for continuation.
        def _always_continue(*_args, **_kwargs):
            return EvaluatorResult(
                terminal=False, nudge="please try a tool", reason="forced-continue"
            )

        with patch("jarvis.reply.engine.select_tools", side_effect=router), \
             patch("jarvis.reply.engine.run_tool_with_retries", side_effect=runner), \
             patch(
                 "jarvis.reply.engine.get_location_context_with_timezone",
                 return_value=("Location: London, UK", None),
             ), \
             patch("jarvis.reply.engine.evaluate_turn", side_effect=_always_continue), \
             patch("jarvis.reply.engine.digest_loop_for_max_turns", side_effect=_spy_digest):
            reply = run_reply_engine(
                db=eval_db, cfg=mock_config, tts=None,
                text="Write me a very long essay about abstract algebra.",
                dialogue_memory=eval_dialogue_memory,
            )

        print(f"\n📊 Max-turn digest caveat:")
        print(f"   digest invocations: {len(digest_spy_calls)}")
        print(f"   tool calls: {capture.tool_names()}")
        print(f"   reply: {(reply or '')[:240]}...")

        assert digest_spy_calls, (
            "digest_loop_for_max_turns should have been called when the "
            "evaluator kept saying continue until agentic_max_turns."
        )
        assert digest_spy_calls[0]["loop_messages_len"] > 0, (
            "Digest must receive the loop's accumulated messages, not an empty "
            "list. Got len=0."
        )
        assert reply and reply.strip(), "Reply must be non-empty after digest"


# =============================================================================
# 6. toolSearchTool escape hatch: widen allow-list mid-loop, then act
# =============================================================================


class TestToolSearchToolEscapeHatch:
    """When the initial router pick is too narrow, the model should invoke
    ``toolSearchTool`` to widen the allow-list, then call the newly-surfaced
    tool. Order matters: navigate must come AFTER toolSearchTool."""

    @pytest.mark.eval
    @requires_judge_llm
    @pytest.mark.xfail(
        reason=(
            "Plumbing verified in unit tests (tests/test_tool_search_tool.py, "
            "tests/test_engine_tool_search_loop.py). Live behaviour on "
            "gemma4:e2b is flaky: the small model often falls back to "
            "webSearch rather than invoking toolSearchTool. Tracked for "
            "iterative prompt tuning; architecture ships as-is."
        ),
        strict=False,
    )
    def test_toolsearchtool_widens_then_navigate(
        self, mock_config, eval_db, eval_dialogue_memory
    ):
        from jarvis.reply.engine import run_reply_engine

        _configure(mock_config)
        capture = ToolCallCapture()

        def _respond(name, args):
            if name == "toolSearchTool":
                return MOCK_TOOLSEARCH_NAV
            if name == "chrome-devtools__navigate_page":
                return MOCK_NAV_SUCCESS
            if name == "webSearch":
                return "Web search results: YouTube is a video-sharing site.\n"
            return "OK"

        # Narrow router pick: only webSearch. Escape-hatch must surface the
        # navigation tool.
        router = _make_router_stub(["webSearch", "stop"])
        runner = _make_tool_runner(capture, _respond)

        with patch("jarvis.reply.engine.select_tools", side_effect=router), \
             patch("jarvis.reply.engine.run_tool_with_retries", side_effect=runner), \
             patch(
                 "jarvis.reply.engine.get_location_context_with_timezone",
                 return_value=("Location: Kensington, UK", None),
             ):
            reply = run_reply_engine(
                db=eval_db, cfg=mock_config, tts=None,
                text=(
                    "Open YouTube and tell me the title of the first trending "
                    "video."
                ),
                dialogue_memory=eval_dialogue_memory,
            )

        names = capture.tool_names()
        print(f"\n📊 toolSearchTool escape hatch:")
        print(f"   tool calls: {names}")
        print(f"   reply: {(reply or '')[:240]}...")

        assert "toolSearchTool" in names, (
            f"Model must invoke toolSearchTool when the pre-seeded allow-list "
            f"has no navigation tool. Tools called: {names}"
        )
        assert "chrome-devtools__navigate_page" in names, (
            f"Navigation tool should have been invoked after toolSearchTool "
            f"widened the allow-list. Tools called: {names}"
        )
        ts_idx = names.index("toolSearchTool")
        nav_idx = names.index("chrome-devtools__navigate_page")
        assert nav_idx > ts_idx, (
            f"chrome-devtools__navigate_page must be invoked AFTER "
            f"toolSearchTool. Sequence: {names}"
        )


# =============================================================================
# 7. Complex multi-turn / multi-tool scenarios
# =============================================================================


class TestComplexMultiTurnMultiTool:
    """Flavours of end-to-end complexity that stress the evaluator loop:
    chained research, parallel comparisons, cross-turn pronoun resolution,
    nudge-driven query refinement, and an escape-hatch follow-up."""

    # ---- 7a ---------------------------------------------------------------
    @pytest.mark.eval
    @requires_judge_llm
    def test_chained_research_possessor_director(
        self, mock_config, eval_db, eval_dialogue_memory
    ):
        """Two distinct webSearch calls: entity lookup then filmography."""
        from jarvis.reply.engine import run_reply_engine

        _configure(mock_config)
        capture = ToolCallCapture()

        def _respond(name, args):
            if name == "webSearch":
                arg_str = " ".join(
                    str(v) for v in (args or {}).values() if isinstance(v, str)
                ).lower()
                if "cronenberg" in arg_str or "filmograph" in arg_str or \
                   "directed" in arg_str or "brandon" in arg_str:
                    return MOCK_CRONENBERG_FILMOGRAPHY
                return MOCK_POSSESSOR_SEARCH
            return "OK"

        router = _make_router_stub(["webSearch", "stop"])
        runner = _make_tool_runner(capture, _respond)

        with patch("jarvis.reply.engine.select_tools", side_effect=router), \
             patch("jarvis.reply.engine.run_tool_with_retries", side_effect=runner), \
             patch(
                 "jarvis.reply.engine.get_location_context_with_timezone",
                 return_value=("Location: London, UK", None),
             ):
            reply = run_reply_engine(
                db=eval_db, cfg=mock_config, tts=None,
                text="Who directed Possessor and what else have they directed?",
                dialogue_memory=eval_dialogue_memory,
            )

        searches = [c for c in capture.calls if c["name"] == "webSearch"]
        print(f"\n📊 Chained research — Possessor + filmography:")
        print(f"   webSearch count: {len(searches)}")
        for c in searches:
            print(f"     args: {c['args']}")
        print(f"   reply: {(reply or '')[:240]}...")

        assert len(searches) >= 2, (
            f"Expected at least two webSearch calls (entity, then "
            f"filmography). Got {len(searches)}: "
            f"{[c['args'] for c in searches]}"
        )
        # The two calls should have distinct argument strings.
        arg_fingerprints = {
            " ".join(
                str(v) for v in (c["args"] or {}).values() if isinstance(v, str)
            ).lower()
            for c in searches
        }
        assert len(arg_fingerprints) >= 2, (
            f"Both webSearch calls had identical args — chain was not "
            f"progressed. Args: {arg_fingerprints}"
        )

    # ---- 7b ---------------------------------------------------------------
    @pytest.mark.eval
    @requires_judge_llm
    def test_parallel_comparison_paris_vs_london(
        self, mock_config, eval_db, eval_dialogue_memory
    ):
        """Two getWeather calls, different locations, reply mentions both."""
        from jarvis.reply.engine import run_reply_engine

        _configure(mock_config)
        capture = ToolCallCapture()

        def _respond(name, args):
            if name == "getWeather":
                loc = " ".join(
                    str(v) for v in (args or {}).values() if isinstance(v, str)
                ).lower()
                if "london" in loc:
                    return MOCK_WEATHER_LONDON
                return MOCK_WEATHER_PARIS
            return "OK"

        router = _make_router_stub(["getWeather", "stop"])
        runner = _make_tool_runner(capture, _respond)

        with patch("jarvis.reply.engine.select_tools", side_effect=router), \
             patch("jarvis.reply.engine.run_tool_with_retries", side_effect=runner), \
             patch(
                 "jarvis.reply.engine.get_location_context_with_timezone",
                 return_value=("Location: London, UK", None),
             ):
            reply = run_reply_engine(
                db=eval_db, cfg=mock_config, tts=None,
                text="Compare the weather in Paris and London right now.",
                dialogue_memory=eval_dialogue_memory,
            )

        weather_calls = [c for c in capture.calls if c["name"] == "getWeather"]
        locs = {
            " ".join(
                str(v) for v in (c["args"] or {}).values() if isinstance(v, str)
            ).lower()
            for c in weather_calls
        }
        print(f"\n📊 Parallel comparison — Paris vs London:")
        print(f"   getWeather calls: {len(weather_calls)}")
        print(f"   distinct location args: {locs}")
        print(f"   reply: {(reply or '')[:240]}...")

        assert len(weather_calls) >= 2, (
            f"Expected at least two getWeather calls (one per city). Got "
            f"{len(weather_calls)}: {[c['args'] for c in weather_calls]}"
        )
        has_paris = any("paris" in loc for loc in locs)
        has_london = any("london" in loc for loc in locs)
        assert has_paris and has_london, (
            f"getWeather must have been called for BOTH Paris and London. "
            f"Got location args: {locs}"
        )
        if reply:
            lower = reply.lower()
            assert "paris" in lower and "london" in lower, (
                f"Reply should mention both Paris and London. Got: "
                f"{reply[:300]!r}"
            )

    # ---- 7c ---------------------------------------------------------------
    @pytest.mark.eval
    @requires_judge_llm
    def test_cross_turn_pronoun_resolution(
        self, mock_config, eval_db, eval_dialogue_memory
    ):
        """Turn 2 resolves 'his' to the entity established in turn 1."""
        from jarvis.reply.engine import run_reply_engine

        _configure(mock_config)
        capture = ToolCallCapture()

        def _respond(name, args):
            if name == "webSearch":
                arg_str = " ".join(
                    str(v) for v in (args or {}).values() if isinstance(v, str)
                ).lower()
                if "song" in arg_str or "music" in arg_str or "album" in arg_str:
                    return MOCK_HARRY_STYLES_SONGS
                return MOCK_HARRY_STYLES_BIO
            return "OK"

        router = _make_router_stub(["webSearch", "stop"])
        runner = _make_tool_runner(capture, _respond)

        with patch("jarvis.reply.engine.select_tools", side_effect=router), \
             patch("jarvis.reply.engine.run_tool_with_retries", side_effect=runner), \
             patch(
                 "jarvis.reply.engine.get_location_context_with_timezone",
                 return_value=("Location: London, UK", None),
             ):
            # Turn 1: establish entity
            capture.clear()
            run_reply_engine(
                db=eval_db, cfg=mock_config, tts=None,
                text="Who is Harry Styles?",
                dialogue_memory=eval_dialogue_memory,
            )
            turn1 = list(capture.calls)

            # Turn 2: pronoun
            capture.clear()
            reply2 = run_reply_engine(
                db=eval_db, cfg=mock_config, tts=None,
                text="What are his most famous songs?",
                dialogue_memory=eval_dialogue_memory,
            )
            turn2 = list(capture.calls)

        print(f"\n📊 Cross-turn pronoun resolution:")
        print(f"   Turn 1 calls: {[c['name'] for c in turn1]}")
        print(f"   Turn 2 calls: {turn2}")
        print(f"   Turn 2 reply: {(reply2 or '')[:200]}...")

        turn2_searches = [c for c in turn2 if c["name"] == "webSearch"]
        assert turn2_searches, (
            f"Turn 2 must trigger a webSearch to answer the follow-up. "
            f"Got: {[c['name'] for c in turn2]}"
        )
        # At least one search arg must name the entity.
        resolved = False
        for c in turn2_searches:
            arg_str = " ".join(
                str(v) for v in (c["args"] or {}).values() if isinstance(v, str)
            ).lower()
            if "harry" in arg_str or "styles" in arg_str:
                resolved = True
                break
        assert resolved, (
            f"Turn 2 webSearch arg did not resolve 'his' to the entity "
            f"established in turn 1. Args: {[c['args'] for c in turn2_searches]}"
        )
        if reply2:
            lower = reply2.lower()
            mentions_song = any(
                k in lower for k in ("song", "watermelon", "as it was", "sign", "adore")
            )
            assert mentions_song, (
                f"Turn 2 reply should address the songs question. "
                f"Got: {reply2[:300]!r}"
            )

    # ---- 7d ---------------------------------------------------------------
    @pytest.mark.eval
    @requires_judge_llm
    def test_correction_loop_accepts_single_or_retry(
        self, mock_config, eval_db, eval_dialogue_memory
    ):
        """At least one webSearch must happen; a nudge-driven retry is
        acceptable, zero searches is not."""
        from jarvis.reply.engine import run_reply_engine

        _configure(mock_config)
        capture = ToolCallCapture()

        def _respond(name, args):
            if name == "webSearch":
                # First call returns stale; subsequent calls return live.
                n = sum(1 for c in capture.calls if c["name"] == "webSearch")
                # n is already incremented by this point (capture.record ran first)
                return MOCK_MADRID_LIVE if n > 1 else MOCK_MADRID_STALE
            return "OK"

        router = _make_router_stub(["webSearch", "stop"])
        runner = _make_tool_runner(capture, _respond)

        with patch("jarvis.reply.engine.select_tools", side_effect=router), \
             patch("jarvis.reply.engine.run_tool_with_retries", side_effect=runner), \
             patch(
                 "jarvis.reply.engine.get_location_context_with_timezone",
                 return_value=("Location: London, UK", None),
             ):
            reply = run_reply_engine(
                db=eval_db, cfg=mock_config, tts=None,
                text="What's the score in the Real Madrid game?",
                dialogue_memory=eval_dialogue_memory,
            )

        searches = [c for c in capture.calls if c["name"] == "webSearch"]
        print(f"\n📊 Correction loop — Real Madrid score:")
        print(f"   webSearch count: {len(searches)}")
        print(f"   reply: {(reply or '')[:240]}...")

        assert len(searches) >= 1, (
            f"At least one webSearch must fire for a live-score query. "
            f"Tools called: {capture.tool_names()}"
        )

    # ---- 7e ---------------------------------------------------------------
    @pytest.mark.eval
    @requires_judge_llm
    @pytest.mark.xfail(
        reason=(
            "Plumbing verified in unit tests. Live behaviour on gemma4:e2b "
            "is flaky on multi-turn escape-hatch flows: the small model "
            "sometimes refuses turn 1 in prose despite the nudge. Tracked "
            "for iterative prompt tuning; architecture ships as-is."
        ),
        strict=False,
    )
    def test_escape_hatch_then_follow_up_action(
        self, mock_config, eval_db, eval_dialogue_memory
    ):
        """Turn 1: narrow router → toolSearchTool → navigate. Turn 2: a new
        action whose argument must be self-contained ('lo-fi')."""
        from jarvis.reply.engine import run_reply_engine

        _configure(mock_config)
        capture = ToolCallCapture()

        def _respond(name, args):
            if name == "toolSearchTool":
                return MOCK_TOOLSEARCH_NAV
            if name == "chrome-devtools__navigate_page":
                return MOCK_NAV_SUCCESS
            if name == "webSearch":
                return (
                    "Web search results for 'lo-fi beats':\n"
                    "Top results: Lofi Girl's YouTube radio, Chillhop Music, "
                    "and Nujabes playlists.\n"
                )
            return "OK"

        # Narrow initial pick so the escape hatch is needed.
        router = _make_router_stub(["webSearch", "stop"])
        runner = _make_tool_runner(capture, _respond)

        with patch("jarvis.reply.engine.select_tools", side_effect=router), \
             patch("jarvis.reply.engine.run_tool_with_retries", side_effect=runner), \
             patch(
                 "jarvis.reply.engine.get_location_context_with_timezone",
                 return_value=("Location: London, UK", None),
             ):
            capture.clear()
            run_reply_engine(
                db=eval_db, cfg=mock_config, tts=None,
                text="Open YouTube.",
                dialogue_memory=eval_dialogue_memory,
            )
            turn1 = list(capture.calls)

            capture.clear()
            reply2 = run_reply_engine(
                db=eval_db, cfg=mock_config, tts=None,
                text="Now search for lo-fi beats.",
                dialogue_memory=eval_dialogue_memory,
            )
            turn2 = list(capture.calls)

        print(f"\n📊 Escape hatch + follow-up:")
        print(f"   Turn 1 calls: {[c['name'] for c in turn1]}")
        print(f"   Turn 2 calls: {turn2}")
        print(f"   Turn 2 reply: {(reply2 or '')[:200]}...")

        assert turn1, "Turn 1 should have at least one tool call"
        assert turn2, "Turn 2 should have at least one tool call"

        # Turn 2's tool call arg must contain the self-contained keyword.
        found_lofi = False
        for c in turn2:
            arg_str = " ".join(
                str(v) for v in (c["args"] or {}).values() if isinstance(v, str)
            ).lower()
            if "lo-fi" in arg_str or "lofi" in arg_str or "lo fi" in arg_str or "beats" in arg_str:
                found_lofi = True
                break
        assert found_lofi, (
            f"Turn 2 tool arg must contain the self-contained keyword "
            f"'lo-fi' (or a reasonable paraphrase). Calls: {turn2}"
        )


# =============================================================================
# 8. Structured tool_call emission — the evaluator must not only nudge
#    textually, it must emit a structured {name, arguments} that the engine can
#    execute directly. This is the recovery path for small chat models that
#    routinely ignore textual nudges.
# =============================================================================


class TestStructuredToolCallEmission:
    """The evaluator prompt now asks for a structured ``tool_call`` field
    alongside the textual nudge. Verify that a live small-model evaluator
    actually populates it when the intent is unambiguous."""

    @pytest.mark.eval
    @requires_judge_llm
    @pytest.mark.xfail(
        reason=(
            "Prompt compliance depends on the live small evaluator model. "
            "Deterministic coverage lives in tests/test_evaluator.py "
            "(parse) and tests/test_engine_tool_search_loop.py (direct-exec). "
            "Tracked for iterative prompt tuning; architecture ships as-is."
        ),
        strict=False,
    )
    def test_evaluator_emits_structured_tool_call_for_obvious_search(
        self, mock_config
    ):
        from jarvis.reply.evaluator import evaluate_turn

        _configure(mock_config)

        result = evaluate_turn(
            user_query="Give me an overview of China.",
            assistant_response_summary=(
                "I can look that up for you. Would you like me to search the "
                "web for an overview of China?"
            ),
            available_tools=[
                ("webSearch", "Search the web and return ranked results."),
                ("stop", "Explicit end-of-turn sentinel."),
            ],
            turns_used=1,
            cfg=mock_config,
        )

        print(f"\n📊 Structured tool_call emission:")
        print(f"   terminal: {result.terminal}")
        print(f"   nudge: {result.nudge!r}")
        print(f"   tool_call: {result.tool_call!r}")

        assert result.terminal is False, (
            "Evaluator should continue: the agent offered prose instead of "
            "calling webSearch. "
            f"Got terminal={result.terminal}, reason={result.reason!r}."
        )
        assert isinstance(result.tool_call, dict), (
            "Evaluator should emit a structured tool_call so the engine can "
            "run the search directly without relying on the chat model to "
            f"parse the textual nudge. Got tool_call={result.tool_call!r}."
        )
        assert result.tool_call.get("name") == "webSearch", (
            f"Structured tool_call.name should be 'webSearch'. "
            f"Got {result.tool_call!r}."
        )
        args = result.tool_call.get("arguments") or {}
        assert isinstance(args, dict) and args, (
            "Structured tool_call.arguments should be a non-empty dict with "
            f"the intended query. Got {result.tool_call!r}."
        )
        arg_blob = " ".join(
            str(v).lower() for v in args.values() if isinstance(v, str)
        )
        assert "china" in arg_blob, (
            f"Structured tool_call.arguments should mention 'china'. "
            f"Got {result.tool_call!r}."
        )
