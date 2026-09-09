"""Tests for the ProactiveToasterService (see src/jarvis/proactive.spec.md).

Behaviour-focused: which events produce an unsolicited remark, which the
lightweight policy suppresses, direct-command folding, mode semantics, and
what the model pass is fed. No real LLM is involved.
"""

import pytest

from jarvis.proactive import (
    DEMO_SCRIPTS,
    ProactiveToasterService,
    _day_window,
    emit_remark,
    format_bytes,
    format_hhmm,
    is_directive,
    run_periodic_checks,
)


def make_service(reply="Toast je hotový.", **kwargs):
    """Service wired to a fake chat callable plus a manual monotonic clock."""
    box = {"t": 0.0}
    calls = []

    def chat(messages, **kw):
        calls.append(messages)
        return {"choices": [{"message": {"content": reply}}]}

    svc = ProactiveToasterService(chat, time_source=lambda: box["t"], **kwargs)
    return svc, calls, box


def ev(type_, context=None, timestamp=None):
    event = {"type": type_}
    if timestamp is not None:
        event["timestamp"] = timestamp
    if context is not None:
        event["context"] = context
    return event


class TestPolicyCore:
    @pytest.mark.unit
    def test_startup_event_returns_remark_and_calls_model_once(self):
        svc, calls, _clock = make_service()
        remark = svc.handle_event(ev("app.startup", {"model": "gemma4:e2b", "whisper_model": "medium"}))
        assert remark == "Toast je hotový."
        assert len(calls) == 1
        assert calls[0][0]["role"] == "system"
        assert "startup" in calls[0][1]["content"].lower()

    @pytest.mark.unit
    def test_unknown_type_is_suppressed_without_model_call(self):
        svc, calls, _clock = make_service()
        assert svc.handle_event(ev("nope.unknown")) is None
        assert calls == []
        assert svc.stats()["suppressed"].get("unknown-type") == 1

    @pytest.mark.unit
    def test_malformed_events_suppressed(self):
        svc, calls, _clock = make_service()
        assert svc.handle_event(None) is None
        assert svc.handle_event("") is None
        assert svc.handle_event({"no_type": 1}) is None
        assert svc.handle_event({"type": 42}) is None
        assert calls == []

    @pytest.mark.unit
    def test_min_gap_suppresses_second_remark(self):
        svc, calls, clock = make_service()
        assert svc.handle_event(ev("app.startup")) == "Toast je hotový."
        clock["t"] = 2.0  # inside the 90 s authentic gap
        assert svc.handle_event(ev("user.login", {}, "2026-09-09T06:00:00+00:00")) is None
        assert len(calls) == 1
        assert svc.stats()["suppressed"].get("gap") == 1

    @pytest.mark.unit
    def test_gap_beyond_min_gap_allows_second(self):
        svc, calls, clock = make_service()
        svc.handle_event(ev("app.startup"))
        clock["t"] = 100.0  # past the 90 s authentic gap
        assert svc.handle_event(ev("microphone.available")) == "Toast je hotový."
        assert len(calls) == 2

    @pytest.mark.unit
    def test_custom_min_gap_override(self):
        svc, calls, clock = make_service(min_gap_sec=5.0)
        svc.handle_event(ev("app.startup"))
        clock["t"] = 6.0
        assert svc.handle_event(ev("microphone.available")) is not None

    @pytest.mark.unit
    def test_hour_limit_caps_remarks(self):
        svc, calls, clock = make_service(hour_limit=2)
        svc.handle_event(ev("app.startup"))
        clock["t"] = 91.0
        svc.handle_event(ev("microphone.available"))
        clock["t"] = 182.0
        assert svc.handle_event(ev("user.login", {}, "2026-09-09T06:00:00+00:00")) is None
        assert len(calls) == 2  # third event skipped by ceiling, no LLM call

    @pytest.mark.unit
    def test_dedup_suppresses_identical_context_within_window(self):
        svc, calls, clock = make_service(min_gap_sec=1.0)
        event = ev("system.temperature_high", {"cpu_celsius": 91, "foreground_app": "Visual Studio Code", "seconds_since_last_interruption": 420})
        assert svc.handle_event(event) == "Toast je hotový."
        clock["t"] = 5.0  # past the custom gap, inside the 30 s dedup window
        assert svc.handle_event(event) is None
        assert len(calls) == 1
        assert svc.stats()["suppressed"].get("dedup") == 1
        clock["t"] = 160.0  # stamp age 155 > 30 → unfolds, one more LLM pass
        assert svc.handle_event(event) == "Toast je hotový."
        assert len(calls) == 2

    @pytest.mark.unit
    def test_dedup_in_window_folds(self):
        svc, calls, clock = make_service(min_gap_sec=1.0)
        event = ev("system.temperature_high", {"cpu_celsius": 91})
        assert svc.handle_event(event) == "Toast je hotový."
        clock["t"] = 5.0  # gap ok (5 > 1), dedup window 30 → folded
        assert svc.handle_event(event) is None
        assert len(calls) == 1

    @pytest.mark.unit
    def test_temperature_gate_only_above_threshold(self):
        svc, calls, clock = make_service()
        assert svc.handle_event(ev("system.temperature_high", {"cpu_celsius": 82})) is None
        assert calls == []
        clock["t"] = 100.0
        assert svc.handle_event(ev("system.temperature_high", {"cpu_celsius": 91})) == "Toast je hotový."
        assert "91" in calls[0][1]["content"]

    @pytest.mark.unit
    def test_battery_low_only_when_unplugged(self):
        svc, calls, clock = make_service()
        assert svc.handle_event(ev("battery.low", {"level_percent": 10, "plugged_in": False})) == "Toast je hotový."
        clock["t"] = 100.0
        # Plugged in at 40% is not "low battery" for the policy.
        assert svc.handle_event(ev("battery.low", {"level_percent": 40, "plugged_in": True})) is None
        assert len(calls) == 1

    @pytest.mark.unit
    def test_charger_connected_skipped_when_full(self):
        svc, calls, clock = make_service()
        assert svc.handle_event(ev("charger.connected", {"level_percent": 100, "plugged_in": True})) is None
        clock["t"] = 100.0
        assert svc.handle_event(ev("charger.connected", {"level_percent": 62, "plugged_in": True})) == "Toast je hotový."
        assert len(calls) == 1

    @pytest.mark.unit
    def test_inactivity_gate_and_note(self):
        svc, calls, clock = make_service()
        assert svc.handle_event(ev("user.inactivity", {"seconds_since_last_interruption": 30})) is None
        clock["t"] = 100.0
        assert svc.handle_event(ev("user.inactivity", {"seconds_since_last_interruption": 300})) == "Toast je hotový."
        assert "300" in calls[0][1]["content"]

    @pytest.mark.unit
    def test_day_window_notes(self):
        svc, calls, _clock = make_service()
        assert svc.handle_event(ev("day.morning", {"hour": 7})) == "Toast je hotový."
        content = calls[0][1]["content"]
        assert "Morning" in content and "7" in content

    @pytest.mark.unit
    def test_day_night_not_a_supported_window(self):
        svc, _calls, _clock = make_service()
        assert svc.handle_event(ev("day.night", {"hour": 23})) is None

    @pytest.mark.unit
    def test_network_pair_both_spoken(self):
        svc, calls, clock = make_service()
        assert svc.handle_event(ev("network.disconnected", {"seconds_down": 12})) == "Toast je hotový."
        clock["t"] = 100.0
        assert svc.handle_event(ev("network.restored", {"seconds_down": 12})) == "Toast je hotový."
        assert len(calls) == 2

    @pytest.mark.unit
    def test_apps_switching_note_carries_count(self):
        svc, calls, _clock = make_service()
        assert svc.handle_event(ev("apps.switching", {"count": 5, "apps": ["Code", "Chrome", "Code", "Chrome", "Code"]})) == "Toast je hotový."
        assert "5" in calls[0][1]["content"]

    @pytest.mark.unit
    def test_microphone_available_spoken(self):
        svc, calls, _clock = make_service()
        assert svc.handle_event(ev("microphone.available", {"sample_rate": 16000})) == "Toast je hotový."

    @pytest.mark.unit
    def test_download_completed_note_formats_bytes(self):
        svc, calls, _clock = make_service()
        svc.handle_event(ev("download.completed", {"name": "model.bin", "bytes": 2048}))
        content = calls[0][1]["content"]
        assert "model.bin" in content
        assert "2.0 KB" in content

    @pytest.mark.unit
    def test_build_failed_note_carries_error(self):
        svc, calls, _clock = make_service()
        svc.handle_event(ev("build.failed", {"error": "missing brace on line 12"}))
        assert "missing brace on line 12" in calls[0][1]["content"]

    @pytest.mark.unit
    def test_tool_completed_note_carries_tool_and_outcome(self):
        svc, calls, _clock = make_service()
        svc.handle_event(ev("tool.completed", {"tool": "getWeather", "success": False}))
        content = calls[0][1]["content"]
        assert "getWeather" in content
        assert "failed" in content

    @pytest.mark.unit
    def test_login_note_carries_utc_time(self):
        svc, calls, _clock = make_service()
        svc.handle_event(ev("user.login", {}, "2026-09-09T06:00:00+00:00"))
        assert "06:00" in calls[0][1]["content"]

    @pytest.mark.unit
    def test_previous_remarks_passed_for_variety(self):
        svc, calls, clock = make_service(reply="PrníRemark.")
        svc.handle_event(ev("app.startup"))
        clock["t"] = 150.0
        svc.handle_event(ev("microphone.available"))
        assert "PrníRemark." in calls[1][1]["content"]

    @pytest.mark.unit
    def test_chat_failure_fails_open(self):
        def boom(messages, **kw):
            raise RuntimeError("server down")

        svc = ProactiveToasterService(boom, time_source=lambda: 0.0)
        assert svc.handle_event(ev("app.startup")) is None
        assert svc.stats()["errors"] == 1

    @pytest.mark.unit
    def test_empty_model_output_is_none(self):
        def empty(messages, **kw):
            return {"choices": [{"message": {"content": "   "}}]}

        svc = ProactiveToasterService(empty, time_source=lambda: 0.0)
        assert svc.handle_event(ev("app.startup")) is None
        assert svc.stats()["suppressed"].get("empty") == 1

    @pytest.mark.unit
    def test_plain_string_model_reply_accepted(self):
        def plain(messages, **kw):
            return "Řádek."

        svc = ProactiveToasterService(plain, time_source=lambda: 0.0)
        assert svc.handle_event(ev("app.startup")) == "Řádek."

    @pytest.mark.unit
    def test_stats_count_spoken_and_suppressed(self):
        svc, _calls, clock = make_service()
        svc.handle_event(ev("app.startup"))
        clock["t"] = 1.0
        svc.handle_event(ev("microphone.available"))  # inside the 90 s gap
        stats = svc.stats()
        assert stats["spoken"] == 1
        assert stats["suppressed"]["gap"] == 1


class TestInterruptionModes:
    @pytest.mark.unit
    def test_polite_mode_only_critical_events_speak(self):
        svc, calls, clock = make_service(mode="polite")
        # Non-critical is skipped by type.
        assert svc.handle_event(ev("day.morning", {"hour": 7})) is None
        assert svc.stats()["suppressed"].get("polite-noncritical") == 1
        # Critical one speaks (2 s gap passes).
        clock["t"] = 2.0
        assert svc.handle_event(ev("app.error", {"message": "boom"})) == "Toast je hotový."
        assert len(calls) == 1

    @pytest.mark.unit
    def test_authentic_is_default_mode(self):
        svc, _calls, _clock = make_service()
        assert svc.mode == "authentic"

    @pytest.mark.unit
    def test_unknown_mode_falls_back_to_authentic(self):
        svc, _calls, _clock = make_service(mode="nope")
        assert svc.mode == "authentic"


class TestDirectCommands:
    @pytest.mark.unit
    def test_ticho_session_directive(self):
        svc, calls, clock = make_service()
        assert svc.apply_directive("Ticho") is True
        svc.handle_event(ev("app.startup"))  # suppressed by directive
        assert calls == []
        assert svc.stats()["suppressed"].get("directive") == 1
        # Session cooldown (default 600 s) expiry returns the persona, i.e.
        # the refusal is not a permanent reset.
        clock["t"] = 601.0
        assert svc.handle_event(ev("app.startup")) == "Toast je hotový."

    @pytest.mark.unit
    def test_directive_folds_case_and_diacritics(self):
        svc, _calls, _clock = make_service()
        for text in ("TICHO", "tiš", "TIŠ", "Přestaň nabízet toast", "prestan nabizet toast"):
            assert svc.apply_directive(text) is True, text
        assert is_directive("TED NE") is True

    @pytest.mark.unit
    def test_one_shot_now_not_command(self):
        svc, calls, clock = make_service()
        assert svc.apply_directive("Teď ne") is True
        assert svc.handle_event(ev("app.startup")) is None  # single pass
        assert svc.stats()["suppressed"].get("directive-skip") == 1
        # Next event flows again (the refusal was temporary by design).
        assert svc.handle_event(ev("microphone.available")) == "Toast je hotový."
        assert len(calls) == 1

    @pytest.mark.unit
    def test_non_directive_is_noop(self):
        svc, _calls, _clock = make_service()
        assert svc.apply_directive("jaké je počasí") is False


class TestDemoMode:
    @pytest.mark.unit
    def test_scripted_triggers_return_canonical_text(self):
        calls = []
        svc = ProactiveToasterService(lambda *a, **k: calls.append(a), mode="demo")
        assert svc.demo_trigger("temperature") == DEMO_SCRIPTS["temperature"][2]
        assert svc.demo_trigger("battery_low") == DEMO_SCRIPTS["battery_low"][2]
        assert svc.demo_trigger("build_success") == DEMO_SCRIPTS["build_success"][2]
        assert svc.demo_trigger("inactivity") == DEMO_SCRIPTS["inactivity"][2]
        # The demo path never touches the LLM.
        assert calls == []

    @pytest.mark.unit
    def test_demo_triggers_have_no_gap(self):
        svc = ProactiveToasterService(None, mode="demo")
        assert svc.demo_trigger("startup") == DEMO_SCRIPTS["startup"][2]
        assert svc.demo_trigger("weather") == DEMO_SCRIPTS["weather"][2]  # 0 s gap

    @pytest.mark.unit
    def test_demo_unknown_name_is_none(self):
        svc = ProactiveToasterService(None, mode="demo")
        assert svc.demo_trigger("nope") is None

    @pytest.mark.unit
    def test_demo_reset_clears_counters(self):
        svc = ProactiveToasterService(None, mode="demo")
        svc.demo_trigger("temperature")
        assert svc.stats()["demo_uttered"] == 1
        svc.demo_reset()
        stats = svc.stats()
        assert stats["demo_uttered"] == 0
        assert svc.recent_records() == []

    @pytest.mark.unit
    def test_handle_event_in_demo_uses_script(self):
        calls = []
        svc = ProactiveToasterService(lambda *a, **k: calls.append(a), mode="demo")
        text = svc.handle_event(ev("system.temperature_high", {"cpu_celsius": 91}))
        assert text == DEMO_SCRIPTS["temperature"][2]
        assert calls == []


class TestStructuredLogs:
    @pytest.mark.unit
    def test_record_per_spoken_event(self):
        svc, _calls, _clock = make_service()
        svc.handle_event(ev("app.startup"))
        rec = svc.recent_records()[-1]
        assert rec["decision"] == "spoken"
        assert rec["event"] == "app.startup"
        assert rec["reason"] == "llm"
        assert "min_gap_sec" in rec["cooldown"] and "hour_limit" in rec["cooldown"]

    @pytest.mark.unit
    def test_record_per_suppression_with_reason(self):
        svc, _calls, _clock = make_service()
        svc.handle_event(ev("nope.unknown"))
        rec = svc.recent_records()[-1]
        assert rec["decision"] == "suppressed"
        assert rec["reason"] == "unknown-type"

    @pytest.mark.unit
    def test_mark_user_response_flags_latest_spoken_record(self):
        svc, _calls, _clock = make_service()
        svc.handle_event(ev("app.startup"))
        assert svc.recent_records()[-1]["responded"] is None
        svc.mark_user_response()
        assert svc.recent_records()[-1]["responded"] is True

    @pytest.mark.unit
    def test_directive_state_visible_in_cooldown(self):
        svc, _calls, _clock = make_service()
        svc.apply_directive("Ticho")
        rec = svc.recent_records()[-1]
        assert "directive_remaining_sec" in rec["cooldown"]


class TestPeriodicChecks:
    @pytest.mark.unit
    def test_inactivity_and_day_window_feed_the_policy(self):
        import time as _time

        calls = []

        def chat(messages, **kw):
            calls.append(messages)
            return {"choices": [{"message": {"content": "R."}}]}

        svc = ProactiveToasterService(chat, time_source=_time.monotonic)

        class _DM:  # minimal stand-in carrying only the activity stamp
            _last_activity_time = _time.time() - 120

        remarks = run_periodic_checks(svc, dialogue_memory=_DM(), llm_base_url="", tts=None)
        assert remarks == ["R."]
        assert "Inactive for 120" in calls[0][1]["content"]

    @pytest.mark.unit
    def test_no_activity_stamp_means_no_inactivity_event(self):
        import time as _time

        calls = []

        def chat(messages, **kw):
            calls.append(messages)
            return {"choices": [{"message": {"content": "R."}}]}

        svc = ProactiveToasterService(chat, time_source=_time.monotonic)
        run_periodic_checks(svc, dialogue_memory=None, llm_base_url="")
        assert all("Inactive for" not in m[1]["content"] for m in calls)


class TestHelpers:
    @pytest.mark.unit
    def test_day_window_mapping(self):
        assert _day_window(7) == "morning"
        assert _day_window(13) == "lunch"
        assert _day_window(18) == "evening"
        assert _day_window(15) is None
        assert _day_window(23) is None
        assert _day_window(4) is None

    @pytest.mark.unit
    def test_format_hhmm_iso(self):
        assert format_hhmm("2026-09-09T06:00:00+00:00") == "06:00"

    @pytest.mark.unit
    def test_format_hhmm_epoch_and_fallback(self):
        assert format_hhmm(0) == "00:00"
        assert format_hhmm("abc") == "abc"

    @pytest.mark.unit
    def test_format_bytes(self):
        assert format_bytes(2048) == "2.0 KB"
        assert format_bytes(1572864) == "1.5 MB"
        assert format_bytes(999) == "999 B"

    @pytest.mark.unit
    def test_emit_remark_prints_emoji_and_label(self, capsys):
        emit_remark("Pozor,!", reason_label="CPU temperature")
        out = capsys.readouterr().out
        assert "🍞 Pozor,!" in out
        assert "🏷️ CPU temperature" in out

    @pytest.mark.unit
    def test_emit_remark_noop_without_text(self, capsys):
        emit_remark(None)
        assert capsys.readouterr().out == ""
