"""Phase 3A: unit tests for TTSCoordinator (generation-guarded, channel-scoped)."""

from __future__ import annotations

from jarvis.output.tts_coordinator import (
    SpeakStatus, TtsChannel, TTSCoordinator,
)


class MockEngine:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self._speaking = False
        self.spoken = []
        self.interrupts = 0
        self._cb = None

    def is_speaking(self):
        return self._speaking

    def speak(self, text, completion_callback=None, duration_callback=None):
        self.spoken.append(text)
        self._cb = completion_callback
        self._speaking = True

    def interrupt(self):
        self.interrupts += 1
        self._speaking = False

    def fire_completion(self):
        """Simulate Piper finishing the CURRENT slot's item."""
        self._speaking = False
        if self._cb is not None:
            self._cb()
            self._cb = None


def test_speak_returns_speaking_and_calls_engine():
    eng = MockEngine()
    co = TTSCoordinator(eng)
    tok = co.speak(turn_id="t1", channel=TtsChannel.CHAT, text="hello")
    assert tok.status is SpeakStatus.SPEAKING
    assert eng.spoken == ["hello"]


def test_speak_off_is_skipped_and_engine_untouched():
    eng = MockEngine()
    co = TTSCoordinator(eng)
    tok = co.speak(turn_id="t1", channel=TtsChannel.CHAT, text="hi", speak_enabled=False)
    assert tok.status is SpeakStatus.SKIPPED
    assert eng.spoken == []


def test_blank_text_is_skipped():
    eng = MockEngine()
    co = TTSCoordinator(eng)
    assert co.speak(turn_id="t", channel=TtsChannel.CHAT, text="   ").status is SpeakStatus.SKIPPED


def test_disabled_engine_is_skipped():
    co = TTSCoordinator(MockEngine(enabled=False))
    assert co.speak(turn_id="t", channel=TtsChannel.CHAT, text="x").status is SpeakStatus.SKIPPED


def test_stale_callback_is_suppressed_by_generation_guard():
    """When a newer owned playback supersedes an older one, only the newer
    callback fires (generation guard); the engine is NOT force-interrupted."""
    eng = MockEngine()
    co = TTSCoordinator(eng)
    first_done, second_done = [], []
    co.speak(turn_id="t1", channel=TtsChannel.CHAT, text="a", on_complete=lambda: first_done.append(1))
    co.speak(turn_id="t2", channel=TtsChannel.CHAT, text="b", on_complete=lambda: second_done.append(1))
    assert eng.interrupts == 0  # no preempt-interrupt anymore (single-flight; mode-C fix)
    eng.fire_completion()  # engine fires the current slot (t2 wrapper)
    assert first_done == []      # superseded callback suppressed (stale generation)
    assert second_done == [1]    # only the current callback runs


def test_release_if_owned_clears_without_interrupt():
    eng = MockEngine()
    co = TTSCoordinator(eng)
    done = []
    co.speak(turn_id="t1", channel=TtsChannel.CHAT, text="x", on_complete=lambda: done.append(1))
    assert co.release_if_owned("t1") is True
    assert co.owns_active_playback() is False
    assert eng.interrupts == 0  # released WITHOUT interrupting audio
    eng.fire_completion()  # any late guarded callback must now be a no-op
    assert done == []


def test_release_if_owned_wrong_turn_is_noop():
    eng = MockEngine()
    co = TTSCoordinator(eng)
    co.speak(turn_id="t1", channel=TtsChannel.CHAT, text="x")
    assert co.release_if_owned("OTHER") is False
    assert co.owns_active_playback() is True


def test_skip_when_unowned_playback_is_live():
    """Mode-A fix: chat must not overwrite a live voice utterance the coordinator
    does not own (engine speaking, active_turn_id is None)."""
    eng = MockEngine()
    eng._speaking = True  # unmanaged (voice) audio playing
    co = TTSCoordinator(eng)
    tok = co.speak(turn_id="c1", channel=TtsChannel.CHAT, text="chat")
    assert tok.status is SpeakStatus.SKIPPED
    assert eng.spoken == [] and eng.interrupts == 0


def test_completion_clears_active_playback():
    eng = MockEngine()
    co = TTSCoordinator(eng)
    co.speak(turn_id="t1", channel=TtsChannel.CHAT, text="x")
    assert co.owns_active_playback() is True
    eng.fire_completion()
    assert co.owns_active_playback() is False


def test_interrupt_matching_turn_stops_audio():
    eng = MockEngine()
    co = TTSCoordinator(eng)
    co.speak(turn_id="t1", channel=TtsChannel.CHAT, text="x")
    assert co.interrupt(turn_id="t1", channel=TtsChannel.CHAT) is True
    assert eng.interrupts == 1
    assert co.owns_active_playback() is False


def test_interrupt_wrong_turn_is_noop():
    eng = MockEngine()
    co = TTSCoordinator(eng)
    co.speak(turn_id="t1", channel=TtsChannel.CHAT, text="x")
    assert co.interrupt(turn_id="OTHER", channel=TtsChannel.CHAT) is False
    assert eng.interrupts == 0  # did not touch the engine


def test_chat_interrupt_cannot_silence_voice():
    eng = MockEngine()
    co = TTSCoordinator(eng)
    co.speak(turn_id="v1", channel=TtsChannel.VOICE, text="reply")
    # a chat Stop (channel=CHAT) must not kill an active VOICE playback
    assert co.interrupt(channel=TtsChannel.CHAT) is False
    assert eng.interrupts == 0


def test_interrupt_suppresses_pending_callback():
    eng = MockEngine()
    co = TTSCoordinator(eng)
    done = []
    co.speak(turn_id="t1", channel=TtsChannel.CHAT, text="x", on_complete=lambda: done.append(1))
    co.interrupt(turn_id="t1", channel=TtsChannel.CHAT)
    eng.fire_completion()  # engine may still fire; wrapper must be a no-op now
    assert done == []


def test_engine_speak_failure_returns_failed():
    class Boom(MockEngine):
        def speak(self, *a, **k):
            raise RuntimeError("boom")
    co = TTSCoordinator(Boom())
    tok = co.speak(turn_id="t1", channel=TtsChannel.CHAT, text="x")
    assert tok.status is SpeakStatus.FAILED
    assert co.owns_active_playback() is False  # bookkeeping cleaned on failure


def test_does_not_preempt_unmanaged_voice_playback():
    """SAFETY: a chat speak must NOT force-interrupt a voice utterance the
    coordinator did not start (engine.is_speaking() True but not owned)."""
    eng = MockEngine()
    eng._speaking = True  # an external (voice) utterance is playing, unmanaged
    co = TTSCoordinator(eng)
    co.speak(turn_id="c1", channel=TtsChannel.CHAT, text="chat")
    assert eng.interrupts == 0  # never interrupted the unmanaged voice audio
