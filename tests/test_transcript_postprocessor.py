"""Unit tests for the offline Hunspell transcript post-processor.

Behaviour-level: what the corrected transcript looks like, which tokens are
protected, how the bypass paths behave, and what the structured event carries.
The dictionaries are the vendored pairs under ``jarvis/resources/hunspell``.
"""

import pytest

from jarvis.listening.transcript_postprocessor import (
    WHISPER_TO_DICTIONARY,
    TranscriptCorrection,
    correct_transcript,
    format_correction_event,
)


@pytest.mark.unit
class TestDictionaryMapping:
    def test_four_supported_languages(self):
        assert set(WHISPER_TO_DICTIONARY.values()) == {"en_US", "cs_CZ", "vi_VN", "sk_SK"}

    def test_script_and_tag_spellings_share_one_id(self):
        assert WHISPER_TO_DICTIONARY["cs"] == WHISPER_TO_DICTIONARY["cs-CZ"] == "cs_CZ"
        assert WHISPER_TO_DICTIONARY["vi"] == WHISPER_TO_DICTIONARY["vi_VN"] == "vi_VN"


@pytest.mark.unit
class TestCorrections:
    def test_single_unique_typo_is_repaired(self):
        result = correct_transcript("the weathr is overcast", "en")
        assert result.corrected == "the weather is overcast"
        assert result.replacements == (("weathr", "weather"),)

    def test_raw_text_always_survives(self):
        raw = "the weathr is overcast"
        result = correct_transcript(raw, "en")
        assert result.raw == raw
        assert isinstance(result, TranscriptCorrection)

    def test_ambiguous_candidates_keep_the_original(self):
        # Two Hunspell candidates sit at distance 1, so no rewrite happens.
        result = correct_transcript("helo world", "en")
        assert result.corrected == "helo world"
        assert result.replacements == ()

    def test_casing_shape_is_preserved(self):
        result = correct_transcript("Weathr is nice", "en")
        assert result.corrected == "Weather is nice"
        assert result.replacements == (("Weathr", "Weather"),)

    def test_czech_dictionary_repairs_diritics(self):
        result = correct_transcript("Jake pocasy", "cs")
        assert result.language == "cs"
        assert result.corrected != "Jake pocasy"

    def test_slovak_and_vietnamese_load_their_pairs(self):
        assert correct_transcript("Hello world", "sk").corrected == "Hello world"
        assert correct_transcript("xin chao", "vi").corrected.startswith("xin")


@pytest.mark.unit
class TestProtectedTokens:
    def test_protected_terms_are_never_rewritten(self):
        result = correct_transcript(
            "toustovac said hello", "en", protected_terms=frozenset({"toustovac"})
        )
        assert result.replacements == ()
        assert "toustovac" in result.corrected

    def test_technical_tokens_survive(self):
        text = "I like Python and JavaScript and NASA"
        assert correct_transcript(text, "en").corrected == text

    def test_url_and_email_tokens_survive(self):
        text = "see https://example.com and a@b.com"
        assert correct_transcript(text, "en").corrected == text

    def test_protection_comparison_is_case_insensitive(self):
        result = correct_transcript(
            "Toustovac here", "en", protected_terms=frozenset({"toustovac"})
        )
        assert result.corrected == "Toustovac here"


@pytest.mark.unit
class TestBypassPaths:
    def test_unsupported_language_bypasses(self):
        result = correct_transcript("hello", "pl")
        assert result.corrected == result.raw
        assert result.replacements == ()

    def test_missing_language_bypasses(self):
        result = correct_transcript("the weathr is overcast", None)
        assert result.corrected == result.raw

    def test_disabled_bypasses(self):
        result = correct_transcript("the weathr is overcast", "en", enabled=False)
        assert result.corrected == result.raw
        assert result.replacements == ()

    def test_empty_text_is_returned_as_is(self):
        result = correct_transcript("", "en")
        assert result.corrected == ""
        assert result.replacements == ()

    def test_broken_dictionary_warns_once_and_returns_raw(self, monkeypatch):
        from jarvis.listening import transcript_postprocessor as pp

        def boom(_dictionary_id: str):
            raise OSError("missing dictionary")

        monkeypatch.setattr(pp, "_load_dictionary", boom)
        result = pp.correct_transcript("the weathr is overcast", "en")
        assert result.corrected == result.raw
        assert result.replacements == ()


@pytest.mark.unit
class TestWorkBounds:
    @staticmethod
    def _words(count: int, misspelled_index: int) -> str:
        tokens = ["word"] * count
        tokens[misspelled_index] = "weathr"
        return " ".join(tokens)

    def test_first_256_words_are_processed(self):
        result = correct_transcript(self._words(256, 255), "en")
        assert result.replacements == (("weathr", "weather"),)

    def test_words_past_the_cap_ride_verbatim(self):
        result = correct_transcript(self._words(300, 299), "en")
        assert result.replacements == ()
        assert result.corrected.endswith("weathr")

    def test_five_candidates_are_enough_for_the_gate(self):
        # 'helo' has several distance-1 candidates; the gate keeps the original.
        assert correct_transcript("helo", "en").replacements == ()


@pytest.mark.unit
class TestStructuredEvent:
    def test_event_carries_both_values_when_text_logging_is_on(self):
        result = correct_transcript("the weathr is overcast", "en")
        line = format_correction_event(result, include_text=True, latency_ms=1.5)
        assert "event=speech_transcript_corrected" in line
        assert "language=en" in line
        assert "replacement_count=1" in line
        assert "raw_text=the weathr is overcast" in line
        assert "corrected_text=the weather is overcast" in line

    def test_event_is_counts_only_when_text_logging_is_off(self):
        result = correct_transcript("the weathr is overcast", "en")
        line = format_correction_event(result, include_text=False, latency_ms=1.5)
        assert "replacement_count=1" in line
        assert "latency_ms=" in line
        assert "raw_text" not in line
        assert "corrected_text" not in line

    def test_nfc_only_normalization_changes_no_case(self):
        result = correct_transcript("janův", "cs")
        assert result.corrected == "janův"
        assert result.replacements == ()


@pytest.mark.unit
class TestProtectedPrefixCompletion:
    """A Hunspell miss is first completed from the protected vocabulary.

    The protected term is the authoritative spelling of its own prefix, so a
    truncated wake word becomes the full term even when the dictionary ranking
    is blocked by the unique-best gate.
    """

    _WAKE_TERMS = frozenset({
        "toustovač",
        "toustovači",
        "toastovač",
        "toastovači",
        "hej toustovač",
        "hej toustovači",
    })

    def test_unfinished_token_is_completed_from_wake_terms(self):
        result = correct_transcript("Hey toastova, ", "cs", protected_terms=self._WAKE_TERMS)
        assert result.corrected == "Hey toastovač, "
        assert result.replacements == (("toastova", "toastovač"),)
        assert result.raw == "Hey toastova, "

    def test_completion_preserves_the_casing_shape(self):
        result = correct_transcript("Toastova 11", "cs", protected_terms=self._WAKE_TERMS)
        assert result.corrected == "Toastovač 11"
        assert result.replacements == (("Toastova", "Toastovač"),)
        assert result.raw == "Toastova 11"

    def test_second_wake_spelling_completes_the_same_way(self):
        result = correct_transcript("Hey toustova,", "cs", protected_terms=self._WAKE_TERMS)
        assert result.corrected == "Hey toustovač,"
        assert result.replacements == (("toustova", "toustovač"),)
        assert result.raw == "Hey toustova,"

    def test_two_same_length_extensions_block_the_completion(self):
        result = correct_transcript("ab", "cs", protected_terms=frozenset({"abc", "abd"}))
        assert result.corrected == "ab"
        assert result.replacements == ()
        assert result.raw == "ab"

    def test_multi_word_term_does_not_complete_a_single_token(self):
        result = correct_transcript("hej", "cs", protected_terms=frozenset({"hej toustovač"}))
        assert result.corrected == "hej"
        assert result.replacements == ()
        assert result.raw == "hej"
