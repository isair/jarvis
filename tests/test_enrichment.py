"""Tests for reply enrichment helpers."""

import pytest

from jarvis.reply.engine import _match_question


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
