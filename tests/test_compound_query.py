"""Tests for compound-query decomposition used by small models."""

import pytest

from jarvis.reply.compound_query import (
    CJK_MIN_CLAUSE_CHARS,
    DEFAULT_MIN_CLAUSE_CHARS,
    MIN_CLAUSE_CHARS,
    split_compound_query,
)


class TestSplitCompoundQuery:
    """Behaviour-level tests for the compound-query splitter."""

    # ── English: positive cases ────────────────────────────────────────────
    def test_multi_part_entity_query_splits(self):
        parts = split_compound_query(
            "Who directed Possessor and what other films has that director made?",
            language="en",
        )
        assert len(parts) == 2
        assert parts[0].startswith("Who directed Possessor")
        assert parts[1].startswith("what other films")

    def test_and_is_case_insensitive(self):
        parts = split_compound_query(
            "Show me the weather AND list my reminders for today",
            language="en",
        )
        assert len(parts) == 2

    def test_extra_whitespace_around_and(self):
        parts = split_compound_query(
            "Tell me about Britney Spears  and  what her best song is",
            language="en",
        )
        assert len(parts) == 2

    # ── English: negative cases (idioms / short clauses) ───────────────────
    def test_rock_and_roll_does_not_split(self):
        """Short left clause guards against idiomatic 'X and Y' phrases."""
        assert split_compound_query("Rock and roll history", language="en") == []

    def test_pros_and_cons_does_not_split(self):
        """Short left clause ('pros' = 4 chars) keeps this as a single query."""
        assert split_compound_query("pros and cons of remote work", language="en") == []

    def test_short_left_side_does_not_split(self):
        """Boundary: left clause below MIN_CLAUSE_CHARS prevents split."""
        short = "x" * (MIN_CLAUSE_CHARS - 1)
        long = "x" * (MIN_CLAUSE_CHARS + 5)
        assert split_compound_query(f"{short} and {long}", language="en") == []

    def test_short_right_side_does_not_split(self):
        short = "x" * (MIN_CLAUSE_CHARS - 1)
        long = "x" * (MIN_CLAUSE_CHARS + 5)
        assert split_compound_query(f"{long} and {short}", language="en") == []

    def test_both_at_threshold_splits(self):
        at_threshold = "x" * MIN_CLAUSE_CHARS
        parts = split_compound_query(f"{at_threshold} and {at_threshold}", language="en")
        assert len(parts) == 2

    def test_multiple_ands_only_first_split(self):
        """First ' and ' wins — keeps the splitter deterministic."""
        parts = split_compound_query(
            "Tell me about dogs and cats and also birds please",
            language="en",
        )
        assert len(parts) == 2
        assert "cats" in parts[1]
        assert "birds" in parts[1]  # second ' and ' stays in right clause

    def test_empty_and_none_are_safe(self):
        assert split_compound_query("", language="en") == []
        assert split_compound_query(None, language="en") == []  # type: ignore[arg-type]

    def test_no_conjunction_returns_empty(self):
        assert split_compound_query("What is the weather today?", language="en") == []

    def test_bare_and_without_whitespace_does_not_split(self):
        """We require whitespace boundaries to avoid splitting 'command' etc."""
        assert split_compound_query("commandline tools are useful", language="en") == []

    # ── Whitespace-separated supported languages ───────────────────────────
    @pytest.mark.parametrize("language,query", [
        # Germanic / Romance
        ("es", "Quién dirigió Possessor y qué otras películas ha hecho"),
        ("fr", "Qui a réalisé Possessor et quels autres films a-t-il faits"),
        ("de", "Wer führte Regie bei Possessor und welche anderen Filme hat er"),
        ("pt", "Quem dirigiu Possessor e quais outros filmes fez o diretor"),
        ("it", "Chi ha diretto Possessor e quali altri film ha fatto"),
        ("nl", "Wie regisseerde Possessor en welke andere films maakte hij"),
        ("sv", "Vem regisserade Possessor och vilka andra filmer har han gjort"),
        ("no", "Hvem regisserte Possessor og hvilke andre filmer har han laget"),
        ("da", "Hvem instruerede Possessor og hvilke andre film har han lavet"),
        ("fi", "Kuka ohjasi Possessorin ja mitä muita elokuvia hän on tehnyt"),
        # Slavic
        ("ru", "Кто снял фильм Поссессор и какие другие фильмы он снял"),
        ("uk", "Хто зняв фільм Поссессор і які інші фільми він зробив"),
        ("pl", "Kto wyreżyserował Possessor i jakie inne filmy zrobił"),
        ("cs", "Kdo režíroval Possessor a jaké další filmy natočil"),
        ("sk", "Kto režíroval Possessor a aké ďalšie filmy natočil"),
        ("bg", "Кой режисира Поссесор и какви други филми е направил"),
        ("hr", "Tko je režirao Possessor i koje druge filmove je snimio"),
        ("sl", "Kdo je režiral Possessor in katere druge filme je posnel"),
        # Other European
        ("el", "Ποιος σκηνοθέτησε το Possessor και ποιες άλλες ταινίες έχει κάνει"),
        ("tr", "Possessor filmini kim yönetti ve başka hangi filmleri yaptı"),
        ("hu", "Ki rendezte a Possessort és milyen más filmeket csinált"),
        ("ro", "Cine a regizat Possessor și ce alte filme a făcut"),
        # Asian whitespace-separated
        ("vi", "Ai đạo diễn Possessor và đạo diễn đó đã làm phim nào khác"),
        ("id", "Siapa sutradara Possessor dan film apa lagi yang sudah dibuat"),
        ("ms", "Siapa pengarah Possessor dan filem apa lagi yang telah dibuat"),
        ("hi", "पोसेसर का निर्देशन किसने किया और निर्देशक ने और कौन सी फिल्में बनाई"),
    ])
    def test_supported_languages_split(self, language, query):
        parts = split_compound_query(query, language=language)
        assert len(parts) == 2, f"{language}: expected split, got {parts!r}"

    def test_italian_ed_variant(self):
        """Italian uses 'ed' before vowels."""
        parts = split_compound_query(
            "Parlami della storia ed anche della geografia del paese",
            language="it",
        )
        assert len(parts) == 2

    # ── Non-English: unsupported / unknown languages ───────────────────────
    def test_unsupported_language_does_not_split(self):
        """Unknown language codes fall back to no-decomposition rather than
        mis-applying English rules — graceful degradation per spec."""
        # Japanese, Korean, Chinese, Russian — not in our conjunction table.
        # We do NOT want to split on ' and ' for these; the text below is
        # contrived to contain English 'and' but a Japanese language code.
        parts = split_compound_query(
            "some long query and another long query", language="ja",
        )
        assert parts == []

    def test_invalid_language_code_defaults_to_english(self):
        """Single-character or malformed codes normalise to None → English default."""
        parts = split_compound_query(
            "Tell me about cats and also about dogs please",
            language="x",
        )
        assert len(parts) == 2

    def test_none_language_defaults_to_english(self):
        """Non-voice entrypoints pass language=None; we default to English."""
        parts = split_compound_query(
            "Who is Britney Spears and what is her best song",
            language=None,
        )
        assert len(parts) == 2

    def test_uppercase_language_code_normalises(self):
        parts = split_compound_query(
            "Quién dirigió Possessor y qué otras películas ha hecho",
            language="ES",
        )
        assert len(parts) == 2

    def test_language_with_region_suffix_normalises(self):
        """en-US style codes should normalise to 'en'."""
        parts = split_compound_query(
            "Who is Britney Spears and what is her best song",
            language="en-US",
        )
        assert len(parts) == 2

    # ── Non-English: idioms should not false-positive ──────────────────────
    def test_french_va_et_vient_short_left_side(self):
        """'va' is only 2 chars so it won't split — guard by length."""
        assert split_compound_query("va et vient", language="fr") == []

    # ── CJK (no whitespace around conjunctions) ────────────────────────────
    def test_chinese_character_level_conjunction_splits(self):
        """Chinese 和 appears between words without whitespace."""
        parts = split_compound_query("电影的历史和音乐的发展", language="zh")
        assert len(parts) == 2
        assert "电影" in parts[0]
        assert "音乐" in parts[1]

    def test_chinese_short_clauses_do_not_split(self):
        """'我和他' — 1-char clauses should not split (below CJK threshold)."""
        assert split_compound_query("我和他", language="zh") == []

    def test_chinese_threshold_is_lower_than_default(self):
        """CJK threshold must be smaller than Latin default — Han chars pack
        more meaning per character."""
        assert CJK_MIN_CLAUSE_CHARS < DEFAULT_MIN_CLAUSE_CHARS

    def test_chinese_multi_char_conjunction_splits(self):
        parts = split_compound_query(
            "请告诉我关于狗的信息并且告诉我关于猫的信息", language="zh",
        )
        assert len(parts) == 2

    def test_japanese_freestanding_conjunction_splits(self):
        parts = split_compound_query(
            "犬について教えてそして猫についても教えて", language="ja",
        )
        assert len(parts) == 2

    def test_japanese_enclitic_particle_does_not_split(self):
        """と/や are noun-attached particles — we intentionally don't split
        on them to avoid fragmenting noun phrases like '犬と猫'."""
        # This phrase contains と between 犬 and 猫; our rules skip と
        # on purpose, so this should NOT split.
        assert split_compound_query("犬と猫が好きです", language="ja") == []

    def test_korean_freestanding_conjunction_splits(self):
        parts = split_compound_query(
            "개에 대해 알려주세요 그리고 고양이에 대해서도 알려주세요",
            language="ko",
        )
        assert len(parts) == 2

    def test_korean_postpositional_particle_does_not_split(self):
        """와/과 are postpositional particles — intentionally not split on
        (same reason as Japanese と/や)."""
        assert split_compound_query("개와 고양이를 좋아해요", language="ko") == []

    # ── Unsupported languages with enclitic conjunctions ───────────────────
    @pytest.mark.parametrize("language", ["ar", "he", "th", "km", "lo"])
    def test_enclitic_languages_return_empty(self, language):
        """Arabic / Hebrew use an enclitic conjunction prefix (و / ו) that
        a regex can't safely split without a morphological tokenizer. Thai
        / Khmer / Lao lack inter-word whitespace and the conjunctions
        overlap syllable boundaries. We intentionally do not support
        these yet — the splitter must return [] rather than mis-split.
        """
        parts = split_compound_query(
            "some long query and another long query", language=language,
        )
        assert parts == []
