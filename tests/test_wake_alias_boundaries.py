"""Configured alias variants must not expand into unrelated ambient words."""
import pytest
from jarvis.config import get_default_config
from jarvis.listening.wake_detection import is_wake_word_detected

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('text', [
    "Doesn't matter if her oppresses A, first joins",
    'the first person joins the game',
    'jarvisian is a made-up word',
    'the chavisville address',
])
def test_ambient_words_do_not_trigger_default_wake_detection(text):
    cfg = get_default_config()
    assert not is_wake_word_detected(text.lower(), cfg['wake_word'], cfg['wake_aliases'],
                                     cfg['wake_fuzzy_ratio'])


@pytest.mark.parametrize('alias', get_default_config()['wake_aliases'])
def test_each_configured_alias_remains_a_valid_address(alias):
    cfg = get_default_config()
    assert is_wake_word_detected(f'{alias}, what time is it?', cfg['wake_word'],
                                 cfg['wake_aliases'], cfg['wake_fuzzy_ratio'])


@pytest.mark.parametrize('name,alias,near_primary', [
    ('jarvis', 'joris', 'jarvas'),
    ('friday', 'computer', 'fridai'),
])
def test_primary_approximation_and_exact_alias_remain_usable(name, alias, near_primary):
    assert is_wake_word_detected(f'{near_primary}, hello', name, [alias])
    assert is_wake_word_detected(f'{alias}, hello', name, [alias])


@pytest.mark.parametrize('text,name,alias', [
    ('FRİDAY, hello', 'jarvis', 'friday'),
    ('ΓΕΙΑ, ΑΘΗΝΑ!', 'athena', 'αθηνα'),
    ('你好，小助手！', 'jarvis', '小助手'),
    ('Jarvis, hello', 'jarvis', ''),
])
def test_unicode_punctuation_and_case_are_supported(text, name, alias):
    assert is_wake_word_detected(text, name, [alias])


def test_empty_alias_cannot_wake_ambient_speech():
    assert not is_wake_word_detected('ambient discussion', 'jarvis', [''])


@pytest.mark.parametrize('text,expected', [
    ('jarvis explain jarvisian philosophy', 'explain jarvisian philosophy'),
    ('jarvis tell me about chavisville', 'tell me about chavisville'),
])
def test_query_extraction_keeps_names_inside_other_words(text, expected):
    from jarvis.listening.wake_detection import extract_query_after_wake
    cfg = get_default_config()
    assert extract_query_after_wake(text, cfg['wake_word'], cfg['wake_aliases']) == expected


def test_explicit_fuzzy_threshold_is_respected():
    assert not is_wake_word_detected('jarvas hello', 'jarvis', ['joris'], fuzzy_ratio=1.0)
