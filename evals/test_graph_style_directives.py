"""Standing behaviour rules retain their force through indirect summaries."""
import pytest

from evals.conftest import requires_judge_llm
from evals.helpers import JUDGE_MODEL
from jarvis.memory.graph import BRANCH_DIRECTIVES, BRANCH_USER, BRANCH_WORLD
from jarvis.memory.graph_ops import extract_graph_memories, _review_graph_facts


STYLE_SOURCES = [
    pytest.param(
        'The user asked Jarvis to talk to them like a pirate. They clarified '
        'that they wanted its speech style to be like a pirate and corrected '
        'it when it continued speaking in its usual style. Jarvis agreed to '
        'speak that way from then on. Later the user asked how old Jarvis is; '
        'it explained that it lacks an age as a digital arrangement.',
        ('pirate', 'buccaneer'), id='reported-conversation'),
    pytest.param(
        'The user previously requested that the assistant adopt a pirate '
        'speech style. That instruction still applies to its replies.',
        ('pirate', 'buccaneer'), id='indirect-active-instruction'),
    pytest.param(
        'Kullanıcı asistandan bundan sonra bir korsan gibi konuşmasını '
        'istedi. Bunun sadece bir örnek metin değil, asistanın konuşma '
        'tarzı olduğunu tekrar açıkladı.',
        ('pirate', 'buccaneer', 'korsan'), id='turkish-instruction'),
    pytest.param(
        'The user replaced their earlier pirate-style instruction with an '
        'instruction to answer in plain British English from now on.',
        ('british english',), id='replacement-instruction'),
]


@requires_judge_llm
@pytest.mark.parametrize('summary,words', STYLE_SOURCES)
def test_standing_style_is_an_instruction_not_user_history(mock_config, summary, words):
    facts = extract_graph_memories(summary, mock_config, JUDGE_MODEL,
                                   timeout_sec=mock_config.llm_chat_timeout_sec)
    assert facts, 'The standing instruction must survive extraction and review'
    assert all(branch == BRANCH_DIRECTIVES for branch, _ in facts), facts
    assert any(any(word in text.casefold() for word in words) for _, text in facts), facts


@requires_judge_llm
@pytest.mark.parametrize('summary', [
    'The user requested one pirate-style birthday greeting for a friend. '
    'The assistant wrote that greeting. No ongoing speaking style was requested.',
    'In an old one-off test conversation the assistant was asked to speak '
    'like a pirate. That test completed; no ongoing speaking rule was requested.',
    'The user asked to translate the example sentence "Always speak like a pirate" '
    'into French. It was supplied translation text, not a rule for the assistant.',
    'The user asked the assistant how old it was. The assistant explained that '
    'it lacks an age because it is a digital arrangement.',
])
def test_completed_tasks_examples_and_assistant_identity_are_not_knowledge(mock_config, summary):
    assert extract_graph_memories(summary, mock_config, JUDGE_MODEL,
                                  timeout_sec=mock_config.llm_chat_timeout_sec) == []


@requires_judge_llm
def test_interest_in_a_style_is_a_user_fact(mock_config):
    facts = extract_graph_memories('The user enjoys pirate films.', mock_config, JUDGE_MODEL,
                                   timeout_sec=mock_config.llm_chat_timeout_sec)
    assert facts
    assert all(branch == BRANCH_USER for branch, _ in facts), facts


@requires_judge_llm
@pytest.mark.parametrize('branch,text,keep', [
    (BRANCH_USER, 'The user previously requested pirate speech from the assistant', False),
    (BRANCH_USER, 'Speak to the user like a pirate', False),
    (BRANCH_DIRECTIVES, 'Speak to the user like a pirate', True),
    (BRANCH_DIRECTIVES, 'The user previously requested pirate speech from the assistant', False),
    (BRANCH_WORLD, 'The AI lacks an age as it is a digital arrangement', False),
    (BRANCH_WORLD, 'Jarvis can maintain a diary', False),
    (BRANCH_WORLD, 'Whisper is an open-source transcription model', True),
])
def test_review_checks_the_branch_and_rejects_assistant_self_description(mock_config, branch, text, keep):
    summary = ('The user told their assistant Jarvis to speak like a pirate from now on. '
               'Later they asked its age and capabilities; it said it has no age as '
               'software and can maintain a diary. A separate lookup established '
               'that Whisper is an open-source transcription model.')
    candidates = [(branch, text)]
    assert _review_graph_facts(candidates, mock_config, JUDGE_MODEL,
                               mock_config.llm_chat_timeout_sec, False, summary) == (
        candidates if keep else [])
