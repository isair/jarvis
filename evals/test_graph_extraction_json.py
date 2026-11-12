"""Composite personal facts survive extraction and semantic review."""
import unicodedata

import pytest

from conftest import requires_judge_llm
from helpers import voice_config
from jarvis.memory.graph_ops import extract_graph_memories

pytestmark = [pytest.mark.eval, requires_judge_llm]


@pytest.mark.parametrize(('summary', 'keywords'), [
    (
        "The user is planning to move from London to Tbilisi, Georgia in June 2026. "
        "They've already secured a flat in Vera district for 800 USD per month. "
        "They work remotely as a software engineer for a UK-based startup called Equals Money.",
        ('Tbilisi', '800', 'Equals Money'),
    ),
    (
        "The user lives in Leeds, has a dog named Juniper, and works as a nurse "
        "at Northgate Clinic. They prefer replies in Spanish.",
        ('Leeds', 'Juniper', 'Northgate', 'Spanish'),
    ),
    (
        "Kullanıcı İzmir'de yaşıyor. Atlas adlı bir kedisi var. "
        "Kuzey Koleji'nde matematik öğretmeni olarak çalışıyor.",
        ('İzmir', 'Atlas', 'Kuzey'),
    ),
])
def test_composite_personal_facts_reach_memory(summary, keywords):
    cfg = voice_config()
    facts = extract_graph_memories(
        summary, cfg, cfg.llm_chat_model, timeout_sec=cfg.llm_chat_timeout_sec,
        thinking=False, date_utc='2026-04-12',
    )
    def comparable(text):
        return ''.join(
            char for char in unicodedata.normalize('NFKD', text.casefold())
            if not unicodedata.combining(char)
        )

    text = comparable(' '.join(fact for _, fact in facts))
    for keyword in keywords:
        assert comparable(keyword) in text, facts
