"""Run memory-digest evals through the configured local model."""
from evals.helpers import voice_config
from jarvis.reply import enrichment


def digest_for_eval(query: str, diary_entries: list[str]) -> str:
    """Keep short fixtures above the computed threshold so digestion runs."""
    entries = list(diary_entries)
    neutral = '[Reference] This entry supplies no additional personal facts or preferences.'
    while sum(len(entry) + 1 for entry in entries) < enrichment._DIGEST_MIN_CHARS:
        entries.append(neutral)
    cfg = voice_config()
    return enrichment.digest_memory_for_query(
        query=query, diary_entries=entries, graph_parts=[],
        cfg=cfg, chat_model=cfg.llm_chat_model, timeout_sec=60.0,
    )
