"""Local inference retains complete rewrites and rejects explicitly capped answers."""

from conftest import requires_judge_llm
from helpers import voice_config
from jarvis.llm import get_llm_backend


@requires_judge_llm
def test_explicitly_truncated_direct_answer_is_unusable():
    cfg = voice_config()
    backend = get_llm_backend(cfg)
    instructions = 'Copy every word of the supplied text exactly. Return only the full copied text.'
    text = 'red green blue orange purple yellow silver gold white black'
    raw = backend.chat(cfg.llm_chat_model, [
        {'role': 'system', 'content': instructions},
        {'role': 'user', 'content': text},
    ], thinking=False, extra_options={'max_tokens': 1, 'temperature': 0})
    assert raw and raw.get('message', {}).get('content', '').strip(), 'Probe must contain actual partial text'
    choices = raw.get('choices')
    reason = choices[0].get('finish_reason') if choices else raw.get('done_reason')
    assert reason == 'length', 'Probe must be explicitly truncated by the server'
    assert backend.direct(cfg.llm_chat_model, instructions, text,
                          thinking=False, temperature=0, max_tokens=1) is None


@requires_judge_llm
def test_complete_direct_rewrite_retains_every_word():
    cfg = voice_config()
    text = 'red green blue orange purple yellow silver gold white black'
    answer = get_llm_backend(cfg).direct(
        cfg.llm_chat_model,
        'Copy every word of the supplied text exactly. Return only the full copied text.',
        text, thinking=False, temperature=0, max_tokens=64,
    )
    assert answer and text.casefold() in answer.casefold()
