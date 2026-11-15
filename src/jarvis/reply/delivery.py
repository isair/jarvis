"""Bounded, provider-agnostic rendering for shared reply delivery."""
import json
import re
import time

from ..debug import debug_log
from ..llm import get_llm_backend, resolve_model, Tier
from ..system_prompt import ENGLISH_SPEECH_RENDER_PROMPT, REPLY_RENDER_PROMPT, WRITTEN_LANGUAGE_PROMPT

_MAX_RENDER_INPUT_BYTES = 4096
_QUOTED_LITERAL = re.compile(r'"(?:\\.|[^"\\])*"|“[^”]*”|«[^»]*»')


def _prepare(text):
    content = json.dumps({'text_to_translate': text}, ensure_ascii=False)
    if len(content.encode('utf-8')) > _MAX_RENDER_INPUT_BYTES:
        return None
    prefix = 'JARVIS_LITERAL_'
    while prefix in text:
        prefix = '_' + prefix
    literals = []
    def protect(match):
        marker = f'{prefix}{len(literals)}_END'
        literals.append((marker, match.group()))
        return marker
    masked = _QUOTED_LITERAL.sub(protect, text)
    content = json.dumps({'text_to_translate': masked}, ensure_ascii=False)
    if len(content.encode('utf-8')) > _MAX_RENDER_INPUT_BYTES:
        return None
    return content, literals, prefix


def _render(cfg, text, prompt, timeout, prepared):
    if prepared is None or timeout <= 0:
        debug_log('reply rendering: packet exceeds budget or deadline', 'planning')
        return text
    content, literals, prefix = prepared
    try:
        rendered = get_llm_backend(cfg).direct(
            resolve_model(cfg, Tier.CHAT), prompt, content, timeout_sec=timeout,
            thinking=False, num_ctx=8192, temperature=0.0, max_tokens=2048,
        )
    except Exception as exc:
        debug_log(f'reply rendering unavailable: {type(exc).__name__}', 'planning')
        return text
    if not isinstance(rendered, str) or not rendered.strip():
        return text
    markers = [marker for marker, _ in literals]
    positions = [rendered.find(marker) for marker in markers]
    if (any(rendered.count(marker) != 1 for marker in markers)
            or positions != sorted(positions) or rendered.count(prefix) != len(markers)):
        debug_log('reply rendering: literal preservation failed', 'planning')
        return text
    for marker, literal in literals:
        rendered = rendered.replace(marker, literal)
    debug_log('reply rendering: translated reply prepared', 'planning')
    return rendered.strip()


def render_english_speech(cfg, text: str) -> str:
    """Translate completed spoken output without execution authority."""
    return _render(cfg, text, ENGLISH_SPEECH_RENDER_PROMPT, float(cfg.llm_chat_timeout_sec), _prepare(text))


def render_written_reply(cfg, request: str, text: str, cancel_event=None) -> str:
    """Select the request's written language, then translate the completed text."""
    prepared = _prepare(text)
    reference = json.dumps({'current_request': request}, ensure_ascii=False)
    if prepared is None or len(reference.encode('utf-8')) > _MAX_RENDER_INPUT_BYTES:
        debug_log('written rendering: packet exceeds budget', 'planning')
        return text
    deadline = time.monotonic() + float(cfg.llm_chat_timeout_sec)
    try:
        language = get_llm_backend(cfg).direct(
            resolve_model(cfg, Tier.CHAT), WRITTEN_LANGUAGE_PROMPT, reference,
            timeout_sec=float(cfg.llm_chat_timeout_sec), thinking=False,
            num_ctx=8192, temperature=0.0, max_tokens=64,
        )
    except Exception as exc:
        debug_log(f'written language unavailable: {type(exc).__name__}', 'planning')
        return text
    if cancel_event is not None and cancel_event.is_set():
        return text
    if not isinstance(language, str):
        return text
    language = language.strip()
    if (not language or len(language.encode('utf-8')) > 80
            or not all(c.isalpha() or c in " -" for c in language)):
        debug_log('written language: invalid classification', 'planning')
        return text
    debug_log('written language selected for translation', 'planning')
    return _render(cfg, text, REPLY_RENDER_PROMPT.format(language=language),
                   deadline - time.monotonic(), prepared)
