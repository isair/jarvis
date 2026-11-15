"""Bounded, provider-agnostic rendering for shared reply delivery."""
import json
import re
import time

from ..debug import debug_log
from ..llm import get_llm_backend, resolve_model, Tier
from ..system_prompt import ENGLISH_SPEECH_RENDER_PROMPT, REPLY_RENDER_PROMPT, WRITTEN_LANGUAGE_PROMPT, REQUEST_LANGUAGE_PROMPT

_MAX_RENDER_INPUT_BYTES = 4096
_QUOTED_LITERAL = re.compile(r'"(?:\\.|[^"\\])*"|“[^”]*”|«[^»]*»')
_QUOTED_OR_NUMERIC_LITERAL = re.compile(_QUOTED_LITERAL.pattern + r'|\b\d+(?:\.\d+)?\b')


def _prepare(text, protect_numbers=False):
    packet = {'text_to_translate': text}
    content = json.dumps(packet, ensure_ascii=False)
    if len(content.encode('utf-8')) > _MAX_RENDER_INPUT_BYTES:
        return None
    prefix = 'JARVIS_LITERAL_'
    while prefix in content:
        prefix = '_' + prefix
    literals = []
    def protect(match):
        marker = f'{prefix}{len(literals)}_END'
        literals.append((marker, match.group()))
        return marker
    masked = (_QUOTED_OR_NUMERIC_LITERAL if protect_numbers else _QUOTED_LITERAL).sub(protect, text)
    content = json.dumps({**packet, 'text_to_translate': masked}, ensure_ascii=False)
    if len(content.encode('utf-8')) > _MAX_RENDER_INPUT_BYTES:
        return None
    return content, literals, prefix


def _render(cfg, text, prompt, timeout, prepared):
    if prepared is None or timeout <= 0:
        debug_log('reply rendering: packet exceeds budget or deadline', 'planning')
        return text
    content, literals, prefix = prepared
    if literals:
        prompt += (" Copy the supplied literal markers exactly, once each, in the same order: "
                   + ", ".join(marker for marker, _ in literals)
                   + ". They represent protected literals. Do not translate, expand or invent markers.")
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


def render_english_speech(cfg, text: str, protect_numbers=False, deadline=None) -> str:
    """Translate completed spoken output without execution authority."""
    timeout = float(cfg.llm_chat_timeout_sec) if deadline is None else deadline - time.monotonic()
    return _render(cfg, text, ENGLISH_SPEECH_RENDER_PROMPT, timeout, _prepare(text, protect_numbers))


def render_written_reply(cfg, request: str, text: str, cancel_event=None, protect_numbers=False, deadline=None) -> str:
    """Select the request's written language, then translate the completed text."""
    return _render_request_text(cfg, request, text, WRITTEN_LANGUAGE_PROMPT, None, cancel_event, protect_numbers, deadline)


def render_timer_announcement(cfg, request: str, text: str, language=None, cancel_event=None, deadline=None) -> str:
    """Translate future expiry speech into the captured or inferred request language."""
    return _render_request_text(cfg, request, text, REQUEST_LANGUAGE_PROMPT, language, cancel_event, True, deadline)


def _render_request_text(cfg, request, text, language_prompt, known_language, cancel_event, protect_numbers, deadline):
    prepared = _prepare(text, protect_numbers)
    fallback = text
    if deadline is None:
        deadline = time.monotonic() + float(cfg.llm_chat_timeout_sec)
    if cancel_event is not None and cancel_event.is_set():
        return fallback
    if known_language is not None:
        if not isinstance(known_language, str) or not re.fullmatch(r'[A-Za-z]{2,3}', known_language):
            return fallback
        return _render(cfg, fallback, REPLY_RENDER_PROMPT.format(language=known_language),
                       deadline - time.monotonic(), prepared)
    reference = json.dumps({'current_request': request}, ensure_ascii=False)
    if prepared is None or len(reference.encode('utf-8')) > _MAX_RENDER_INPUT_BYTES:
        debug_log('written rendering: packet exceeds budget', 'planning')
        return fallback
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return fallback
    try:
        language = get_llm_backend(cfg).direct(
            resolve_model(cfg, Tier.CHAT), language_prompt, reference,
            timeout_sec=remaining, thinking=False,
            num_ctx=8192, temperature=0.0, max_tokens=8,
        )
    except Exception as exc:
        debug_log(f'written language unavailable: {type(exc).__name__}', 'planning')
        return fallback
    if cancel_event is not None and cancel_event.is_set():
        return fallback
    if not isinstance(language, str):
        return fallback
    language = language.strip()
    if not re.fullmatch(r'[A-Za-z]{2,3}', language):
        debug_log('written language: invalid classification', 'planning')
        return fallback
    debug_log('written language selected for translation', 'planning')
    return _render(cfg, fallback, REPLY_RENDER_PROMPT.format(language=language),
                   deadline - time.monotonic(), prepared)
