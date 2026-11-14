"""Bounded, provider-agnostic English rendering for local speech delivery."""
import json
import re

from ..debug import debug_log
from ..llm import get_llm_backend, resolve_model, Tier
from ..system_prompt import ENGLISH_SPEECH_RENDER_PROMPT


_MAX_RENDER_INPUT_BYTES = 4096
_QUOTED_LITERAL = re.compile(r'"(?:\\.|[^"\\])*"|“[^”]*”|«[^»]*»')


def render_english_speech(cfg, text: str) -> str:
    """Translate a completed reply without reinterpreting or executing its request."""
    content = json.dumps({'text_to_translate': text}, ensure_ascii=False)
    if len(content.encode('utf-8')) > _MAX_RENDER_INPUT_BYTES:
        debug_log('speech rendering: input exceeds bounded context, retaining original reply', 'planning')
        return text
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
        debug_log('speech rendering: protected input exceeds bounded context, retaining original reply', 'planning')
        return text
    try:
        rendered = get_llm_backend(cfg).direct(
            resolve_model(cfg, Tier.CHAT), ENGLISH_SPEECH_RENDER_PROMPT, content,
            timeout_sec=float(cfg.llm_chat_timeout_sec), thinking=False,
            num_ctx=8192, temperature=0.0, max_tokens=2048,
        )
    except Exception as exc:
        debug_log(f'speech rendering unavailable: {type(exc).__name__}', 'planning')
        return text
    if not isinstance(rendered, str) or not rendered.strip():
        debug_log('speech rendering unavailable: retaining original reply', 'planning')
        return text
    markers = [marker for marker, _ in literals]
    positions = [rendered.find(marker) for marker in markers]
    if (any(rendered.count(marker) != 1 for marker in markers)
            or positions != sorted(positions)
            or rendered.count(prefix) != len(markers)):
        debug_log('speech rendering: literal preservation failed, retaining original reply', 'planning')
        return text
    for marker, literal in literals:
        rendered = rendered.replace(marker, literal)
    debug_log('speech rendering: English reply prepared', 'planning')
    return rendered.strip()
