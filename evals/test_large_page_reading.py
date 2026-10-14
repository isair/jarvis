"""Local chat uses streamed page text and preserves partial-read uncertainty."""
from contextlib import nullcontext
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from evals.conftest import requires_judge_llm
from evals.helpers import voice_config
from jarvis.llm import get_llm_backend
from jarvis.system_prompt import build_system_prompt
from jarvis.tools.builtin import fetch_web_page

pytestmark = [pytest.mark.eval, requires_judge_llm]


@pytest.mark.parametrize('partial', [False, True])
@pytest.mark.parametrize('sample', range(3))
def test_reading_large_pages_preserves_facts_and_unread_footer(partial, sample):
    article = b'<p>The assembly status indicator is amber.</p>'
    script = b'<script>'+b'x'*(3*1024*1024)+b'</script>'
    body = b'<html><body>'+(article+script if partial else script+article)+b'</body></html>'
    response = SimpleNamespace(
        raise_for_status=lambda: None, encoding='utf-8',
        headers={'Content-Type': 'text/html; charset=utf-8'}, url='https://example.test/article',
        iter_content=lambda chunk_size: (body[i:i+chunk_size] for i in range(0,len(body),chunk_size)),
    )
    with patch.object(fetch_web_page.requests, 'get', return_value=nullcontext(response)), patch.object(
            fetch_web_page, '_MAX_FETCH_BYTES', 4096 if partial else 4*1024*1024):
        result = fetch_web_page.FetchWebPageTool().run(
            {'url': response.url}, SimpleNamespace(user_print=Mock()),
        )
    assert result.success, result.reply_text
    cfg = voice_config()
    answer = get_llm_backend(cfg).direct(
        cfg.llm_chat_model, build_system_prompt(),
        'Using only this page result, return a JSON object with indicator (colour), '
        'footer_code (null if unavailable) and partial (whether the page read is incomplete).\n\n'
        +result.reply_text,
        timeout_sec=60, temperature=0.0, max_tokens=1024,
    )
    assert answer, 'No reply'
    start = answer.find('{')
    assert start >= 0, answer
    parsed, _ = json.JSONDecoder().raw_decode(answer[start:])
    assert parsed == {'indicator': 'amber', 'footer_code': None, 'partial': partial}, answer
