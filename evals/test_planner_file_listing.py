"""Planned local listings honour the user's requested recursion boundary."""
import os

import pytest

from evals.helpers import voice_config
from evals.tool_routing import requires_judge_llm
from jarvis.reply.planner import plan_query, resolve_next_tool_call, tool_steps_of
from jarvis.tools.base import ToolContext
from jarvis.tools.builtin.local_files import LocalFilesTool

pytestmark = [pytest.mark.eval, requires_judge_llm]
CASES = [
    (False, 'List the files directly in {path}, without listing files inside subfolders.'),
    (True, 'List the files in {path} recursively, including files inside subfolders.'),
    (False, '{path} klasöründeki dosyaları listele, alt klasörlerin içine girme.'),
]


@pytest.mark.parametrize('recursive, utterance', CASES)
def test_planned_file_listing_preserves_recursion_choice(monkeypatch, tmp_path, recursive, utterance):
    (tmp_path / 'visible.txt').write_text('top-level fixture')
    nested = tmp_path / 'nested'
    nested.mkdir()
    (nested / 'hidden.txt').write_text('nested fixture')
    original_expand = os.path.expanduser
    monkeypatch.setattr(os.path, 'expanduser', lambda path: str(tmp_path) if path == '~' else original_expand(path))
    cfg = voice_config()
    tool = LocalFilesTool()
    monkeypatch.chdir(tmp_path)
    query = utterance.format(path='.')
    plan = plan_query(cfg, query, '', [(tool.name, tool.description)], timeout_sec=60.0)
    steps = tool_steps_of(plan)
    assert steps, f'📂 A requested listing needs an executable plan: {plan}'
    schema = [{'type': 'function', 'function': {
        'name': tool.name, 'description': tool.description, 'parameters': tool.inputSchema,
    }}]
    resolved = resolve_next_tool_call(cfg, steps[0], [], schema, timeout_sec=60.0)
    assert resolved is not None and resolved[0] == tool.name, f'📂 Listing must remain executable: {plan}'
    context = ToolContext(None, cfg, '', query, query, 0, lambda message: None)
    result = tool.run(resolved[1], context)
    assert result.success, f'📂 Planned listing failed: {result.reply_text}'
    assert 'visible.txt' in result.reply_text, f'📂 Top-level file was lost: {resolved}'
    assert ('hidden.txt' in result.reply_text) is recursive, f'📂 Recursion choice was changed: {resolved}'
