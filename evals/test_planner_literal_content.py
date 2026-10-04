"""Resolved file-writing steps preserve literal quotation and punctuation."""
import os

import pytest

from evals.helpers import voice_config
from evals.tool_routing import requires_judge_llm
from jarvis.reply.planner import resolve_next_tool_call
from jarvis.tools.base import ToolContext
from jarvis.tools.builtin.local_files import LocalFilesTool

pytestmark = [pytest.mark.eval, requires_judge_llm]
CONTENTS = ["It's a quiet morning.", "L'été arrive bientôt.", 'He said "hello".']


@pytest.mark.parametrize('content', CONTENTS)
def test_resolved_write_preserves_the_complete_literal(monkeypatch, tmp_path, content):
    original_expand = os.path.expanduser
    monkeypatch.setattr(os.path, 'expanduser', lambda path: str(tmp_path) if path == '~' else original_expand(path))
    monkeypatch.chdir(tmp_path)
    cfg = voice_config()
    tool = LocalFilesTool()
    schema = [{'type': 'function', 'function': {
        'name': tool.name, 'description': tool.description, 'parameters': tool.inputSchema,
    }}]
    if '"' in content:
        quoted = '"' + content.replace('"', '\\"') + '"'
    else:
        quoted = "'" + content + "'"
    step = f"localFiles operation='write' path='note.txt' content={quoted}"
    resolved = resolve_next_tool_call(cfg, step, [], schema, timeout_sec=60.0)
    assert resolved is not None and resolved[0] == tool.name, f'📝 Complete content needs a resolved write: {step}'
    result = tool.run(resolved[1], ToolContext(None, cfg, '', '', '', 0, lambda message: None))
    assert result.success, f'📝 Private write failed: {result.reply_text}'
    assert (tmp_path / 'note.txt').read_text() == content, f'📝 Literal content was altered: {resolved}'
