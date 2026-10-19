"""Local reads retain their text limit without allocating the whole source."""
import os
import re
import tracemalloc

import pytest

from jarvis.tools.base import ToolContext
from jarvis.tools.builtin.local_files import LocalFilesTool

pytestmark = pytest.mark.unit


def test_large_local_read_keeps_allocation_bounded(tmp_path, monkeypatch):
    pattern = '🦦 café\n'
    source = tmp_path / 'large.txt'
    with source.open('w', encoding='utf-8') as stream:
        for _ in range(256):
            stream.write(pattern * 2048)
    expanduser = os.path.expanduser
    monkeypatch.setattr(os.path, 'expanduser', lambda path: str(tmp_path) if path == '~' else expanduser(path))
    context = ToolContext(None, None, '', '', '', 1, lambda message: None)
    tracemalloc.start()
    try:
        result = LocalFilesTool().run({'operation': 'read', 'path': str(source)}, context)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert result.success
    match = re.fullmatch(r'(.*)\n\.\.\. \(truncated, showing first (\d+) chars\)', result.reply_text, re.DOTALL)
    assert match
    returned, limit = match.group(1), int(match.group(2))
    assert returned == (pattern * (limit // len(pattern) + 1))[:limit]
    assert peak < source.stat().st_size // 4, 'A short preview must not allocate the entire source'

    for size in (limit - 1, limit, limit + 1):
        text = (pattern * (size // len(pattern) + 1))[:size]
        source.write_text(text, encoding='utf-8')
        preview = LocalFilesTool().run({'operation': 'read', 'path': str(source)}, context)
        assert preview.success
        if size <= limit:
            assert preview.reply_text == text
        else:
            assert preview.reply_text == text[:limit] + match.group(0)[len(returned):]
