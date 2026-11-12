"""Malformed file paths cannot select or alter stringified filenames."""
import os
import pytest
from jarvis.tools.base import ToolContext
from jarvis.tools.builtin.local_files import LocalFilesTool
pytestmark = pytest.mark.unit

@pytest.fixture
def private_files(monkeypatch, tmp_path, mock_config):
    original_expand = os.path.expanduser
    monkeypatch.setattr(os.path, 'expanduser', lambda path: str(tmp_path) if path == '~' else original_expand(path))
    monkeypatch.chdir(tmp_path)
    context = ToolContext(None, mock_config, '', '', '', 0, lambda message: None)
    return LocalFilesTool(), context, tmp_path

@pytest.mark.parametrize('path', [123, 1.25, True, ['notes.txt'], {'file': 'notes.txt'}])
@pytest.mark.parametrize('operation', ['list', 'read', 'write', 'append', 'delete'])
def test_non_string_paths_cannot_select_stringified_files(private_files, path, operation):
    tool, context, home = private_files
    target = home / str(path)
    target.write_text('original private fixture')
    result = tool.run({'operation': operation, 'path': path, 'content': 'changed'}, context)
    assert not result.success, '📁 A non-string path must return a correction error'
    assert target.read_text() == 'original private fixture', '📁 Malformed paths must never alter files'
    assert 'original private fixture' not in result.reply_text
    assert 'string' in result.reply_text.casefold()

@pytest.mark.parametrize('operation', ['list', 'read', 'write', 'append', 'delete'])
@pytest.mark.parametrize('relative', [False, True])
def test_string_paths_keep_file_operations(private_files, operation, relative):
    tool, context, home = private_files
    target = home / ('123' if relative else 'notes.txt')
    target.write_text('original')
    path = target.name if relative else str(target)
    result = tool.run({'operation': operation, 'path': path, 'content': 'changed'}, context)
    assert result.success, result.reply_text
    if operation == 'read': assert result.reply_text == 'original'
    elif operation == 'write': assert target.read_text() == 'changed'
    elif operation == 'append': assert target.read_text() == 'originalchanged'
    elif operation == 'delete': assert not target.exists()
    else: assert target.name in result.reply_text
