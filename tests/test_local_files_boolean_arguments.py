"""Local listings honour the schema's recursion boundary."""
import os

import pytest
from jarvis.tools.base import ToolContext
from jarvis.tools.builtin.local_files import LocalFilesTool

pytestmark = pytest.mark.unit


@pytest.fixture
def private_listing(monkeypatch, tmp_path, mock_config):
    (tmp_path / 'visible.txt').write_text('top-level')
    nested = tmp_path / 'nested'
    nested.mkdir()
    (nested / 'hidden.txt').write_text('nested')
    original_expand = os.path.expanduser
    monkeypatch.setattr(os.path, 'expanduser', lambda path: str(tmp_path) if path == '~' else original_expand(path))
    context = ToolContext(None, mock_config, '', '', '', 0, lambda message: None)
    return LocalFilesTool(), context, str(tmp_path)


@pytest.mark.parametrize('recursive', ['false', 'true', '', None, 0, 1, [], [False], {}, {'flag': False}])
def test_non_boolean_recursion_cannot_return_a_listing(private_listing, recursive):
    tool, context, path = private_listing
    result = tool.run({'operation': 'list', 'path': path, 'recursive': recursive}, context)
    assert not result.success, '📁 Malformed recursion needs a correctable tool error'
    assert 'boolean' in result.reply_text.casefold()
    assert 'visible.txt' not in result.reply_text and 'hidden.txt' not in result.reply_text


@pytest.mark.parametrize('arguments, includes_nested', [({}, False), ({'recursive': False}, False), ({'recursive': True}, True)])
def test_valid_recursion_and_default_keep_their_listing_boundary(private_listing, arguments, includes_nested):
    tool, context, path = private_listing
    result = tool.run({'operation': 'list', 'path': path, **arguments}, context)
    assert result.success
    assert 'visible.txt' in result.reply_text
    assert ('hidden.txt' in result.reply_text) is includes_nested
