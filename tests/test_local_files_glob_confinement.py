"""Local listing patterns cannot expose files outside the configured home."""
import os

import pytest
from jarvis.tools.base import ToolContext
from jarvis.tools.builtin.local_files import LocalFilesTool

pytestmark = pytest.mark.unit


@pytest.fixture
def listing_sandbox(monkeypatch, tmp_path, mock_config):
    home = tmp_path / 'home'
    home.mkdir()
    (home / 'visible.txt').write_text('allowed')
    nested = home / 'nested'
    nested.mkdir()
    (nested / 'inside.txt').write_text('allowed nested')
    outside = tmp_path / 'outside'
    outside.mkdir()
    (outside / 'private_payload.txt').write_text('private fixture')
    original_expand = os.path.expanduser
    monkeypatch.setattr(os.path, 'expanduser', lambda path: str(home) if path == '~' else original_expand(path))
    context = ToolContext(None, mock_config, '', '', '', 0, lambda message: None)
    return LocalFilesTool(), context, home, outside


@pytest.mark.parametrize('recursive', [False, True])
def test_parent_glob_cannot_return_outside_names(listing_sandbox, recursive):
    tool, context, home, outside = listing_sandbox
    result = tool.run({'operation': 'list', 'path': str(home), 'glob': '../outside/*', 'recursive': recursive}, context)
    assert not result.success, '🔒 Parent glob must not expose a sibling directory'
    assert 'private_payload.txt' not in result.reply_text
    assert str(outside) not in result.reply_text


@pytest.mark.parametrize('recursive', [False, True])
@pytest.mark.parametrize('pattern', ['escape/*', '*/private_payload.txt'])
def test_glob_cannot_follow_a_directory_link_outside_home(listing_sandbox, recursive, pattern):
    tool, context, home, outside = listing_sandbox
    try:
        (home / 'escape').symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip('🔒 Directory symlinks require platform permission')
    result = tool.run({'operation': 'list', 'path': str(home), 'glob': pattern, 'recursive': recursive}, context)
    assert not result.success, '🔒 Listing must not expose an external directory through a link'
    assert 'private_payload.txt' not in result.reply_text
    assert str(outside) not in result.reply_text


@pytest.mark.parametrize('pattern, recursive, expected', [
    ('*', False, ['visible.txt', 'nested']),
    ('nested/*.txt', False, ['nested/inside.txt']),
    ('**/*.txt', False, ['visible.txt', 'nested/inside.txt']),
    ('*.txt', True, ['visible.txt', 'nested/inside.txt']),
])
def test_in_home_patterns_keep_their_matching_entries(listing_sandbox, pattern, recursive, expected):
    tool, context, home, outside = listing_sandbox
    result = tool.run({'operation': 'list', 'path': str(home), 'glob': pattern, 'recursive': recursive}, context)
    assert result.success, result.reply_text
    assert all(entry in result.reply_text.replace('\\', '/') for entry in expected)
    assert 'private_payload.txt' not in result.reply_text



def test_parent_pattern_inside_home_keeps_its_entry(listing_sandbox):
    tool, context, home, outside = listing_sandbox
    result = tool.run({'operation': 'list', 'path': str(home / 'nested'), 'glob': '../*.txt'}, context)
    assert result.success
    assert 'visible.txt' in result.reply_text
    assert 'private_payload.txt' not in result.reply_text


def test_link_to_an_in_home_directory_keeps_matching_children(listing_sandbox):
    tool, context, home, outside = listing_sandbox
    try:
        (home / 'alias').symlink_to(home / 'nested', target_is_directory=True)
    except OSError:
        pytest.skip('🔒 Directory symlinks require platform permission')
    result = tool.run({'operation': 'list', 'path': str(home), 'glob': 'alias/*.txt'}, context)
    assert result.success
    assert 'inside.txt' in result.reply_text


def test_own_directory_link_is_identified_without_listing_external_children(listing_sandbox):
    tool, context, home, outside = listing_sandbox
    try:
        (home / 'escape').symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip('🔒 Directory symlinks require platform permission')
    result = tool.run({'operation': 'list', 'path': str(home)}, context)
    assert result.success
    assert 'LINK: escape' in result.reply_text
    assert 'private_payload.txt' not in result.reply_text


@pytest.mark.parametrize('recursive', [False, True])
def test_exact_parent_selector_cannot_inspect_home_parent(listing_sandbox, recursive):
    tool, context, home, outside = listing_sandbox
    result = tool.run({'operation': 'list', 'path': str(home), 'glob': '..', 'recursive': recursive}, context)
    assert not result.success, '🔒 Exact parent selection must keep the home boundary'
    assert str(home.parent) not in result.reply_text


def test_file_link_reports_only_its_own_name(listing_sandbox):
    tool, context, home, outside = listing_sandbox
    try:
        (home / 'file_alias').symlink_to(outside / 'private_payload.txt')
    except OSError:
        pytest.skip('🔒 Symbolic links require platform permission')
    result = tool.run({'operation': 'list', 'path': str(home), 'glob': 'file_alias'}, context)
    assert result.success
    assert 'LINK: file_alias' in result.reply_text
    assert 'private_payload.txt' not in result.reply_text
    assert str(outside) not in result.reply_text


@pytest.mark.parametrize('pattern', [None, 0, False, []])
def test_invalid_glob_types_remain_correctable_failures(listing_sandbox, pattern):
    tool, context, home, outside = listing_sandbox
    result = tool.run({'operation': 'list', 'path': str(home), 'glob': pattern}, context)
    assert not result.success
    assert 'private_payload.txt' not in result.reply_text



def test_mixed_listing_failure_does_not_return_partial_entries_or_log_private_names(monkeypatch, listing_sandbox):
    tool, context, home, outside = listing_sandbox
    try:
        (home / 'escape').symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip('🔒 Directory symlinks require platform permission')
    messages = []
    monkeypatch.setattr('jarvis.tools.builtin.local_files.debug_log', lambda message, category: messages.append(message))
    result = tool.run({'operation': 'list', 'path': str(home), 'glob': '*/*.txt'}, context)
    assert not result.success
    assert 'inside.txt' not in result.reply_text
    assert 'private_payload.txt' not in result.reply_text
    assert all('private_payload.txt' not in message and str(outside) not in message for message in messages)
