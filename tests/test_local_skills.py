import json
from types import SimpleNamespace

import pytest

from jarvis.skills import SkillSession, without_skill_messages

pytestmark = pytest.mark.unit


def package(root, name='tea-guide', description='Brew sencha tea', body='Use water at 75 degrees.'):
    path = root / name
    path.mkdir(parents=True, exist_ok=True)
    (path / 'SKILL.md').write_text(f'---\nname: {name}\ndescription: {description}\n---\n{body}', encoding='utf-8')
    return path


def session(root, enabled=True):
    return SkillSession(SimpleNamespace(skills_enabled=enabled, skills_paths=[str(root)]))


def test_disabled_reads_nothing(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Disabled skills accessed the filesystem')
    monkeypatch.setattr('jarvis.skills.scan_document_directory', forbidden)
    s = session(tmp_path, False)
    assert s.search('tea') == []
    with pytest.raises(ValueError, match='disabled'):
        s.read('tea-guide')


def test_progressive_disclosure_and_request_scope(tmp_path):
    p = package(tmp_path)
    (p / 'references').mkdir()
    (p / 'references' / 'timing.md').write_text('Steep for 90 seconds.', encoding='utf-8')
    s = session(tmp_path)
    hits = s.search('sencha')
    assert hits[0]['name'] == 'tea-guide'
    assert hits[0]['path'] == hits[0]['id'] + '/SKILL.md'
    assert '75 degrees' not in json.dumps(hits)
    instructions = s.read(hits[0]['id'])
    reference = s.read(hits[0]['id'], 'references/timing.md')
    assert '75 degrees' in instructions['instructions'] and '90 seconds' in reference['instructions']
    assert 'instructions' not in s.read(hits[0]['id'])
    assert session(tmp_path).read('tea-guide') == instructions


def test_unbounded_catalogue_bounded_disclosure(tmp_path):
    for i in range(260):
        package(tmp_path, f'guide-{i}', f'Common guide for activity {i}', 'SECRET BODY')
    s = session(tmp_path)
    assert s.catalogue_size == 260
    assert s.search('guide-259')[0]['name'] == 'guide-259'
    outputs = [s.search('Common') for _ in range(30)]
    assert all(len(hits) <= 5 for hits in outputs)
    assert sum(len(json.dumps(hits, ensure_ascii=False).encode()) for hits in outputs if hits) <= s.METADATA_BYTES
    assert 'SECRET BODY' not in json.dumps(outputs)


def test_long_unicode_description_remains_discoverable(tmp_path):
    package(tmp_path, description='茶' * 1023 + '湯')
    hits = session(tmp_path).search('茶' * 1023 + '湯')
    assert len(hits) == 1
    assert hits[0]['description_truncated'] is True
    assert hits[0]['name'] == 'tea-guide'


def test_unicode_metadata_and_ambiguous_names(tmp_path):
    package(tmp_path / 'one', description='İstanbul çay hazırlama')
    package(tmp_path / 'two', description='İstanbul çay hazırlama')
    s = SkillSession(SimpleNamespace(skills_enabled=True, skills_paths=[str(tmp_path/'one'), str(tmp_path/'two')]))
    hits = s.search('çay')
    assert len(hits) == 2
    with pytest.raises(ValueError, match='ambiguous'):
        s.read('tea-guide')
    s.read(hits[0]['id'])


@pytest.mark.parametrize('relative', ['../outside.md', '/outside.md', 'references/../../outside.md', 'C:\\outside.md'])
def test_reference_confinement(tmp_path, relative):
    package(tmp_path)
    with pytest.raises(ValueError):
        session(tmp_path).read('tea-guide', relative)


def test_symlinks_and_oversized_files_are_not_loaded(tmp_path):
    p = package(tmp_path, body='x' * 20000)
    outside = tmp_path / 'outside.md'
    outside.write_text('PRIVATE', encoding='utf-8')
    (p / 'reference.md').symlink_to(outside)
    s = session(tmp_path)
    with pytest.raises(ValueError, match='large'):
        s.read('tea-guide')
    with pytest.raises((OSError, ValueError)):
        s.read('tea-guide', 'reference.md')


def test_native_mixed_batch_carryover_remains_paired():
    messages = [
        {'role': 'assistant', 'content': 'SECRET SKILL INSTRUCTIONS', 'thinking': 'SECRET SKILL INSTRUCTIONS', 'tool_calls': [
            {'id': 's', 'function': {'name': 'readSkill', 'arguments': {'path': 'tea-guide/SKILL.md'}}},
            {'id': 'w', 'function': {'name': 'getWeather', 'arguments': {}}},
        ]},
        {'role': 'tool', 'tool_call_id': 's', 'tool_name': 'readSkill', 'content': 'loaded'},
        {'role': 'tool', 'tool_call_id': 'w', 'tool_name': 'getWeather', 'content': 'sunny'},
    ]
    kept = without_skill_messages(messages)
    assert len(kept) == 2
    assert [call['id'] for call in kept[0]['tool_calls']] == ['w']
    assert kept[1]['tool_call_id'] == 'w'
    assert kept[0]['content'] == ''
    assert 'thinking' not in kept[0]
    assert len(messages[0]['tool_calls']) == 2


def test_skill_reasoning_is_not_saved_in_later_native_calls():
    from jarvis.memory.conversation import DialogueMemory
    messages = [
        {'role': 'tool', 'tool_name': 'readSkill', 'tool_call_id': 's', 'content': 'SECRET SKILL INSTRUCTIONS'},
        {'role': 'assistant', 'content': 'SECRET SKILL INSTRUCTIONS', 'thinking': 'SECRET SKILL INSTRUCTIONS',
         'tool_calls': [{'id': 'w', 'function': {'name': 'getWeather', 'arguments': {}}}]},
        {'role': 'tool', 'tool_call_id': 'w', 'tool_name': 'getWeather', 'content': 'sunny'},
    ]
    memory = DialogueMemory()
    memory.record_tool_turn(without_skill_messages(messages))
    carried = memory.get_recent_turns_with_tools()
    assert 'SECRET SKILL INSTRUCTIONS' not in json.dumps(carried)
    assert 'sunny' in json.dumps(carried)


def test_yaml_aliases_are_rejected_before_metadata_construction(tmp_path):
    p = package(tmp_path)
    (p / 'SKILL.md').write_text(
        '---\nbase: &base {x: y}\nmerged: {<<: [*base, *base]}\n'
        'name: tea-guide\ndescription: Brew tea\n---\nInstructions', encoding='utf-8')
    assert session(tmp_path).search('tea') == []


def test_folded_yaml_description_is_supported(tmp_path):
    p = package(tmp_path)
    (p / 'SKILL.md').write_text(
        '---\nname: tea-guide\ndescription: >\n  Brew Japanese tea\n  with local instructions\n---\nBrew it.', encoding='utf-8')
    assert session(tmp_path).search('Japanese')[0]['name'] == 'tea-guide'


def test_unrelated_carryover_is_unchanged():
    message = {'role': 'assistant', 'content': 'Checking the forecast', 'tool_calls': [
        {'id': 'w', 'function': {'name': 'getWeather', 'arguments': {}}}]}
    assert without_skill_messages([message]) == [message]


def test_permission_metadata_does_not_grant_execution(tmp_path):
    p = package(tmp_path)
    (p / 'SKILL.md').write_text('---\nname: tea-guide\ndescription: Tea\nallowed-tools: shell(*)\n---\nBrew tea.', encoding='utf-8')
    s = session(tmp_path)
    result = s.read('tea-guide')
    assert s.search('shell') == []
    assert result['skill'] == 'tea-guide'
    assert set(result) == {'skill', 'file', 'instructions'}


def test_loading_budget_rejects_whole_file_without_losing_prior_guidance(tmp_path):
    p = package(tmp_path, body='x' * 10000)
    (p / 'extra.md').write_text('y' * 10000, encoding='utf-8')
    s = session(tmp_path)
    instructions = s.read('tea-guide')
    with pytest.raises(ValueError, match='budget'):
        s.read('tea-guide', 'extra.md')
    assert s.read('tea-guide')['already_loaded'] is True
    assert len(instructions['instructions']) > 10000


def test_cancelled_session_cannot_publish_new_guidance(tmp_path):
    import threading
    package(tmp_path)
    event = threading.Event()
    s = SkillSession(SimpleNamespace(skills_enabled=True, skills_paths=[str(tmp_path)]), event)
    event.set()
    assert s.search('tea') == []
    with pytest.raises(ValueError, match='cancelled'):
        s.read('tea-guide')


def test_disabled_catalogue_and_direct_dispatch_grant_no_access(tmp_path):
    from jarvis.tools.registry import available_builtin_tools, run_tool_with_retries
    cfg = SimpleNamespace(skills_enabled=False, skills_paths=[str(tmp_path)])
    assert not {'readSkill', 'searchSkills'} & available_builtin_tools(cfg).keys()
    result = run_tool_with_retries(None, cfg, 'readSkill', {'skill': 'tea-guide'}, '', '', '')
    assert not result.success
    assert 'disabled' in result.error_message


def test_skill_tool_reads_complete_relative_paths(tmp_path):
    from jarvis.tools.builtin.skills import ReadSkillTool
    p = package(tmp_path)
    (p / 'references').mkdir()
    (p / 'references' / 'detail.md').write_text('Owned reference', encoding='utf-8')
    s = session(tmp_path)
    context = SimpleNamespace(cfg=SimpleNamespace(skills_enabled=True), skill_session=s)
    result = ReadSkillTool().run({'path': 'tea-guide/references/detail.md'}, context)
    assert result.success
    assert json.loads(result.reply_text)['instructions'] == 'Owned reference'
    invalid = ReadSkillTool().run({'path': 'tea-guide/../outside.md'}, context)
    assert not invalid.success


@pytest.mark.parametrize('value', [False, 'true', 1, [], None])
def test_configuration_requires_boolean_opt_in(tmp_path, monkeypatch, value):
    from jarvis.config import load_settings
    path = tmp_path / 'config.json'
    path.write_text(json.dumps({'skills_enabled': value, 'skills_paths': ['  chosen  ', 7, '']}))
    monkeypatch.setenv('JARVIS_CONFIG_PATH', str(path))
    cfg = load_settings()
    assert cfg.skills_enabled is False
    assert cfg.skills_paths == ['chosen']
