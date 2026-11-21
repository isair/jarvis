"""Strict native progressive-disclosure cases through the shared reply engine."""
from dataclasses import replace
import json
import os
import re

import pytest

from evals.helpers import assert_not_fallback_reply, isolated_voice_settings
from jarvis.memory.conversation import DialogueMemory
from jarvis.memory.db import Database
from jarvis.reply import engine
from jarvis.skills import SkillSession

pytestmark = pytest.mark.eval


@pytest.mark.parametrize('spoken_output', [False, True])
@pytest.mark.parametrize('explicit', [False, True])
def test_local_skill_instructions_and_reference_are_used(
    tmp_path, monkeypatch, record_property, spoken_output, explicit,
):
    package = tmp_path / 'kettle-care'
    (package / 'references').mkdir(parents=True)
    (package / 'SKILL.md').write_text(
        '---\nname: kettle-care\ndescription: Local kettle descaling recipe and quantities\n---\n'
        'When asked for the kettle descaling recipe, read references/recipe.md '
        'before answering. Use the quantities in that file. Do not guess quantities.', encoding='utf-8')
    recipe = {'citric_acid_grams': 12, 'water_millilitres': 750}
    (package / 'references' / 'recipe.md').write_text(
        'Dissolve 12 grams of citric acid in 750 millilitres of water. '
        'Follow the kettle manufacturer\'s cleaning instructions.', encoding='utf-8')
    # A large unrelated library must not inflate model context or replace the match.
    for i in range(80):
        folder = tmp_path / f'archive-{i}'
        folder.mkdir()
        (folder / 'SKILL.md').write_text(
            f'---\nname: archive-{i}\ndescription: Organise archived postage stamps\n---\n'
            'UNRELATED-INSTRUCTION-MARKER', encoding='utf-8')
    loaded = []
    original = SkillSession.read

    def read(self, skill, relative='SKILL.md'):
        result = original(self, skill, relative)
        loaded.append((result['skill'], result['file']))
        return result

    monkeypatch.setattr(SkillSession, 'read', read)
    cfg = replace(isolated_voice_settings(),
                  skills_enabled=os.environ.get('SKILLS_EVAL_BASELINE') != '1',
                  skills_paths=[str(tmp_path)], planner_enabled=False,
                  tool_result_digest_enabled=False, mcps={}, location_enabled=False)
    query = ('Use the kettle-care skill to tell me the local kettle descaling recipe and quantities.'
             if explicit else 'What are the quantities in my local kettle descaling recipe?')
    db = Database(':memory:')
    try:
        reply = engine.run_reply_engine(db, cfg, None, query, DialogueMemory(),
                                        language='en', spoken_output=spoken_output)
    finally:
        db.close()
    record_property('reply', reply)
    record_property('loaded_files', json.dumps(loaded))
    assert reply
    assert_not_fallback_reply(reply, 'local skill instructions')
    assert ('kettle-care', 'SKILL.md') in loaded
    assert ('kettle-care', 'references/recipe.md') in loaded
    assert re.search(r'\b12\b|\btwelve\b', reply, re.I), recipe
    assert re.search(r'\b750\b|\bseven hundred(?: and)? fifty\b', reply, re.I), recipe
    assert 'citric' in reply.lower()
    assert all(name == 'kettle-care' for name, _ in loaded)
