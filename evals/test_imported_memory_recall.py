"""Reviewed imported personal facts and instructions are usable in replies."""
from contextlib import closing
import json
import pytest
from evals.conftest import requires_judge_llm
from evals.helpers import voice_config
from jarvis.llm import get_llm_backend
from jarvis.memory.graph import GraphMemoryStore
from jarvis.memory.graph_ops import build_warm_profile, format_warm_profile_block
from jarvis.memory.imports import ImportChoice, commit_import, preview_text
from jarvis.system_prompt import build_system_prompt

pytestmark = [pytest.mark.eval, requires_judge_llm]

@pytest.mark.parametrize('provider', ['ChatGPT', 'Claude', 'Claude Code'])
@pytest.mark.parametrize('sample',range(2))
def test_imported_personal_facts_and_address_preference(tmp_path,provider,sample):
    facts = 'My name is Mira. I live in London. I go to the office three days a week.'
    rule = 'Address me as Captain.'
    preview = preview_text(provider,facts+'\n\n'+rule,label='Synthetic memory text')
    with closing(GraphMemoryStore(str(tmp_path/'graph.db'))) as store:
        commit_import(store,preview,[ImportChoice(preview.items[0].id,facts,'user',None),
                                     ImportChoice(preview.items[1].id,rule,'directives',None)])
        profile = format_warm_profile_block(build_warm_profile(store))
    cfg=voice_config()
    answer=get_llm_backend(cfg).direct(cfg.llm_chat_model,build_system_prompt()+'\n'+profile,
        'What do you know about my name, home city and weekly office routine? '
        'Return a JSON object with name, home, office_days (a number), and address '
        '(the form of address I asked you to use).',
        timeout_sec=60,temperature=0.0,max_tokens=1024)
    assert answer,'No reply'
    start=answer.find('{')
    assert start>=0,answer
    result,_=json.JSONDecoder().raw_decode(answer[start:])
    assert result=={'name':'Mira','home':'London','office_days':3,'address':'Captain'},answer
