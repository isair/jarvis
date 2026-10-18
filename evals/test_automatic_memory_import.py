"""Automatic source updates remain usable through the production profile format."""
from contextlib import closing
import pytest
from evals.conftest import requires_judge_llm
from evals.helpers import voice_config
from jarvis.llm import get_llm_backend
from jarvis.memory import auto_import
from jarvis.memory.graph import GraphMemoryStore
from jarvis.memory.graph_ops import build_warm_profile,format_warm_profile_block
from jarvis.memory.imports import MemorySource,ImportChoice,read_source,commit_import
from jarvis.system_prompt import build_system_prompt

pytestmark=[pytest.mark.eval,requires_judge_llm]


@pytest.mark.parametrize('scenario',['city','protected','directive'])
def test_updated_source_profile_is_used_without_overwriting_human_edits(tmp_path,scenario):
    path=tmp_path/'source.md'
    path.write_text('Start every reply with Ahoy.' if scenario=='directive' else 'I live in London.')
    preview=read_source(MemorySource('Claude','Synthetic source','personal',path))
    branch='directives' if scenario=='directive' else 'user'
    with closing(GraphMemoryStore(str(tmp_path/'graph.db'))) as store:
        choices=[ImportChoice(i.id,i.text,branch,None) for i in preview.items]
        result=commit_import(store,preview,choices)
        grant=auto_import.record_success(store,preview,choices,result)
        grant=auto_import.decide(store,grant.source_id,True)
        if scenario=='protected':
            store.update_node(result.node_ids[0],data='I live in Oxford.')
        path.write_text('Start every reply with Hello.' if scenario=='directive' else 'I live in Bath.')
        state=auto_import.sync_source(store,grant.source_id,grant.generation,read_source(preview.source))
        assert state==('review' if scenario=='protected' else 'updated')
        profile=format_warm_profile_block(build_warm_profile(store))
    cfg=voice_config()
    question='What is three plus three?' if scenario=='directive' else 'What city do I live in?'
    answer=get_llm_backend(cfg).direct(cfg.llm_chat_model,build_system_prompt()+'\n'+profile,
                                      question,timeout_sec=60,temperature=0.0,max_tokens=256)
    assert answer,'No reply'
    if scenario=='directive':
        assert answer.lower().startswith('hello'),answer
        assert 'ahoy' not in answer.lower(),answer
    else:
        expected='oxford' if scenario=='protected' else 'bath'
        assert expected in answer.lower(),answer
        assert 'london' not in answer.lower(),answer
