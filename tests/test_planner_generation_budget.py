"""Planning answers can follow a bounded reasoning prelude on local servers."""
from types import SimpleNamespace
from unittest.mock import MagicMock
import pytest
from jarvis.reply.planner import plan_query, resolve_next_tool_call

pytestmark=pytest.mark.unit

@pytest.mark.parametrize('provider', ['ollama','openai_compatible'])
@pytest.mark.parametrize('context', ['plan','resolver'])
def test_planner_wire_budget_leaves_room_for_structured_answer(monkeypatch,provider,context):
    plan="searchMemory topic='film interests'\nwebSearch query='film recommendations'\nReply to the user with the combined findings."
    resolved='{"name":"webSearch","arguments":{"query":"Brandon Cronenberg filmography"}}'
    answer=plan if context=='plan' else resolved
    reasoning=' '.join(['Compare the allowed tools and resolve the supplied context.'] * 70)
    required_tokens=len(reasoning.split())+len(answer.split())
    def post(url,**kwargs):
        payload=kwargs['json']
        cap=payload.get('max_tokens',payload.get('options',{}).get('num_predict',0))
        content=answer if cap>=required_tokens else ''
        msg={'content':content,'reasoning_content':reasoning}
        response=MagicMock()
        response.__enter__.return_value=response
        response.json.return_value=({'message':msg} if provider=='ollama' else {'choices':[{'message':msg}]})
        return response
    monkeypatch.setattr('requests.post',post)
    cfg=SimpleNamespace(llm_provider=provider,llm_base_url='http://127.0.0.1:1/v1',
                        ollama_base_url='http://127.0.0.1:1',llm_api_key='',
                        llm_chat_model='local-reasoning-model',fast_model='',
                        planner_enabled=True,planner_timeout_sec=3.)
    if context=='plan':
        result=plan_query(cfg, 'recommend a film based on my interests','', [('webSearch','Search the web')])
        assert result==plan.splitlines(), 'Generation ended during reasoning before the plan'
    else:
        schema=[{'type':'function','function':{'name':'webSearch','description':'Search the web',
                 'parameters':{'type':'object','properties':{'query':{'type':'string'}}}}}]
        result=resolve_next_tool_call(cfg,"webSearch query='<director from prior result> filmography'",
                  [('webSearch','Possessor director','Director: Brandon Cronenberg')],schema)
        assert result==('webSearch',{'query':'Brandon Cronenberg filmography'}), 'Generation ended before tool arguments'
