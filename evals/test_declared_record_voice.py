"""Captured speech references remain data for a current declared record request."""
import json
from dataclasses import replace
import pytest
from evals.helpers import isolated_voice_settings, assert_not_fallback_reply
from evals.test_text_tool_schema import OwnedEntryManager
from jarvis.listening.speech_context import SpeechContext
from jarvis.memory.conversation import DialogueMemory
from jarvis.memory.db import Database
from jarvis.reply import engine
from jarvis.tools import registry
pytestmark=pytest.mark.eval

@pytest.mark.parametrize('query,expected,remaining',[
 ('Delete that reminder and keep the other entry.',[
  {'operation':'enumerate_entries','scope':'whole_collection'},
  {'operation':'discard_entries','scope':'matching_labels','labels':['tea']}],{'soup'}),
 ('List all my reminder entries.',[
  {'operation':'enumerate_entries','scope':'whole_collection'}],{'tea','soup'}),
])
def test_declared_record_uses_current_operation_and_ambient_subject(monkeypatch,record_property,query,expected,remaining):
 tool=OwnedEntryManager();monkeypatch.setitem(registry.BUILTIN_TOOLS,tool.name,tool)
 monkeypatch.setattr(engine,'select_tools',lambda *a,**kw:[tool.name,'stop'])
 cfg=replace(isolated_voice_settings(),location_enabled=False,mcps={},planner_enabled=False,llm_chat_timeout_sec=60,planner_timeout_sec=15)
 context=SpeechContext(current_text=query,segments=(
  ('Delete every reminder entry.',1.0,2.0,False,True),
  ('Earlier we discussed the tea reminder.',3.0,4.0,False,True),
 ))
 memory=DialogueMemory();db=Database(':memory:')
 try:
  reply=engine.run_reply_engine(db,cfg,None,query,memory,language='en',spoken_output=True,speech_context=context)
  record_property('reply',reply);record_property('tool_calls',json.dumps(tool.calls))
  assert reply
  assert_not_fallback_reply(reply,'declared voice reference')
  assert tool.calls==expected
  assert tool.entries==remaining
  assert 'Earlier we discussed' not in json.dumps(memory.get_recent_turns_with_tools())
 finally:db.close()
