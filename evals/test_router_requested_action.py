"""Live routing distinguishes requested operations from shared subjects."""

import pytest

from evals.tool_routing import requires_judge_llm, route_tools, routing_config
from jarvis.tools.registry import ToolSpec


@pytest.fixture
def browser_and_transcript_tools():
    descriptions = {
        'chrome-devtools__navigate_page': 'Go to a URL, or back, forward, or reload. Use project URL if not specified otherwise.',
        'chrome-devtools__new_page': 'Open a new tab and load a URL. Use project URL if not specified otherwise.',
        'chrome-devtools__list_pages': 'Get a list of pages open in the browser.',
        'youtube-transcript__get_transcript': 'Extract transcript from a YouTube video URL or ID',
    }
    schemas = {
        'chrome-devtools__navigate_page': {
            'type': 'object', 'properties': {
                'url': {'type': 'string'}, 'pageId': {'type': 'integer'},
            }, 'required': ['pageId'],
        },
        'chrome-devtools__new_page': {
            'type': 'object', 'properties': {'url': {'type': 'string'}},
            'required': ['url'],
        },
        'chrome-devtools__list_pages': {'type': 'object', 'properties': {}},
        'youtube-transcript__get_transcript': {
            'type': 'object', 'properties': {'url': {'type': 'string'}},
            'required': ['url'],
        },
    }
    return {
        name: ToolSpec(name, description, schemas[name])
        for name, description in descriptions.items()
    }


@pytest.mark.eval
@requires_judge_llm
@pytest.mark.parametrize('query', [
    'open YouTube',
    'open youtube on chrome',
    'ouvre YouTube',
    'YouTube öffnen',
    'abre YouTube',
    'YouTubeを開いて',
    'open the BBC website',
    'open a recipe website in Chrome',
])
def test_open_site_routes_to_browser(browser_and_transcript_tools, query):
    selected, reply = route_tools(routing_config(), query, mcp_tools=browser_and_transcript_tools)
    assert set(selected) & {'chrome-devtools__new_page', 'chrome-devtools__navigate_page'}, reply
    assert 'youtube-transcript__get_transcript' not in selected, reply


@pytest.mark.eval
@requires_judge_llm
def test_transcript_request_routes_to_transcription(browser_and_transcript_tools):
    selected, reply = route_tools(
        routing_config(), 'get a transcript of https://www.youtube.com/watch?v=abcdefghijk',
        mcp_tools=browser_and_transcript_tools,
    )
    assert 'youtube-transcript__get_transcript' in selected, reply
    assert not set(selected) & {'chrome-devtools__new_page', 'chrome-devtools__navigate_page'}, reply


@pytest.mark.eval
@requires_judge_llm
@pytest.mark.parametrize('query, required, excluded', [
    ('Show my meals today', 'fetchMeals', {'logMeal', 'deleteMeal'}),
    ('Delete the meal with ID 5', 'deleteMeal', {'logMeal'}),
    ('Log that I had porridge', 'logMeal', {'deleteMeal'}),
])
def test_meal_operations_are_distinct(query, required, excluded):
    selected, reply = route_tools(routing_config(), query)
    assert required in selected, reply
    assert not set(selected) & excluded, reply


@pytest.mark.eval
@requires_judge_llm
@pytest.mark.parametrize('query', ['open youtube on chrome', 'ouvre YouTube'])
def test_browser_open_completes_through_real_router_and_planner(
    monkeypatch, mock_config, eval_db, eval_dialogue_memory,
    configure_mcp_tools, browser_and_transcript_tools, query,
):
    import json
    from urllib.parse import urlparse

    from jsonschema import validate
    from jsonschema.exceptions import ValidationError
    from jarvis.reply.engine import run_reply_engine
    from jarvis.tools.types import ToolExecutionResult
    from helpers import assert_not_fallback_reply, assert_not_max_turns_digest

    catalogue = configure_mcp_tools(*browser_and_transcript_tools.values())
    mock_config.fast_model = mock_config.llm_chat_model
    from jarvis.reply import engine
    from jarvis.llm import get_llm_backend

    original_runner = engine.run_tool_with_retries
    backend = get_llm_backend(mock_config)
    original_direct = backend.direct
    router_answers = []

    def record_direct(*args, **kwargs):
        answer = original_direct(*args, **kwargs)
        if len(args) > 1 and args[1].startswith('You are a tool router.'):
            router_answers.append(answer)
        return answer

    monkeypatch.setattr(backend, 'direct', record_direct)
    monkeypatch.setattr(engine, 'get_llm_backend', lambda cfg: backend)
    opened = []
    transcript_calls = []

    def run_tool(db, cfg, tool_name, tool_args, **kwargs):
        args = tool_args or {}
        if tool_name == 'toolSearchTool':
            return original_runner(db, cfg, tool_name, args, **kwargs)
        if tool_name in catalogue:
            try:
                validate(args, catalogue[tool_name].inputSchema)
            except ValidationError as error:
                return ToolExecutionResult(success=False, reply_text=None, error_message=str(error))
        if tool_name == 'chrome-devtools__new_page':
            opened.append(args['url'])
            return ToolExecutionResult(success=True, reply_text=json.dumps({
                'url': args['url'], 'status': 'opened',
            }))
        if tool_name == 'chrome-devtools__list_pages':
            return ToolExecutionResult(success=True, reply_text='No open pages.')
        if tool_name == 'youtube-transcript__get_transcript':
            transcript_calls.append(args)
        return ToolExecutionResult(success=False, reply_text=None, error_message='No page or transcript available.')

    monkeypatch.setattr('jarvis.reply.engine.run_tool_with_retries', run_tool)
    monkeypatch.setattr(
        'jarvis.reply.engine.get_location_context_with_timezone', lambda cfg: ('', None),
    )
    reply = run_reply_engine(
        db=eval_db, cfg=mock_config, tts=None, text=query,
        dialogue_memory=eval_dialogue_memory,
    )
    assert any(urlparse(url).hostname in {'youtube.com', 'www.youtube.com'} for url in opened), router_answers
    assert not transcript_calls
    assert_not_fallback_reply(reply, context='browser-open')
    assert_not_max_turns_digest(reply, context='browser-open')
