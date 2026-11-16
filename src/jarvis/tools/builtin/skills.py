"""Raw local skill discovery and progressive instruction loading."""
import json

from ..base import Tool
from ..types import ToolExecutionResult


class SearchSkillsTool(Tool):
    name = 'searchSkills'
    description = ('Find local skills by name or words describing a task. Returns a small metadata '
                   'shortlist, not instructions. Use readSkill with the exact returned path before '
                   'following a skill. Try more specific words if no match is found.')
    inputSchema = {'type': 'object', 'properties': {
        'query': {'type': 'string', 'description': 'Skill name or relevant task words'}},
        'required': ['query']}

    def run(self, args, context):
        if context.cfg.skills_enabled is not True or context.skill_session is None:
            return ToolExecutionResult(False, None, error_message='Local skills are disabled or unavailable')
        query = args.get('query') if isinstance(args, dict) else None
        if not isinstance(query, str) or not query.strip():
            return ToolExecutionResult(False, None, error_message='searchSkills requires a non-empty query')
        hits = context.skill_session.search(query)
        return ToolExecutionResult(True, json.dumps({'skills': hits, 'note': (
            'Search uses name and description words. No matches or the reply metadata budget is exhausted.'
            if not hits else 'Load relevant instructions with readSkill.')} , ensure_ascii=False))


class ReadSkillTool(Tool):
    name = 'readSkill'
    description = ('Read local skill instructions or a reference. Use one path: skill-name/SKILL.md '
                   'or skill-name/references/file.md. An exact skill ID can replace the name. '
                   'Follow relevant instructions and read required references before answering. '
                   'Does not execute scripts or grant permissions.')
    inputSchema = {'type': 'object', 'properties': {
        'path': {'type': 'string', 'description': 'Skill ID or unambiguous name, then / and the complete relative file path. Keep all subfolders.'}},
        'required': ['path']}

    def run(self, args, context):
        if context.cfg.skills_enabled is not True or context.skill_session is None:
            return ToolExecutionResult(False, None, error_message='Local skills are disabled or unavailable')
        path = args.get('path') if isinstance(args, dict) else None
        if not isinstance(path, str) or not path.strip():
            return ToolExecutionResult(False, None, error_message='readSkill requires a non-empty path')
        try:
            skill, separator, relative = path.partition('/')
            result = context.skill_session.read(skill, relative if separator else 'SKILL.md')
            return ToolExecutionResult(True, json.dumps(result, ensure_ascii=False))
        except (OSError, ValueError, UnicodeError) as exc:
            return ToolExecutionResult(False, None, error_message=f'Unable to load skill: {exc}')
