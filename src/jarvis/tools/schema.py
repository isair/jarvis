"""Canonical argument constraints and scoped structured tool output."""


from copy import deepcopy
import re
from urllib.parse import unquote

from referencing import Resource
from referencing.jsonschema import DRAFT202012

from ..debug import debug_log


def _supported_local_reference(reference, root):
    if not isinstance(reference, str) or not reference.startswith('#'):
        return False
    fragment = unquote(reference[1:], errors='strict')
    if not fragment:
        return True
    prefix = '/$defs/'
    if not fragment.startswith(prefix):
        return False
    token = fragment[len(prefix):]
    if '/' in token or re.search(r'~(?![01])', token):
        return False
    name = token.replace('~1', '/').replace('~0', '~')
    definitions = root.get('$defs')
    return isinstance(definitions, dict) and name in definitions


def _scoped_schemas(schemas):
    definitions = {}
    branches = []
    for index, name in enumerate(sorted(schemas)):
        namespace = f'tool_{index}'
        try:
            schema = deepcopy(schemas[name])
            pending = [Resource.from_contents(schema, default_specification=DRAFT202012)]
            seen = set()
            while pending:
                resource = pending.pop()
                node = resource.contents
                if id(node) in seen:
                    continue
                seen.add(id(node))
                if isinstance(node, dict):
                    if any(key in node for key in ('$schema', '$id', 'id', '$anchor', '$dynamicAnchor',
                                                    '$dynamicRef', '$recursiveRef', '$recursiveAnchor')):
                        debug_log('structured tool format unavailable: unsupported reference scope', 'planning')
                        return None
                    if '$ref' in node:
                        reference = node['$ref']
                        if not _supported_local_reference(reference, schema):
                            debug_log('structured tool format unavailable: unsupported local or external reference', 'planning')
                            return None
                        node['$ref'] = f'#/$defs/{namespace}' + reference[1:]
                pending.extend(resource.subresources())
        except Exception as exc:
            debug_log(f'structured tool format unavailable: {type(exc).__name__}', 'planning')
            return None
        definitions[namespace] = schema
        branches.append({'type': 'object', 'properties': {
            'name': {'type': 'string', 'enum': [name]},
            'arguments': {'$ref': f'#/$defs/{namespace}'},
        }, 'required': ['name', 'arguments'], 'additionalProperties': False})
    return definitions, branches


def tool_call_output_schema(schemas, *, nullable=False):
    scoped = _scoped_schemas(schemas)
    if scoped is None:
        return None
    definitions, branches = scoped
    return {'$defs': definitions, 'oneOf': ([{'type': 'null'}] if nullable else []) + branches}


def reply_output_schema(schemas, maximum_calls):
    scoped = _scoped_schemas(schemas)
    if scoped is None:
        return None
    definitions, branches = scoped
    call = {'type': 'object', 'properties': {'function': {'oneOf': branches}},
            'required': ['function'], 'additionalProperties': False}
    return {'$defs': definitions, 'oneOf': [
        {'type': 'object', 'properties': {'tool_calls': {'type': 'array', 'items': call,
                                                       'minItems': 1, 'maxItems': maximum_calls}},
         'required': ['tool_calls'], 'additionalProperties': False},
    ] + [
        {'type': 'object', 'properties': {field: {'type': 'string', 'minLength': 1}},
         'required': [field], 'additionalProperties': False}
        for field in ('reply', 'question', 'blocker')
    ]}


def missing_required_arguments(schema: dict, arguments: dict) -> list[str]:
    """Include root requirements and dependencies of supplied fields."""
    if not isinstance(schema, dict):
        return []
    declared = schema.get('required')
    required = {name for name in declared if isinstance(name, str)} if isinstance(declared, list) else set()
    dependencies = schema.get('dependentRequired')
    if isinstance(dependencies, dict):
        for name, fields in dependencies.items():
            if name in arguments and isinstance(fields, list):
                required.update(field for field in fields if isinstance(field, str))
    return sorted(required - arguments.keys())
