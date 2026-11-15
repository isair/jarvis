"""Shared required-field presence checks for tool planning and dispatch."""


def tool_call_output_schema(names):
    return {'type': 'object', 'properties': {
        'name': {'type': 'string', 'enum': sorted(names)},
        'arguments': {'type': 'object', 'additionalProperties': True},
    }, 'required': ['name', 'arguments'], 'additionalProperties': False}


def reply_output_schema(names, maximum_calls):
    call = {'type': 'object', 'properties': {'function': tool_call_output_schema(names)},
            'required': ['function'], 'additionalProperties': False}
    return {'oneOf': [
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
