"""Shared required-field presence checks for tool planning and dispatch."""


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
