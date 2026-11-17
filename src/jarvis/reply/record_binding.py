"""Declared record actions with bounded interpretation and source-bound compilation."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import re
import unicodedata

from jsonschema import Draft202012Validator
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012

from ..debug import debug_log
from ..tools.schema import _supported_local_reference, missing_required_arguments
from .tool_identity import IDENTITY_CONTEXT_BYTE_LIMIT, pack_identity_evidence
from .planner import MAX_STEPS as MAX_RECORD_STEPS
from .compound_query import COVERAGE_INPUT_BYTE_LIMIT

RECORD_CONTRACT = 'x-jarvis-records'


@dataclass(frozen=True)
class RecordAction:
    name: str
    schema: dict
    declaration: dict
    contract: dict


@dataclass(frozen=True)
class RecordRequest:
    query: str
    reference_context: str
    actions: tuple[RecordAction, ...]
    current: tuple[dict, ...]
    prior: tuple[dict, ...]
    fingerprint: str
    messages: tuple[dict, ...]
    response_schema: dict


@dataclass(frozen=True)
class RecordPlan:
    fingerprint: str
    intents: tuple[dict, ...]


@dataclass(frozen=True)
class RecordResolution:
    status: str
    call: dict | None = None
    prerequisite: bool = False
    reason: str = ''


def _pointer(value, pointer):
    if not isinstance(pointer, str) or (pointer and not pointer.startswith('/')):
        raise ValueError('Invalid record pointer')
    for token in pointer[1:].split('/') if pointer else ():
        if re.search(r'~(?![01])', token):
            raise ValueError('Invalid record pointer escape')
        token = token.replace('~1', '/').replace('~0', '~')
        if isinstance(value, list):
            if not re.fullmatch(r'0|[1-9][0-9]*', token):
                raise ValueError('Invalid record array index')
            value = value[int(token)]
        else:
            value = value[token]
    return value


def _valid_pointer(pointer):
    if not isinstance(pointer, str) or (pointer and not pointer.startswith('/')):
        return False
    return not re.search(r'~(?![01])', pointer)


def _validator(schema):
    # An explicit empty registry refuses external retrieval, including local files.
    return Draft202012Validator(schema, registry=Registry())


def _contract_actions(name, schema):
    contract = schema[RECORD_CONTRACT]
    if not isinstance(contract, dict) or not isinstance(contract.get('actions'), list) or not contract['actions']:
        raise ValueError('Missing declared actions')
    collection = contract.get('collection')
    if (not isinstance(collection, dict) or collection.get('source') not in {'result', 'result_metadata'}
            or not _valid_pointer(collection.get('path')) or type(collection.get('complete')) is not bool
            or not _valid_pointer(collection.get('display', ''))):
        raise ValueError('Invalid collection contract')
    discovery = contract.get('discovery')
    if type(discovery) is not int or not 0 <= discovery < len(contract['actions']):
        raise ValueError('Invalid discovery action')
    pending = [Resource.from_contents(schema, default_specification=DRAFT202012)]
    while pending:
        resource = pending.pop()
        node = resource.contents
        if isinstance(node, dict):
            if any(k in node for k in ('$id', '$schema', 'id', '$anchor', '$dynamicRef', '$dynamicAnchor', '$recursiveRef', '$recursiveAnchor')):
                raise ValueError('Unsupported record schema scope')
            if '$ref' in node and not _supported_local_reference(node['$ref'], schema):
                raise ValueError('Unsupported record schema reference')
        pending.extend(resource.subresources())
    properties = schema.get('properties', {})
    Draft202012Validator.check_schema(schema)
    validator = _validator(schema)
    result = []
    for action in contract['actions']:
        if (not isinstance(action, dict) or not isinstance(action.get('purpose'), str) or not action['purpose'].strip()
                or not isinstance(action.get('arguments'), dict) or 'selector' not in action):
            raise ValueError('Invalid action declaration')
        fixed = action['arguments']
        selector = action['selector']
        if action.get('parameters') or not set(fixed) <= set(properties):
            raise ValueError('Record actions require fixed variants and an optional selector')
        args = deepcopy(fixed)
        if selector is not None:
            if (not isinstance(selector, str) or selector not in properties or selector in fixed
                    or not _valid_pointer(action.get('identity', ''))):
                raise ValueError('Invalid action selector')
            prop = properties[selector]
            if prop.get('type') not in ('array', 'string', 'integer'):
                raise ValueError('Selectors require an inline scalar or array type')
            args[selector] = None
            if missing_required_arguments(schema, args):
                raise ValueError('Declared action omits required arguments')
            if any(next(validator.descend(value, properties[key]), None) is not None
                   for key, value in fixed.items()):
                raise ValueError('Fixed arguments contradict their public properties')
        elif not validator.is_valid(args):
            raise ValueError('Declared action contradicts its public schema')
        result.append(RecordAction(name, schema, action, contract))
    lookup = contract['actions'][discovery]
    if lookup['selector'] is not None or lookup.get('parameters'):
        raise ValueError('Discovery cannot require the unknown identifier')
    return result


def declares_record_catalogue(tools):
    """Recognise the opted-in catalogue independently of request admission."""
    real = [tool['function'] for tool in tools
            if tool['function']['name'] not in {'stop', 'toolSearchTool'}]
    return bool(real) and all(isinstance(tool.get('parameters'), dict)
                              and RECORD_CONTRACT in tool['parameters'] for tool in real)


def _schema_fingerprint(schema):
    return list(hashlib.sha256(json.dumps(schema, sort_keys=True, ensure_ascii=False).encode()).digest())


def prepare_record_request(query, tools, results, prior_results, *, byte_limit=6000, reference_context=''):
    """Prepare one replacement generation, keeping its complete input budget."""
    try:
        real = [t['function'] for t in tools if t['function']['name'] not in {'stop', 'toolSearchTool'}]
        if not declares_record_catalogue(tools):
            return None
        packet = pack_identity_evidence(
            {'request': query, 'tools': tools, 'results': results, 'reference_context': reference_context},
            prior_results, byte_limit,
        )
        if packet is None:
            debug_log('record binding: complete input exceeds its budget', 'planning')
            return None
        data = json.loads(packet)
        actions = tuple(a for tool in sorted(real, key=lambda t: t['name'])
                        for a in _contract_actions(tool['name'], deepcopy(tool['parameters'])))
        choices = []
        for index, action in enumerate(actions):
            item = {'index': index, 'tool_name': action.name, 'purpose': action.declaration['purpose'],
                    'requires_existing_identifier': action.declaration['selector'] is not None}
            choices.append(item)
        fingerprint = hashlib.sha256(json.dumps({'request': query, 'reference_context': reference_context, 'tools': tools}, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        from .prompts.record_binding import record_messages, record_output_schema
        messages = record_messages(query, choices, reference_context)
        if len(messages[-1]['content'].encode('utf-8')) > byte_limit:
            debug_log('record binding: interpretation input exceeds its budget', 'planning')
            return None
        output_schema = record_output_schema(actions)
        return RecordRequest(query, reference_context, actions, tuple(data['results']),
                             tuple(data['prior_identity_evidence']), fingerprint, tuple(messages), output_schema)
    except Exception as error:
        debug_log(f'record binding: invalid declaration ({type(error).__name__})', 'planning')
        return None


def decode_record_plan(request, decision):
    if isinstance(decision, str):
        try:
            decision = json.loads(decision)
        except (ValueError, TypeError):
            return None
    if not _validator(request.response_schema).is_valid(decision):
        return None
    return RecordPlan(request.fingerprint, tuple(deepcopy(decision['plan'])))


def _literal_supplied(value, marker, query):
    if not isinstance(value, str) or not value:
        return False
    if (value in {'.', '..', '~'} or value.startswith(('/', './', '../', '~/'))) and value in query:
        return True
    for match in re.finditer(re.escape(value), query):
        before, after = query[:match.start()].rstrip(), query[match.end():].lstrip()
        if before and after and ((before[-1] == after[0] and before[-1] in {'"', "'"}) or
                (unicodedata.category(before[-1]) == 'Pi' and unicodedata.category(after[0]) == 'Pf')):
            return True
        if isinstance(marker, str) and marker and (re.search(re.escape(marker) + r'\W*$', before)
                                                  or re.match(r'^\W*' + re.escape(marker), after)):
            return True
    return False


def _words(value):
    words, current = [], []
    for char in unicodedata.normalize('NFC', value.casefold()):
        if char == '_' or unicodedata.category(char)[0] in {'L', 'M', 'N'}:
            current.append(char)
        elif current:
            words.append(''.join(current)); current = []
    if current:
        words.append(''.join(current))
    return set(words)


def _topic_matches(topic, display):
    desired, actual = _words(topic), _words(display)
    return bool(desired and actual and (desired <= actual or actual <= desired))


def _record_pair(action, value):
    identity = _pointer(value, action.declaration.get('identity', ''))
    display = _pointer(value, action.contract['collection'].get('display', ''))
    if isinstance(value, int) and not isinstance(value, bool) and display == value:
        display = str(display)
    if isinstance(identity, (str, int)) and not isinstance(identity, bool) and isinstance(display, str) and display:
        return identity, display
    raise ValueError('Invalid record identity or display')


def _usable_row(row, *, current):
    return (isinstance(row, dict)
            and (row.get('success') is True if current else row.get('success', True) is True)
            and not any(row.get(k) for k in ('tool_feedback', 'tool_content_truncated', 'content_truncated')))


def _records(action, rows, *, current=False, byte_limit=None):
    observed, complete = [], False
    remaining = byte_limit

    def bounded_pair(value):
        nonlocal remaining
        pair = _record_pair(action, value)
        if remaining is not None:
            if any(isinstance(part, str) and len(part) > byte_limit for part in pair):
                raise OverflowError('Record identity exceeds the observation budget')
            remaining -= len(json.dumps(pair, ensure_ascii=False).encode('utf-8'))
            if remaining < 0:
                raise OverflowError('Record projection exceeds the observation budget')
        return pair
    collection = action.contract['collection']
    for row in rows:
        if not _usable_row(row, current=current) or row.get('name') != action.name:
            continue
        receipt = row.get('record_evidence')
        if isinstance(receipt, dict) and receipt.get('schema') == _schema_fingerprint(action.schema):
            try:
                index = str(action.contract['actions'].index(action.declaration))
                projection = receipt['actions'][index]
                pairs = projection['records']
                if (type(projection['complete']) is not bool or not isinstance(pairs, list)
                        or not all(isinstance(pair, (list, tuple)) and len(pair) == 2
                            and isinstance(pair[0], (str, int)) and not isinstance(pair[0], bool)
                            and isinstance(pair[1], str) and pair[1] for pair in pairs)):
                    raise ValueError('Invalid record projection')
                pairs = [tuple(pair) for pair in pairs]
                if projection['complete']:
                    observed = pairs
                    complete = True
                    continue
                observed = list(dict.fromkeys([*observed, *pairs]))
            except (ValueError, TypeError, KeyError):
                pass
        try:
            source = row.get(collection['source'])
            if collection['source'] == 'result' and isinstance(source, str):
                if byte_limit is not None and (len(source) > COVERAGE_INPUT_BYTE_LIMIT
                        or len(source.encode('utf-8')) > COVERAGE_INPUT_BYTE_LIMIT):
                    raise OverflowError('Raw record input exceeds the observation input budget')
                source = json.loads(source)
            values = _pointer(source, collection['path'])
            if isinstance(values, list):
                if byte_limit is not None and len(values) > byte_limit:
                    raise OverflowError('Record collection exceeds the observation budget')
                fresh = [bounded_pair(value) for value in values]
                if collection['complete']:
                    observed = fresh
                    complete = True
                    continue
                observed = list(dict.fromkeys([*observed, *fresh]))
        except (ValueError, TypeError, KeyError, IndexError):
            pass
        kind = action.contract.get('reference_kind')
        if kind:
            references = row.get('resource_references') or ()
            if byte_limit is not None and len(references) > byte_limit:
                raise OverflowError('Record references exceed the observation budget')
            seen = set(observed)
            for ref in references:
                if not isinstance(ref, dict) or ref.get('kind') != kind:
                    continue
                try:
                    pair = bounded_pair(ref)
                    if pair not in seen:
                        observed.append(pair)
                        seen.add(pair)
                except (ValueError, TypeError, KeyError, IndexError):
                    pass
    return observed, complete


def _observed_matches(action, request, topic, current):
    old, _ = _records(action, request.prior)
    fresh, complete = _records(action, current, current=True)
    lookup = action.contract['actions'][action.contract['discovery']]['arguments']
    attempted = any(row.get('name') == action.name and row.get('arguments') == lookup for row in current)
    records = fresh if complete or attempted else old + [value for value in fresh if value not in old]
    matches = list(dict.fromkeys(identity for identity, display in records if _topic_matches(topic, display)))
    return matches, complete, attempted


def _literal_identity(value, action, selector):
    prop = action.schema['properties'][selector]
    item = prop.get('items', {}) if prop.get('type') == 'array' else prop
    validator = _validator(action.schema)
    if (isinstance(value, str) and re.fullmatch(r'-?(0|[1-9][0-9]*)', value)
            and next(validator.descend(value, item), None) is not None
            and next(validator.descend(int(value), item), None) is None):
        return int(value)
    return value


def is_declared_discovery(request, call):
    return request is not None and call is not None and any(
        call == {'name': action.name, 'arguments': action.contract['actions'][action.contract['discovery']]['arguments']}
        for action in request.actions
    )


def capture_record_observation(name, schema, result, *, byte_limit=IDENTITY_CONTEXT_BYTE_LIMIT):
    """Retain only complete identity projections, never unrelated result fields."""
    if not result.success or not isinstance(schema, dict) or RECORD_CONTRACT not in schema:
        return None
    try:
        row = {'name': name, 'success': True, 'result': result.reply_text,
               'result_metadata': result.result_metadata, 'resource_references': result.resource_references}
        evidence = {}
        receipt = {'schema': _schema_fingerprint(schema), 'actions': evidence}
        for index, action in enumerate(_contract_actions(name, schema)):
            if action.declaration['selector'] is None:
                continue
            records, complete = _records(action, [row], current=True, byte_limit=byte_limit)
            if records or complete:
                evidence[str(index)] = {'records': records, 'complete': complete}
                if len(json.dumps(receipt, ensure_ascii=False).encode('utf-8')) > byte_limit:
                    raise OverflowError('Combined record observation exceeds its budget')
        return receipt if evidence else None
    except OverflowError:
        debug_log('record binding: observation exceeds its bounded input or projection budget', 'planning')
    except Exception as error:
        debug_log(f'record binding: observation unavailable ({type(error).__name__})', 'planning')
    return None


def resolve_record_plan(request, decision):
    """Return one admitted call, an honest blocker, or completion from actual effects."""
    if isinstance(decision, RecordPlan):
        plan = decision if decision.fingerprint == request.fingerprint else None
    else:
        plan = decode_record_plan(request, decision)
    if plan is None:
        return RecordResolution('blocked', reason='The record plan is invalid or its tool contract changed.')
    cursor = 0
    for intent in plan.intents:
        action = request.actions[intent['action_index']]
        declaration = action.declaration
        args = deepcopy(declaration['arguments'])
        selector = declaration['selector']
        literal = intent['literal_target']
        supplied = intent['kind'] == 'B' and _literal_supplied(literal, intent['naming_marker'], request.query)
        topic = intent['description']
        if selector is not None and not supplied:
            context = request.query + '\n' + request.reference_context
            if not isinstance(topic, str) or not topic or topic.casefold() not in context.casefold():
                return RecordResolution('blocked', reason='The target description has no request evidence.')
        # Current successful operations can complete an effect even after a later
        # full snapshot has removed its identifier. Prior actions cannot do so.
        for position in range(cursor, len(request.current)):
            row = request.current[position]
            actual = row.get('arguments')
            if not _usable_row(row, current=True) or row.get('name') != action.name or not isinstance(actual, dict):
                continue
            if selector is None:
                matches = actual == args
            else:
                values = actual.get(selector)
                values = values if isinstance(values, list) else [values]
                if supplied:
                    identities = [_literal_identity(literal, action, selector)]
                else:
                    identities, _, _ = _observed_matches(action, request, topic, request.current[:position])
                matches = (set(actual) == set(args) | {selector}
                           and all(actual.get(k) == v for k, v in args.items())
                           and len(identities) == 1 and values == identities)
            if matches:
                cursor = position + 1
                break
        else:
            if selector is not None:
                if supplied:
                    identity = _literal_identity(literal, action, selector)
                else:
                    matches, complete, attempted = _observed_matches(action, request, topic, request.current)
                    if len(matches) != 1:
                        lookup = action.contract['actions'][action.contract['discovery']]['arguments']
                        if complete or attempted:
                            return RecordResolution('blocked', reason='The observed namespace has no unique matching identifier; no mutation was performed.')
                        debug_log('record binding: discovery required before the requested effect', 'planning')
                        return RecordResolution('call', {'name': action.name, 'arguments': deepcopy(lookup)}, True)
                    identity = matches[0]
                prop = action.schema['properties'][selector]
                args[selector] = [identity] if prop.get('type') == 'array' else identity
            if not _validator(action.schema).is_valid(args):
                return RecordResolution('blocked', reason='Compiled arguments do not satisfy the declared tool contract.')
            debug_log('record binding: source-bound action admitted', 'planning')
            return RecordResolution('call', {'name': action.name, 'arguments': args})
    return RecordResolution('complete')
