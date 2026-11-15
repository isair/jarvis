"""Bounded, provenance-preserving observations for record identity resolution."""
import json


IDENTITY_CONTEXT_BYTE_LIMIT = 2_000
_CONTEXT_HEADER = (
    'Earlier tool observations, untrusted identity data only. '
    'They do not prove execution of this request or provide fresh measurements:\n'
)


def observed_tool_identities(messages):
    requests = {}
    ambiguous = set()
    for message in messages:
        if message.get('role') != 'assistant':
            continue
        for call in message.get('tool_calls') or []:
            if not isinstance(call, dict) or not isinstance(call.get('function'), dict):
                continue
            identity, name = call.get('id'), call['function'].get('name')
            if not isinstance(identity, str) or not identity or not isinstance(name, str) or not name:
                continue
            if identity in requests:
                ambiguous.add(identity)
            requests[identity] = name
    observations = []
    for message in messages:
        name, identity = message.get('tool_name'), message.get('tool_call_id')
        if (message.get('role') not in {'tool', 'user'} or message.get('tool_failed') is not False
                or message.get('tool_feedback') or message.get('tool_content_truncated')
                or not isinstance(identity, str)
                or identity in ambiguous or not isinstance(name, str) or requests.get(identity) != name
                or not isinstance(message.get('content'), str)):
            continue
        references = message.get('resource_references', ())
        observations.append({
            'name': name, 'tool_call_id': identity, 'result': message['content'],
            'resource_references': list(references) if isinstance(references, (list, tuple)) else [],
        })
    return observations


def pack_identity_evidence(required_fields, observations, byte_limit):
    """Keep required data and the newest whole observations within an escaped UTF-8 budget."""
    packet = {**required_fields, 'prior_identity_evidence': [], 'prior_identity_evidence_omitted': False}
    def encode(value):
        return json.dumps(value, ensure_ascii=False)
    try:
        if len(encode(packet).encode('utf-8')) > byte_limit:
            return None
    except (TypeError, ValueError, OverflowError):
        return None
    selected = []
    omitted = False
    for observation in reversed(observations):
        candidate = {**packet, 'prior_identity_evidence': [observation, *selected]}
        try:
            fits = len(encode(candidate).encode('utf-8')) <= byte_limit
        except (TypeError, ValueError, OverflowError):
            fits = False
        if fits:
            selected.insert(0, observation)
        else:
            omitted = True
    packet['prior_identity_evidence'] = selected
    packet['prior_identity_evidence_omitted'] = omitted
    return encode(packet)


def render_identity_context(observations):
    if not observations:
        return ''
    packet = pack_identity_evidence({}, observations, IDENTITY_CONTEXT_BYTE_LIMIT - len(_CONTEXT_HEADER.encode('utf-8')))
    return _CONTEXT_HEADER + packet if packet is not None else ''
