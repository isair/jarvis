"""Classify transport failures without exposing URLs or credentials."""

import requests
from urllib3.exceptions import ReadTimeoutError


def is_timeout_error(error: BaseException) -> bool:
    """Recognise typed timeouts wrapped by Requests while reading a body.

    Requests can raise ConnectionError(ReadTimeoutError(...)) after receiving
    response headers. Inspect explicit causes and exception arguments, not
    message text or unrelated exceptions in the handling context.
    """
    pending = [error]
    seen = set()
    while pending:
        current = pending.pop()
        if not isinstance(current, BaseException) or id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, (requests.exceptions.Timeout, ReadTimeoutError)):
            return True
        pending.extend(current.args)
        pending.append(current.__cause__)
    return False
