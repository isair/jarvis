"""Elapsed-budget checks shared by local model stream transports."""

import time
from collections.abc import Iterator

import requests


def check_stream_deadline(deadline: float) -> None:
    """Reject expired output, including time spent in token callbacks."""
    if time.monotonic() >= deadline:
        raise requests.exceptions.Timeout('Model streaming budget exhausted')


def iter_stream_lines(response: requests.Response, deadline: float) -> Iterator[bytes]:
    """Check all delivered lines, including keep-alives and empty events.

    Requests' read timeout bounds idle socket reads. Elapsed-budget checks run
    at line boundaries; they cannot interrupt a pending read or user callback.
    """
    check_stream_deadline(deadline)
    for line in response.iter_lines():
        check_stream_deadline(deadline)
        yield line
        check_stream_deadline(deadline)
