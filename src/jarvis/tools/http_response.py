"""Response hooks for streaming web tools."""
import requests

from ..debug import debug_log


def discard_redirect_body(response: requests.Response, *args, **kwargs) -> requests.Response:
    """Release redirect bodies before Requests prepares the next request."""
    if response.is_redirect:
        debug_log('Web fetch: discarding redirect body before handling Location', 'web')
        response.close()
    return response
