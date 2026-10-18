"""Response-scoped HTTP transport for configured model servers."""

import ipaddress
import os
import socket
from contextlib import contextmanager
from urllib.parse import urlsplit

import requests


def _is_loopback(url: str) -> bool:
    try:
        host = urlsplit(url).hostname or ""
    except ValueError:
        return False
    if host.lower().rstrip(".") == "localhost":
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        # Numeric address spellings accepted by the socket layer, without DNS.
        try:
            address = ipaddress.ip_address(socket.getaddrinfo(
                host, None, flags=socket.AI_NUMERICHOST,
            )[0][4][0])
        except OSError:
            return False
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    return address.is_loopback


@contextmanager
def _request(method: str, url: str, **kwargs):
    with requests.Session() as session:
        if _is_loopback(url):
            session.trust_env = False
            # Explicit certificate trust remains available for local HTTPS.
            if "verify" not in kwargs:
                bundle = os.environ.get("REQUESTS_CA_BUNDLE") or os.environ.get("CURL_CA_BUNDLE")
                if bundle:
                    kwargs["verify"] = bundle
        with session.request(method, url, **kwargs) as response:
            yield response


def get(url: str, **kwargs):
    """Keep the response and its session alive throughout consumption."""
    return _request("GET", url, **kwargs)


def post(url: str, **kwargs):
    """Keep the response and its session alive throughout consumption."""
    return _request("POST", url, **kwargs)
