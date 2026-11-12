"""HTTP page transport confined to validated public network addresses."""

import ipaddress
import socket
from urllib.parse import urlsplit

import requests
from urllib3.connection import HTTPConnection, HTTPSConnection
from urllib3.connectionpool import HTTPConnectionPool, HTTPSConnectionPool
from urllib3.exceptions import NewConnectionError
from urllib3.util import connection

from ..debug import debug_log


def _public_addresses(host: str, port: int) -> list[str]:
    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        answers = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        resolved = [answer[4][0] for answer in answers]
    else:
        resolved = [str(literal)]
    addresses = []
    for resolved_address in resolved:
        address = ipaddress.ip_address(resolved_address)
        classified = getattr(address, 'ipv4_mapped', None) or address
        if (not classified.is_global or classified.is_multicast or classified.is_reserved
                or getattr(classified, 'is_site_local', False)):
            raise ValueError('Web pages must use public network addresses')
        addresses.append(str(address))
    if not addresses:
        raise ValueError('Web page hostname has no public network addresses')
    return list(dict.fromkeys(addresses))


def _page_target(url: str) -> tuple[str, int]:
    parsed = urlsplit(url)
    if (parsed.scheme not in {'http', 'https'} or not parsed.hostname
            or parsed.username is not None or parsed.password is not None):
        raise ValueError('Web pages require an HTTP(S) URL without credentials')
    return parsed.hostname, parsed.port if parsed.port is not None else (443 if parsed.scheme == 'https' else 80)


def _is_public_url(url: str) -> bool:
    try:
        _public_addresses(*_page_target(url))
        return True
    except (OSError, ValueError):
        debug_log('Web fetch rejected an invalid or non-public destination', 'web')
        return False


def _open_public_socket(owner):
    try:
        addresses = _public_addresses(owner._dns_host, owner.port)
    except (OSError, ValueError) as exc:
        debug_log('Web connection rejected an unresolved or non-public destination', 'web')
        raise NewConnectionError(owner, str(exc)) from exc
    last_error = None
    for address in addresses:
        try:
            # Dial the validated numeric address, retaining the original
            # connection hostname for HTTP Host, TLS SNI and certificate checks.
            return connection.create_connection(
                (address, owner.port), owner.timeout,
                source_address=owner.source_address, socket_options=owner.socket_options,
            )
        except OSError as exc:
            last_error = exc
    raise NewConnectionError(owner, 'Could not connect to a public web address') from last_error


class _PublicHTTPConnection(HTTPConnection):
    def _new_conn(self):
        return _open_public_socket(self)


class _PublicHTTPSConnection(HTTPSConnection):
    def _new_conn(self):
        return _open_public_socket(self)


class _PublicHTTPPool(HTTPConnectionPool):
    ConnectionCls = _PublicHTTPConnection


class _PublicHTTPSPool(HTTPSConnectionPool):
    ConnectionCls = _PublicHTTPSConnection


class _PublicAdapter(requests.adapters.HTTPAdapter):
    def init_poolmanager(self, *args, **kwargs):
        super().init_poolmanager(*args, **kwargs)
        self.poolmanager.pool_classes_by_scheme = {
            'http': _PublicHTTPPool, 'https': _PublicHTTPSPool,
        }

    def send(self, request, **kwargs):
        try:
            _page_target(request.url)
        except ValueError as exc:
            raise requests.exceptions.InvalidURL(str(exc)) from exc
        return super().send(request, **kwargs)


def get(url: str, **kwargs) -> requests.Response:
    """Fetch a page with public-only connections, including every redirect."""
    with requests.Session() as session:
        session.trust_env = False
        adapter = _PublicAdapter()
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session.get(url, **kwargs)
