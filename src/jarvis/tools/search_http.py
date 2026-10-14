"""Bounded HTML requests with matching browser TLS and HTTP fingerprints."""
from curl_cffi import requests
from curl_cffi.curl import CURL_WRITEFUNC_ERROR


def download_search_html(url: str, timeout_sec: float, max_bytes: int = 512 * 1024) -> tuple[int, bytes]:
    if timeout_sec <= 0 or max_bytes <= 0:
        raise ValueError('Search download requires a positive time and byte budget')
    body = bytearray()
    oversized = False

    def receive(chunk: bytes) -> int:
        nonlocal oversized
        if len(body) + len(chunk) > max_bytes:
            oversized = True
            return CURL_WRITEFUNC_ERROR
        body.extend(chunk)
        return len(chunk)

    with requests.Session() as session:
        try:
            response = session.get(url, impersonate='chrome', timeout=timeout_sec,
                                   allow_redirects=False, content_callback=receive)
        except requests.RequestsError:
            if oversized:
                raise ValueError('Search response exceeds the decoded byte budget') from None
            raise
        return response.status_code, bytes(body)
