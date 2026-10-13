"""Configured transport policy preserves explicit trust and response lifetime."""

from contextlib import contextmanager

import pytest

from jarvis.llm import http

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('url,local', [
    ('http://localhost:11434/api/chat', True),
    ('http://LOCALHOST.:1234/v1', True),
    ('http://127.2.3.4:1234/v1', True),
    ('http://[::1]:1234/v1', True),
    ('https://model.example/v1', False),
    ('http://localhost.example/v1', False),
])
@pytest.mark.parametrize('method', ['get', 'post'])
def test_transport_policy_and_lifetime(monkeypatch, url, local, method):
    events = []
    response = object()

    class Session:
        trust_env = True

        def __enter__(self):
            events.append('session open')
            return self

        def __exit__(self, *args):
            events.append('session closed')

        @contextmanager
        def request(self, verb, endpoint, **kwargs):
            assert verb == method.upper()
            assert endpoint == url
            assert self.trust_env is (not local)
            assert kwargs['headers'] == {'Authorization': 'Bearer explicit'}
            events.append('response open')
            try:
                yield response
            finally:
                events.append('response closed')

    monkeypatch.setattr(http.requests, 'Session', Session)
    with pytest.raises(RuntimeError, match='consumer interrupted'):
        with getattr(http, method)(url, headers={'Authorization': 'Bearer explicit'}) as actual:
            assert actual is response
            assert events == ['session open', 'response open']
            raise RuntimeError('consumer interrupted')
    assert events == ['session open', 'response open', 'response closed', 'session closed']


@pytest.mark.parametrize('request_bundle,curl_bundle,explicit,expected', [
    ('selected-ca.pem', 'fallback-ca.pem', None, 'selected-ca.pem'),
    ('', 'fallback-ca.pem', None, 'fallback-ca.pem'),
    ('selected-ca.pem', 'fallback-ca.pem', 'explicit-ca.pem', 'explicit-ca.pem'),
    ('selected-ca.pem', 'fallback-ca.pem', False, False),
])
def test_local_https_preserves_selected_certificate_trust(monkeypatch, request_bundle, curl_bundle, explicit, expected):
    monkeypatch.setenv('REQUESTS_CA_BUNDLE', request_bundle)
    monkeypatch.setenv('CURL_CA_BUNDLE', curl_bundle)
    observed = []

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        @contextmanager
        def request(self, method, url, **kwargs):
            observed.append(kwargs['verify'])
            yield object()

    monkeypatch.setattr(http.requests, 'Session', Session)
    kwargs = {} if explicit is None else {'verify': explicit}
    with http.get('https://localhost:1234/v1', **kwargs):
        pass
    assert observed == [expected]
