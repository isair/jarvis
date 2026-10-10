"""Structural redaction protects carried tool results and diagnostic content."""

import pytest
from urllib.parse import urlsplit

from src.jarvis.utils.redact import redact, scrub_secrets


@pytest.mark.unit
class TestVendorAccessKeys:
    def test_aws_akia_key_redacted(self):
        out = redact("key=AKIAIOSFODNN7EXAMPLE rest")
        assert "AKIAIOSFODNN7EXAMPLE" not in out
        assert "[REDACTED_AWS_KEY]" in out

    def test_aws_asia_key_redacted(self):
        out = redact("ASIAIOSFODNN7EXAMPLE")
        assert "ASIAIOSFODNN7EXAMPLE" not in out
        assert "[REDACTED_AWS_KEY]" in out

    def test_stripe_live_secret_redacted(self):
        token = "sk_live_" + "a" * 24
        out = redact(f"see {token} please")
        assert token not in out
        assert "[REDACTED_STRIPE_KEY]" in out

    def test_stripe_test_publishable_redacted(self):
        token = "pk_test_" + "Z" * 24
        out = redact(token)
        assert token not in out
        assert "[REDACTED_STRIPE_KEY]" in out

    def test_github_pat_redacted(self):
        token = "ghp_" + "A" * 36
        out = redact(token)
        assert token not in out
        assert "[REDACTED_GH_TOKEN]" in out

    def test_openai_key_redacted(self):
        token = "sk-" + "A" * 40
        out = redact(token)
        assert token not in out
        assert "[REDACTED_OPENAI_KEY]" in out

    def test_google_api_key_redacted(self):
        token = "AIza" + "B" * 35
        out = redact(token)
        assert token not in out
        assert "[REDACTED_GOOG_KEY]" in out


@pytest.mark.unit
class TestAuthorizationHeaders:
    def test_bearer_header_redacted(self):
        out = scrub_secrets("Authorization: Bearer abc.def.ghi")
        assert "abc.def.ghi" not in out
        assert "Authorization: Bearer [REDACTED]" in out

    def test_basic_header_redacted(self):
        out = scrub_secrets("Authorization: Basic dXNlcjpwYXNz")
        assert "dXNlcjpwYXNz" not in out
        assert "Authorization: Basic [REDACTED]" in out


@pytest.mark.unit
class TestKeywordAnchoredCredentials:
    def test_refresh_token_keyword_redacted(self):
        out = redact("refresh_token=abcdef123456")
        assert "abcdef123456" not in out
        assert "refresh_token=[REDACTED]" in out

    def test_access_token_keyword_redacted(self):
        out = redact("access_token: zzz999")
        assert "zzz999" not in out
        assert "access_token=[REDACTED]" in out

    def test_session_id_redacted(self):
        out = redact("session_id=deadbeefcafe")
        assert "deadbeefcafe" not in out
        assert "session_id=[REDACTED]" in out

    def test_oauth_token_redacted(self):
        out = redact("oauth_token=qwertyuiop")
        assert "qwertyuiop" not in out
        assert "oauth_token=[REDACTED]" in out


@pytest.mark.unit
@pytest.mark.parametrize('scrubber', [redact, scrub_secrets])
@pytest.mark.parametrize('url,endpoint', [
    ('http://alice:p%40ss@localhost:8080/v1', 'localhost:8080/v1'),
    ('HTTPS://alice:p@ss@192.0.2.1/api', '192.0.2.1/api'),
    ('ws://alice:short@[::1]:9000/events', '[::1]:9000/events'),
    ('custom+local://alice:short@intranet/rpc', 'intranet/rpc'),
    ('https://alice:short@example.test/path', 'example.test/path'),
])
def test_url_credentials_are_removed_without_losing_endpoint(scrubber, url, endpoint):
    result = scrubber('Connection failed: ' + url)
    assert 'alice' not in result
    assert 'p%40ss' not in result and 'p@ss' not in result and 'short' not in result
    assert urlsplit(result.removeprefix('Connection failed: ')).username is None
    assert endpoint in result
    assert scrubber(result) == result


@pytest.mark.unit
@pytest.mark.parametrize('scrubber', [redact, scrub_secrets])
@pytest.mark.parametrize('url', [
    'http://localhost:8080/v1', 'https://example.test/path@segment',
    'http://localhost/search?q=alice@intranet',
])
def test_non_credential_urls_remain_usable(scrubber, url):
    assert scrubber(url) == url
