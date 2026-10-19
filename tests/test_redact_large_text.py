"""Large local text remains responsive while structural secrets are masked."""
import subprocess
import sys
from pathlib import Path
import pytest

from jarvis.utils.redact import redact, scrub_secrets

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('ending', ['', '@localhost', '@example.test'])
def test_large_text_completes_with_correct_redaction(ending):
    # A child bounds the pathological control so a failure cannot stall pytest.
    script = """
import sys
sys.path.insert(0, sys.argv[1])
from jarvis.utils.redact import redact, scrub_secrets
text = 'g' * 131072 + sys.argv[2]
expected = '[REDACTED_EMAIL]' if sys.argv[2] == '@example.test' else text
assert scrub_secrets(text) == expected
assert redact(text, max_len=len(text)+100) == expected
print('completed')
"""
    source_root = Path(__file__).resolve().parents[1] / 'src'
    result = subprocess.run([sys.executable, '-c', script, str(source_root), ending],
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'completed'


@pytest.mark.parametrize('scrubber', [redact, scrub_secrets])
@pytest.mark.parametrize('address', ['first.last+tag@example.test', 'user_name@example.test',
                                      'USER%alias@sub-domain.example.test', 'İuser@example.test'])
def test_adjacent_punctuation_and_unicode_keep_email_coverage(scrubber, address):
    result = scrubber('Before ('+address+'), after')
    assert address not in result
    assert result == 'Before ([REDACTED_EMAIL]), after'
    assert scrubber(result) == result


@pytest.mark.parametrize('scrubber', [redact, scrub_secrets])
@pytest.mark.parametrize('text', ['first@a.test+second@b.test',
                                  'first@a.test_second@b.test',
                                  'first@a.test.second+third@b.test'])
def test_adjacent_addresses_are_all_masked(scrubber,text):
    assert scrubber(text)=='[REDACTED_EMAIL][REDACTED_EMAIL]'
