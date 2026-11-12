"""Unicode setup preserves writer ownership and tolerates custom capture sinks."""

import io
import tracemalloc
from types import SimpleNamespace

import pytest

from jarvis.utils.console import configure_utf8_streams


pytestmark = pytest.mark.unit


def test_output_without_a_console_does_not_accumulate_captured_text():
    owner = SimpleNamespace(stdout=None, stderr=None)
    configure_utf8_streams(owner)
    chunk = '.' * 32768
    tracemalloc.start()
    try:
        for index in range(16):
            owner.stdout.write(chunk + str(index))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 4 * len(chunk), f'Unavailable console retained {peak} bytes of output'


def test_existing_writers_emit_utf8_after_configuration_and_restoration():
    buffers = [io.BytesIO(), io.BytesIO()]
    streams = [io.TextIOWrapper(buffer, encoding='cp1252') for buffer in buffers]
    owner = SimpleNamespace(stdout=streams[0], stderr=streams[1])
    try:
        configure_utf8_streams(owner)
        owner.stdout, owner.stderr = streams
        for stream in streams:
            stream.write('🧠 Ελληνικά, 日本語')
            stream.flush()
        assert all(buffer.getvalue().decode('utf-8') == '🧠 Ελληνικά, 日本語' for buffer in buffers)
    finally:
        for stream in streams:
            stream.close()


@pytest.mark.parametrize('failure', [OSError, ValueError])
def test_one_unconfigurable_capture_does_not_break_other_output(failure):
    class Capture(io.StringIO):
        def reconfigure(self, **kwargs):
            raise failure('Owned unsupported configuration')

    captured = Capture()
    buffer = io.BytesIO()
    other = io.TextIOWrapper(buffer, encoding='cp1252')
    try:
        configure_utf8_streams(SimpleNamespace(stdout=captured, stderr=other))
        captured.write('🧠 Captured text')
        other.write('🧠 Other output')
        other.flush()
        assert captured.getvalue() == '🧠 Captured text'
        assert buffer.getvalue().decode('utf-8') == '🧠 Other output'
    finally:
        captured.close()
        other.close()
