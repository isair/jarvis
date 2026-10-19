"""Long source lines obey the same embedding bound as ordinary text."""
import pytest

from jarvis.memory import document_index

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('unit', ['a', '🦦'])
def test_long_lines_are_bounded_without_losing_their_text(unit):
    limit = document_index._MAX_CHUNK_CHARS
    text = unit * (limit * 3 + 7)
    chunks = document_index.DocumentIndex._chunks(text)
    assert chunks
    assert all(len(value) <= limit for _, _, value in chunks)
    assert ''.join(value for _, _, value in chunks) == text
    assert all(start == end == 1 for start, end, _ in chunks)


def test_ordinary_line_ranges_preserve_all_source_lines():
    limit = document_index._MAX_CHUNK_LINES
    lines = [f'line {i}' for i in range(limit * 2 + 3)]
    chunks = document_index.DocumentIndex._chunks('\n'.join(lines))
    assert '\n'.join(value for _, _, value in chunks).splitlines() == lines
    for start, end, value in chunks:
        assert value.splitlines() == lines[start - 1:end]
        assert end - start + 1 <= limit


def test_exact_character_boundary_keeps_the_next_line():
    text = 'x' * document_index._MAX_CHUNK_CHARS
    assert document_index.DocumentIndex._chunks(text + '\ntail') == [
        (1, 1, text), (2, 2, 'tail'),
    ]
