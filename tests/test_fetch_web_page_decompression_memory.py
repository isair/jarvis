"""Compressed downloads are rejected without expanding the complete response."""
import gzip
import threading
import tracemalloc
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from jarvis.tools.base import ToolContext
from jarvis.tools.builtin import fetch_web_page

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('encoding', ['gzip', 'deflate'])
@pytest.mark.parametrize('chunked', [False, True])
def test_compressed_page_rejection_bounds_decoder_allocations(mock_config, encoding, chunked):
    limit = fetch_web_page._MAX_FETCH_BYTES
    expanded = b' ' * (10 * limit)
    body = gzip.compress(expanded) if encoding == 'gzip' else zlib.compress(expanded)
    del expanded

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Content-Encoding', encoding)
            if chunked:
                self.send_header('Transfer-Encoding', 'chunked')
            else:
                self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            try:
                if chunked:
                    self.wfile.write(f'{len(body):x}\r\n'.encode() + body + b'\r\n0\r\n\r\n')
                else:
                    self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    context = ToolContext(None, mock_config, '', '', '', 0, lambda message: None)
    try:
        tracemalloc.start()
        try:
            result = fetch_web_page.FetchWebPageTool().run(
                {'url': f'http://127.0.0.1:{server.server_port}'}, context,
            )
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        assert not result.success and 'download limit' in result.reply_text
        assert peak < 4 * (limit + fetch_web_page._READ_CHUNK_BYTES), (
            f'🌐 Decoder allocated {peak:,} bytes before rejecting the compressed page'
        )
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)
