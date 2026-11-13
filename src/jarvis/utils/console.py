"""Prepare Unicode console output without taking ownership of existing buffers."""

import os

from ..debug import debug_log


def configure_utf8_streams(stream_owner) -> None:
    """Reconfigure open text writers in place and provide sinks when unavailable."""
    for name in ('stdout', 'stderr'):
        stream = getattr(stream_owner, name)
        if stream is None or getattr(stream, 'closed', False):
            setattr(stream_owner, name, open(os.devnull, 'w', encoding='utf-8', errors='replace'))
            continue
        reconfigure = getattr(stream, 'reconfigure', None)
        if callable(reconfigure):
            try:
                reconfigure(encoding='utf-8', errors='replace')
            except (OSError, ValueError) as exc:
                debug_log(f'⚠️ Console {name} encoding unavailable: {type(exc).__name__}', 'startup')
    debug_log('🖥️ Console setup preserved ownership of open output streams', 'startup')
