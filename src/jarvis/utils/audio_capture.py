"""Shared microphone selection, format negotiation and mono conversion."""

from contextlib import nullcontext

from ..debug import debug_log
from .audio_lock import portaudio_lock


def resolve_input_device(sounddevice, voice_device, devices=None):
    """Resolve one input index for every phase of a capture session."""
    selected = str(voice_device if voice_device is not None else '').strip()
    if not selected or selected.lower() in ('default', 'system'):
        try:
            info = sounddevice.query_devices(kind='input')
            if info['index'] < 0 or info['max_input_channels'] <= 0:
                raise ValueError('Default device has no microphone input')
        except Exception as exc:
            raise ValueError(
                'System default microphone unavailable. Choose an available input '
                'in Jarvis Settings or set a default microphone in system Settings.'
            ) from exc
        debug_log(f"Resolved default microphone: {info.get('name', 'Unknown')} (index {info['index']})", 'audio')
        return {'device': info['index']}
    try:
        return {'device': int(selected)}
    except ValueError:
        pass

    if devices is None:
        devices = sounddevice.query_devices()
    requested = selected.casefold()
    partial_match = None
    for index, device in enumerate(devices):
        if (device.get('max_input_channels') or 0) <= 0:
            continue
        name = str(device.get('name') or '').casefold()
        if requested == name:
            debug_log(f'Resolved microphone name to input index {index} (exact match)', 'audio')
            return {'device': index}
        if partial_match is None and requested in name:
            partial_match = index
    if partial_match is not None:
        debug_log(f'Resolved microphone name to input index {partial_match} (partial match)', 'audio')
        return {'device': partial_match}
    raise ValueError('Selected microphone not found. Choose an available input in Settings.')


def _is_input_format_error(exc):
    """Distinguish unsupported capture formats from access and device failures."""
    code = exc.args[1] if len(exc.args) > 1 else None
    message = str(exc).lower()
    return code in (-9998, -9997) or any(part in message for part in (
        'invalid number of channels', 'invalid channel count',
        'invalid sample rate', 'paerrorcode -9998', 'paerrorcode -9997',
    ))


def open_input_stream(sounddevice, sample_rate, frame_ms, device_kwargs, *,
                      callback=None, serialise=True, log_category='voice', fallback_rate=None):
    """Try bounded channel/rate formats on one selected input device."""
    candidates = [(sample_rate, 1)]
    first_error = None
    for rate, channels in candidates:
        try:
            with portaudio_lock if serialise else nullcontext():
                stream = sounddevice.InputStream(
                    samplerate=rate, channels=channels, dtype='float32',
                    blocksize=max(1, int(rate * frame_ms / 1000)),
                    callback=callback, **device_kwargs,
                )
            debug_log(f'Input format accepted: {rate} Hz, {channels} channel(s)', log_category)
            return stream, rate, channels
        except Exception as exc:
            if not _is_input_format_error(exc):
                raise
            if first_error is None:
                first_error = exc
            debug_log(f'Input format rejected: {rate} Hz, {channels} channel(s): {exc}', log_category)
            if len(candidates) == 1:
                try:
                    info = (sounddevice.query_devices(device_kwargs['device'])
                            if 'device' in device_kwargs else sounddevice.query_devices(kind='input'))
                    native_rate = int(info.get('default_samplerate', sample_rate))
                    max_channels = int(info.get('max_input_channels', 1))
                except Exception:
                    raise first_error
                rates = list(dict.fromkeys(
                    rate for rate in (sample_rate, native_rate, fallback_rate)
                    if rate is not None and rate > 0
                ))
                counts = list(dict.fromkeys(
                    count for count in (1, 2, max_channels) if 0 < count <= max_channels
                ))
                candidates.extend((r, c) for c in counts for r in rates if (r, c) != candidates[0])
    raise first_error


def mono_capture(indata):
    """Retain microphone signal from every captured channel."""
    return indata.mean(axis=1) if indata.ndim > 1 else indata.flatten()
