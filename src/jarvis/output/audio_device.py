"""Output-device selection shared by TTS and UI sounds."""
from __future__ import annotations

import sys
import threading
from typing import Any

from ..debug import debug_log


_selection_lock = threading.Lock()
_last_logged_selection: tuple[int | None, str] | None = None


def windows_default_output(sd: Any) -> int | None:
    """Return the real Windows WASAPI default output device.

    PortAudio's global ``sd.default.device`` can resolve to the default of a
    different host API (ASIO, DirectSound, WDM-KS) on studio machines. Edge and
    the Windows volume flyout use the WASAPI multimedia default, so Windows
    playback is deliberately pinned to that host API. Failure is explicit:
    silently falling back to an RME ASIO/ADAT port makes successful TTS
    inaudible while the application reports no error.
    """
    if sys.platform != "win32":
        try:
            device_id = int((sd.default.device or (-1, -1))[1])
            return device_id if device_id >= 0 else None
        except Exception:
            return None

    hostapis = sd.query_hostapis()
    wasapi = next(
        (
            api
            for api in hostapis
            if "windows wasapi" in str(api.get("name", "")).casefold()
        ),
        None,
    )
    if wasapi is None:
        raise RuntimeError("Windows WASAPI host API is unavailable")

    try:
        device_id = int(wasapi.get("default_output_device", -1))
    except (TypeError, ValueError):
        device_id = -1
    if device_id < 0:
        raise RuntimeError("Windows has no WASAPI default output device")

    info = sd.query_devices(device_id)
    if int(info.get("max_output_channels", 0) or 0) < 1:
        raise RuntimeError(
            f"Windows WASAPI default is not an output device (index {device_id})"
        )

    name = str(info.get("name", f"device {device_id}"))
    global _last_logged_selection
    selection = (device_id, name)
    with _selection_lock:
        if selection != _last_logged_selection:
            debug_log(
                f"Windows default output: {name} (WASAPI index {device_id})",
                "tts",
            )
            print(
                f"  🔊 Windows default output: {name} (WASAPI)",
                flush=True,
            )
            _last_logged_selection = selection
    return device_id


def output_stream_samplerate(sd: Any, device_id: Any, model_rate: int) -> int:
    """Sample rate to open ``device_id`` with, given a model rate.

    PortAudio's ``Windows WASAPI`` host accepts only the endpoint's default mix
    format rate; any other value makes ``Pa_OpenStream`` fail with
    ``paInvalidSampleRate`` (PaErrorCode -9997). WASAPI was chosen on purpose
    (see :func:`windows_default_output`), so the device rate wins over the
    model rate. Non-WASAPI hosts and unknown info fall back to the model rate,
    which those hosts accept directly.
    """
    if device_id is None:
        return int(model_rate)
    try:
        info = sd.query_devices(int(device_id))
        rate = int(info.get("default_samplerate") or 0)
    except Exception:
        return int(model_rate)
    if rate <= 0:
        return int(model_rate)
    hostapi_idx = info.get("hostapi")
    host_name = ""
    try:
        if hostapi_idx is not None:
            host_name = str(
                sd.query_hostapis()[int(hostapi_idx)].get("name", "")
            ).casefold()
    except Exception:
        pass
    if "wasapi" in host_name:
        return rate
    return int(model_rate)


def resample_int16(np_module: Any, samples: Any, src_hz: int, dst_hz: int) -> Any:
    """Linearly resample an int16 mono buffer; no-op when rates match."""
    src_hz, dst_hz = int(src_hz), int(dst_hz)
    if dst_hz == src_hz or src_hz <= 0 or samples.size == 0:
        return samples
    n_in = int(samples.size)
    n_out = max(1, int(round(n_in * dst_hz / src_hz)))
    x_in = np_module.linspace(0.0, 1.0, num=n_in)
    x_out = np_module.linspace(0.0, 1.0, num=n_out)
    return np_module.interp(x_out, x_in, samples).astype(np_module.int16)
