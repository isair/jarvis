"""Parity tests for the native audio surface and its bridge mirror.

Runs with the real ``jarvis_audio_engine.dll``: same functions on both
module objects must produce the same results.
"""

from __future__ import annotations

import ctypes  # noqa: F401  (kept for struct-size asserts below)
import threading
import time

from jarvis import native_audio as _na
from jarvis import native_bridge as _nb


# ---------------------------------------------------------------------------
# module shape
# ---------------------------------------------------------------------------

def test_both_modules_load_one_abi_two_dll() -> None:
    assert _na.load() is True
    assert _nb.load() is True
    assert _na.is_loaded() is True
    assert _nb.is_loaded() is True
    assert _na.ABI_VERSION == _nb.ABI_VERSION == 2
    assert _na.supported_abis() == _nb.supported_abis() == (1, 2)
    # The title line of each module docstring differs by exactly one sentence.
    assert (_na.__doc__ or "").startswith(
        "In-process ctypes binding to the Toustovač native audio engine.")
    assert (_nb.__doc__ or "").startswith(
        "Bridge to the Toustovač Windows audio stack: native audio engine")
    assert (_nb.__doc__ or "").find("virtual microphone") > 0


def test_bridge_mirror_named_same_api() -> None:
    assert _nb.__all__ == _na_names_in_origin_order()


def _na_names_in_origin_order() -> list[str]:
    # Same order as the mirrored module; the parity list is fixed by the API.
    return [
        "AEC_RATE_HZ", "AEC_FRAME_MS", "AEC_FRAME_SAMPLES", "ASR_RATE_HZ",
        "ASR_FRAME_SAMPLES", "CLEANED_RING_FRAMES", "ABI_VERSION",
        "AEC_MODE_OFF", "AEC_MODE_WEBRTC_AEC3",
        "AEC_MODE_WINDOWS_ENDPOINT_AEC", "PROFILE_STUDIO", "PROFILE_ASSISTANT",
        "PROFILE_HOSTILE_PLAYBACK", "CONV_DISABLED", "CONV_ACQUIRING",
        "CONV_CONVERGED", "CONV_DOUBLE_TALK", "CONV_RECONVERGING",
        "CONV_FAILED", "CONV_LABEL", "NAMED_STATUS", "SOURCE_LOCAL_WASAPI",
        "SOURCE_SATELLITE", "REF_TAP_UNKNOWN", "REF_TAP_POST_VOLUME",
        "REF_TAP_PRE_VOLUME", "REF_TAP_INJECTED", "TTSREF_LOOPBACK",
        "TTSREF_INJECTED", "CHANNEL_MODES", "DUCK_OFF", "DUCK_ACTIVE",
        "DUCK_RELEASING", "DUCK_RESTORED_OK", "DUCK_RESTORE_PARTIAL",
        "CAP_RAW_CAPTURE", "CAP_LOOPBACK", "CAP_NATIVE_AEC",
        "CAP_ENDPOINT_REF_CTRL", "CAP_POST_VOLUME_REF", "OK", "STATUS_NAMES",
        "Cfg", "Telemetry", "ShmLayout", "EndpointInfo", "EngineConfigV2",
        "LaneConfigV2", "AudioPacketV2", "LaneTelemetryV2",
        "EngineTelemetryV2", "load", "is_loaded", "supported_abis", "create",
        "run", "destroy", "set_profile", "set_listening", "capabilities",
        "last_status", "telemetry", "shm_layout", "pop_asr",
        "status_summary", "enumerate_endpoints", "endpoint_lines",
        "engine_create", "engine_run", "engine_destroy", "engine_telemetry",
        "lane_create", "lane_destroy", "lane_push_capture",
        "lane_push_reference", "lane_pop_clean", "lane_telemetry",
        "lane_reset", "engine_dump",
    ]


# ---------------------------------------------------------------------------
# constants and struct sizes — one value for both modules
# ---------------------------------------------------------------------------

def test_constants_parity() -> None:
    for name in _nb.__all__:
        if name in ("ABI_VERSION",) or callable(getattr(_na, name, None)):
            continue
        if isinstance(getattr(_na, name), type) or not isinstance(
            getattr(_na, name), (int, str, dict, tuple, list)
        ):
            continue
        assert getattr(_na, name) == getattr(_nb, name), name


def test_struct_layout_parity() -> None:
    for name in "Cfg", "Telemetry", "ShmLayout", "EndpointInfo", \
            "EngineConfigV2", "LaneConfigV2", "AudioPacketV2", \
            "LaneTelemetryV2", "EngineTelemetryV2":
        mod = getattr(_na, name)
        brg = getattr(_nb, name)
        assert mod is not brg
        assert ctypes.sizeof(mod) == ctypes.sizeof(brg)


# ---------------------------------------------------------------------------
# v1 path (same singleton DLL, so both modules see the same state)
# ---------------------------------------------------------------------------

def test_v1_parity_create_and_reads() -> None:
    st_a = _na.create()
    assert st_a in (0, 2)          # 2 = shared mix without the RAW option
    st_b = _nb.create()
    assert st_b == st_a
    caps_a, caps_b = _na.capabilities(), _nb.capabilities()
    assert caps_a == caps_b
    if st_a == 0:
        assert caps_a & _na.CAP_NATIVE_AEC
    stat_a, stat_b = _na.last_status(), _nb.last_status()
    assert stat_a == stat_b
    sa, sb = _na.status_summary(), _nb.status_summary()
    assert sa == sb
    assert sa["abi"] == 2 and sa["capability_bits"] == caps_a
    assert set(sa["capabilities"]) <= {
        "raw_capture", "loopback", "native_aec", "endpoint_ref_ctrl",
        "post_volume_ref",
    }
    _na.set_profile(_na.PROFILE_ASSISTANT)
    _nb.set_profile(_na.PROFILE_ASSISTANT)
    _na.set_listening(1)
    _nb.set_listening(1)
    rate, arr = _na.pop_asr(max_frames=2)
    assert (rate, arr) == _nb.pop_asr(max_frames=2)
    _na.destroy()
    # same global v1 singleton through both bindings
    assert _na.last_status() == _nb.last_status() == 0


# ---------------------------------------------------------------------------
# v2 path — engine create, telemetry; lane + packet work
# ---------------------------------------------------------------------------

def test_v2_create_engine_endpoint_lists_match() -> None:
    eps_a = _na.enumerate_endpoints()
    eps_b = _nb.enumerate_endpoints()
    assert len(eps_a) == len(eps_b) > 0
    assert [{k: e[k] for k in sorted(e)} for e in eps_a] == \
        [{k: e[k] for k in sorted(e)} for e in eps_b]
    lines_a = _na.endpoint_lines(2) + _na.endpoint_lines(3)
    lines_b = _nb.endpoint_lines(2) + _nb.endpoint_lines(3)
    assert lines_a == lines_b

    st, engine = _na.engine_create()
    assert (st, engine is not None) == (0, True)
    st_b, engine_b = _nb.engine_create()
    assert st_b == st
    # the same process-wide engine truth: telemetry from both modules
    tel_a, tel_b = _na.engine_telemetry(engine), _nb.engine_telemetry(engine_b)
    assert tel_a == tel_b
    assert tel_a is not None
    assert tel_a["generation"] >= 1 and tel_a["lane_count"] >= 0

    # ``engine_run`` is the engine's endless pump pass: one daemon thread per
    # engine (exactly like :func:`jarvis.listening.audio_io.native_create`).
    threading.Thread(target=_na.engine_run, args=(engine,), daemon=True).start()
    threading.Thread(target=_nb.engine_run, args=(engine_b,), daemon=True).start()

    lst, lane = _na.lane_create(engine, source_type=_na.SOURCE_LOCAL_WASAPI)
    assert lst == 0 and lane is not None
    lst_b, lane_b = _nb.lane_create(engine_b, source_type=_nb.SOURCE_LOCAL_WASAPI)
    assert lst_b == 0 and lane_b is not None
    block = [float(i % 16) / 16.0 for i in range(480)]
    assert _na.lane_push_capture(lane, block, 48000) == 0
    assert _nb.lane_push_capture(lane_b, block, 48000) == 0

    time.sleep(0.08)

    def first_pop(mod, lane_handle):
        rate, blk = mod.lane_pop_clean(lane_handle)
        return (rate, None if blk is None else blk.size)

    def next_pop(mod, lane_handle):
        for _ in range(8):
            time.sleep(0.02)
            rate, blk = mod.lane_pop_clean(lane_handle)
            if blk is not None:
                return (rate, blk.size)
        return first_pop(mod, lane_handle)

    # the same 16k stream shape from both lanes: 16 kHz, 160-sample blocks
    a_first, b_first = next_pop(_na, lane), next_pop(_nb, lane_b)
    assert a_first[0] == b_first[0] == 16000
    assert a_first[1] == b_first[1] == 160
    # per-module readbacks agree on each lane (same handle, both bindings)
    assert first_pop(_na, lane) == first_pop(_nb, lane)
    assert first_pop(_na, lane_b) == first_pop(_nb, lane_b)

    lt_a = _na.lane_telemetry(int(lane))
    lt_b = _nb.lane_telemetry(int(lane))
    assert lt_a is not None and lt_b is not None
    # parity on the stable identity/routing fields of each lane
    stable = (
        "engine_generation", "lane_id", "source_type", "device_id",
        "connection_generation", "session_generation",
        "capture_endpoint_id", "capture_endpoint_name",
        "render_endpoint_id", "render_endpoint_name",
        "capture_native_rate_hz", "capture_native_channels",
        "capture_native_format", "render_native_rate_hz",
        "render_native_channels", "render_native_format",
        "capture_channel_mode", "capture_channel_index",
        "reference_tap", "reference_active", "tts_ref_mode",
    )
    assert {k: lt_a[k] for k in stable} == {k: lt_b[k] for k in stable}
    assert _na.lane_telemetry(int(lane)) is not None

    lt_na_b = _na.lane_telemetry(int(lane_b))
    lt_nb_b = _nb.lane_telemetry(int(lane_b))
    assert {k: lt_na_b[k] for k in stable} == {k: lt_nb_b[k] for k in stable}

    _nb.lane_reset(int(lane))
    _nb.lane_reset(int(lane_b))
    _na.engine_dump(engine, "parity_a")
    _nb.engine_dump(engine_b, "parity_b")
    # The pump is the engine's endless loop and the daemon threads keep it
    # running; the engines live for the rest of this process.
    assert _na.capabilities() == _nb.capabilities()
    assert _na.last_status() == _nb.last_status()
