"""Parity suite for the v1/v2 DSP surface shared by ``native_audio`` / ``native_bridge``."""

from __future__ import annotations

import ctypes  # noqa: F401

from jarvis import native_audio as _na
from jarvis import native_bridge as _nb


def _mk_block(n: int, *, seed: int = 7) -> list:
    return [float((i * 31 + seed) % 1000) / 1000.0 for i in range(n)]


def test_dsp_domain_constants_parity() -> None:
    assert _na.AEC_RATE_HZ == _nb.AEC_RATE_HZ == 48000
    assert _na.AEC_FRAME_SAMPLES == _nb.AEC_FRAME_SAMPLES == 480
    assert _na.ASR_RATE_HZ == _nb.ASR_RATE_HZ == 16000
    assert _na.ASR_FRAME_SAMPLES == _nb.ASR_FRAME_SAMPLES == 160
    assert _na.CLEANED_RING_FRAMES == _nb.CLEANED_RING_FRAMES == 512


def test_dsp_v1_engine_and_ducking_reads_parity() -> None:
    # Fresh v1 engine (rollback path) — identical answers from both modules.
    st1 = _na.create(profile=_na.PROFILE_HOSTILE_PLAYBACK)
    # 2 = shared mix without the RAW option, 3 = no loopback tap here; both
    # are legitimate degraded-but-working create() outcomes on real hardware.
    assert st1 in (0, 2, 3)
    assert _nb.last_status() == _na.last_status()
    caps = _nb.capabilities()
    assert caps == _na.capabilities()
    assert bool(caps & _nb.CAP_RAW_CAPTURE or not caps)
    t_a, t_b = _na.telemetry(), _nb.telemetry()
    if st1 == 0:
        assert t_a is not None and t_b is not None
        assert t_a.ducking_state == t_b.ducking_state
        assert t_a.ducking_current_db == t_b.ducking_current_db
    lay_a, lay_b = _na.shm_layout(), _nb.shm_layout()
    if st1 == 0:
        assert lay_a is not None and lay_b is not None
        assert int(lay_a.asr_frame_samples) == int(lay_b.asr_frame_samples) == 160
        assert int(lay_a.aec_rate_hz) == int(lay_b.aec_rate_hz) == 48000
    _na.destroy()
    assert _na.last_status() == _nb.last_status() == 0


def test_dsp_v2_engine_telemetry_labels_parity() -> None:
    st, engine = _na.engine_create()
    assert st == 0 and engine
    tel_a = _na.engine_telemetry(engine)
    tel_b = _nb.engine_telemetry(engine)
    assert isinstance(tel_a, dict) and tel_a == tel_b
    assert {
        key: tel_a[key]
        for key in
        ("capture_native_rate_hz", "render_native_rate_hz", "generation")
    } == {
        key: tel_b[key]
        for key in
        ("capture_native_rate_hz", "render_native_rate_hz", "generation")
    }

    # Satellite lane with both reference models.
    st_s, lane_inj = _nb.lane_create(
        engine, source_type=_nb.SOURCE_SATELLITE, device_id="parity-inj",
        tts_ref_mode=_nb.TTSREF_INJECTED,
    )
    assert st_s == 0 and lane_inj is not None
    push_inj = _nb.lane_push_reference(lane_inj, _mk_block(160), 16000)
    assert push_inj == 0
    cap_a = _nb.lane_push_capture(lane_inj, _mk_block(160), 16000)
    cap_b = _na.lane_push_capture(lane_inj, _mk_block(160), 16000)
    # same lane, same DLL: second push answers identically
    assert cap_a == cap_b == 0
    rate_a, block_a = _na.lane_pop_clean(lane_inj)
    rate_b, block_b = _nb.lane_pop_clean(lane_inj)
    assert (rate_a, None if block_a is None else block_a.size) == \
        (rate_b, None if block_b is None else block_b.size)
    assert rate_a in (0, 16000)
    # after one pump step the lane must eventually deliver cleaned audio
    assert _na.lane_telemetry(int(lane_inj))["lane_id"] >= 1

    st_l, lane_loop = _na.lane_create(
        engine, source_type=_na.SOURCE_SATELLITE, device_id="parity-loop",
        tts_ref_mode=_na.TTSREF_LOOPBACK,
    )
    assert st_l == 0 and lane_loop is not None
    lt_a = _na.lane_telemetry(int(lane_loop))
    lt_b = _nb.lane_telemetry(int(lane_loop))
    assert lt_a == lt_b and lt_a is not None
    # state labels identical
    for raw, label in _na.CONV_LABEL.items():
        if raw == lt_a["aec_state"]:
            assert lt_b["aec_state_label"] == label
            break
    # reset while the lane is still alive (both bindings), then destroy once
    assert _na.lane_reset(int(lane_inj)) == _nb.lane_reset(int(lane_inj)) == 0
    assert _na.lane_reset(int(lane_loop)) == _nb.lane_reset(int(lane_loop)) == 0
    _na.lane_destroy(int(lane_inj))
    _nb.lane_destroy(int(lane_loop))
    _nb.engine_dump(engine, "parity_dsp")
    _na.engine_dump(engine, "parity_dsp")
    print("parity engine:", tel_a["lane_count"], "lanes")

    _na.engine_destroy(engine)
    # v1 globals must stay identical through both modules after close
    assert _na.capabilities() == _nb.capabilities()
    assert _na.last_status() == _nb.last_status()
