"""Behaviour of the particle orb.

The orb's job is to make the assistant's state legible without words, so
these assert the contrasts a user actually perceives — speaking moves
more than idle, the far surface recedes — rather than exact pixels.
Geometry is pure and deterministic, so none of this needs a display.
"""

from __future__ import annotations

import math

import pytest

from src.desktop_app.face_widget import JarvisState
from src.desktop_app.orb_widget import (
    MOTION,
    PARTICLE_COUNT,
    OrbMotion,
    project,
    unit_sphere_points,
)


@pytest.mark.unit
class TestSphereDistribution:
    def test_every_point_lies_on_the_unit_sphere(self):
        for x, y, z in unit_sphere_points(200):
            assert math.isclose(math.sqrt(x * x + y * y + z * z), 1.0, abs_tol=1e-6)

    def test_points_are_spread_not_clumped_at_the_poles(self):
        """Equal-area banding is the whole point of the Fibonacci spiral.

        Split the sphere into three equal-height bands; a naive lat/long
        distribution piles up at the poles, an even one does not.
        """
        ys = [y for _, y, _ in unit_sphere_points(600)]
        low = sum(1 for y in ys if y < -1 / 3)
        mid = sum(1 for y in ys if -1 / 3 <= y <= 1 / 3)
        high = sum(1 for y in ys if y > 1 / 3)
        assert min(low, mid, high) > 0.28 * len(ys)

    def test_default_count_is_used(self):
        assert len(unit_sphere_points()) == PARTICLE_COUNT

    def test_degenerate_counts_do_not_raise(self):
        assert unit_sphere_points(0) == []
        assert len(unit_sphere_points(1)) == 1


@pytest.mark.unit
class TestProjection:
    def _project(self, state: JarvisState, t: float = 1.0):
        return project(
            unit_sphere_points(240), motion=MOTION[state], t=t, radius=100.0,
        )

    def test_depth_is_normalised(self):
        for _, _, depth in self._project(JarvisState.IDLE):
            assert 0.0 <= depth <= 1.0

    def test_both_near_and_far_surfaces_are_present(self):
        """Without a depth spread the cloud reads flat, not spherical."""
        depths = [d for _, _, d in self._project(JarvisState.IDLE)]
        assert min(depths) < 0.15
        assert max(depths) > 0.85

    def test_points_stay_within_the_drawn_radius(self):
        """Motion must not fling particles outside the widget."""
        radius = 100.0
        for state in JarvisState:
            for x, y, _ in project(
                unit_sphere_points(240), motion=MOTION[state], t=2.5, radius=radius,
            ):
                assert math.hypot(x, y) <= radius * 1.35, f"{state} escaped the frame"

    def test_the_orb_turns_over_time(self):
        first = self._project(JarvisState.IDLE, t=0.0)
        later = self._project(JarvisState.IDLE, t=2.0)
        assert any(
            abs(a[0] - b[0]) > 1.0 for a, b in zip(first, later)
        ), "orb is static; it should drift even when idle"


@pytest.mark.unit
class TestStateReadsDifferently:
    """Each state must be distinguishable at a glance."""

    def _deformation(self, state: JarvisState, t: float) -> float:
        """How far the surface departs from a smooth sphere.

        Spread of the per-point radius within one frame. This is what
        reads as motion: a ripple pushes some points out while pulling
        others in, so it deforms the surface without changing its
        average size. Uniform breathing scales every point together and
        scores ~0 here, which is correct — inflating is not talking.
        """
        pts = project(unit_sphere_points(240), motion=MOTION[state], t=t, radius=100.0)
        radii = []
        for x, y, depth in pts:
            z = (depth * 2.0 - 1.0) * 100.0
            radii.append(math.sqrt(x * x + y * y + z * z))
        mean = sum(radii) / len(radii)
        return math.sqrt(sum((r - mean) ** 2 for r in radii) / len(radii))

    def test_speaking_visibly_moves_while_idle_holds_steady(self):
        """The reason the orb exists: you can see it talking.

        Speaking must deform the surface by an order of magnitude more
        than idle, otherwise a reply looks identical to silence.
        """
        times = [i * 0.12 for i in range(24)]
        speaking = min(self._deformation(JarvisState.SPEAKING, t) for t in times)
        idle = max(self._deformation(JarvisState.IDLE, t) for t in times)

        assert speaking > idle * 10.0, (
            f"speaking deforms by {speaking:.2f}, idle by {idle:.2f} — "
            "not distinguishable enough to read as talking"
        )

    def test_idle_stays_a_smooth_sphere(self):
        """Idle breathes, but must not ripple — that cue belongs to speech."""
        for t in [i * 0.2 for i in range(15)]:
            assert self._deformation(JarvisState.IDLE, t) < 1.0

    def test_thinking_churns_without_looking_like_speech(self):
        """Turbulence must be visible, yet clearly below the speaking ripple."""
        times = [i * 0.12 for i in range(24)]
        thinking = sum(self._deformation(JarvisState.THINKING, t) for t in times) / len(times)
        speaking = sum(self._deformation(JarvisState.SPEAKING, t) for t in times) / len(times)
        idle = sum(self._deformation(JarvisState.IDLE, t) for t in times) / len(times)

        assert idle < thinking < speaking

    def test_thinking_spins_faster_than_idle(self):
        assert MOTION[JarvisState.THINKING].spin > MOTION[JarvisState.IDLE].spin

    def test_asleep_is_the_dimmest_state(self):
        dimmest = min(MOTION.values(), key=lambda m: m.brightness)
        assert dimmest is MOTION[JarvisState.ASLEEP]

    def test_speaking_is_the_brightest_state(self):
        assert MOTION[JarvisState.SPEAKING].brightness == max(
            m.brightness for m in MOTION.values()
        )

    def test_only_speaking_ripples(self):
        """The ripple is the talking cue; another state using it would muddy that."""
        rippling = [s for s, m in MOTION.items() if m.ripple > 0]
        assert rippling == [JarvisState.SPEAKING]

    def test_every_state_has_motion_defined(self):
        for state in JarvisState:
            assert isinstance(MOTION[state], OrbMotion), f"{state} has no motion"


@pytest.mark.unit
class TestStateIsReadCrossProcess:
    """The daemon usually runs in another process, so the orb must poll.

    `JarvisStateManager.set_state` writes a file for cross-process
    readers and emits `state_changed` only for same-process ones. In dev
    mode `app.py` starts the daemon with `subprocess.Popen`, so the
    signal never arrives in the UI process — a widget that only listens
    to it stays frozen at ASLEEP through every reply.
    """

    def test_orb_follows_state_written_by_another_process(self, qapp, monkeypatch, tmp_path):
        from src.desktop_app import face_widget
        from src.desktop_app.orb_widget import ParticleOrbWidget

        state_file = tmp_path / "jarvis_state"
        monkeypatch.setattr(
            face_widget, "_get_jarvis_state_file", lambda: str(state_file)
        )
        face_widget._jarvis_state_instance = None

        orb = ParticleOrbWidget()

        # Stand in for the daemon process: write the file directly, with
        # no signal, exactly as a separate process would.
        state_file.write_text(JarvisState.SPEAKING.value)
        orb._tick()

        assert orb.motion is MOTION[JarvisState.SPEAKING], (
            "orb ignored a state change made by another process"
        )

    def test_orb_recovers_from_an_unreadable_state_file(self, qapp, monkeypatch, tmp_path):
        """Garbage on disk must not crash the paint loop."""
        from src.desktop_app import face_widget
        from src.desktop_app.orb_widget import ParticleOrbWidget

        state_file = tmp_path / "jarvis_state"
        monkeypatch.setattr(
            face_widget, "_get_jarvis_state_file", lambda: str(state_file)
        )
        face_widget._jarvis_state_instance = None

        orb = ParticleOrbWidget()
        state_file.write_text("not-a-state")
        orb._tick()  # must not raise

        assert isinstance(orb.motion, OrbMotion)


@pytest.mark.unit
def test_frames_are_reproducible():
    """Same state and clock produce the same frame, so renders are testable."""
    a = project(unit_sphere_points(60), motion=MOTION[JarvisState.SPEAKING], t=1.234, radius=80.0)
    b = project(unit_sphere_points(60), motion=MOTION[JarvisState.SPEAKING], t=1.234, radius=80.0)
    assert a == b
