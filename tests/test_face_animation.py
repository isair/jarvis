"""Rendered behaviour of the quiet desktop companion, with an isolated clock."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QImage, QPainter, QRegion

from desktop_app import face_widget
from desktop_app.face_widget import JarvisState


pytestmark = pytest.mark.unit


@dataclass
class Clock:
    now: float = 1000.0


@dataclass
class LocalState:
    state: JarvisState = JarvisState.ASLEEP


@pytest.fixture
def scene(qapp, monkeypatch):
    """Never construct the file-backed singleton or inspect the user's state."""
    clock = Clock()
    state = LocalState()
    monkeypatch.setattr(face_widget, 'get_jarvis_state', lambda: state)
    monkeypatch.setattr(face_widget._time, 'monotonic', lambda: clock.now)
    monkeypatch.setattr(face_widget.random, 'uniform', lambda low, high: (low + high) / 2)
    window = face_widget.FaceWindow()
    window.show()
    qapp.processEvents()
    window.face._animation_timer.stop()
    yield window, state, clock
    window.close()


def pixels(image):
    image = image.convertToFormat(QImage.Format.Format_RGBA8888)
    return np.frombuffer(image.constBits().asstring(image.sizeInBytes()), dtype=np.uint8).reshape(
        image.height(), image.width(), 4).copy()


def render(widget):
    # Fix only the pixel density; the real widget supplies all geometry and ink.
    image = QImage(widget.size(), QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    widget.render(image, QPoint(), QRegion(widget.rect()),
                  widget.RenderFlag.DrawChildren | widget.RenderFlag.IgnoreMask)
    return image


def advance(scene, state, seconds, fps=30):
    window, manager, clock = scene
    manager.state = state
    window.face._animate()
    started = clock.now
    for frame in range(1, round(seconds * fps) + 1):
        clock.now = started + frame / fps
        window.face._animate()


def test_idle_mouth_is_a_small_relaxed_smile(scene):
    """The resting expression reads as welcoming at actual desktop size."""
    window, _, _ = scene
    advance(scene, JarvisState.IDLE, 2)
    image = QImage(220, 280, QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    window.face._draw_mouth(painter, 110, 140, 154, 190)
    painter.end()
    alpha = pixels(image)[:, :, 3]
    occupied_x = np.flatnonzero(alpha.max(axis=0) > 30)
    centre = round((occupied_x[0] + occupied_x[-1]) / 2)
    rows = np.arange(alpha.shape[0])
    heights = [np.average(rows, weights=alpha[:, x])
               for x in (occupied_x[0] + 3, centre, occupied_x[-1] - 3)]
    assert heights[1] > max(heights[0], heights[2]) + 2
    assert occupied_x[-1] - occupied_x[0] < 154 * 0.5


@pytest.mark.parametrize('state', [JarvisState.LISTENING, JarvisState.THINKING, JarvisState.SPEAKING])
def test_motion_follows_elapsed_time_instead_of_frame_count(scene, state):
    """A busy desktop and a fast desktop show the same expression at the same time."""
    window, manager, clock = scene
    advance(scene, state, 2.5, fps=30)
    normal = pixels(render(window)).astype(float)
    window.close()
    clock.now = 1000.0
    manager.state = JarvisState.ASLEEP
    second = face_widget.FaceWindow()
    second.show()
    second.face._animation_timer.stop()
    try:
        advance((second, manager, clock), state, 2.5, fps=60)
        faster = pixels(render(second)).astype(float)
        assert np.abs(normal - faster).mean() < 0.25
    finally:
        second.close()


def test_hiding_the_face_suspends_its_animation_and_show_observes_current_state(scene):
    window, manager, clock = scene
    advance(scene, JarvisState.IDLE, 2)
    window.face._animation_timer.start()
    window.hide()
    assert not window.face._animation_timer.isActive()
    manager.state = JarvisState.LISTENING
    clock.now += 600
    observed = []
    window.face.state_observed.connect(observed.append)
    window.show()
    assert observed == [JarvisState.LISTENING.value]
    assert window.face._animation_timer.isActive()


def test_sleep_has_no_continuing_motion(scene):
    window, _, _ = scene
    advance(scene, JarvisState.ASLEEP, 5)
    first = pixels(render(window))
    advance(scene, JarvisState.ASLEEP, 20)
    assert np.array_equal(first, pixels(render(window)))


def test_empty_space_beside_the_face_does_not_capture_desktop_clicks(scene):
    window, _, _ = scene
    # Blank lateral margins are just as important as the four empty corners.
    assert not window.mask().contains(QPoint(round(window.width() * 0.1), window.height() // 2))
    assert window.mask().contains(window.rect().center())


def test_idle_motion_does_not_roam_across_the_desktop(scene):
    window, _, _ = scene
    edges = []
    for _ in range(80):
        advance(scene, JarvisState.IDLE, 0.25, fps=20)
        y, x = np.nonzero(pixels(render(window))[:, :, 3] > 20)
        edges.append((x.min(), y.min(), x.max(), y.max()))
    assert np.max(np.ptp(edges, axis=0)) <= window.width() * 0.015


@pytest.mark.parametrize('state', list(JarvisState))
def test_every_state_leaves_the_desktop_visible_and_stays_inside_its_footprint(scene, state):
    window, _, _ = scene
    advance(scene, state, 2.5)
    alpha = pixels(render(window))[:, :, 3]
    assert 0 < np.count_nonzero(alpha) < alpha.size * 0.22
    assert alpha[0].max() == alpha[-1].max() == 0
    assert alpha[:, 0].max() == alpha[:, -1].max() == 0
    assert alpha[window.height() // 2, window.width() // 2] == 0
    y, x = np.nonzero(alpha > 10)
    assert all(window.mask().contains(QPoint(int(px), int(py))) for px, py in zip(x, y))


@pytest.mark.parametrize('state', list(JarvisState))
@pytest.mark.parametrize('background', [(18, 20, 26), (242, 240, 235)])
def test_rendered_states_match_the_reviewed_light_and_dark_appearance(scene, state, background):
    """Allow minor rasteriser differences while protecting the actual character."""
    window, _, _ = scene
    advance(scene, state, 2.5)
    actual = pixels(render(window)).astype(float)
    reference = QImage(str(Path(__file__).parent / 'fixtures' / 'face' / f'{state.value}.png'))
    assert not reference.isNull()
    expected = pixels(reference).astype(float)
    assert actual.shape == expected.shape

    def on_desktop(rgba):
        alpha = rgba[:, :, 3:] / 255 * window.windowOpacity()
        return rgba[:, :, :3] * alpha + np.array(background) * (1 - alpha)

    difference = np.abs(on_desktop(actual) - on_desktop(expected))
    assert difference.mean() < 0.5
    assert np.mean(np.max(difference, axis=2) > 20) < 0.005
