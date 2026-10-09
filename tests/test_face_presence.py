"""The always-visible face blends into the desktop without taking focus."""
import pytest
from PyQt6.QtCore import Qt, QPoint, QPointF
from PyQt6.QtGui import QContextMenuEvent, QMouseEvent
from PyQt6.QtWidgets import QMenu
from desktop_app.face_widget import Expression, JarvisState

pytestmark = pytest.mark.unit


@pytest.fixture
def face(qapp, monkeypatch, tmp_path):
    from desktop_app import face_widget
    monkeypatch.setattr(face_widget, '_get_jarvis_state_file', lambda: str(tmp_path / 'state'))
    monkeypatch.setattr(face_widget, '_jarvis_state_instance', None)
    window = face_widget.FaceWindow()
    window.show()
    qapp.processEvents()
    window.face._animation_timer.stop()
    yield window
    window.close()
    window.deleteLater()
    qapp.processEvents()


def test_face_is_frameless_transparent_and_does_not_accept_focus(face):
    assert face.windowFlags() & Qt.WindowType.FramelessWindowHint
    assert face.windowFlags() & Qt.WindowType.WindowStaysOnTopHint
    assert face.windowFlags() & Qt.WindowType.WindowDoesNotAcceptFocus
    assert face.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
    assert face.testAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
    image = face.grab().toImage()
    assert image.pixelColor(0, 0).alpha() == 0
    assert image.pixelColor(image.width() // 2, image.height() // 2).alpha() == 0
    assert any(image.pixelColor(x, y).alpha() > 0
               for x in range(image.width()) for y in range(image.height()))


def test_face_has_a_small_desktop_footprint(face):
    screen = face.screen().availableGeometry()
    assert face.width() < screen.width() / 3
    assert face.height() < screen.height() / 2
    assert face.face.geometry() == face.contentsRect()


def test_context_menu_can_hide_and_tray_can_show_the_face(face, qapp):
    event = QContextMenuEvent(QContextMenuEvent.Reason.Mouse, QPoint(20, 20),
                              face.mapToGlobal(QPoint(20, 20)))
    face.contextMenuEvent(event)
    qapp.processEvents()
    menu = next(menu for menu in face.findChildren(QMenu) if menu.isVisible())
    action = next(action for action in menu.actions() if 'hide' in action.text().lower())
    action.trigger()
    menu.close()
    assert not face.isVisible()
    face.show()
    assert face.isVisible()


def test_drag_moves_face_without_window_chrome(face, monkeypatch):
    # Exercise the fallback used when a compositor cannot start a system move.
    monkeypatch.setattr(face, 'windowHandle', lambda: None)
    before = face.pos()
    local = QPointF(40, 40)
    global_pos = QPointF(face.mapToGlobal(QPoint(40, 40)))
    face.mousePressEvent(QMouseEvent(QMouseEvent.Type.MouseButtonPress, local,
        global_pos, Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier))
    delta = QPointF(-35, 25)
    face.mouseMoveEvent(QMouseEvent(QMouseEvent.Type.MouseMove, local + delta,
        global_pos + delta, Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier))
    assert face.pos() == before + delta.toPoint()


def test_empty_corners_do_not_occupy_the_native_input_region(face):
    assert not face.mask().contains(QPoint(0, 0))
    assert face.mask().contains(face.rect().center())


def test_resting_presence_is_quieter_than_active_states(face):
    from desktop_app.face_widget import JarvisState
    manager = face.face._state_manager
    manager.set_state(JarvisState.ASLEEP)
    face.face._animate()
    asleep = face.windowOpacity()
    manager.set_state(JarvisState.IDLE)
    face.face._animate()
    idle = face.windowOpacity()
    for state in (JarvisState.LISTENING, JarvisState.THINKING,
                  JarvisState.SPEAKING, JarvisState.DICTATING,
                  JarvisState.DICTATION_PROCESSING):
        manager.set_state(state)
        face.face._animate()
        assert 0 < asleep < idle < face.windowOpacity() <= 1


def test_tray_show_does_not_request_focus(face):
    from desktop_app.app import JarvisSystemTray
    tray = object.__new__(JarvisSystemTray)
    tray.face_window = face
    face.hide()
    # Reject an explicit activation request while exercising the real tray action.
    face.activateWindow = lambda: pytest.fail('The face must not interrupt the active desktop application')
    tray.show_face_window()
    assert face.isVisible()


def test_subprocess_state_file_updates_presence_without_a_qt_signal(face):
    from pathlib import Path
    from desktop_app.face_widget import JarvisState
    before = face.windowOpacity()
    Path(face.face._state_manager._state_file).write_text(JarvisState.LISTENING.value)
    face.face._animate()
    assert face.windowOpacity() > before
    assert face.face._listening_started_at is not None


@pytest.mark.parametrize('state', list(JarvisState))
@pytest.mark.parametrize('expression', list(Expression))
@pytest.mark.parametrize('blink_factor', [0.0, 1.0])
@pytest.mark.parametrize('scale', [1, 2])
def test_eye_surroundings_stay_transparent(face, state, expression,
                                         blink_factor, scale):
    """Only eye strokes and pupils cover the desktop, including at high DPI."""
    from PyQt6.QtGui import QImage, QPainter

    widget = face.face
    widget._jarvis_state = state
    widget._expression = expression
    widget._activation_level = 0.0 if state == JarvisState.ASLEEP else 1.0
    size = 20
    centre = size * 2
    image = QImage(centre * 2 * scale, centre * 2 * scale,
                   QImage.Format.Format_ARGB32_Premultiplied)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.scale(scale, scale)
    try:
        widget._draw_eye(painter, centre, centre, size, blink_factor, is_left=True)
    finally:
        painter.end()

    # The horizontal margin is outside every expression's outline, but close
    # enough to catch a filled halo behind an open or closed eye.
    margin_x = round((centre + size * 1.2) * scale)
    assert all(image.pixelColor(margin_x, y).alpha() == 0
               for y in range(image.height()))
    if expression == Expression.NEUTRAL and blink_factor == 0.0 and state != JarvisState.ASLEEP:
        # Desktop content is visible between the pupil and the eye outline.
        gap_x = round((centre + size * 0.6) * scale)
        assert image.pixelColor(gap_x, centre * scale).alpha() == 0
    assert any(image.pixelColor(x, y).alpha() > 0
               for x in range(image.width()) for y in range(image.height()))
