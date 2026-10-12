"""Memory opens in its usable surface without a second placeholder window."""

from types import SimpleNamespace
import webbrowser

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("desktop_log_home", "no_network")]


@pytest.fixture
def viewer(qapp, monkeypatch):
    from desktop_app import app

    monkeypatch.setattr(app, "HAS_WEBENGINE", False)
    window = app.MemoryViewerWindow()
    window.is_server_running = True
    yield window
    window.close()


def open_from_tray(window):
    from desktop_app.app import JarvisSystemTray

    JarvisSystemTray.show_memory_viewer(SimpleNamespace(memory_viewer=window))


def test_browser_viewer_opens_without_a_native_window(viewer, monkeypatch):
    urls = []
    monkeypatch.setattr(webbrowser, "open", lambda url: urls.append(url) or True)
    open_from_tray(viewer)
    open_from_tray(viewer)
    assert urls == [f"http://localhost:{viewer.MEMORY_VIEWER_PORT}"] * 2
    assert not viewer.isVisible()
    assert viewer.is_server_running


@pytest.mark.parametrize("failure", ["server", "browser_false", "browser_exception"])
def test_browser_failure_is_actionable_without_a_placeholder(viewer, monkeypatch, failure):
    from PyQt6.QtWidgets import QMessageBox
    warnings = []
    opened = []
    monkeypatch.setattr(QMessageBox, "warning", lambda parent, title, message: warnings.append(message))
    if failure == "server":
        monkeypatch.setattr(viewer, "start_server", lambda: False)
    def launch(url):
        opened.append(url)
        if failure == "browser_exception":
            raise OSError("Synthetic browser failure")
        return False
    monkeypatch.setattr(webbrowser, "open", launch)
    open_from_tray(viewer)
    assert not viewer.isVisible()
    assert len(warnings) == 1
    if failure == "server":
        assert not opened
        assert "start" in warnings[0].lower()
    else:
        assert f"http://localhost:{viewer.MEMORY_VIEWER_PORT}" in warnings[0]


def test_embedded_viewer_loads_and_reopens_without_stopping_server(viewer, monkeypatch):
    from PyQt6.QtWidgets import QWidget

    urls = []
    embedded = QWidget()
    embedded.setUrl = lambda url: urls.append(url.toString())
    viewer.web_view = embedded
    viewer.setCentralWidget(embedded)
    monkeypatch.setattr(webbrowser, "open", lambda url: pytest.fail("Embedded viewer must stay in the app"))
    open_from_tray(viewer)
    assert viewer.isVisible()
    viewer.close()
    assert viewer.is_server_running
    open_from_tray(viewer)
    assert viewer.isVisible()
    assert urls == [f"http://localhost:{viewer.MEMORY_VIEWER_PORT}"] * 2


@pytest.mark.parametrize("mode", ["macos_bundle", "webengine_failure"])
def test_webengine_fallback_uses_only_the_browser(qapp, monkeypatch, mode):
    from desktop_app import app

    monkeypatch.setattr(app, "HAS_WEBENGINE", True)
    if mode == "macos_bundle":
        monkeypatch.setattr(app.sys, "platform", "darwin")
        monkeypatch.setattr(app.sys, "frozen", True, raising=False)
    def unavailable():
        raise RuntimeError("Synthetic WebEngine failure")
    monkeypatch.setattr(app, "QWebEngineView", unavailable, raising=False)
    window = app.MemoryViewerWindow()
    window.is_server_running = True
    urls = []
    monkeypatch.setattr(webbrowser, "open", lambda url: urls.append(url) or True)
    open_from_tray(window)
    assert urls == [f"http://localhost:{window.MEMORY_VIEWER_PORT}"]
    assert not window.isVisible()
