"""Readable download output and desktop progress presentation."""

import io
import pytest

from desktop_app.log_output import LogStream, parse_progress

pytestmark = pytest.mark.unit


def test_stream_handles_terminal_updates_and_partial_lines():
    lines = []
    stream = LogStream(lines.append)
    stream.write("hello")
    stream.write(" world\n\rweights: 10%|x| 1M/10M [00:01<00:09, 1MB/s]")
    stream.flush()
    stream.write("\x1b[A\rweights: 20%|xx| 2M/10M [00:02<00:08, 1MB/s]\n")
    assert lines == ["hello world\n", "weights: 10%|x| 1M/10M [00:01<00:09, 1MB/s]\n",
                     "weights: 20%|xx| 2M/10M [00:02<00:08, 1MB/s]\n"]


def test_progress_preserves_byte_counts_and_rate():
    result = parse_progress("weights.npz:  24%|██▍| 366M/1.52G [00:30<01:35, 12.1MB/s]")
    assert result.name == "weights.npz"
    assert result.percent == 24
    assert "366M/1.52G" in result.detail
    assert "12.1MB/s" in result.detail
    assert parse_progress("❌ Download failed: connection lost") is None


def test_progress_card_coalesces_updates_and_keeps_errors(qapp):
    from desktop_app.app import LogViewerWindow
    window = LogViewerWindow()
    window.append_log("weights.npz: 10%|x| 1M/10M [00:01<00:09, 1MB/s]\n")
    window.append_log("weights.npz: 20%|xx| 2M/10M [00:02<00:08, 1MB/s]\n")
    window.append_log("⚠️ Optional feature unavailable\n")
    assert window.download_bar.value() == 20
    assert "2M/10M" in window.download_detail.text()
    assert "10%" not in window.log_display.toPlainText()
    assert "Optional feature unavailable" in window.log_display.toPlainText()
    window.append_log("weights.npz: 100%|xxxxxxxxxx| 10M/10M [00:10<00:00, 1MB/s]\n")
    window.append_log("weights.npz: 100%|xxxxxxxxxx| 10M/10M [00:10<00:00, 1MB/s]\n")
    assert window.log_display.toPlainText().count("Downloaded weights.npz") == 1
    window.clear_logs()
    assert window.download_card.isHidden()
    window.close()


def test_cached_whisper_path_and_loading_message(monkeypatch, capsys, tmp_path):
    from jarvis.listening.model_download import prepare_mlx_model
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda **kwargs: str(tmp_path))
    assert prepare_mlx_model("test/whisper") == str(tmp_path)
    output = capsys.readouterr().out
    assert "Checking Whisper model files" in output
    assert "Loading Whisper into memory" in output


def test_download_failure_does_not_claim_model_loaded(monkeypatch, capsys):
    from jarvis.listening.model_download import prepare_mlx_model
    import huggingface_hub
    import pytest
    def fail(**kwargs):
        raise OSError("offline")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", fail)
    with pytest.raises(OSError):
        prepare_mlx_model("test/whisper")
    assert "Loading Whisper into memory" not in capsys.readouterr().out


def test_completion_of_small_file_does_not_hide_active_download(qapp):
    from desktop_app.app import LogViewerWindow
    window = LogViewerWindow()
    window.append_log("weights.npz: 20%|xx| 2M/10M [00:02<00:08, 1MB/s]\n")
    window.append_log("config.json: 100%|xxx| 2k/2k [00:01<00:00, 2kB/s]\n")
    assert "weights.npz" in window.download_title.text()
    assert window.download_bar.value() == 20
    window.close()


def test_unknown_size_is_indeterminate(qapp):
    from desktop_app.app import LogViewerWindow
    window = LogViewerWindow()
    window.append_log("weights.npz: 12.0MB [00:03, 4.0MB/s]\n")
    assert window.download_bar.maximum() == 0
    assert "12.0MB" in window.download_detail.text()
    window.close()


def test_slow_download_does_not_fake_progress(qapp, monkeypatch):
    from desktop_app.app import LogViewerWindow
    window = LogViewerWindow()
    window.append_log("weights.npz: 20%|xx| 2M/10M [00:02<00:08, 1MB/s]\n")
    monkeypatch.setattr("desktop_app.app.time.monotonic", lambda: window._last_progress_at + 30)
    window._refresh_download_status()
    assert "No new progress for 30s" in window.download_detail.text()
    assert window.download_bar.value() == 20
    window.close()


def test_native_hub_byte_progress_reaches_non_terminal_desktop(monkeypatch):
    """Exercise the installed Hub HTTP transport without network or model data."""
    from desktop_app import app  # Configures non-terminal progress before daemon launch.
    from huggingface_hub.file_download import http_get
    from unittest.mock import Mock
    response = Mock(status_code=200, headers={"Content-Length": "1000000"})
    response.iter_content.return_value = [b"x" * 1_000_000]
    monkeypatch.setattr("huggingface_hub.file_download._request_wrapper", lambda **kw: response)
    lines = []
    monkeypatch.setattr("sys.stderr", LogStream(lines.append))
    destination = io.BytesIO()
    http_get("https://example.invalid/model", destination, displayed_filename="weights.npz")
    progress = [parse_progress(line) for line in lines if parse_progress(line)]
    assert progress[-1].percent == 100
    assert "1.00M/1.00M" in progress[-1].detail
    assert len(destination.getvalue()) == 1_000_000


@pytest.mark.parametrize("known_size", [True, False])
def test_piper_updates_are_throttled_and_complete(tmp_path, monkeypatch, known_size):
    from jarvis.output import tts
    from unittest.mock import Mock
    response = Mock(headers={"content-length": "1000000"} if known_size else {})
    response.iter_content.side_effect = lambda **kw: iter([b"x" * 1000] * 1000)
    monkeypatch.setattr("requests.get", lambda *args, **kwargs: response)
    monkeypatch.setattr(tts, "_get_piper_models_dir", lambda: tmp_path)
    monkeypatch.setattr(tts.time, "monotonic", lambda: 1.0)
    logs = []
    assert tts._download_piper_voice("en_GB-alan-medium", logs.append)
    assert len(logs) < 15
    assert (tmp_path / "en_GB-alan-medium.onnx").stat().st_size == 1_000_000
    assert any("100%" in line if known_size else "total size unknown" in line for line in logs)


def test_loading_then_failure_is_not_reported_as_ready(qapp):
    from desktop_app.app import LogViewerWindow
    window = LogViewerWindow()
    window.append_log("🎤 Loading Whisper into memory and warming up speech recognition...\n")
    assert window.download_bar.maximum() == 0
    window.append_log("❌ Failed to initialise MLX Whisper: not enough memory\n")
    assert window.download_bar.maximum() == 100
    assert window.download_bar.format() == 'Interrupted'
    assert 'not enough memory' in window.log_display.toPlainText()
    window.close()


def test_report_redacts_progress_details(qapp, monkeypatch):
    from desktop_app.app import LogViewerWindow
    from urllib.parse import parse_qs, urlparse
    window = LogViewerWindow()
    window.append_log("weights.npz: 20%|xx| 2M/10M · token=private-value\n")
    opened = []
    monkeypatch.setattr('desktop_app.app.webbrowser.open', opened.append)
    window._report_issue()
    body = parse_qs(urlparse(opened[0]).query)['body'][0]
    assert 'private-value' not in body
    assert '2M/10M' in body
    assert '[REDACTED]' in body
    window.close()


def test_subprocess_carriage_returns_reach_progress_card(qapp):
    import subprocess
    import sys
    from desktop_app.app import LogViewerWindow
    window = LogViewerWindow()
    process = subprocess.Popen(
        [sys.executable, '-c', "print('\\rweights.npz: 20%|xx| 2M/10M [00:02<00:08, 1MB/s]\\rweights.npz: 30%|xxx| 3M/10M [00:03<00:07, 1MB/s]')"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    for line in process.stdout:
        window.append_log(line)
    process.wait(timeout=5)
    assert window.download_bar.value() == 30
    assert window.log_display.toPlainText().count('Downloading weights.npz') == 1
    window.close()


def test_native_xet_byte_progress_reaches_desktop(monkeypatch, tmp_path):
    """The large-file transport reports bytes as well as ordinary HTTP."""
    from desktop_app import app
    from huggingface_hub.file_download import xet_get
    from types import SimpleNamespace
    import sys
    connection = SimpleNamespace(endpoint='unused', access_token='unused', expiration_unix_epoch=0)
    monkeypatch.setattr('huggingface_hub.file_download.refresh_xet_connection_info', lambda **kw: connection)
    def download_files(*args, **kwargs):
        kwargs['progress_updater'][0](250_000)
        kwargs['progress_updater'][0](750_000)
    monkeypatch.setitem(sys.modules, 'hf_xet', SimpleNamespace(
        PyXetDownloadInfo=lambda **kwargs: kwargs, download_files=download_files,
    ))
    lines = []
    monkeypatch.setattr('sys.stderr', LogStream(lines.append))
    xet_get(incomplete_path=tmp_path / 'weights.incomplete',
            xet_file_data=SimpleNamespace(file_hash='unused'), headers={},
            expected_size=1_000_000, displayed_filename='weights.npz')
    progress = [parse_progress(line) for line in lines if parse_progress(line)]
    assert progress[-1].percent == 100
    assert '1.00M/1.00M' in progress[-1].detail
