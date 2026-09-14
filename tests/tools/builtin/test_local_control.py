from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from jarvis.tools.base import ToolContext
from jarvis.tools.builtin.local_control import LocalControlTool


def _context(**overrides):
    values = {
        "local_control_enabled": True,
        "local_control_require_approval": True,
        "local_control_allowed_applications": ["notepad.exe"],
        "local_control_allowed_roots": ["~"],
    }
    values.update(overrides)
    context = SimpleNamespace(
        cfg=SimpleNamespace(**values),
        original_prompt=overrides.get(
            "original_prompt", "Please open it. I approve this local action."
        ),
        user_print=Mock(),
    )
    if "approval_callback" in overrides:
        context.approval_callback = overrides["approval_callback"]
    return context


class TestLocalControlTool:
    def test_requires_feature_flag(self):
        result = LocalControlTool().run(
            {"operation": "open_url", "url": "https://example.test"},
            _context(local_control_enabled=False),
        )
        assert not result.success
        assert "disabled" in result.reply_text.lower()

    @patch("jarvis.tools.builtin.local_control.socket.getaddrinfo")
    def test_requires_explicit_user_approval(self, getaddrinfo):
        getaddrinfo.return_value = [(2, 1, 6, "", ("127.0.0.1", 443))]
        result = LocalControlTool().run(
            {"operation": "open_url", "url": "https://example.test"},
            _context(original_prompt="Open the URL."),
        )
        assert not result.success
        assert "approval" in result.reply_text.lower()

    @patch("jarvis.tools.builtin.local_control.socket.getaddrinfo")
    @patch("jarvis.tools.builtin.local_control.webbrowser.open")
    def test_opens_allowed_url_after_approval(self, open_url, getaddrinfo):
        getaddrinfo.return_value = [(2, 1, 6, "", ("127.0.0.1", 443))]
        result = LocalControlTool().run(
            {"operation": "open_url", "url": "https://example.test"},
            _context(),
        )
        assert result.success
        open_url.assert_called_once_with("https://example.test")
        assert "opened" in result.reply_text.lower()

    def test_rejects_disallowed_url_scheme(self):
        result = LocalControlTool().run(
            {"operation": "open_url", "url": "file:///etc/passwd"},
            _context(),
        )
        assert not result.success
        assert "scheme" in result.reply_text.lower()

    @patch("jarvis.tools.builtin.local_control.subprocess.Popen")
    def test_launches_allowlisted_application_without_shell(self, popen):
        result = LocalControlTool().run(
            {"operation": "open_application", "application": "notepad.exe"},
            _context(),
        )
        assert result.success
        popen.assert_called_once_with(["notepad.exe"], shell=False)

    def test_rejects_application_not_in_allowlist(self):
        result = LocalControlTool().run(
            {"operation": "open_application", "application": "powershell.exe"},
            _context(),
        )
        assert not result.success
        assert "allowlist" in result.reply_text.lower()

    def test_rejects_path_outside_configured_roots(self, tmp_path):
        result = LocalControlTool().run(
            {"operation": "reveal_path", "path": str(tmp_path / "secret.txt")},
            _context(local_control_allowed_roots=["~/Documents"]),
        )
        assert not result.success
        assert "allowed" in result.reply_text.lower()

    @patch("jarvis.tools.builtin.local_control.os.startfile")
    @patch("jarvis.tools.builtin.local_control.sys.platform", "win32")
    def test_opens_existing_path_under_allowed_root(self, startfile, tmp_path):
        target = tmp_path / "notes.txt"
        target.write_text("local")
        result = LocalControlTool().run(
            {"operation": "reveal_path", "path": str(target)},
            _context(local_control_allowed_roots=[str(tmp_path)]),
        )
        assert result.success
        startfile.assert_called_once_with(str(target))

    def test_schema_exposes_only_narrow_operations(self):
        schema = LocalControlTool().inputSchema
        assert set(schema["properties"]["operation"]["enum"]) == {
            "open_application", "open_url", "reveal_path",
            "clipboard_read", "clipboard_write",
            "copy_file", "move_file", "rename_file",
            "list_windows", "focus_window", "minimize_window",
        }

    @patch("jarvis.tools.builtin.local_control._read_clipboard", create=True)
    def test_reads_clipboard_without_approval(self, read_clipboard):
        read_clipboard.return_value = "copied text"
        context = _context(original_prompt="Read the clipboard.")

        result = LocalControlTool().run({"operation": "clipboard_read"}, context)

        assert result.success
        assert result.reply_text == "Clipboard: copied text"
        read_clipboard.assert_called_once_with()

    @patch("jarvis.tools.builtin.local_control._write_clipboard", create=True)
    def test_writes_clipboard_with_exact_approval_summary(self, write_clipboard):
        approval = Mock(return_value=True)
        context = _context(approval_callback=approval)

        result = LocalControlTool().run(
            {"operation": "clipboard_write", "text": "secret note"}, context
        )

        assert result.success
        write_clipboard.assert_called_once_with("secret note")
        request = approval.call_args.args[0]
        assert request == {
            "operation": "clipboard_write",
            "summary": "Write clipboard text: 11 characters",
            "risk": "Replaces text currently available to other local applications.",
            "reason": "The requested clipboard text is ready to be written.",
        }

    def test_file_mutations_reject_paths_outside_allowlisted_root(self, tmp_path):
        outside = tmp_path / "outside.txt"
        inside = tmp_path / "inside.txt"
        result = LocalControlTool().run(
            {"operation": "copy_file", "source": str(outside), "destination": str(inside)},
            _context(local_control_allowed_roots=[str(tmp_path / "allowed")]),
        )
        assert not result.success
        assert "allowed" in result.reply_text.lower()

    @patch("jarvis.tools.builtin.local_control.shutil.copy2", create=True)
    def test_copy_file_is_approval_gated_and_dry_run_summary_is_exact(self, copy2, tmp_path):
        source = tmp_path / "source.txt"
        destination = tmp_path / "destination.txt"
        source.write_text("data")
        approval = Mock(return_value=False)

        result = LocalControlTool().run(
            {"operation": "copy_file", "source": str(source), "destination": str(destination)},
            _context(
                approval_callback=approval,
                local_control_allowed_roots=[str(tmp_path)],
            ),
        )

        assert not result.success
        copy2.assert_not_called()
        request = approval.call_args.args[0]
        assert request["summary"] == f"Copy file: {source} -> {destination}"
        assert request["risk"] == "Copies a local file and may overwrite the destination."
        assert "approval" not in result.reply_text.lower()

    @patch("jarvis.tools.builtin.local_control.shutil.move")
    @pytest.mark.parametrize("operation", ["move_file", "rename_file"])
    def test_move_and_rename_are_approval_gated(self, move, operation, tmp_path):
        source = tmp_path / "source.txt"
        destination = tmp_path / "destination.txt"
        source.write_text("data")
        approval = Mock(return_value=True)

        result = LocalControlTool().run(
            {"operation": operation, "source": str(source), "destination": str(destination)},
            _context(
                approval_callback=approval,
                local_control_allowed_roots=[str(tmp_path)],
            ),
        )

        assert result.success
        move.assert_called_once_with(source, destination)
        assert approval.call_args.args[0]["operation"] == operation
        assert approval.call_args.args[0]["summary"].startswith(
            f"{operation.replace('_', ' ').capitalize().replace(' File', ' file')}:"
        )

    @patch("jarvis.tools.builtin.local_control._list_windows", create=True)
    def test_lists_windows_without_approval(self, list_windows):
        list_windows.return_value = ["Editor - notes", "Terminal"]
        result = LocalControlTool().run(
            {"operation": "list_windows"}, _context(original_prompt="List windows.")
        )
        assert result.success
        assert "Editor - notes" in result.reply_text
        list_windows.assert_called_once_with()

    @patch("jarvis.tools.builtin.local_control._focus_window", create=True)
    def test_focus_window_is_approval_gated(self, focus_window):
        approval = Mock(return_value=True)
        result = LocalControlTool().run(
            {"operation": "focus_window", "title": "Editor"},
            _context(approval_callback=approval),
        )
        assert result.success
        focus_window.assert_called_once_with("Editor")
        request = approval.call_args.args[0]
        assert request["summary"] == "Focus window: Editor"
        assert request["risk"] == "Changes the active window in the local desktop."

    @patch("jarvis.tools.builtin.local_control._minimize_window", create=True)
    def test_minimize_window_is_approval_gated(self, minimize_window):
        approval = Mock(return_value=True)
        result = LocalControlTool().run(
            {"operation": "minimize_window", "title": "Terminal"},
            _context(approval_callback=approval),
        )
        assert result.success
        minimize_window.assert_called_once_with("Terminal")
        assert approval.call_args.args[0]["risk"] == "Changes window state in the local desktop."

    def test_window_actions_are_rejected_off_windows(self, monkeypatch):
        monkeypatch.setattr(
            "jarvis.tools.builtin.local_control.sys.platform", "linux"
        )
        result = LocalControlTool().run(
            {"operation": "minimize_window", "title": "Terminal"}, _context()
        )
        assert not result.success
        assert "windows" in result.reply_text.lower()
