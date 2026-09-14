from types import SimpleNamespace
from unittest.mock import Mock, patch

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
            {"operation": "open_url", "url": "http://127.0.0.1:8765"},
            _context(local_control_enabled=False),
        )
        assert not result.success
        assert "disabled" in result.reply_text.lower()

    def test_requires_explicit_user_approval(self):
        result = LocalControlTool().run(
            {"operation": "open_url", "url": "http://127.0.0.1:8765"},
            _context(original_prompt="Open the URL."),
        )
        assert not result.success
        assert "approval" in result.reply_text.lower()

    @patch("jarvis.tools.builtin.local_control.socket.getaddrinfo")
    @patch("jarvis.tools.builtin.local_control.webbrowser.open")
    def test_opens_allowed_url_after_approval(self, open_url, getaddrinfo):
        getaddrinfo.return_value = [(2, 1, 6, "", ("127.0.0.1", 8765))]
        result = LocalControlTool().run(
            {"operation": "open_url", "url": "http://127.0.0.1:8765"},
            _context(),
        )
        assert result.success
        open_url.assert_called_once_with("http://127.0.0.1:8765")
        assert "opened" in result.reply_text.lower()

    @patch("jarvis.tools.builtin.local_control.socket.getaddrinfo")
    @patch("jarvis.tools.builtin.local_control.webbrowser.open")
    def test_reports_dns_failure_without_opening_browser(self, open_url, getaddrinfo):
        import socket

        getaddrinfo.side_effect = socket.gaierror("name or service not known")
        result = LocalControlTool().run(
            {"operation": "open_url", "url": "https://offline.invalid"},
            _context(),
        )
        assert not result.success
        assert "dns" in result.reply_text.lower()
        open_url.assert_not_called()

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
            "open_application",
            "open_url",
            "reveal_path",
        }

    def test_desktop_approval_broker_receives_exact_action_summary_before_launch(self):
        approval = Mock(return_value=True)
        context = _context(approval_callback=approval)

        with patch("jarvis.tools.builtin.local_control.subprocess.Popen") as popen:
            result = LocalControlTool().run(
                {"operation": "open_application", "application": "notepad.exe"},
                context,
            )

        assert result.success
        approval.assert_called_once()
        request = approval.call_args.args[0]
        assert request["operation"] == "open_application"
        assert request["summary"] == "Open application: notepad.exe"
        assert request["risk"] == "Launches a local application."
        assert "allowlisted" in request["reason"].lower()
        popen.assert_called_once_with(["notepad.exe"], shell=False)

    def test_rejected_desktop_approval_never_launches(self):
        approval = Mock(return_value=False)
        context = _context(approval_callback=approval)

        with patch("jarvis.tools.builtin.local_control.subprocess.Popen") as popen:
            result = LocalControlTool().run(
                {"operation": "open_application", "application": "notepad.exe"},
                context,
            )

        assert not result.success
        assert "rejected" in result.reply_text.lower()
        popen.assert_not_called()
