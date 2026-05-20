"""Tests for manageWorkQueue built-in tool."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from jarvis.operator import work_queue as wq
from jarvis.tools.builtin.manage_work_queue import ManageWorkQueueTool


@pytest.mark.unit
class TestManageWorkQueueTool:
    def test_summary_operation(self, tmp_path, monkeypatch):
        monkeypatch.setattr(wq, "_queue_path", lambda: tmp_path / "work_queue.json")
        wq.create_item(title="Test task")

        tool = ManageWorkQueueTool()
        ctx = MagicMock()
        ctx.cfg.work_queue_enabled = True
        ctx.user_print = lambda _m: None

        result = tool.run({"operation": "summary"}, ctx)
        assert result.success
        assert "total_active" in result.reply_text
