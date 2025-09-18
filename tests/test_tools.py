import types
import pytest

from jarvis.tools.registry import run_tool_with_retries, ToolExecutionResult


class DummyCfg:
    def __init__(self):
        self.voice_debug = False
        self.ollama_base_url = "http://localhost"
        self.ollama_chat_model = "test"
        self.llm_chat_timeout_sec = 5.0
        self.location_enabled = False
        self.location_ip_address = None
        self.location_auto_detect = False
        self.use_stdin = True
        self.web_search_enabled = False
        self.mcps = {}


class DummyDB:
    def get_meals_between(self, since, until):
        return []

    def delete_meal(self, mid: int) -> bool:
        return mid == 1


@pytest.mark.unit
def test_delete_meal_success(monkeypatch):
    db = DummyDB()
    cfg = DummyCfg()
    res = run_tool_with_retries(
        db=db,
        cfg=cfg,
        tool_name="deleteMeal",
        tool_args={"id": 1},
        system_prompt="",
        original_prompt="",
        redacted_text="",
        max_retries=0,
    )
    assert isinstance(res, ToolExecutionResult)
    assert res.success is True
    assert "deleted" in (res.reply_text or "").lower()


@pytest.mark.unit
def test_delete_meal_failure(monkeypatch):
    db = DummyDB()
    cfg = DummyCfg()
    res = run_tool_with_retries(
        db=db,
        cfg=cfg,
        tool_name="deleteMeal",
        tool_args={"id": 2},
        system_prompt="",
        original_prompt="",
        redacted_text="",
        max_retries=0,
    )
    assert res.success is False


@pytest.mark.unit
def test_local_files_list_and_read(tmp_path):
    # Arrange
    root = tmp_path / "notes"
    root.mkdir()
    f1 = root / "a.txt"
    f2 = root / "b.md"
    f1.write_text("hello", encoding="utf-8")
    f2.write_text("world", encoding="utf-8")

    db = DummyDB()
    cfg = DummyCfg()

    # Monkeypatch expanduser to point to tmp home
    import jarvis.tools.registry as tools_mod
    import builtins
    from pathlib import Path as _P

    orig_expanduser = tools_mod.os.path.expanduser
    tools_mod.os.path.expanduser = lambda p: str(tmp_path) if p == "~" or p.startswith("~") else orig_expanduser(p)

    try:
        # list
        res_list = run_tool_with_retries(
            db=db,
            cfg=cfg,
            tool_name="localFiles",
            tool_args={"operation": "list", "path": "~/notes", "glob": "*.txt", "recursive": False},
            system_prompt="",
            original_prompt="",
            redacted_text="",
            max_retries=0,
        )
        assert res_list.success is True
        assert "a.txt" in (res_list.reply_text or "")

        # read
        res_read = run_tool_with_retries(
            db=db,
            cfg=cfg,
            tool_name="localFiles",
            tool_args={"operation": "read", "path": "~/notes/a.txt"},
            system_prompt="",
            original_prompt="",
            redacted_text="",
            max_retries=0,
        )
        assert res_read.success is True
        assert (res_read.reply_text or "").strip() == "hello"
    finally:
        tools_mod.os.path.expanduser = orig_expanduser


@pytest.mark.unit
def test_local_files_write_append_delete(tmp_path):
    db = DummyDB()
    cfg = DummyCfg()
    import jarvis.tools.registry as tools_mod

    orig_expanduser = tools_mod.os.path.expanduser
    tools_mod.os.path.expanduser = lambda p: str(tmp_path) if p == "~" or p.startswith("~") else orig_expanduser(p)
    try:
        # write
        res_write = run_tool_with_retries(
            db=db,
            cfg=cfg,
            tool_name="localFiles",
            tool_args={"operation": "write", "path": "~/x/y.txt", "content": "abc"},
            system_prompt="",
            original_prompt="",
            redacted_text="",
            max_retries=0,
        )
        assert res_write.success is True

        # append
        res_append = run_tool_with_retries(
            db=db,
            cfg=cfg,
            tool_name="localFiles",
            tool_args={"operation": "append", "path": "~/x/y.txt", "content": "def"},
            system_prompt="",
            original_prompt="",
            redacted_text="",
            max_retries=0,
        )
        assert res_append.success is True

        # read back
        res_read = run_tool_with_retries(
            db=db,
            cfg=cfg,
            tool_name="localFiles",
            tool_args={"operation": "read", "path": "~/x/y.txt"},
            system_prompt="",
            original_prompt="",
            redacted_text="",
            max_retries=0,
        )
        assert res_read.success is True
        assert (res_read.reply_text or "").strip() == "abcdef"

        # delete
        res_del = run_tool_with_retries(
            db=db,
            cfg=cfg,
            tool_name="localFiles",
            tool_args={"operation": "delete", "path": "~/x/y.txt"},
            system_prompt="",
            original_prompt="",
            redacted_text="",
            max_retries=0,
        )
        assert res_del.success is True
    finally:
        tools_mod.os.path.expanduser = orig_expanduser