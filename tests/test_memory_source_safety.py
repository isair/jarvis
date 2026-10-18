"""Selected memory files reject filesystem indirection on every supported OS."""
import os
from pathlib import Path
import stat
import subprocess
import sys
from types import SimpleNamespace
import pytest
from jarvis.memory import imports

pytestmark=[pytest.mark.unit,pytest.mark.usefixtures('no_network')]


@pytest.mark.parametrize('component',['file','parent'])
def test_reparse_attributes_cannot_be_accepted_as_a_regular_source(tmp_path,monkeypatch,component):
    directory=tmp_path/'selected';directory.mkdir()
    source=directory/'memory.md';source.write_text('Owned synthetic memory.')
    flagged=source if component=='file' else directory
    original=Path.lstat
    def observed(path,*args,**kwargs):
        result=original(path,*args,**kwargs)
        if path==flagged:
            return SimpleNamespace(st_mode=result.st_mode,st_file_attributes=0x400)
        return result
    monkeypatch.setattr(Path,'lstat',observed)
    with pytest.raises(imports.MemoryImportError):
        imports.read_source(imports.MemorySource('Claude','Selected memory','personal',source))


@pytest.mark.skipif(sys.platform!='win32',reason='Requires the Windows filesystem')
def test_windows_directory_junction_is_not_a_memory_source(tmp_path):
    target=tmp_path/'owned-target';target.mkdir()
    (target/'memory.md').write_text('Do not follow the junction.')
    alias=tmp_path/'junction'
    subprocess.run(['cmd','/d','/c','mklink','/J',str(alias),str(target)],check=True,capture_output=True)
    try:
        with pytest.raises(imports.MemoryImportError):
            imports.read_source(imports.MemorySource('Claude','Selected memory','personal',alias/'memory.md'))
    finally:alias.rmdir()


@pytest.mark.skipif(sys.platform!='win32',reason='Requires the Windows filesystem')
def test_windows_parent_swap_after_inventory_cannot_change_selected_source(tmp_path,monkeypatch):
    selected=tmp_path/'selected';selected.mkdir()
    alternate=tmp_path/'alternate';alternate.mkdir()
    (selected/'memory.md').write_text('Selected owned file.')
    (alternate/'memory.md').write_text('Do not follow the substituted parent.')
    original=imports._has_symlink
    def swap(path):
        safe=original(path)
        selected.rename(tmp_path/'kept')
        subprocess.run(['cmd','/d','/c','mklink','/J',str(selected),str(alternate)],check=True,capture_output=True)
        return safe
    monkeypatch.setattr(imports,'_has_symlink',swap)
    try:
        with pytest.raises(imports.MemoryImportError):
            imports.read_source(imports.MemorySource('Claude','Selected memory','personal',selected/'memory.md'))
    finally:
        if selected.exists():selected.rmdir()


def test_replacing_path_during_a_read_cannot_publish_an_old_snapshot(tmp_path,monkeypatch):
    source=tmp_path/'memory.md';source.write_text('Original owned memory.')
    alternate=tmp_path/'replacement.md';alternate.write_text('New owned memory.')
    original=imports.os.fstat
    snapshots=[]
    def observed(fd):
        value=original(fd)
        snapshots.append(value)
        if len(snapshots)==1:
            alternate.replace(source)
        return value
    monkeypatch.setattr(imports.os,'fstat',observed)
    with pytest.raises(imports.MemoryImportError):
        imports.read_source(imports.MemorySource('Claude','Selected memory','personal',source))


def test_regular_selected_utf8_file_is_read_exactly(tmp_path):
    path=tmp_path/'记忆.md'
    text='I live in London.\n\nKısa yanıtları tercih ediyorum.'
    path.write_text(text,encoding='utf-8')
    preview=imports.read_source(imports.MemorySource('Claude','Selected memory','personal',path))
    assert [item.text for item in preview.items]==text.split('\n\n')
    assert path.read_text(encoding='utf-8')==text
