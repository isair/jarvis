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


@pytest.mark.skipif(sys.platform!='win32',reason='Requires the Windows filesystem')
def test_windows_selected_file_stays_readonly_until_its_descriptor_closes(tmp_path):
    from jarvis.utils.windows_files import open_readonly_no_links
    path=tmp_path/'memory.md';path.write_text('Owned original memory.')
    with os.fdopen(open_readonly_no_links(path),'rb') as source:
        with pytest.raises(PermissionError):path.write_text('Competing write.')
        with pytest.raises(PermissionError):path.rename(tmp_path/'moved.md')
        assert source.read()==b'Owned original memory.'
    path.write_text('Owned later version.')
    assert path.read_text()=='Owned later version.'


@pytest.mark.skipif(sys.platform!='win32',reason='Requires the Windows filesystem')
@pytest.mark.parametrize('mutation',['rename','reparse'])
def test_windows_checked_parent_cannot_change_before_leaf_selection(tmp_path,monkeypatch,mutation):
    """Race the native selection boundary using only owned directories."""
    import ctypes
    from ctypes import wintypes
    import struct
    from jarvis.utils import windows_files
    selected=tmp_path/'selected';selected.mkdir()
    alternate=tmp_path/'alternate';alternate.mkdir()
    (alternate/'memory.md').write_text('Unselected memory.')
    path=selected/'memory.md'
    checked=windows_files._open_checked
    attempted=[]

    def mutate(kernel,parent,*,directory):
        handle=checked(kernel,parent,directory=directory)
        if directory and parent==selected:
            try:
                if mutation=='rename':
                    moved=tmp_path/'moved'
                    try:selected.rename(moved)
                    except PermissionError:attempted.append('blocked')
                    else:
                        moved.rename(selected)
                        attempted.append('allowed')
                else:
                    writer=kernel.CreateFileW(str(selected),0x40000000,0x7,None,3,
                                              0x02200000,None)
                    if writer in (None,-1,ctypes.c_void_p(-1).value):
                        error=ctypes.get_last_error()
                        assert error==32, f'Expected a sharing violation, got {error}'
                        attempted.append('blocked')
                    else:
                        ioctl=kernel.DeviceIoControl
                        ioctl.argtypes=[wintypes.HANDLE,wintypes.DWORD,wintypes.LPVOID,
                                        wintypes.DWORD,wintypes.LPVOID,wintypes.DWORD,
                                        ctypes.POINTER(wintypes.DWORD),wintypes.LPVOID]
                        ioctl.restype=wintypes.BOOL
                        substitute=('\\??\\'+str(alternate)).encode('utf-16-le')
                        display=str(alternate).encode('utf-16-le')
                        names=substitute+b'\0\0'+display+b'\0\0'
                        mount=struct.pack('<HHHH',0,len(substitute),len(substitute)+2,len(display))+names
                        data=struct.pack('<IHH',0xa0000003,len(mount),0)+mount
                        returned=wintypes.DWORD()
                        try:
                            buffer=ctypes.create_string_buffer(data)
                            if not ioctl(writer,0x000900a4,buffer,len(data),None,0,
                                         ctypes.byref(returned),None):
                                raise ctypes.WinError()
                            attempted.append('allowed')
                            delete=ctypes.create_string_buffer(struct.pack('<IHH',0xa0000003,0,0))
                            if not ioctl(writer,0x000900ac,delete,8,None,0,
                                         ctypes.byref(returned),None):
                                raise ctypes.WinError()
                        finally:kernel.CloseHandle(writer)
                assert attempted==['blocked'], 'Checked ancestry must resist mutation until leaf selection.'
                path.write_text('Selected memory.')
            except BaseException:
                windows_files._close(kernel,handle)
                raise
        return handle
    monkeypatch.setattr(windows_files,'_open_checked',mutate)
    with os.fdopen(windows_files.open_readonly_no_links(path),'rb') as source:
        assert source.read()==b'Selected memory.'
    assert attempted==['blocked']
    assert (alternate/'memory.md').read_text()=='Unselected memory.'
    # Parent handles are temporary and do not block later source changes.
    selected.rename(tmp_path/'after-read')


@pytest.mark.skipif(sys.platform!='win32',reason='Requires the Windows filesystem')
@pytest.mark.parametrize('access',[0x40000000,0x00010000],ids=['writer','deleter'])
def test_windows_existing_parent_mutator_prevents_source_selection(tmp_path,access):
    import ctypes
    from jarvis.utils import windows_files
    selected=tmp_path/'selected';selected.mkdir()
    path=selected/'memory.md';path.write_text('Owned selected memory.')
    kernel=windows_files._kernel32()
    competing=kernel.CreateFileW(str(selected),access,0x7,None,3,0x02200000,None)
    if competing in (None,-1,ctypes.c_void_p(-1).value):raise ctypes.WinError()
    try:
        with pytest.raises(OSError):
            with os.fdopen(windows_files.open_readonly_no_links(path),'rb'):
                pass
    finally:kernel.CloseHandle(competing)
    assert path.read_text()=='Owned selected memory.'
