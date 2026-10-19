"""Owned native-handle doubles enforce reparse rejection and transfer ownership."""
from pathlib import Path
import ctypes
import os
import stat
from types import SimpleNamespace
import pytest

pytestmark=[pytest.mark.unit,pytest.mark.usefixtures('no_network'),
            pytest.mark.skipif(os.name!='posix',reason='Uses owned POSIX descriptors to model Windows handles')]


@pytest.fixture
def native(tmp_path,monkeypatch):
    from jarvis.utils import windows_files
    class Kernel:
        def __init__(self):
            self.handles={};self.reparse=set();self.fail=None
        def CreateFileW(self,name,access,sharing,security,disposition,flags,template):
            path=Path(name)
            if path==self.fail:return -1
            fd=os.open(path,os.O_RDONLY|getattr(os,'O_NONBLOCK',0))
            self.handles[fd]=(path,sharing)
            return fd
        def GetFileType(self,handle):return 1
        def GetFileInformationByHandleEx(self,handle,info_class,buffer,size):
            path,_=self.handles[handle]
            info=ctypes.cast(buffer,ctypes.POINTER(windows_files._AttributeTagInfo)).contents
            info.attributes=0x10 if stat.S_ISDIR(os.fstat(handle).st_mode) else 0
            if path in self.reparse:info.attributes|=0x400
            return 1
        def CloseHandle(self,handle):
            self.handles.pop(handle);os.close(handle);return 1
    kernel=Kernel()
    monkeypatch.setattr(windows_files,'_kernel32',lambda:kernel)
    monkeypatch.setattr(windows_files,'_native_error',lambda:OSError('Owned native open failed'))
    def transfer(handle):
        kernel.handles.pop(handle)
        return handle
    monkeypatch.setattr(windows_files,'_to_fd',transfer)
    yield windows_files,kernel
    assert not kernel.handles,'Every owned native handle must be released or transferred.'


def test_regular_source_transfers_only_the_leaf_descriptor(tmp_path,native):
    module,kernel=native
    parent=tmp_path/'selected';parent.mkdir();path=parent/'memory.md';path.write_text('Owned memory.')
    with os.fdopen(module.open_readonly_no_links(path),'rb') as source:
        assert source.read()==b'Owned memory.'
    assert not kernel.handles


@pytest.mark.parametrize('component',['file','parent'])
def test_reparse_handle_cannot_be_followed_after_the_initial_check(tmp_path,native,component):
    module,kernel=native
    parent=tmp_path/'selected';parent.mkdir();path=parent/'memory.md';path.write_text('Do not import.')
    kernel.reparse.add(path if component=='file' else parent)
    with pytest.raises(OSError):module.open_readonly_no_links(path)
    assert path.read_text()=='Do not import.'


def test_native_open_failure_releases_already_retained_parents(tmp_path,native):
    module,kernel=native
    parent=tmp_path/'selected';parent.mkdir();path=parent/'memory.md';path.write_text('Owned memory.')
    kernel.fail=path
    with pytest.raises(OSError,match='Owned native open failed'):module.open_readonly_no_links(path)


def test_descriptor_transfer_failure_also_closes_the_leaf(tmp_path,native,monkeypatch):
    module,kernel=native
    path=tmp_path/'memory.md';path.write_text('Owned memory.')
    def reject(handle):raise OSError('Owned descriptor transfer failed')
    monkeypatch.setattr(module,'_to_fd',reject)
    with pytest.raises(OSError,match='transfer failed'):module.open_readonly_no_links(path)


def test_native_metadata_failure_cannot_leak_the_opened_source(tmp_path,native,monkeypatch):
    module,kernel=native
    path=tmp_path/'memory.md';path.write_text('Owned memory.')
    original=kernel.GetFileInformationByHandleEx
    def unavailable(handle,*args):
        return 0 if kernel.handles[handle][0]==path else original(handle,*args)
    monkeypatch.setattr(kernel,'GetFileInformationByHandleEx',unavailable)
    with pytest.raises(OSError,match='Owned native open failed'):
        module.open_readonly_no_links(path)


def test_non_disk_handle_cannot_supply_memory_text(tmp_path,native,monkeypatch):
    module,kernel=native
    path=tmp_path/'memory.md';path.write_text('Owned memory.')
    monkeypatch.setattr(kernel,'GetFileType',lambda handle:2 if kernel.handles[handle][0]==path else 1)
    with pytest.raises(OSError):module.open_readonly_no_links(path)
