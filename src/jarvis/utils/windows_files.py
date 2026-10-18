"""Open selected Windows files without following reparse ancestors."""
from contextlib import ExitStack
import ctypes
from ctypes import wintypes
from functools import lru_cache
import os
from pathlib import Path

_GENERIC_READ = 0x80000000
_SHARE_READ = 0x1
_SHARE_READ_WRITE = _SHARE_READ | 0x2
_OPEN_EXISTING = 3
_OPEN_REPARSE_POINT = 0x00200000
_BACKUP_SEMANTICS = 0x02000000
_ATTRIBUTE_DIRECTORY = 0x10
_ATTRIBUTE_REPARSE_POINT = 0x400
_FILE_ATTRIBUTE_TAG_INFO = 9
_FILE_TYPE_DISK = 1


class _AttributeTagInfo(ctypes.Structure):
    _fields_ = [('attributes', ctypes.c_uint32), ('reparse_tag', ctypes.c_uint32)]


@lru_cache(maxsize=1)
def _kernel32():
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                  wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD,
                                  wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.GetFileInformationByHandleEx.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                                   wintypes.LPVOID, wintypes.DWORD]
    kernel.GetFileInformationByHandleEx.restype = wintypes.BOOL
    kernel.GetFileType.argtypes = [wintypes.HANDLE]
    kernel.GetFileType.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    return kernel


def _native_error():
    return ctypes.WinError()


def _to_fd(handle):
    import msvcrt
    return msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)


def _close(kernel, handle):
    if not kernel.CloseHandle(handle):
        raise _native_error()


def _open_checked(kernel, path, *, directory):
    access = 0 if directory else _GENERIC_READ
    flags = _OPEN_REPARSE_POINT | (_BACKUP_SEMANTICS if directory else 0)
    sharing = _SHARE_READ_WRITE if directory else _SHARE_READ
    handle = kernel.CreateFileW(str(path), access, sharing, None,
                                _OPEN_EXISTING, flags, None)
    if handle in (None, -1, ctypes.c_void_p(-1).value):
        raise _native_error()
    try:
        info = _AttributeTagInfo()
        if not kernel.GetFileInformationByHandleEx(handle, _FILE_ATTRIBUTE_TAG_INFO,
                                                   ctypes.byref(info), ctypes.sizeof(info)):
            raise _native_error()
        if (info.attributes & _ATTRIBUTE_REPARSE_POINT or
                bool(info.attributes & _ATTRIBUTE_DIRECTORY) != directory or
                kernel.GetFileType(handle) != _FILE_TYPE_DISK):
            raise OSError('The selected path must contain regular files and directories without links.')
    except BaseException:
        _close(kernel, handle)
        raise
    return handle


def open_readonly_no_links(path: Path) -> int:
    """Return an owned descriptor after checking and retaining every ancestor.

    Parents deny delete sharing while later components are opened. Opening each
    component's reparse point itself prevents junction and link substitution.
    The leaf denies write/delete sharing until the caller closes its descriptor.
    All native parent handles close before ownership transfers.
    """
    if not path.is_absolute():
        raise OSError('Choose an absolute file path.')
    kernel = _kernel32()
    leaf = None
    try:
        with ExitStack() as parents:
            for parent in reversed(path.parents):
                handle = _open_checked(kernel, parent, directory=True)
                parents.callback(_close, kernel, handle)
            leaf = _open_checked(kernel, path, directory=False)
        descriptor = _to_fd(leaf)
        leaf = None
        return descriptor
    finally:
        if leaf is not None:
            _close(kernel, leaf)
