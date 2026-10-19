"""Pinned filesystem capabilities for authorised document traversal and reads."""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
import os
from pathlib import Path
import stat


if os.name == 'nt':
    import ctypes
    from ctypes import wintypes
    import msvcrt

    _kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    _kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                   wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD,
                                   wintypes.HANDLE]
    _kernel.CreateFileW.restype = wintypes.HANDLE
    _kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    _kernel.CloseHandle.restype = wintypes.BOOL
    _kernel.GetFileInformationByHandleEx.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                                   wintypes.LPVOID, wintypes.DWORD]
    _kernel.GetFileInformationByHandleEx.restype = wintypes.BOOL
    _kernel.GetFileType.argtypes = [wintypes.HANDLE]
    _kernel.GetFileType.restype = wintypes.DWORD

    class _AttributeTag(ctypes.Structure):
        _fields_ = [('attributes', wintypes.DWORD), ('tag', wintypes.DWORD)]


def _open_entry(path: Path, parent, *, directory: bool):
    """Open one entry without following a reparse point or symbolic link."""
    if os.name != 'nt':
        flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
        if directory:
            flags |= os.O_DIRECTORY
        return os.open(path.name if parent is not None else path, flags, dir_fd=parent)

    # Ancestor handles deny deletion while this absolute name is used. Both
    # directories and files open the reparse point itself, which is rejected.
    name = str(path)
    if not name.startswith('\\\\?\\'):
        name = ('\\\\?\\UNC\\' + name[2:] if name.startswith('\\\\')
                else '\\\\?\\' + name)
    handle = _kernel.CreateFileW(name, 0x80 if directory else 0x80000000,
                                 0x1 | 0x2, None, 3, 0x02000000 | 0x00200000, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        attributes = _AttributeTag()
        if not _kernel.GetFileInformationByHandleEx(handle, 9, ctypes.byref(attributes),
                                                    ctypes.sizeof(attributes)):
            raise ctypes.WinError(ctypes.get_last_error())
        if attributes.attributes & 0x400 or bool(attributes.attributes & 0x10) != directory:
            raise OSError('Document entry is a reparse point or has the wrong type')
        if _kernel.GetFileType(handle) != 1:
            raise OSError('Document entry is not a disk file')
    except BaseException:
        _kernel.CloseHandle(handle)
        raise
    if directory:
        return handle
    try:
        return msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except BaseException:
        _kernel.CloseHandle(handle)
        raise


@contextmanager
def _directory(path: Path, parent=None):
    """Pin a child of an owned parent, or walk canonical ancestors from the anchor."""
    if not path.is_absolute():
        raise ValueError('Document capabilities require absolute paths')
    with ExitStack() as stack:
        parts = path.parts[1:] if parent is None else ()
        current = Path(path.anchor) if parent is None else path
        parent = _open_entry(current, parent, directory=True)
        close = _kernel.CloseHandle if os.name == 'nt' else os.close
        stack.callback(close, parent)
        for part in parts:
            current = current / part
            parent = _open_entry(current, parent, directory=True)
            stack.callback(close, parent)
        yield parent


@contextmanager
def open_document_file(path: Path):
    """Read a regular file through pinned ancestors and a non-following leaf."""
    with _directory(path.parent) as parent:
        fd = _open_entry(path, parent, directory=False)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise OSError('Document entry is not a regular file')
            stream = os.fdopen(fd, 'rb')
        except BaseException:
            os.close(fd)
            raise
        with stream:
            yield stream


@contextmanager
def scan_document_directory(path: Path, parent=None):
    """Enumerate a pinned directory, never a substituted ancestor target."""
    with _directory(path, parent) as directory:
        with os.scandir(path if os.name == 'nt' else directory) as entries:
            yield directory, entries
