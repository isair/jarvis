"""Pinned filesystem capabilities for authorised document traversal and reads."""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
import os
from pathlib import Path
import stat
from typing import NamedTuple


class DocumentEntry(NamedTuple):
    name: str
    directory: bool


if os.name == 'nt':
    import ctypes
    from ctypes import wintypes
    import msvcrt

    _kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    _kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    _kernel.CloseHandle.restype = wintypes.BOOL
    _kernel.GetFileInformationByHandleEx.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                                   wintypes.LPVOID, wintypes.DWORD]
    _kernel.GetFileInformationByHandleEx.restype = wintypes.BOOL
    _kernel.GetFileType.argtypes = [wintypes.HANDLE]
    _kernel.GetFileType.restype = wintypes.DWORD
    _kernel.ReOpenFile.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.DWORD, wintypes.DWORD]
    _kernel.ReOpenFile.restype = wintypes.HANDLE

    class _AttributeTag(ctypes.Structure):
        _fields_ = [('attributes', wintypes.DWORD), ('tag', wintypes.DWORD)]

    class _UnicodeString(ctypes.Structure):
        _fields_ = [('length', wintypes.USHORT), ('maximum_length', wintypes.USHORT),
                    ('buffer', wintypes.LPWSTR)]

    class _ObjectAttributes(ctypes.Structure):
        _fields_ = [('length', wintypes.ULONG), ('root', wintypes.HANDLE),
                    ('name', ctypes.POINTER(_UnicodeString)), ('attributes', wintypes.ULONG),
                    ('security', wintypes.LPVOID), ('quality', wintypes.LPVOID)]

    class _IoStatus(ctypes.Structure):
        _fields_ = [('status', ctypes.c_void_p), ('information', ctypes.c_size_t)]

    class _DirectoryInfo(ctypes.Structure):
        _fields_ = [('next', wintypes.DWORD), ('index', wintypes.DWORD),
                    ('creation', ctypes.c_int64), ('access', ctypes.c_int64),
                    ('write', ctypes.c_int64), ('change', ctypes.c_int64),
                    ('size', ctypes.c_int64), ('allocation', ctypes.c_int64),
                    ('attributes', wintypes.DWORD), ('name_bytes', wintypes.DWORD),
                    ('ea_size', wintypes.DWORD), ('short_length', ctypes.c_byte),
                    ('short_name', wintypes.WCHAR * 12), ('file_id', ctypes.c_int64),
                    ('name', wintypes.WCHAR * 1)]

    _nt = ctypes.WinDLL('ntdll')
    _nt.NtCreateFile.argtypes = [ctypes.POINTER(wintypes.HANDLE), wintypes.DWORD,
                                ctypes.POINTER(_ObjectAttributes), ctypes.POINTER(_IoStatus),
                                wintypes.LPVOID, wintypes.ULONG, wintypes.ULONG,
                                wintypes.ULONG, wintypes.ULONG, wintypes.LPVOID, wintypes.ULONG]
    _nt.NtCreateFile.restype = ctypes.c_long
    _nt.RtlNtStatusToDosError.argtypes = [ctypes.c_long]
    _nt.RtlNtStatusToDosError.restype = wintypes.ULONG

    def _windows_handle(name: str, parent, *, directory: bool):
        # RootDirectory binds a single child name to the already pinned object.
        # OBJ_DONT_REPARSE rejects traversal at the kernel boundary.
        encoded = name.encode('utf-16-le')
        if len(encoded) > 65532:
            raise OSError('Document entry name is too long')
        buffer = ctypes.create_unicode_buffer(name, len(encoded) // 2 + 1)
        text = _UnicodeString(len(encoded), len(encoded) + 2,
                              ctypes.cast(buffer, wintypes.LPWSTR))
        attributes = _ObjectAttributes(ctypes.sizeof(_ObjectAttributes), parent,
                                        ctypes.pointer(text), 0x40 | 0x1000, None, None)
        handle = wintypes.HANDLE()
        status_block = _IoStatus()
        access = 0x80 | 0x100000 | (0x20 if directory else 0x1)
        status = _nt.NtCreateFile(ctypes.byref(handle), access, ctypes.byref(attributes),
                                  ctypes.byref(status_block), None, 0, 0x1 | 0x2, 1,
                                  0x200000 | 0x20 | (0x1 if directory else 0x40), None, 0)
        if status < 0:
            raise ctypes.WinError(_nt.RtlNtStatusToDosError(status))
        return handle.value

    def _windows_entries(directory):
        # Reopen the same object with list permission, without resolving a pathname.
        handle = _kernel.ReOpenFile(directory, 0x1 | 0x80, 0x1 | 0x2, 0x02200000)
        if handle == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            # Fixed-size batches retain the index's per-entry cancellation/budget checks.
            buffer = ctypes.create_string_buffer(65536)
            while True:
                if not _kernel.GetFileInformationByHandleEx(handle, 10, buffer, len(buffer)):
                    error = ctypes.get_last_error()
                    if error == 18:  # ERROR_NO_MORE_FILES
                        return
                    raise ctypes.WinError(error)
                offset = 0
                while True:
                    info = _DirectoryInfo.from_buffer(buffer, offset)
                    start = offset + _DirectoryInfo.name.offset
                    end = start + info.name_bytes
                    if (info.name_bytes % 2 or end > len(buffer)
                            or (info.next and (info.next < end - offset
                                               or info.next % 8 or offset + info.next >= len(buffer)))):
                        raise OSError('Invalid document directory information')
                    name = ctypes.string_at(ctypes.addressof(buffer) + start, info.name_bytes).decode('utf-16-le')
                    if name not in ('.', '..'):
                        yield DocumentEntry(name, bool(info.attributes & 0x10)
                                            and not bool(info.attributes & 0x400))
                    if not info.next:
                        break
                    offset += info.next
        finally:
            _kernel.CloseHandle(handle)


def _open_entry(path: Path, parent, *, directory: bool):
    """Open one entry without following a reparse point or symbolic link."""
    if os.name != 'nt':
        flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
        if directory:
            flags |= os.O_DIRECTORY
        return os.open(path.name if parent is not None else path, flags, dir_fd=parent)

    if parent is None:
        name = str(path)
        name = ('\\??\\UNC\\' + name[2:] if name.startswith('\\\\')
                else '\\??\\' + name)
    else:
        name = path.name
        if not name or name in ('.', '..') or any(char in name for char in ('/', '\\', ':')):
            raise ValueError('Document capabilities require a single child name')
    handle = _windows_handle(name, parent, directory=directory)
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
        if os.name == 'nt':
            entries = _windows_entries(directory)
            try:
                yield directory, entries
            finally:
                entries.close()
        else:
            with os.scandir(directory) as entries:
                yield directory, (DocumentEntry(entry.name, entry.is_dir(follow_symlinks=False))
                                  for entry in entries)
