"""Native Windows document capabilities, using only owned synthetic files."""
from contextlib import contextmanager
import ctypes
from ctypes import wintypes
import importlib.util
import os
from pathlib import Path
import struct
import sys
import tempfile


@contextmanager
def owned_runtime(root):
    runtime = root / 'runtime'
    runtime.mkdir()
    names = ('TMPDIR', 'TEMP', 'TMP', 'XDG_CONFIG_HOME', 'JARVIS_CONFIG_PATH')
    old = {key: os.environ.get(key) for key in names}
    old_temp = tempfile.tempdir
    for key in names:
        os.environ[key] = str(runtime / 'config.json' if key == 'JARVIS_CONFIG_PATH' else runtime)
    tempfile.tempdir = str(runtime)
    try:
        yield
    finally:
        tempfile.tempdir = old_temp
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def main():
    assert os.name == 'nt', 'This qualification requires native Windows'
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                  wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.DeviceIoControl.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPVOID,
                                      wintypes.DWORD, wintypes.LPVOID, wintypes.DWORD,
                                      ctypes.POINTER(wintypes.DWORD), wintypes.LPVOID]
    kernel.DeviceIoControl.restype = wintypes.BOOL

    def junction(path, target, access):
        substitute = ('\\??\\' + str(target)).encode('utf-16-le')
        display = str(target).encode('utf-16-le')
        names = substitute + b'\0\0' + display + b'\0\0'
        body = struct.pack('<HHHH', 0, len(substitute), len(substitute) + 2, len(display)) + names
        payload = struct.pack('<IHH', 0xA0000003, len(body), 0) + body
        handle = kernel.CreateFileW(str(path), access, 7, None, 3, 0x02200000, None)
        if handle == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            buffer = ctypes.create_string_buffer(payload)
            returned = wintypes.DWORD()
            if not kernel.DeviceIoControl(handle, 0x000900A4, buffer, len(payload),
                                          None, 0, ctypes.byref(returned), None):
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            kernel.CloseHandle(handle)

    with tempfile.TemporaryDirectory(prefix='jarvis-document-native-') as temporary:
        root = Path(temporary).resolve()
        with owned_runtime(root):
            source = Path(__file__).resolve().parents[2] / 'src/jarvis/memory/document_files.py'
            spec = importlib.util.spec_from_file_location('document_files_probe', source)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            outside = root / 'outside'
            outside.mkdir()
            sentinel = b'owned-outside-document-sentinel'
            (outside / 'source.txt').write_bytes(sentinel)
            # Each mutation permission has an independent positive control.
            for access in (0x2, 0x100):
                control = root / f'control-{access}'
                control.mkdir()
                try:
                    junction(control, outside, access)
                    assert (control / 'source.txt').read_bytes() == sentinel
                    print(f'✅ Junction control succeeded, access={access}', flush=True)
                finally:
                    os.rmdir(control)  # Remove the junction itself, never recurse through it.

            ordinary = root / 'ordinary.txt'
            ordinary.write_bytes(b'owned ordinary document')
            with module.open_document_file(ordinary) as stream:
                assert stream.read() == ordinary.read_bytes()
            listing = root / 'listing'
            listing.mkdir()
            names = {'café.txt', '東京.md', '🌱.md', 'child'}
            (listing / 'café.txt').write_text('owned accent', encoding='utf-8')
            (listing / '東京.md').write_text('owned Unicode', encoding='utf-8')
            (listing / '🌱.md').write_bytes(b'owned astral Unicode')
            (listing / 'child').mkdir()
            with module.scan_document_directory(listing) as (_, entries):
                listed = {entry.name: entry.directory for entry in entries}
            assert set(listed) == names and listed['child']
            assert not listed['café.txt'] and not listed['東京.md']
            with module.open_document_file(listing / '東京.md') as stream:
                assert stream.read() == b'owned Unicode'
            with module.open_document_file(listing / '🌱.md') as stream:
                assert stream.read() == b'owned astral Unicode'
            print('✅ Ordinary and Unicode reads/enumeration passed', flush=True)
            original = module._open_entry
            failures = []
            for access in (0x2, 0x100):
                for action in ('read', 'scan'):
                    nested = root / f'allowed-{action}-{access}'
                    nested.mkdir()
                    mutated = False

                    def mutate_after_open(path, parent, *, directory, list_directory=False):
                        nonlocal mutated
                        handle = original(path, parent, directory=directory, list_directory=list_directory)
                        if directory and path == nested:
                            try:
                                try:
                                    os.rename(nested, nested.with_name(nested.name + '-renamed'))
                                except OSError as error:
                                    assert error.winerror in (5, 32), error
                                else:
                                    renamed = nested.with_name(nested.name + '-renamed')
                                    os.rename(renamed, nested)
                                    print('🧪 Attribute-only handle permitted rename', flush=True)
                                junction(nested, outside, access)
                                assert (nested / 'source.txt').read_bytes() == sentinel
                                mutated = True
                                print(f'✅ In-place mutation succeeded, {action}, access={access}', flush=True)
                            except BaseException as error:
                                kernel.CloseHandle(handle)
                                raise RuntimeError('Native mutation/control did not qualify') from error
                        return handle

                    module._open_entry = mutate_after_open
                    leaked = False
                    try:
                        try:
                            if action == 'read':
                                with module.open_document_file(nested / 'source.txt') as stream:
                                    leaked = sentinel in stream.read()
                            else:
                                with module.scan_document_directory(nested) as (_, entries):
                                    leaked = 'source.txt' in [entry.name for entry in entries]
                        except OSError as error:
                            assert mutated, f'Mutation did not qualify: {error}'
                        assert mutated, 'Native mutation was not exercised'
                        if leaked:
                            failures.append((action, access))
                    finally:
                        module._open_entry = original
                        os.rmdir(nested)
            assert not failures, f'Outside document boundary crossed: {failures}'
            assert (outside / 'source.txt').read_bytes() == sentinel
            print('✅ Native document capability boundaries passed', flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
