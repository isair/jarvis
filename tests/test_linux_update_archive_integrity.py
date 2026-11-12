"""Linux update staging rejects unsafe archives before launching installation."""
import io
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import stat
import tarfile

import pytest

from desktop_app import updater

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures('no_network'), pytest.mark.skipif(os.name == 'nt', reason='Linux archive filesystem semantics')]


@pytest.fixture
def package(tmp_path, monkeypatch):
    staging = tmp_path / 'staging'
    staging.mkdir()
    outside = tmp_path / 'outside'
    outside.mkdir()
    marker = outside / 'marker.bin'
    marker.write_bytes(b'original')
    installed = tmp_path / 'installed'
    installed.mkdir()
    (installed / 'Jarvis').write_bytes(b'installed original')
    launched = Mock()
    monkeypatch.setattr(updater, 'get_app_path', lambda: installed)
    monkeypatch.setattr(updater.tempfile, 'mkdtemp', lambda: str(staging))
    monkeypatch.setattr(updater.subprocess, 'Popen', launched)
    return SimpleNamespace(staging=staging, outside=outside, marker=marker,
                           installed=installed, launched=launched, archive=tmp_path / 'update.tar.gz')


def write_archive(package, kind):
    def regular(tf, name, body, mode=0o644):
        member = tarfile.TarInfo(str(name))
        member.size = len(body)
        member.mode = mode
        tf.addfile(member, io.BytesIO(body))

    def link(tf, name, target, hard=False):
        member = tarfile.TarInfo(name)
        member.type = tarfile.LNKTYPE if hard else tarfile.SYMTYPE
        member.linkname = str(target)
        tf.addfile(member)

    with tarfile.open(package.archive, 'w:gz') as tf:
        if kind == 'root_file':
            regular(tf, 'Jarvis', b'not a directory')
            return
        if kind == 'root_symlink':
            regular(tf, 'Payload/Jarvis', b'owned launcher', 0o755)
            link(tf, 'Jarvis', 'Payload')
            return
        if kind == 'missing_root':
            regular(tf, 'readme.txt', b'owned fixture')
            return
        regular(tf, 'Jarvis/Jarvis', b'owned launcher', 0o755)
        if kind in ('parent', 'nested_parent', 'absolute'):
            target = {'parent': '../outside/marker.bin',
                      'nested_parent': 'Jarvis/../../outside/marker.bin',
                      'absolute': str(package.marker)}[kind]
            regular(tf, target, b'changed')
        elif kind in ('symlink', 'absolute_symlink'):
            link(tf, 'Jarvis/link', package.outside if kind.startswith('absolute') else '../../outside')
            regular(tf, 'Jarvis/link/marker.bin', b'changed')
        elif kind in ('hardlink', 'absolute_hardlink'):
            link(tf, 'Jarvis/link.bin', package.marker if kind.startswith('absolute') else '../outside/marker.bin', hard=True)
            regular(tf, 'Jarvis/link.bin', b'changed')
        elif kind == 'fifo':
            member = tarfile.TarInfo('Jarvis/pipe')
            member.type = tarfile.FIFOTYPE
            tf.addfile(member)
        elif kind == 'valid':
            regular(tf, 'Jarvis/assets.bin', b'owned asset')
            link(tf, 'Jarvis/assets-link.bin', 'assets.bin')
            link(tf, 'Jarvis/assets-hard.bin', 'Jarvis/assets.bin', hard=True)


@pytest.mark.parametrize('kind', ['parent', 'nested_parent', 'symlink', 'absolute_symlink',
                                   'hardlink', 'absolute_hardlink', 'fifo', 'root_file',
                                   'root_symlink', 'missing_root'])
def test_unsafe_or_malformed_update_is_not_installed(package, kind):
    write_archive(package, kind)
    result = updater.install_update_linux(package.archive)
    assert package.marker.read_bytes() == b'original', 'Archive changed an outside staging file'
    assert result is False
    package.launched.assert_not_called()
    assert not package.staging.exists(), 'Rejected staging content was retained'
    assert (package.installed / 'Jarvis').read_bytes() == b'installed original'


def test_absolute_member_cannot_overwrite_an_outside_file(package):
    write_archive(package, 'absolute')
    updater.install_update_linux(package.archive)
    assert package.marker.read_bytes() == b'original'


def test_valid_bundle_retains_executable_and_contained_links(package):
    write_archive(package, 'valid')
    assert updater.install_update_linux(package.archive) is True
    root = package.staging / 'Jarvis'
    assert (root / 'Jarvis').read_bytes() == b'owned launcher'
    assert (root / 'Jarvis').stat().st_mode & stat.S_IXUSR
    assert (root / 'assets-link.bin').is_symlink()
    assert (root / 'assets-link.bin').read_bytes() == b'owned asset'
    assert (root / 'assets-hard.bin').samefile(root / 'assets.bin')
    assert (package.staging / 'update.sh').is_file()
    assert package.launched.call_args.args[0] == [str(package.staging / 'update.sh')]
    assert (package.installed / 'Jarvis').read_bytes() == b'installed original'
