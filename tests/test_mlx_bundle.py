"""Apple Silicon builds retain the speech backend and its required resources."""
import platform
from pathlib import Path
import runpy
import sys
from types import ModuleType
import pytest

pytestmark=pytest.mark.unit
ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def collector(monkeypatch,tmp_path):
    monkeypatch.setattr(sys,'platform','darwin')
    monkeypatch.setattr(platform,'machine',lambda:'arm64')
    mlx=ModuleType('mlx')
    mlx.__path__=[str(tmp_path/'mlx')]
    monkeypatch.setitem(sys.modules,'mlx',mlx)
    lib=tmp_path/'mlx'/'lib'
    lib.mkdir(parents=True)
    (lib/'mlx.metallib').write_bytes(b'metal shader resource')
    hooks=ModuleType('PyInstaller.utils.hooks')
    def submodules(name,filter=lambda name:True):
        # A broad MLX import walk is unsafe for native type registration.
        assert name!='mlx'
        modules=[name,name+'.runtime']
        if name=='mlx_whisper':
            modules+=['mlx_whisper.torch_whisper','mlx_whisper.torch_whisper.extra']
        return [name for name in modules if filter(name)]
    hooks.collect_submodules=submodules
    hooks.collect_data_files=lambda name:[(f'{name}/assets/data',f'{name}/assets')]
    hooks.collect_dynamic_libs=lambda name:[(str(lib/'libmlx.dylib'),'mlx/lib')]
    monkeypatch.setitem(sys.modules,'PyInstaller',ModuleType('PyInstaller'))
    monkeypatch.setitem(sys.modules,'PyInstaller.utils',ModuleType('PyInstaller.utils'))
    monkeypatch.setitem(sys.modules,'PyInstaller.utils.hooks',hooks)
    return lambda:runpy.run_path(str(ROOT/'installer'/'mlx_bundle.py'))['collect_mlx_whisper']()


def test_apple_silicon_retains_native_backend_and_assets(collector):
    imports,data,binaries=collector()
    assert {'mlx_whisper','mlx.core','mlx.nn','mlx.utils','numba'}<=set(imports)
    assert not any(name.startswith('mlx_whisper.torch_whisper') for name in imports)
    assert any(Path(source).name=='mlx.metallib' and dest=='mlx/lib' for source,dest in data)
    assert any(Path(source).name=='libmlx.dylib' for source,dest in binaries)
    assert any(dest=='mlx_whisper/assets' for source,dest in data)


def test_missing_metal_resource_stops_build(collector,tmp_path):
    (tmp_path/'mlx'/'lib'/'mlx.metallib').unlink()
    with pytest.raises(RuntimeError,match='Metal shader'):
        collector()


@pytest.mark.parametrize('system,machine',[('linux','x86_64'),('win32','AMD64'),('darwin','x86_64')])
def test_other_platforms_do_not_collect_apple_backend(collector,monkeypatch,system,machine):
    monkeypatch.setattr(sys,'platform',system)
    monkeypatch.setattr(platform,'machine',lambda:machine)
    assert collector()==([],[],[])


def test_native_extension_python_helpers_are_discovered_without_importing_them(collector,tmp_path):
    root=tmp_path/"mlx"
    helpers=["__array_api_info", "_extension_helper"]
    for name in helpers:
        (root/f"{name}.py").write_text("raise RuntimeError('Build must not import this helper')")
    imports,_,_=collector()
    assert {f"mlx.{name}" for name in helpers}<=set(imports)
