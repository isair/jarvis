"""Collect the Apple Silicon speech backend for a frozen desktop application."""
import platform
from pathlib import Path
import sys


def collect_mlx_whisper():
    """Return hidden imports, data and native libraries for the current build."""
    if sys.platform != 'darwin' or platform.machine() != 'arm64':
        return [], [], []

    from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules
    import mlx

    metallib = next((Path(root) / 'lib' / 'mlx.metallib' for root in mlx.__path__
                     if (Path(root) / 'lib' / 'mlx.metallib').is_file()), None)
    if metallib is None:
        raise RuntimeError('❌ MLX Metal shader library is missing; reinstall the MLX build dependencies')

    # The native MLX namespace is explicit, avoiding recursive native imports.
    hiddenimports = ['mlx', 'mlx.core', 'mlx._reprlib_fix', 'mlx.nn', 'mlx.utils', 'numba']
    # Native extensions can import package-level Python helpers dynamically.
    for root in mlx.__path__:
        hiddenimports += [f'mlx.{path.stem}' for path in Path(root).glob('*.py')
                          if path.stem != '__init__']
    hiddenimports += collect_submodules(
        'mlx_whisper', filter=lambda name: not name.startswith('mlx_whisper.torch_whisper'),
    )
    hiddenimports += collect_submodules('tiktoken') + collect_submodules('tiktoken_ext')
    datas = [(str(metallib), 'mlx/lib')]
    datas += collect_data_files('mlx_whisper') + collect_data_files('tiktoken')
    binaries = collect_dynamic_libs('mlx')
    print('🎤 Collecting Apple Silicon speech recognition and Metal resources', flush=True)
    return list(dict.fromkeys(hiddenimports)), datas, binaries
