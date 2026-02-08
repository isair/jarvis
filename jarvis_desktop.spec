# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Jarvis Desktop App
Builds a standalone executable for Windows, macOS, and Linux
"""

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Get the project root directory
project_root = Path('.').absolute()
src_path = project_root / 'src'

# Create qt.conf for macOS to help Qt find plugins correctly
if sys.platform == 'darwin':
    qt_conf_path = project_root / 'qt.conf'
    qt_conf_path.write_text("""[Paths]
Prefix = .
Plugins = PyQt6/Qt6/plugins
""")
    print(f"Created qt.conf at {qt_conf_path}")

# Collect all necessary data files
# Note: Let PyInstaller's built-in hooks handle sounddevice, ctranslate2, and Qt WebEngine
# Manual collection can conflict with hooks and cause crashes
datas = [
    (str(src_path / 'desktop_app' / 'desktop_assets' / '*.png'), 'desktop_app/desktop_assets'),
]

# Collect Piper TTS data files (espeak-ng-data is required for phonemization)
try:
    import piper
    piper_path = Path(piper.__file__).parent
    # espeak-ng-data contains phoneme data needed for TTS
    espeak_data = piper_path / 'espeak-ng-data'
    if espeak_data.exists():
        datas.append((str(espeak_data), 'piper/espeak-ng-data'))
        print(f"Bundling Piper espeak-ng-data from {espeak_data}")
    # tashkeel contains Arabic diacritization data
    tashkeel_data = piper_path / 'tashkeel'
    if tashkeel_data.exists():
        datas.append((str(tashkeel_data), 'piper/tashkeel'))
        print(f"Bundling Piper tashkeel from {tashkeel_data}")
except ImportError:
    print("Warning: piper not installed, TTS may not work in bundle")

# Add qt.conf for macOS
if sys.platform == 'darwin':
    datas.append((str(project_root / 'qt.conf'), '.'))

# Collect Qt plugins for system tray functionality
try:
    import PyQt6
    qt_path = Path(PyQt6.__file__).parent
    # Add Qt plugins for platform integration (needed for system tray on macOS)
    # Only add directories that actually exist (e.g., 'styles' may not exist on Linux)
    qt_plugin_dirs = [
        ('platforms', 'PyQt6/Qt6/plugins/platforms'),
        ('styles', 'PyQt6/Qt6/plugins/styles'),
    ]
    for plugin_name, dest_path in qt_plugin_dirs:
        plugin_path = qt_path / 'Qt6' / 'plugins' / plugin_name
        if plugin_path.exists():
            datas.append((str(plugin_path), dest_path))
        else:
            print(f"Info: Qt plugin directory '{plugin_name}' not found, skipping")
except Exception as e:
    print(f"Warning: Could not collect Qt plugins: {e}")

# Note: Qt WebEngine resources are handled by PyInstaller's hook-PyQt6.QtWebEngineWidgets.py
# Manual collection can conflict with the hook and cause crashes

# Hidden imports that PyInstaller might miss
hiddenimports = [
    # Jarvis core modules
    'jarvis',
    'jarvis._version',
    'jarvis.daemon',
    'jarvis.config',
    'jarvis.debug',
    'jarvis.llm',
    'jarvis.main',
    # Desktop app modules
    'desktop_app',
    'desktop_app.app',
    'desktop_app.splash_screen',
    'desktop_app.setup_wizard',
    'desktop_app.updater',
    'desktop_app.update_dialog',
    'desktop_app.themes',
    'desktop_app.face_widget',
    'desktop_app.diary_dialog',
    'desktop_app.memory_viewer',
    # Listening modules
    'jarvis.listening',
    'jarvis.listening.echo_detection',
    'jarvis.listening.listener',
    'jarvis.listening.state_manager',
    'jarvis.listening.wake_detection',
    'jarvis.listening.transcript_buffer',
    'jarvis.listening.intent_judge',
    'jarvis.listening.wake_detector',
    # openWakeWord for audio-level wake detection
    'openwakeword',
    'openwakeword.model',
    # Memory modules
    'jarvis.memory',
    'jarvis.memory.conversation',
    'jarvis.memory.db',
    'jarvis.memory.embeddings',
    # Output modules
    'jarvis.output',
    'jarvis.output.tts',
    'jarvis.output.tune_player',
    # Piper TTS (local neural TTS)
    'piper',
    'piper.voice',
    'piper.config',
    'piper.download',
    'piper.download_voices',
    'piper.phonemize_espeak',
    'piper.phoneme_ids',
    # ONNX Runtime (required by Piper for model inference)
    'onnxruntime',
    'onnxruntime.capi',
    'onnxruntime.capi._pybind_state',
    # Profile modules
    'jarvis.profile',
    'jarvis.profile.profiles',
    # Reply modules
    'jarvis.reply',
    'jarvis.reply.engine',
    'jarvis.reply.enrichment',
    # Tools modules
    'jarvis.tools',
    'jarvis.tools.base',
    'jarvis.tools.registry',
    'jarvis.tools.types',
    'jarvis.tools.builtin',
    'jarvis.tools.builtin.fetch_web_page',
    'jarvis.tools.builtin.local_files',
    'jarvis.tools.builtin.nutrition',
    'jarvis.tools.builtin.nutrition.delete_meal',
    'jarvis.tools.builtin.nutrition.fetch_meals',
    'jarvis.tools.builtin.nutrition.log_meal',
    'jarvis.tools.builtin.recall_conversation',
    'jarvis.tools.builtin.refresh_mcp_tools',
    'jarvis.tools.builtin.screenshot',
    'jarvis.tools.builtin.web_search',
    'jarvis.tools.external',
    'jarvis.tools.external.mcp_client',
    # Utils modules
    'jarvis.utils',
    'jarvis.utils.fast_vector_store',
    'jarvis.utils.fuzzy_search',
    'jarvis.utils.location',
    'jarvis.utils.redact',
    'jarvis.utils.vector_store',
    # PyQt6
    'PyQt6.QtCore',
    'PyQt6.QtGui',
    'PyQt6.QtWidgets',
    'PyQt6.sip',
    # PyQt6 WebEngine (for embedded memory viewer)
    'PyQt6.QtWebEngineWidgets',
    'PyQt6.QtWebEngineCore',
    'PyQt6.QtWebChannel',
    # Audio dependencies (critical for voice input)
    'sounddevice',
    '_sounddevice_data',
    '_sounddevice_data.portaudio-binaries',
    'webrtcvad',
    # Speech recognition (faster-whisper backend)
    'faster_whisper',
    'ctranslate2',
    'huggingface_hub',
    'huggingface_hub.file_download',
    'huggingface_hub.hf_api',
    'huggingface_hub.utils',
    'tokenizers',
    # Third-party dependencies
    'dotenv',
    'psutil',
    'requests',
    'numpy',
    'PIL',
    'PIL.Image',
    'rapidfuzz',
    'rapidfuzz.fuzz',
    'bs4',
    'lxml',
    'html2text',
    'faiss',
    'sqlite3',
    'json',
    'asyncio',
    'threading',
    'subprocess',
    'geoip2',
    'geoip2.database',
    'miniupnpc',
    # Flask for memory viewer
    'flask',
    'flask.json',
    'werkzeug',
    'werkzeug.serving',
    'werkzeug.routing',
    'werkzeug.utils',
    'werkzeug.datastructures',
    'werkzeug.wrappers',
    'werkzeug.exceptions',
    'jinja2',
    'markupsafe',
    'itsdangerous',
    'click',
    'blinker',
]

a = Analysis(
    ['src/desktop_app/app.py'],
    pathex=[str(src_path)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['src/desktop_app/pyinstaller_runtime_hook.py'],
    excludes=[
        # Exclude heavy packages to keep bundle size reasonable
        'psycopg2',  # Not used and causes OpenSSL conflicts
        'torch',  # PyTorch is 1.5-2GB - chatterbox TTS is optional
        'torchaudio',
        'torchvision',
        'chatterbox',  # Optional TTS engine (uses PyTorch)
        'transformers',  # Heavy ML library (not needed, faster_whisper uses ctranslate2)
        'safetensors',
        'accelerate',
        'cv2',  # OpenCV - not needed for core functionality
        'opencv-python',
        'matplotlib',  # Not needed for core app
        'notebook',
        'jupyter',
        'IPython',
        'scipy',  # Large, only used by optional features
        'sklearn',
        'scikit-learn',
        # Note: Keep huggingface_hub - needed by faster_whisper for model downloads
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Filter out heavy binaries on all platforms to reduce bundle size
# Note: Be careful not to exclude libs needed by numpy/faster-whisper
excluded_binary_patterns = [
    'torch', 'libtorch', 'libcaffe2',  # PyTorch (~1.5GB)
    'torchaudio', 'torchvision',
    'cv2', 'opencv', 'libopencv',  # OpenCV (~500MB)
    'sklearn', 'scikit',  # scikit-learn
    'transformers',  # Heavy ML library
    'chatterbox',
    'matplotlib',
    # Note: Keep huggingface_hub (needed by faster_whisper for model downloads)
    # Note: Keep libopenblas (needed by numpy) and libfreetype (needed by av/ffmpeg)
]

filtered_binaries = []
for binary in a.binaries:
    name = binary[0].lower()
    binary_path = str(binary[1]).lower() if len(binary) > 1 else ''

    # Check if this binary should be excluded
    should_exclude = False
    for pattern in excluded_binary_patterns:
        if pattern in name or pattern in binary_path:
            print(f"Excluding heavy binary: {binary[0]}")
            should_exclude = True
            break

    if not should_exclude:
        filtered_binaries.append(binary)

a.binaries = filtered_binaries

# Note: VC++ runtime DLL handling on Windows is managed by PyInstaller 6.13.0+
# which has built-in pre-loading of system VC runtime DLLs

# On Windows, filter out C:\Windows\System32\onnxruntime.dll.
# PyInstaller's dependency scanner resolves onnxruntime_pybind11_state.pyd's
# dependency on onnxruntime.dll and may pick up the System32 copy (an older,
# smaller DirectML/Windows ML build).  The correct pip-installed version is
# collected by hook-onnxruntime.py via collect_dynamic_libs("onnxruntime").
# If both end up in the bundle, the System32 copy lands at _MEIPASS root and
# is found first by LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR, causing an API-mismatch
# access violation crash.
if sys.platform == 'win32':
    _pre = len(a.binaries)
    a.binaries = [
        b for b in a.binaries
        if not (
            'onnxruntime' in b[0].lower()
            and len(b) > 1
            and 'system32' in str(b[1]).lower()
        )
    ]
    _removed = _pre - len(a.binaries)
    if _removed:
        print(f"Excluded {_removed} System32 onnxruntime binary(ies) to prevent DLL shadowing")

# On macOS, ensure OpenSSL libraries are bundled properly
if sys.platform == 'darwin':
    # Remove any psycopg2 binaries and OpenCV's bundled OpenSSL (should be excluded already, but be safe)
    filtered_binaries = []
    for binary in a.binaries:
        name = binary[0]
        # Exclude psycopg2 entirely
        if 'psycopg2' in name.lower():
            print(f"Excluding psycopg2: {name}")
            continue
        filtered_binaries.append(binary)

    # Find and bundle OpenSSL libraries from Python's dependencies
    # Python's SSL module needs these, and they should come from Python's installation
    python_executable = sys.executable
    python_lib_dir = Path(python_executable).parent.parent / 'lib'

    # Try to find OpenSSL in Python's lib directory or common locations
    openssl_candidates = [
        # Check Python's lib directory (pyenv, virtualenv, etc.)
        python_lib_dir / 'libssl.3.dylib',
        python_lib_dir / 'libcrypto.3.dylib',
        # Check Homebrew locations (will bundle these into the app)
        Path('/opt/homebrew/opt/openssl@3/lib/libssl.3.dylib'),
        Path('/opt/homebrew/opt/openssl@3/lib/libcrypto.3.dylib'),
        Path('/opt/homebrew/lib/libssl.3.dylib'),
        Path('/opt/homebrew/lib/libcrypto.3.dylib'),
        # Check system locations
        Path('/usr/local/lib/libssl.3.dylib'),
        Path('/usr/local/lib/libcrypto.3.dylib'),
    ]

    openssl_libs = {
        'libssl.3.dylib': None,
        'libcrypto.3.dylib': None,
    }

    # Find existing OpenSSL libraries
    for candidate in openssl_candidates:
        lib_name = candidate.name
        if lib_name in openssl_libs and candidate.exists() and openssl_libs[lib_name] is None:
            openssl_libs[lib_name] = candidate
            print(f"Found OpenSSL library: {candidate}")

    # Remove any existing libssl/libcrypto entries first
    filtered_binaries = [b for b in filtered_binaries
                        if not (b[0] == 'libssl.3.dylib' or b[0] == 'libcrypto.3.dylib')]

    # Add found OpenSSL libraries
    for lib_name, lib_path in openssl_libs.items():
        if lib_path and lib_path.exists():
            print(f"Bundling OpenSSL: {lib_path} as {lib_name}")
            filtered_binaries.append((lib_name, str(lib_path), 'BINARY'))
        else:
            print(f"Warning: OpenSSL library {lib_name} not found - SSL may not work!")

    a.binaries = filtered_binaries

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Platform-specific configurations
if sys.platform == 'darwin':
    # macOS: Create .app bundle
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name='Jarvis',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        console=False,  # No console for production
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon=str(src_path / 'desktop_app' / 'desktop_assets' / 'icon_idle.png'),
    )

    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=False,
        upx=True,
        upx_exclude=[],
        name='Jarvis',
    )

    app = BUNDLE(
        coll,
        name='Jarvis.app',
        icon=str(src_path / 'desktop_app' / 'desktop_assets' / 'icon_idle.png'),
        bundle_identifier='com.jarvis.assistant',
        info_plist={
            'NSHighResolutionCapable': 'True',
            'LSUIElement': '1',  # Hide from dock
            'NSMicrophoneUsageDescription': 'Jarvis needs microphone access to listen for voice commands.',
            'NSScreenCaptureUsageDescription': 'Jarvis needs screen capture access to read text from your screen via OCR.',
        },
    )

    # Post-build: Ensure OpenSSL libraries are correct and remove conflicting ones
    import shutil
    frameworks_dir = Path('dist/Jarvis.app/Contents/Frameworks')

    # Remove OpenCV's bundled OpenSSL libraries (they conflict with Python's SSL)
    # Try both possible directory names
    for dylibs_dir_name in ['__dot__dylibs', '.dylibs']:
        cv2_dylibs_dir = frameworks_dir / 'cv2' / dylibs_dir_name
        if cv2_dylibs_dir.exists():
            for lib_name in ['libssl.3.dylib', 'libcrypto.3.dylib']:
                cv2_lib = cv2_dylibs_dir / lib_name
                if cv2_lib.exists():
                    cv2_lib.unlink()
                    print(f"Removed OpenCV bundled OpenSSL: {cv2_lib}")

    # Also check Resources directory
    resources_dir = Path('dist/Jarvis.app/Contents/Resources')
    cv2_resources_dylibs = resources_dir / 'cv2' / '.dylibs'
    if cv2_resources_dylibs.exists():
        for lib_name in ['libssl.3.dylib', 'libcrypto.3.dylib']:
            cv2_lib = cv2_resources_dylibs / lib_name
            if cv2_lib.exists():
                cv2_lib.unlink()
                print(f"Removed OpenCV bundled OpenSSL from Resources: {cv2_lib}")

    # Find OpenSSL libraries that were bundled (from the binaries we added)
    bundled_openssl = {}
    for binary in a.binaries:
        if binary[0] in ['libssl.3.dylib', 'libcrypto.3.dylib']:
            bundled_openssl[binary[0]] = Path(binary[1])

    # Also check the source paths we used during build
    openssl_source_paths = {
        'libssl.3.dylib': Path('/opt/homebrew/opt/openssl@3/lib/libssl.3.dylib'),
        'libcrypto.3.dylib': Path('/opt/homebrew/opt/openssl@3/lib/libcrypto.3.dylib'),
    }
    # Fallback to homebrew lib if openssl@3 not found
    if not openssl_source_paths['libssl.3.dylib'].exists():
        openssl_source_paths = {
            'libssl.3.dylib': Path('/opt/homebrew/lib/libssl.3.dylib'),
            'libcrypto.3.dylib': Path('/opt/homebrew/lib/libcrypto.3.dylib'),
        }

    # Fix any broken symlinks in Frameworks and ensure correct libraries are in place
    for lib_name in ['libssl.3.dylib', 'libcrypto.3.dylib']:
        lib_path = frameworks_dir / lib_name
        if lib_path.exists():
            if lib_path.is_symlink():
                # Check if symlink is broken
                try:
                    lib_path.resolve(strict=True)
                    # Symlink is valid, skip
                    continue
                except (OSError, RuntimeError):
                    # Broken symlink - remove it
                    lib_path.unlink()
                    print(f"Removed broken symlink: {lib_path}")
            else:
                # File exists and is not a symlink, check if it's valid
                if lib_path.stat().st_size > 0:
                    # File looks valid, skip
                    continue

        # Library doesn't exist or was removed - copy from source
        source_lib = None
        if lib_name in bundled_openssl and bundled_openssl[lib_name].exists():
            source_lib = bundled_openssl[lib_name]
        elif lib_name in openssl_source_paths and openssl_source_paths[lib_name].exists():
            source_lib = openssl_source_paths[lib_name]

        if source_lib and source_lib.exists():
            shutil.copy2(source_lib, lib_path)
            print(f"Fixed OpenSSL library: {source_lib} -> {lib_path}")
        else:
            print(f"Warning: Could not find source for {lib_name}")

elif sys.platform == 'win32':
    # Windows: Create single executable
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.zipfiles,
        a.datas,
        [],
        name='Jarvis',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        upx_exclude=[],
        runtime_tmpdir=None,
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon=str(src_path / 'desktop_app' / 'desktop_assets' / 'icon_idle.ico'),
    )

else:
    # Linux: Create directory-based distribution (more reliable than one-file)
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name='Jarvis',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )

    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=False,
        upx=False,
        upx_exclude=[],
        name='Jarvis',
    )

