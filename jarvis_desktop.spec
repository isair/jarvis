# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Jarvis Desktop App
Builds a standalone executable for Windows, macOS, and Linux
"""

import sys
from pathlib import Path

block_cipher = None

# Get the project root directory
project_root = Path('.').absolute()
src_path = project_root / 'src'

# Collect all necessary data files
datas = [
    (str(src_path / 'jarvis' / 'desktop_assets' / '*.png'), 'jarvis/desktop_assets'),
]

# Hidden imports that PyInstaller might miss
hiddenimports = [
    'jarvis',
    'jarvis.daemon',
    'jarvis.config',
    'jarvis.debug',
    'jarvis.listening',
    'jarvis.listening.streaming_transcriber',
    'jarvis.listening.streaming_voice_listener',
    'jarvis.listening.wake_detection',
    'jarvis.listening.state_manager',
    'jarvis.memory',
    'jarvis.memory.conversation',
    'jarvis.memory.db',
    'jarvis.memory.embeddings',
    'jarvis.output',
    'jarvis.output.tts',
    'jarvis.output.tune_player',
    'jarvis.profile',
    'jarvis.profile.profiles',
    'jarvis.reply',
    'jarvis.reply.engine',
    'jarvis.reply.enrichment',
    'jarvis.tools',
    'jarvis.tools.base',
    'jarvis.tools.registry',
    'jarvis.tools.types',
    'jarvis.tools.builtin',
    'jarvis.tools.builtin.fetch_web_page',
    'jarvis.tools.builtin.local_files',
    'jarvis.tools.builtin.nutrition',
    'jarvis.tools.builtin.recall_conversation',
    'jarvis.tools.builtin.screenshot',
    'jarvis.tools.builtin.web_search',
    'jarvis.tools.external',
    'jarvis.tools.external.mcp_client',
    'jarvis.utils',
    'jarvis.utils.fast_vector_store',
    'jarvis.utils.fuzzy_search',
    'jarvis.utils.location',
    'jarvis.utils.redact',
    'jarvis.utils.vector_store',
    'PyQt6.QtCore',
    'PyQt6.QtGui',
    'PyQt6.QtWidgets',
    'psutil',
]

a = Analysis(
    ['src/jarvis/desktop_app.py'],
    pathex=[str(src_path)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

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
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon=str(src_path / 'jarvis' / 'desktop_assets' / 'icon_idle.png'),
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
        icon=str(src_path / 'jarvis' / 'desktop_assets' / 'icon_idle.png'),
        bundle_identifier='com.jarvis.assistant',
        info_plist={
            'NSHighResolutionCapable': 'True',
            'LSUIElement': '1',  # Hide from dock
        },
    )

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
        icon=str(src_path / 'jarvis' / 'desktop_assets' / 'icon_idle.png'),
    )

else:
    # Linux: Create single executable
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
    )

