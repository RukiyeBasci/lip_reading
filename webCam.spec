# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['webCam.py'],
    pathex=[],
    binaries=[],
    datas=[('statistics\\\\vocab.txt', 'statistics'), ('Cascade_Files/shape_predictor_68_face_landmarks.dat', 'Cascade_Files'), ('VideoProc/vid2frames.py', 'VideoProc'), ('VideoProc/lips_detection.py', 'VideoProc'), ('best_model3.pth', '.')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='webCam',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
