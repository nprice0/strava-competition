# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Strava Segment Competition Tool.

This creates a standalone Windows executable that includes all dependencies.

Run from project root:
    pyinstaller --clean scripts/strava_competition.spec

Or use the build script:
    python scripts/build_windows.py
"""
import os
from pathlib import Path

os.chdir(project_root)
# Change to project root (spec file is in scripts/)
spec_dir = os.path.dirname(os.path.abspath(SPEC))
project_root = str(Path(spec_dir).parent)
os.chdir(project_root)

block_cipher = None

# Hidden imports - packages PyInstaller may not auto-detect
hidden_imports = [
    'pandas',
    'openpyxl',
    'numpy',
    'scipy',
    'shapely',
    'pyproj',
    'geopy',
    'polyline',
    'folium',
    'flask',
    'werkzeug',
    'requests',
    'urllib3',
    'cachetools',
    'defusedxml',
    'dotenv',
    # Numpy and scipy internals
    'numpy.core._methods',
    'numpy.lib.format',
    'scipy.special._ufuncs_cxx',
    'scipy.linalg.cython_blas',
    'scipy.linalg.cython_lapack',
    'scipy.integrate',
    'scipy.integrate.quadrature',
    'scipy.integrate.odepack',
    'scipy.integrate._odepack',
    'scipy.integrate.quadpack',
    'scipy.integrate._quadpack',
    'scipy.integrate._ode',
    'scipy.integrate.vode',
    'scipy.integrate._dop',
    'scipy.integrate.lsoda',
    # Pandas internals
    'pandas._libs.tslibs.timedeltas',
    'pandas._libs.tslibs.np_datetime',
    'pandas._libs.tslibs.nattype',
    'pandas._libs.skiplist',
    # Shapely and pyproj
    'shapely.geometry',
    'shapely._geos',
    'pyproj.crs',
    'pyproj._crs',
    'pyproj.database',
    # Openpyxl
    'openpyxl.cell._writer',
    # Flask/Werkzeug for OAuth flow
    'flask.json',
    'werkzeug.serving',
    'werkzeug.debug',
]

# Modules to exclude (reduce size)
excludes = [
    'tkinter',
    'matplotlib',
    'PIL',
    'cv2',
    'IPython',
    'jupyter',
    'notebook',
    'pytest',
    'mypy',
    'ruff',
    'bandit',
]

# Data files to include (add paths here if needed)
# Format: (source, destination_folder)
datas = [
    # Example: ('assets/icon.png', 'assets'),
]

# Binary files to include
binaries = []

a = Analysis(
    [str(Path(project_root) / 'run.py')],
    pathex=[project_root],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='StravaCompetition',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Keep console for logging output
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add icon path here: icon='assets/icon.ico'
)

# Uncomment below for a directory-based distribution instead of single exe
# coll = COLLECT(
#     exe,
#     a.binaries,
#     a.zipfiles,
#     a.datas,
#     strip=False,
#     upx=True,
#     upx_exclude=[],
#     name='StravaCompetition',
# )
