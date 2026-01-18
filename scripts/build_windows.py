#!/usr/bin/env python3
"""
Build script for creating a Windows distribution of the Strava Competition Tool.

This script:
1. Creates a standalone executable using PyInstaller
2. Packages it with necessary files into a distributable zip

Prerequisites:
- Python 3.10+ installed
- All dependencies from requirements.txt installed
- PyInstaller installed: pip install pyinstaller

Usage (from project root):
    python scripts/build_windows.py
    python scripts/build_windows.py --check-imports  # Check for missing imports in spec

The output will be in the 'dist' folder at project root.
"""

import ast
import importlib
import logging
import os
import re
import secrets
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
SPEC_FILE = SCRIPT_DIR / "strava_competition.spec"
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"
SOURCE_DIR = PROJECT_ROOT / "strava_competition"
DIST_DIR = PROJECT_ROOT / "dist"
BUILD_DIR = PROJECT_ROOT / "build"

MIN_PYTHON_VERSION = (3, 10)
EXECUTABLE_NAME = "StravaCompetition.exe"
APP_NAME = "Strava Competition Tool"

# Packages required for the build (subset of requirements.txt)
REQUIRED_BUILD_PACKAGES = [
    "pandas",
    "openpyxl",
    "requests",
    "flask",
    "shapely",
    "pyproj",
]

# Dev-only packages to exclude when checking requirements coverage
DEV_ONLY_PACKAGES = frozenset(
    {
        "pytest",
        "ruff",
        "mypy",
        "bandit",
        "types-cachetools",
        "types-requests",
    }
)

# Map import names to PyPI package names (where they differ)
IMPORT_TO_PACKAGE_MAP: dict[str, str] = {
    "cv2": "opencv-python",
    "PIL": "pillow",
    "sklearn": "scikit-learn",
    "yaml": "pyyaml",
    "dotenv": "python-dotenv",
}

# Python standard library modules (common subset)
STDLIB_MODULES = frozenset(
    {
        "abc",
        "argparse",
        "ast",
        "asyncio",
        "base64",
        "collections",
        "contextlib",
        "copy",
        "csv",
        "dataclasses",
        "datetime",
        "decimal",
        "enum",
        "functools",
        "glob",
        "hashlib",
        "html",
        "http",
        "io",
        "itertools",
        "json",
        "logging",
        "math",
        "os",
        "pathlib",
        "pickle",
        "queue",
        "re",
        "shutil",
        "socket",
        "sqlite3",
        "ssl",
        "string",
        "subprocess",
        "sys",
        "tempfile",
        "threading",
        "time",
        "typing",
        "unittest",
        "urllib",
        "uuid",
        "warnings",
        "xml",
        "zipfile",
    }
)

# Environment variables to include in distribution
ESSENTIAL_ENV_VARS = ("STRAVA_CLIENT_ID", "STRAVA_CLIENT_SECRET")

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger(__name__)

# Ensure we're working from project root
os.chdir(PROJECT_ROOT)


def get_spec_hidden_imports() -> set[str]:
    """
    Parse hidden_imports from the spec file.

    Returns:
        Set of top-level package names declared in the spec file.
    """
    if not SPEC_FILE.exists():
        return set()

    content = SPEC_FILE.read_text()
    match = re.search(r"hidden_imports\s*=\s*\[(.*?)\]", content, re.DOTALL)
    if not match:
        return set()

    imports: set[str] = set()
    for line in match.group(1).split("\n"):
        for module in re.findall(r"'([^']+)'", line):
            imports.add(module.split(".")[0])  # Get top-level package
    return imports


def get_requirements_packages() -> set[str]:
    """
    Parse package names from requirements.txt.

    Returns:
        Set of lowercase package names from requirements.
    """
    if not REQUIREMENTS_FILE.exists():
        return set()

    packages: set[str] = set()
    for line in REQUIREMENTS_FILE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            # Extract package name (before any version specifier)
            pkg = re.split(r"[<>=!~\[]", line)[0].strip().lower()
            if pkg:
                packages.add(pkg)
    return packages


def get_code_imports() -> set[str]:
    """
    Scan all Python files in the source directory for third-party imports.

    Returns:
        Set of third-party package names imported in the codebase.
    """
    imports: set[str] = set()

    for py_file in SOURCE_DIR.rglob("*.py"):
        try:
            tree = ast.parse(py_file.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name.split(".")[0])
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.add(node.module.split(".")[0])
        except SyntaxError:
            logger.warning("  Skipping %s (syntax error)", py_file)

    # Filter out stdlib and local imports
    imports -= STDLIB_MODULES
    imports -= {"strava_competition"}
    return imports


def check_imports() -> None:
    """
    Check if all imported packages are covered in the spec file.

    Prints warnings for any imports found in code but not in hidden_imports.
    This helps catch missing dependencies before distribution.
    """
    logger.info("\nChecking imports...")

    spec_imports = get_spec_hidden_imports()
    code_imports = get_code_imports()
    requirements = get_requirements_packages()

    # Find imports used in code but not in spec
    missing_from_spec = code_imports - spec_imports

    if missing_from_spec:
        logger.warning(
            "\n⚠️  Imports found in code but not in spec file hidden_imports:"
        )
        for imp in sorted(missing_from_spec):
            pkg_name = IMPORT_TO_PACKAGE_MAP.get(imp, imp)
            status = (
                "✓ in requirements.txt"
                if pkg_name.lower() in requirements
                else "⚠️ not in requirements.txt"
            )
            logger.warning("    '%s',  # %s", imp, status)
        logger.info(
            "\n  Add these to hidden_imports in scripts/strava_competition.spec if needed."
        )
        logger.info("  (Some may be auto-detected by PyInstaller)")
    else:
        logger.info("  ✓ All code imports appear to be covered in spec file")

    # Also check requirements vs spec (excluding dev-only packages)
    req_not_in_spec = requirements - spec_imports - DEV_ONLY_PACKAGES
    if req_not_in_spec:
        logger.info(
            "\n  Note: These requirements.txt packages aren't explicitly in hidden_imports:"
        )
        for pkg in sorted(req_not_in_spec):
            logger.info("    %s", pkg)
        logger.info("  (PyInstaller usually auto-detects these)")


def check_prerequisites() -> bool:
    """
    Verify all prerequisites are met for building.

    Returns:
        True if all prerequisites are satisfied, False otherwise.
    """
    logger.info("Checking prerequisites...")

    # Check Python version
    if sys.version_info < MIN_PYTHON_VERSION:
        logger.error(
            "ERROR: Python %d.%d+ required, found %s",
            MIN_PYTHON_VERSION[0],
            MIN_PYTHON_VERSION[1],
            sys.version,
        )
        return False
    logger.info("  ✓ Python %d.%d", sys.version_info.major, sys.version_info.minor)

    # Check PyInstaller
    try:
        pyinstaller = importlib.import_module("PyInstaller")
        logger.info("  ✓ PyInstaller %s", pyinstaller.__version__)
    except ImportError:
        logger.error("  ✗ PyInstaller not found. Install with: pip install pyinstaller")
        return False

    # Check key dependencies
    for pkg in REQUIRED_BUILD_PACKAGES:
        try:
            importlib.import_module(pkg)
            logger.info("  ✓ %s", pkg)
        except ImportError:
            logger.error("  ✗ %s not found. Install dependencies first.", pkg)
            return False

    return True


def clean_build_dirs() -> None:
    """Remove previous build artifacts from build/ and dist/ directories."""
    logger.info("\nCleaning previous builds...")
    for dir_path in [BUILD_DIR, DIST_DIR]:
        if dir_path.exists():
            shutil.rmtree(dir_path)
            logger.info("  Removed %s/", dir_path.name)


def run_pyinstaller() -> bool:
    """
    Run PyInstaller to create the executable.

    Returns:
        True if build succeeded, False otherwise.
    """
    logger.info("\nBuilding executable with PyInstaller...")
    logger.info("  This may take several minutes...\n")

    if not SPEC_FILE.exists():
        logger.error("ERROR: %s not found", SPEC_FILE)
        logger.error("  Generate one with: pyi-makespec --onefile run.py")
        return False

    result = subprocess.run(
        [sys.executable, "-m", "PyInstaller", "--clean", str(SPEC_FILE)],
        check=False,
    )

    return result.returncode == 0


def create_minimal_env(source_env: Path) -> str:
    """
    Create a minimal .env with only essential variables for distribution.

    Args:
        source_env: Path to the source .env file containing credentials.

    Returns:
        String content for the minimal .env file.
    """
    lines = ["# Strava Competition Tool - Configuration\n\n"]

    # Parse existing values from source
    existing: dict[str, str] = {}
    for line in source_env.read_text().splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            var_name = stripped.split("=", maxsplit=1)[0].strip()
            existing[var_name] = line

    # Add credentials section
    lines.append("# Strava API Credentials\n")
    for var in ESSENTIAL_ENV_VARS:
        if var in existing:
            lines.append(existing[var] + "\n")
    lines.append("\n")

    # Add file paths section with defaults for distribution
    lines.append("# File paths (relative to this folder)\n")
    lines.append("INPUT_FILE=competition_input.xlsx\n")
    lines.append("OUTPUT_FILE=competition_results\n")
    lines.append("\n")

    # Generate a new unique salt for this distribution
    new_salt = f"dist-{secrets.token_hex(16)}"
    lines.append("# Cache identifier salt (auto-generated for this distribution)\n")
    lines.append(f"STRAVA_CACHE_ID_SALT={new_salt}\n")

    return "".join(lines)


def create_distribution_package() -> Path | None:
    """
    Create the final distribution zip file.

    Returns:
        Path to the created zip file, or None if creation failed.
    """
    logger.info("\nCreating distribution package...")

    exe_path = DIST_DIR / EXECUTABLE_NAME

    if not exe_path.exists():
        logger.error("ERROR: Executable not found at %s", exe_path)
        return None

    # Create a distribution folder with timestamp
    timestamp = datetime.now().strftime("%Y%m%d")
    package_name = f"StravaCompetition_Windows_{timestamp}"
    package_dir = DIST_DIR / package_name
    package_dir.mkdir(exist_ok=True)

    # Copy executable
    shutil.copy(exe_path, package_dir / EXECUTABLE_NAME)
    logger.info("  Copied executable")

    # Copy sample input file if exists
    sample_input = PROJECT_ROOT / "data" / "competition_input.xlsx"
    if sample_input.exists():
        shutil.copy(sample_input, package_dir / "competition_input_SAMPLE.xlsx")
        logger.info("  Copied sample input file")

    # Create minimal .env with only essential credentials
    env_source = PROJECT_ROOT / ".env"
    if env_source.exists():
        env_content = create_minimal_env(env_source)
        (package_dir / ".env").write_text(env_content)
        logger.info("  Created .env with credentials")
    else:
        logger.warning(
            "  WARNING: .env not found - users will need to create their own"
        )

    # Create README
    readme_content = _create_readme_content()
    (package_dir / "README.txt").write_text(readme_content)
    logger.info("  Created README.txt")

    # Create run batch file
    batch_content = _create_batch_file()
    (package_dir / "Run_Strava_Competition.bat").write_text(batch_content)
    logger.info("  Created Run_Strava_Competition.bat")

    # Create zip file
    zip_path = DIST_DIR / f"{package_name}.zip"
    with ZipFile(zip_path, "w") as zf:
        for file_path in package_dir.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(package_dir)
                zf.write(file_path, arcname)

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    logger.info("\n✓ Distribution package created: %s", zip_path)
    logger.info("  Size: %.1f MB", size_mb)

    return zip_path


def _create_readme_content() -> str:
    """
    Generate README content for end users.

    Returns:
        Plain-text README content for the distribution.
    """
    return f"""
================================================================================
                    {APP_NAME.upper()}
================================================================================

HOW TO USE
----------
1. Copy your "competition_input.xlsx" file into this folder
2. Double-click "Run_Strava_Competition.bat"
3. Wait for the process to finish
4. Your results will appear as "competition_results_[date].xlsx"

That's it!


USING FILES FROM A DIFFERENT FOLDER (optional)
----------------------------------------------
If you want to keep your input file somewhere else:

1. Open the ".env" file in Notepad
2. Change INPUT_FILE to point to your file, for example:
   INPUT_FILE=C:/Users/YourName/Documents/competition_input.xlsx
3. Save and close

You can also change where results are saved by editing OUTPUT_FILE.

Note: Use forward slashes (/) not backslashes (\\) in paths.


IF SOMETHING GOES WRONG
-----------------------
- Look in the "logs" folder - there's a file with details about what happened
- Send that log file to the competition administrator for help


================================================================================
"""


def _create_batch_file() -> str:
    """
    Generate a Windows batch file for easy execution.

    Returns:
        Batch file content string.
    """
    return f"""@echo off
title {APP_NAME}
echo.
echo ============================================
echo    {APP_NAME.upper()}
echo ============================================
echo.

REM Check if input file exists
if not exist "competition_input.xlsx" (
    echo WARNING: competition_input.xlsx not found!
    echo Please copy your competition input file to this folder.
    echo.
    echo The file should be named: competition_input.xlsx
    echo.
    pause
    exit /b 1
)

echo Starting {APP_NAME}...
echo Logs will be saved to the 'logs' folder.
echo.

{EXECUTABLE_NAME}

echo.
echo ============================================
echo Process complete.
echo.
echo If there were any issues, check the logs folder
echo for detailed information.
echo ============================================
echo Press any key to exit.
pause > nul
"""


def main() -> int:
    """
    Main build process.

    Returns:
        Exit code: 0 for success, 1 for failure.
    """
    logger.info("=" * 60)
    logger.info("  %s - WINDOWS BUILD", APP_NAME.upper())
    logger.info("=" * 60)

    if not check_prerequisites():
        return 1

    clean_build_dirs()

    if not run_pyinstaller():
        logger.error("\n✗ PyInstaller build failed")
        return 1

    package_path = create_distribution_package()
    if not package_path:
        return 1

    logger.info("\n" + "=" * 60)
    logger.info("  BUILD COMPLETE!")
    logger.info("=" * 60)
    logger.info("\nDistribute: %s", package_path)
    logger.info("\nInstructions for users:")
    logger.info("  1. Extract the zip file")
    logger.info("  2. Copy their competition_input.xlsx to the folder")
    logger.info("  3. Double-click 'Run_Strava_Competition.bat'")

    return 0


if __name__ == "__main__":
    if "--check-imports" in sys.argv:
        check_imports()
        sys.exit(0)

    sys.exit(main())
