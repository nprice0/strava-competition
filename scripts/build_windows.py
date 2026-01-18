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
import os
import re
import secrets
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile

# Ensure we're working from project root
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
os.chdir(PROJECT_ROOT)


def get_spec_hidden_imports() -> set[str]:
    """Parse hidden_imports from the spec file."""
    spec_file = SCRIPT_DIR / "strava_competition.spec"
    if not spec_file.exists():
        return set()

    content = spec_file.read_text()
    # Find the hidden_imports list
    match = re.search(r"hidden_imports\s*=\s*\[(.*?)\]", content, re.DOTALL)
    if not match:
        return set()

    imports = set()
    for line in match.group(1).split("\n"):
        # Extract quoted strings
        for m in re.findall(r"'([^']+)'", line):
            imports.add(m.split(".")[0])  # Get top-level package
    return imports


def get_requirements_packages() -> set[str]:
    """Parse packages from requirements.txt."""
    req_file = Path("requirements.txt")
    if not req_file.exists():
        return set()

    packages = set()
    for line in req_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            # Extract package name (before any version specifier)
            pkg = re.split(r"[<>=!~\[]", line)[0].strip().lower()
            if pkg:
                packages.add(pkg)
    return packages


def get_code_imports() -> set[str]:
    """Scan all Python files for third-party imports."""
    stdlib = {
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

    imports = set()
    for py_file in Path("strava_competition").rglob("*.py"):
        try:
            tree = ast.parse(py_file.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name.split(".")[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.add(node.module.split(".")[0])
        except SyntaxError:
            pass

    # Filter out stdlib and local imports
    imports -= stdlib
    imports -= {"strava_competition"}
    return imports


def check_imports() -> None:
    """Check if all imported packages are in the spec file."""
    print("\nChecking imports...")

    spec_imports = get_spec_hidden_imports()
    code_imports = get_code_imports()
    requirements = get_requirements_packages()

    # Map common import names to package names
    import_to_package = {
        "cv2": "opencv-python",
        "PIL": "pillow",
        "sklearn": "scikit-learn",
        "yaml": "pyyaml",
        "dotenv": "python-dotenv",
    }

    # Find imports used in code but not in spec
    missing_from_spec = code_imports - spec_imports

    if missing_from_spec:
        print("\n⚠️  Imports found in code but not in spec file hidden_imports:")
        for imp in sorted(missing_from_spec):
            pkg_name = import_to_package.get(imp, imp)
            in_req = (
                "✓ in requirements.txt"
                if pkg_name.lower() in requirements
                else "⚠️ not in requirements.txt"
            )
            print(f"    '{imp}',  # {in_req}")
        print(
            "\n  Add these to hidden_imports in scripts/strava_competition.spec if needed."
        )
        print("  (Some may be auto-detected by PyInstaller)")
    else:
        print("  ✓ All code imports appear to be covered in spec file")

    # Also check requirements vs spec
    req_not_in_spec = (
        requirements
        - spec_imports
        - {"pytest", "ruff", "mypy", "bandit", "types-cachetools", "types-requests"}
    )
    if req_not_in_spec:
        print(
            "\n  Note: These requirements.txt packages aren't explicitly in hidden_imports:"
        )
        for pkg in sorted(req_not_in_spec):
            print(f"    {pkg}")
        print("  (PyInstaller usually auto-detects these)")


def check_prerequisites() -> bool:
    """Verify all prerequisites are met."""
    print("Checking prerequisites...")

    # Check Python version
    if sys.version_info < (3, 10):
        print(f"ERROR: Python 3.10+ required, found {sys.version}")
        return False
    print(f"  ✓ Python {sys.version_info.major}.{sys.version_info.minor}")

    # Check PyInstaller
    try:
        import PyInstaller

        print(f"  ✓ PyInstaller {PyInstaller.__version__}")
    except ImportError:
        print("  ✗ PyInstaller not found. Install with: pip install pyinstaller")
        return False

    # Check key dependencies
    required = ["pandas", "openpyxl", "requests", "flask", "shapely", "pyproj"]
    for pkg in required:
        try:
            __import__(pkg)
            print(f"  ✓ {pkg}")
        except ImportError:
            print(f"  ✗ {pkg} not found. Install dependencies first.")
            return False

    return True


def clean_build_dirs() -> None:
    """Remove previous build artifacts."""
    print("\nCleaning previous builds...")
    for dir_name in ["build", "dist"]:
        dir_path = Path(dir_name)
        if dir_path.exists():
            shutil.rmtree(dir_path)
            print(f"  Removed {dir_name}/")


def run_pyinstaller() -> bool:
    """Run PyInstaller to create the executable."""
    print("\nBuilding executable with PyInstaller...")
    print("  This may take several minutes...\n")

    spec_file = SCRIPT_DIR / "strava_competition.spec"
    if not spec_file.exists():
        print(f"ERROR: {spec_file} not found")
        print("  Generate one with: pyi-makespec --onefile run.py")
        return False

    result = subprocess.run(
        [sys.executable, "-m", "PyInstaller", "--clean", str(spec_file)],
        capture_output=False,
    )

    return result.returncode == 0


def create_minimal_env(source_env: Path) -> str:
    """Create a minimal .env with only essential variables."""
    # Variables to include in distribution
    essential_vars = {
        "STRAVA_CLIENT_ID",
        "STRAVA_CLIENT_SECRET",
        "STRAVA_CACHE_ID_SALT",
        "INPUT_FILE",
        "OUTPUT_FILE",
    }

    lines = ["# Strava Competition Tool - Configuration\n\n"]

    # Read existing values from source
    existing = {}
    for line in source_env.read_text().splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            var_name = stripped.split("=")[0].strip()
            existing[var_name] = line

    # Add credentials section
    lines.append("# Strava API Credentials\n")
    for var in ["STRAVA_CLIENT_ID", "STRAVA_CLIENT_SECRET"]:
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
    """Create the final distribution zip file."""
    print("\nCreating distribution package...")

    dist_dir = Path("dist")
    exe_path = dist_dir / "StravaCompetition.exe"

    if not exe_path.exists():
        print(f"ERROR: Executable not found at {exe_path}")
        return None

    # Create a distribution folder
    timestamp = datetime.now().strftime("%Y%m%d")
    package_name = f"StravaCompetition_Windows_{timestamp}"
    package_dir = dist_dir / package_name
    package_dir.mkdir(exist_ok=True)

    # Copy executable
    shutil.copy(exe_path, package_dir / "StravaCompetition.exe")
    print(f"  Copied executable")

    # Copy sample input file if exists
    sample_input = Path("data/competition_input.xlsx")
    if sample_input.exists():
        shutil.copy(sample_input, package_dir / "competition_input_SAMPLE.xlsx")
        print(f"  Copied sample input file")

    # Create minimal .env with only essential credentials
    env_source = Path(".env")
    if env_source.exists():
        env_content = create_minimal_env(env_source)
        (package_dir / ".env").write_text(env_content)
        print(f"  Created .env with credentials")
    else:
        print(f"  WARNING: .env not found - users will need to create their own")

    # Create README
    readme_content = create_readme_content()
    (package_dir / "README.txt").write_text(readme_content)
    print(f"  Created README.txt")

    # Create run batch file
    batch_content = create_batch_file()
    (package_dir / "Run_Strava_Competition.bat").write_text(batch_content)
    print(f"  Created Run_Strava_Competition.bat")

    # Create zip file
    zip_path = dist_dir / f"{package_name}.zip"
    with ZipFile(zip_path, "w") as zf:
        for file_path in package_dir.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(package_dir)
                zf.write(file_path, arcname)

    print(f"\n✓ Distribution package created: {zip_path}")
    print(f"  Size: {zip_path.stat().st_size / (1024 * 1024):.1f} MB")

    return zip_path


def create_readme_content() -> str:
    """Generate README content for end users."""
    return """
================================================================================
                    STRAVA SEGMENT COMPETITION TOOL
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


def create_batch_file() -> str:
    """Generate a Windows batch file for easy execution."""
    return """@echo off
title Strava Segment Competition Tool
echo.
echo ============================================
echo    STRAVA SEGMENT COMPETITION TOOL
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

echo Starting Strava Competition Tool...
echo Logs will be saved to the 'logs' folder.
echo.

StravaCompetition.exe

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
    """Main build process."""
    print("=" * 60)
    print("  STRAVA COMPETITION TOOL - WINDOWS BUILD")
    print("=" * 60)

    if not check_prerequisites():
        return 1

    clean_build_dirs()

    if not run_pyinstaller():
        print("\n✗ PyInstaller build failed")
        return 1

    package_path = create_distribution_package()
    if not package_path:
        return 1

    print("\n" + "=" * 60)
    print("  BUILD COMPLETE!")
    print("=" * 60)
    print(f"\nDistribute: {package_path}")
    print("\nInstructions for users:")
    print("  1. Extract the zip file")
    print("  2. Copy their competition_input.xlsx to the folder")
    print("  3. Double-click 'Run_Strava_Competition.bat'")

    return 0


if __name__ == "__main__":
    # Check for --check-imports flag
    if "--check-imports" in sys.argv:
        check_imports()
        sys.exit(0)

    sys.exit(main())
