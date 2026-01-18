@echo off
REM Windows build script for Strava Competition Tool
REM Run this from the project root: scripts\build_windows.bat

echo ============================================
echo  Building Strava Competition Tool
echo ============================================
echo.

REM Change to project root (parent of scripts folder)
cd /d "%~dp0.."

REM Ensure spec file exists
if not exist "scripts\strava_competition.spec" (
    echo ERROR: scripts\strava_competition.spec not found.
    echo Please run this script from the repository root and ensure the file exists.
    pause
    exit /b 1
)

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.10 or later.
    pause
    exit /b 1
)

REM Install build dependencies
echo Installing build dependencies...
pip install pyinstaller --quiet
pip install -r requirements.txt --quiet

REM Run the build script from repo root (ensures run.py is found)
echo.
echo Running build process...
python scripts/build_windows.py

echo.
pause
