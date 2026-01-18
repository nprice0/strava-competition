# Windows Distribution Guide

This guide explains how to build and distribute the Strava Competition Tool for non-technical Windows users.

## Building the Windows Executable

### Prerequisites (Build Machine)

You need a Windows machine (or Windows VM) with:
- Python 3.10 or later
- All project dependencies installed
- PyInstaller

### Option 1: Automated Build (Recommended)

1. Open Command Prompt in the project directory
2. Run the build script:
   ```batch
   build_windows.bat
   ```

Or run the Python build script directly:
```batch
pip install pyinstaller
python build_windows.py
```

### Option 2: Manual Build

```batch
pip install pyinstaller
pyinstaller --clean strava_competition.spec
```

### Build Output

After building, you'll find:
- `dist/StravaCompetition.exe` - The standalone executable
- `dist/StravaCompetition_Windows_YYYYMMDD.zip` - Ready-to-distribute package

## Distribution Package Contents

The zip file includes:
- `StravaCompetition.exe` - Main application
- `Run_Strava_Competition.bat` - Easy launcher with pre-flight checks
- `.env` - Pre-configured Strava API credentials
- `competition_input_SAMPLE.xlsx` - Sample input file (if available)
- `README.txt` - User instructions

## User Setup Instructions

Share these instructions with your users:

### Step 1: Extract the Package
1. Download the zip file
2. Right-click and select "Extract All..."
3. Choose a location (e.g., `C:\StravaCompetition`)

### Step 2: Add Your Input File
1. Copy your `competition_input.xlsx` to the extracted folder
2. The file **must** be named exactly `competition_input.xlsx`
3. Ensure it contains:
   - **Runners** sheet with participant info and refresh tokens
   - **Segments** sheet with segment IDs and groups
   - **Distance Windows** sheet (optional)

### Step 3: Run the Tool
1. Double-click `Run_Strava_Competition.bat`
2. A command window will open showing progress
3. Wait for processing to complete
4. Find results in `competition_results_YYYYMMDD_HHMMSS.xlsx`

That's it! No API setup required - credentials are pre-configured.

## Troubleshooting

### "Windows protected your PC" Warning
This is normal for unsigned executables:
1. Click "More info"
2. Click "Run anyway"

### Check the Logs
Every run creates a log file in the `logs` folder:
- Log files are named `strava_competition_YYYYMMDD_HHMMSS.log`
- They contain detailed information about what happened
- Ask users to send you the log file if they have issues

### Window Closes Immediately
The app now pauses on errors, but if it still closes:
1. Check the `logs` folder for the latest log file
2. Open it in Notepad to see what went wrong

### Antivirus False Positives
PyInstaller executables sometimes trigger antivirus warnings:
- Add an exception for the folder
- Or download from a trusted source (your shared drive, etc.)

### Missing .env File
The .env file with API credentials should be included in the distribution. If it's missing, contact the administrator for a new package.

### API Rate Limits
Strava has API rate limits. If you hit them:
- Wait 15 minutes and try again
- The tool caches responses to minimize API calls

## Building on macOS/Linux for Windows

To cross-compile for Windows from macOS/Linux, you have a few options:

### Option 1: Use a Windows VM
- Install VirtualBox/VMware with Windows
- Build natively on Windows

### Option 2: Use GitHub Actions
Create `.github/workflows/build-windows.yml`:

```yaml
name: Build Windows Executable

on:
  push:
    tags:
      - 'v*'
  workflow_dispatch:

jobs:
  build:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install pyinstaller
          pip install -r requirements.txt
      
      - name: Build executable
        run: python build_windows.py
      
      - name: Upload artifact
        uses: actions/upload-artifact@v4
        with:
          name: StravaCompetition-Windows
          path: dist/*.zip
```

### Option 3: Use Docker with Wine (Advanced)
Not recommended due to complexity, but possible with wine-based PyInstaller.

## Security Considerations

- **Your Strava Client Secret is embedded** in this distribution - only share with trusted users
- The credentials allow access to the Strava API on your behalf
- Consider who you share the distribution with
- If credentials are compromised, regenerate them at https://www.strava.com/settings/api

## Updating the Application

To update:
1. Make code changes
2. Increment version (if tracked)
3. Rebuild with `build_windows.bat`
4. Distribute the new zip file
5. Users extract over their existing installation (keeping their `.env`)
