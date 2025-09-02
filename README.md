# Strava Segment Competition Tool

A small Python app that reads competition inputs from an Excel file, fetches Strava segment efforts for listed runners, and writes per-segment results to an Excel report. It also refreshes and persists Strava refresh tokens back to your input workbook.

## Features
- Read input workbook with:
  - Segments (name, ID, date window)
  - Runners (name, Strava athlete ID, refresh token, team)
- Fetch segment efforts from Strava with pagination, retry, and light rate-limit handling
- Compute attempts, fastest time, and fastest date per runner
- Output results to an Excel file (timestamp optional)
- Update runner refresh tokens back into your input workbook
- OAuth helper (`oauth.py`) to obtain initial refresh tokens securely

---

## Requirements
- Python 3.10+ (tested with 3.13)
- A Strava API application (Client ID/Secret)
- Access to segment efforts via the Strava API requires an active Strava subscription for each athlete whose efforts are requested (Strava restricts this endpoint to subscribed athletes)

Python packages:
- pandas
- openpyxl
- requests
- Flask (for oauth.py)
- Werkzeug (used by oauth.py for clean shutdown)
- urllib3 (used for retry/backoff via `Retry`)
- python-dotenv (to load `.env` automatically)

### Install on macOS (zsh)
```bash
# Create and activate a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies (from pinned list)
pip install -r requirements.txt
```

If your editor shows missing imports, ensure your virtual environment is selected.

### Install on Windows (PowerShell)
```powershell
# Create and activate a virtual environment (recommended)
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies (from pinned list)
pip install -r requirements.txt
```

Notes:
- If activation is blocked, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`
- On Command Prompt (cmd.exe), activate with: `.\.venv\Scripts\activate.bat`

---

## Project structure (key files)
- `strava_competition/` — Python package
  - `__main__.py` — allows running via `python -m strava_competition`
  - `main.py` — entry point; coordinates reading inputs, processing, writing results, and persisting tokens
  - `config.py` — configuration constants (paths, timestamp toggle, Strava API settings)
    - Performance knobs: concurrency, HTTP pool sizes, timeouts, and rate-limiter thresholds
  - `excel_io.py` — Excel read/write utilities
  - `processor.py` — transforms Strava efforts into reportable results
  - `strava_api.py` — Strava API client (pagination, retries, token caching per runner)
  - `auth.py` — token refresh flow (refresh_token -> access_token + possibly new refresh_token)
  - `oauth.py` — local OAuth helper to obtain a refresh token via browser
  - `models.py` — simple data models

---

## Configure the app
Edit `strava_competition/config.py` with your paths and set Strava credentials via environment variables:

```python
# strava_competition/config.py
INPUT_FILE = '/absolute/path/to/competition_input.xlsx'
OUTPUT_FILE = '/absolute/path/for/output/competition_results'
OUTPUT_FILE_TIMESTAMP_ENABLED = True  # if True, appends _YYYYMMDD_HHMMSS

import os
CLIENT_ID = os.getenv('STRAVA_CLIENT_ID', '')
CLIENT_SECRET = os.getenv('STRAVA_CLIENT_SECRET', '')
STRAVA_BASE_URL = 'https://www.strava.com/api/v3'
STRAVA_OAUTH_URL = 'https://www.strava.com/oauth/token'

# Performance tuning
MAX_WORKERS = 8  # threads per segment when fetching runner efforts
HTTP_POOL_CONNECTIONS = 20
HTTP_POOL_MAXSIZE = 20
REQUEST_TIMEOUT = 15  # seconds
RATE_LIMIT_MAX_CONCURRENT = 8
RATE_LIMIT_JITTER_RANGE = (0.05, 0.2)
RATE_LIMIT_NEAR_LIMIT_BUFFER = 3
RATE_LIMIT_THROTTLE_SECONDS = 15
```

Notes:
- `INPUT_FILE` and `OUTPUT_FILE` can point outside this repo. `OUTPUT_FILE` is a base; `.xlsx` is added automatically (with a timestamp suffix if enabled).
- Keep your Client Secret safe. Don’t commit real secrets. Store credentials in a local `.env` file at the project root (recommended; not committed).

Set credentials via .env (macOS and Windows):
```dotenv
STRAVA_CLIENT_ID=<your_id>
STRAVA_CLIENT_SECRET=<your_secret>
```
Notes:
- Create a file named `.env` in the project root. It’s auto-loaded by python-dotenv.
- Don’t commit `.env`. It’s local to your machine and per-project.
- No shell restart is needed; activate your venv and run the app/tests.

### Input workbook format
- Sheet: `Segments`
  - Columns: `Segment ID` (int), `Segment Name` (str), `Start Date` (date), `End Date` (date)
- Sheet: `Runners`
  - Columns: `Name` (str), `Strava ID` (int), `Refresh Token` (str), `Team` (str)

Dates can be Excel dates or ISO-like strings; they are parsed with pandas.

---

## Getting refresh tokens (first-time setup)
Use `strava_competition/oauth.py` (run as a module) to obtain refresh tokens for runners. This script starts a local web server and opens your browser to authorize.

1. Ensure `CLIENT_ID` and `CLIENT_SECRET` are set in `config.py`.
2. In `oauth.py`, you can customize:
   - `OAUTH_PORT` (default 5000)
   - `PRINT_TOKENS` (True to print full tokens after exchange)
3. Make sure your Strava app’s callback URL matches: `http://localhost:<OAUTH_PORT>/callback`
4. Run the script:
   ```bash
  python -m strava_competition.oauth
   ```
5. After authorizing, the script prints the `access_token`, `refresh_token`, and `expires_at` (if `PRINT_TOKENS=True`). Use the `refresh_token` in your input workbook under `Runners`.

Security notes:
- The OAuth flow uses a `state` parameter for CSRF protection.
- The local server shuts down cleanly after token exchange.

Troubleshooting:
- Port in use: change `OAUTH_PORT` or stop the conflicting process. On macOS, AirPlay Receiver can use port 5000.
- Missing Flask/requests: install the packages as shown above.

---

## Running the app
From the project root (with your venv active):

```bash
# easiest
python run.py

# or via the package entry point
python -m strava_competition
```

What it does:
- Reads segments and runners from `INPUT_FILE`
- Fetches segment efforts for each runner and time window (with pagination and retries)
- Writes a results workbook to `<OUTPUT_FILE>.xlsx` (with optional timestamp)
- Persists any updated refresh tokens back to the `Runners` sheet in your input workbook

Logs:
- Logs print to stdout. Authentication logs mask secrets; refresh token endings are logged per runner.

---

## Behavior details
- Token handling
  - `auth.get_access_token` returns `(access_token, refresh_token)` using your runner’s `refresh_token`.
  - `strava_api.get_segment_efforts` caches `access_token` in-memory per runner to avoid redundant refresh calls.
  - If Strava returns 401 once, the app clears the cached token and retries once with a fresh token.
- Pagination
  - Efforts are retrieved with `per_page=200` and page through until no more results.
- Rate limits
  - If 429, the client throttles for `RATE_LIMIT_THROTTLE_SECONDS` and retries. It also inspects headers and slows down if close to the short-window limit (within `RATE_LIMIT_NEAR_LIMIT_BUFFER`). Concurrency is capped both per-segment (`MAX_WORKERS`) and globally (`RATE_LIMIT_MAX_CONCURRENT`).
- Timezones
  - Strava dates are converted to timezone-naive datetimes before writing because Excel doesn’t support TZ-aware datetimes.
- Excel writing
  - Segment sheets use unique names within Excel’s 31-char limit.
  - If a segment has no data, a small message sheet is written instead.
  - Each segment sheet includes overall `Rank` (fastest=1) and per-team `Team Rank` based on `Fastest Time (sec)`; ties share the same rank.

---

## Common issues and fixes
- 401 Unauthorized when refreshing token
  - The refresh token is invalid or expired, or client credentials are wrong. Re-authorize with `oauth.py` and update the workbook.
- 402 Payment Required when calling `segment_efforts`
  - Strava requires a paid subscription to access segment efforts. Use a subscribed account.
- 429 Rate limit reached
  - The client will back off and retry. You may still need to wait if you hit daily limits.
- Excel timezone error
  - Fixed in code: times are normalized to timezone-naive before writing.
- “No visible sheet” on save
  - The app writes a default summary sheet or per-segment messages if there are no results.
- Port 5000 already in use
  - Change `OAUTH_PORT` in `oauth.py` or stop the process using the port.

---

## Running tests
From the project root (with your virtual environment active):

```bash
# install dev deps (if needed)
pip install -r requirements.txt
pip install pytest

# run all tests
pytest -q

# or a single file
pytest -q tests/test_excel_smoke.py
```

Notes:
- Run from the repo root so `tests/conftest.py` is picked up and `strava_competition` imports resolve.
- In VS Code, open the Testing sidebar and enable pytest to run via the UI.
- If `pytest` is not found, ensure your virtual environment is active or install pytest into it.

---

## Development tips
- Prefer running in a virtual environment.
- Avoid committing real Client Secret or actual spreadsheets with tokens.

---

## Quick reference
- Configure: edit `strava_competition/config.py`
- Get refresh token: `python -m strava_competition.oauth`
- Run app: `python -m strava_competition`
- Output: `<OUTPUT_FILE>[_YYYYMMDD_HHMMSS].xlsx` in the configured path
