# Strava Segment & Distance Competition Tool

A Python application that reads competition inputs from an Excel workbook, fetches Strava data for participating runners, and produces an Excel results workbook containing:
* Per‑segment leaderboards (attempts, fastest time & date, global + intra‑team ranks)
* Optional distance/elevation competition window sheets (runs, total distance/elev, threshold run counts)
* Summary sheets (segments aggregate; distance aggregate) if data present
* It also refreshes and persists runner refresh tokens back into the input workbook.

## Features
Core
* Structured input workbook with clearly separated competitions
  * `Segment Series` sheet: segment definitions (ID, name, date window)
  * `Runners` sheet: runner identity, credentials, and independent team membership for each competition
  * Optional `Distance Series` sheet: distance/elevation competition windows including per‑window distance thresholds
* Fetch segment efforts (pagination, retries, adaptive rate limiting, resilient token refresh)
* Compute per‑runner attempts, fastest time & date, rank, and team rank for each segment
* Distance competition: per window total runs, distance (km), elevation (m) and Runs ≥ Threshold (km) count (if threshold provided)
* Single activity fetch per runner across union of distance windows (no double counting overlaps) powering all window sheets + summary
* Token refresh persistence back into input workbook

Additions
* Optional team summary sheet (participation counts, attempts, aggregate & average fastest times)
* Distance summary sheet (total runs, distance, elevation, average distance per run) for distance participants only

Competition participation flexibility
* A runner may only be in none, one, or both competitions; blank team cells exclude them from that competition automatically.
* Threshold column lets you track how many qualifying (≥ threshold km) runs each athlete logs per distance window

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
* `strava_competition/` — Python package
  * `__main__.py` — enables `python -m strava_competition`
  * `main.py` — orchestration entry point (segments + distance competitions)
  * `config.py` — configuration & tuning constants (paths, concurrency, HTTP pools, autosize flags)
  * `excel_reader.py` — Excel input parser and validation
  * `excel_writer.py` — Excel result writer
  * `segment_aggregation.py` — pure segment ranking & summary (mirrors distance_aggregation)
  * `distance_aggregation.py` — distance window & summary builder (threshold aware)
  * `services/` — high-level orchestration services (segment and distance)
  * `strava_api.py` — resilient Strava REST client (pagination, retries, adaptive rate limiting)
  * `auth.py` — token refresh / error decoding
  * `oauth.py` — local OAuth helper for initial refresh token acquisition
  * `models.py` — dataclasses (`Runner`, `Segment`, `SegmentResult`)
* `run.py` — lightweight convenience launcher (optional; mirrors `python -m`)

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
MAX_WORKERS = 4  # threads per segment when fetching runner efforts
HTTP_POOL_CONNECTIONS = 20
HTTP_POOL_MAXSIZE = 20
REQUEST_TIMEOUT = 15  # seconds
RATE_LIMIT_MAX_CONCURRENT = 8
RATE_LIMIT_JITTER_RANGE = (0.05, 0.2)
RATE_LIMIT_NEAR_LIMIT_BUFFER = 3
RATE_LIMIT_THROTTLE_SECONDS = 15

# Retry/backoff (per page) for Strava API
STRAVA_MAX_RETRIES = 3
STRAVA_BACKOFF_MAX_SECONDS = 4.0
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
Sheets (case sensitive):
* `Segment Series`
  * Columns: `Segment ID` (int), `Segment Name` (str), `Start Date` (date), `End Date` (date)
* `Runners`
  * Columns: `Name` (str), `Strava ID` (int), `Refresh Token` (str), `Segment Series Team` (str | blank), `Distance Series Team` (str | blank)
    * Blank segment team => runner excluded from segment competition
    * Blank distance team => runner excluded from distance competition
* Optional `Distance Series`
  * Columns: `Start Date` (date), `End Date` (date), `Distance Threshold (km)` (float|int|blank)
  * Each row defines a distance competition window; threshold (if set) drives extra column `Runs ≥ <threshold> km`

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
From the project root (venv active):

```bash
python -m strava_competition   # preferred
# or
python run.py                  # convenience wrapper
```

What it does:
- Reads segments, runners, and optional distance windows from `INPUT_FILE`
- Fetches segment efforts for each runner and time window (pagination, retries, adaptive rate limiting)
- If a `Distance Series` sheet exists, fetches run activities once per distance runner over the union of all windows, then derives per-window sheets + distance summary (no double counting overlaps)
- Writes a results workbook to `<OUTPUT_FILE>.xlsx` (with optional timestamp)
- Persists any updated refresh tokens back to the `Runners` sheet

Logs:
- Logs print to stdout. Authentication logs mask secrets; refresh token endings are logged per runner.

---

## Behavior details
These subsections describe current, implemented behavior.

### Architecture & services

The project follows a layered architecture:

```
Domain: models.py, errors.py
Infrastructure: auth.py, oauth.py, strava_api.py, utils.py
I/O: excel_reader.py (pure reads), excel_writer.py (writes)
Services: services/segment_service.py, services/distance_service.py
Orchestration: main.py, run.py
```

### Token handling
Immediate per-runner persistence on rotation, plus a final defensive snapshot at shutdown.
  - `auth.get_access_token` returns `(access_token, refresh_token)` using your runner’s `refresh_token`.
  - `strava_api.get_segment_efforts` caches `access_token` in-memory per runner to avoid redundant refresh calls.
  - If Strava returns 401 once, the app clears the cached token and retries once with a fresh token.

### Pagination
Efforts are retrieved with `per_page=200` and page through until no more results.

### Rate limits
- A global `RateLimiter` enforces a soft cap on in-flight HTTP calls (initially `RATE_LIMIT_MAX_CONCURRENT`).
- Runtime resizing: call `from strava_competition.strava_api import set_rate_limiter; set_rate_limiter(4)` to lower or raise concurrency without restarting.
- If 429 or nearing the short-window limit (within `RATE_LIMIT_NEAR_LIMIT_BUFFER`), a brief throttle window (`RATE_LIMIT_THROTTLE_SECONDS`) is applied; small jitter further smooths bursts.

### Timezones
Strava dates are converted to timezone-naive datetimes before writing because Excel doesn’t support TZ-aware datetimes.

### Excel writing
- Segment sheets use unique names within Excel’s 31-char limit.
- If a segment has no data, a small message sheet is written instead.
- Each segment sheet includes overall `Rank` (fastest=1) and per-team `Team Rank` based on `Fastest Time (sec)`; ties share the same rank.
- Segment summary sheet (if enabled) shows per-team: attempts, participating runner count, sum/average fastest times
- Distance summary sheet shows: total runs, total distance (km), total elevation gain (m), average distance per run (km)

---
## Roadmap (future)
- Activity caching abstraction
- Enhanced workbook schema versioning & validation

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
  - Distance sheets are still written (if defined) even when no segment efforts are available.
- Port 5000 already in use
  - Change `OAUTH_PORT` in `oauth.py` or stop the process using the port.

---

## Running tests
From the project root (with your virtual environment active):

```bash
# install dev deps (if needed)
pip install -r requirements.txt
pip install pytest

pytest -q  # run all tests

# or a single file
pytest -q tests/test_excel_smoke.py
```

Notes:
- Run from the repo root so `tests/conftest.py` is picked up and `strava_competition` imports resolve.
- In VS Code, open the Testing sidebar and enable pytest to run via the UI.
- If `pytest` is not found, ensure your virtual environment is active or install pytest into it.
- Key test files:
  - `test_excel_summary.py` – segment summary sheet aggregation
  - `test_rate_limiter.py` – dynamic concurrency resize semantics
  - `test_auth.py` – token refresh success & error cases
  - `test_integration_api_auth.py` – integration of token refresh with effort fetching
  - `test_strava_api_mocked.py` – API pagination & 401/402 handling

Legacy note: prefer importing from `strava_competition.*`. Any old top-level modules are deprecated and may be removed.

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
- Resize concurrency at runtime (optional):
  ```python
  from strava_competition.strava_api import set_rate_limiter
  set_rate_limiter(4)  # reduce global in-flight request cap to 4
  ```
