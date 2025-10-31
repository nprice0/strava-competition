# Strava Segment & Distance Competition Tool

This app reads an Excel workbook, fetches fresh Strava data, and writes a results workbook ready to share. It is aimed at club admins who want accurate segment leaderboards and distance summaries without touching the Strava UI.

You get:

- Per-segment leaderboards with attempts, fastest time, and team rankings
- Distance competition sheets covering each window and an overall summary
- Automatic refresh-token updates so the input workbook always stays current
- A segment matcher fallback that rebuilds efforts with discrete Fréchet and DTW distance checks plus accurate start/finish timing

---

## What it does

- Reads segment, runner, and optional distance data from a structured Excel workbook
- Pulls segment efforts with pagination, retries, and adaptive rate limiting
- Fetches each distance runner’s activities once and reuses them across windows
- Writes segment sheets, team/distance summaries, and persists refreshed tokens back to the workbook

Optional sheets such as team summaries or distance summaries appear only when the source data is present. Blank team cells in the input automatically exclude a runner from that competition.

---

## Requirements

- Python 3.10 or later (tested on 3.13)
- A Strava API application (Client ID and Client Secret)
- Strava subscriptions for any athletes whose segment efforts you need to view (Strava enforces this)

Install the Python packages listed in `requirements.txt`. Key libraries include pandas, openpyxl, requests, Flask, urllib3, and python-dotenv.

### macOS quick start (zsh)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Windows quick start (PowerShell)

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If execution policy blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`. On Command Prompt use `\.venv\Scripts\activate.bat` instead.

---

## Configure the app

Edit `strava_competition/config.py` to point at your input and output files and to set Strava credentials:

```python
INPUT_FILE = "/absolute/path/to/competition_input.xlsx"
OUTPUT_FILE = "/absolute/path/for/output/competition_results"
OUTPUT_FILE_TIMESTAMP_ENABLED = True  # adds _YYYYMMDD_HHMMSS

import os
CLIENT_ID = os.getenv("STRAVA_CLIENT_ID", "")
CLIENT_SECRET = os.getenv("STRAVA_CLIENT_SECRET", "")
```

Performance tuning values such as worker counts, HTTP pool sizes, rate-limit settings, and retry strategy also live in this file. Adjust them only if you know you need to.

Create a `.env` file in the project root so credentials stay out of source control:

```dotenv
STRAVA_CLIENT_ID=<your_id>
STRAVA_CLIENT_SECRET=<your_secret>
```

The app loads `.env` automatically when it starts.

### Workbook layout

All sheet names are case sensitive.

- `Segment Series`: segment ID, segment name, start date, end date
- `Runners`: name, Strava ID, refresh token, segment team, distance team
- Optional `Distance Series`: start date, end date, distance threshold (km)

Leave a team column blank to skip that runner for the related competition. Dates can be Excel dates or ISO strings; pandas handles both.

---

## Getting refresh tokens

The first time you add a runner you need a refresh token. Run the helper:

```bash
python -m strava_competition.oauth
```

It spins up a small web server, opens Strava’s OAuth screen, and prints the tokens once the runner approves. Copy the refresh token into the `Runners` sheet. You can change `OAUTH_PORT` or `PRINT_TOKENS` in `oauth.py` if needed.

Already have an authorisation code? Swap it via curl:

```bash
curl -X POST https://www.strava.com/oauth/token \
  -d client_id=<your_client_id> \
  -d client_secret=<your_client_secret> \
  -d code=<authorisation_code> \
  -d grant_type=authorization_code
```

The response includes the refresh token. Store it securely.

---

## Run it

Activate your virtual environment, then run either command from the repo root:

```bash
python -m strava_competition
# or
python run.py
```

The app reads the workbook, fetches the required Strava data, writes the results workbook named after `OUTPUT_FILE`, and updates the runner tokens inline. Status logs print to stdout with minimal secrets.

---

## How it works

### Data flow

- `excel_reader.py` loads the workbook and validates each sheet
- `services/segment_service.py` and `services/distance_service.py` orchestrate Strava API calls via `strava_api.py`
- Results flow through `segment_aggregation.py`, `distance_aggregation.py`, and finally `excel_writer.py`
- Updated refresh tokens are written back before shutdown

### Segment matcher overview

The fallback matcher kicks in when Strava refuses segment efforts (HTTP 402 or pre-flagged subscriptions). It builds a fresh comparison between the runner’s activity stream and the segment.

- Streams pulled: lat/lng, altitude, distance, and time. We project them into metres using the segment transformer.
- Prepared geometry: `prepare_geometry` simplifies and resamples the segment, and `prepare_activity` applies the same transform to the activity so both paths have similar spacing.
- Coverage trim: `compute_coverage` projects every point onto the segment. `_refine_coverage_window` keeps samples that stay within the offset tolerance and notes the last point before the start and the first point after the finish. That avoids loops.
- Similarity: discrete Fréchet distance is the first check. If it is above the adaptive threshold we fall back to windowed DTW. Both operate on the resampled polylines.
- Timing: `estimate_segment_time` honours the refined bounds. `_resolve_entry_exit` shifts the entry one sample back if needed, nudges the exit forward, or falls back to `_find_entry_event` and `_find_exit_event` which interpolate timestamps. You end up with the closest pre-start to post-finish duration.
- Diagnostics: we log coverage ratios, similarity scores, sample indices, and elapsed times for every attempt.

---

## Troubleshooting

- 401 when refreshing tokens: the refresh token or client credentials are wrong; rerun the OAuth helper
- 402 Payment Required: the athlete needs a paid Strava subscription for segment efforts
- 429 Too Many Requests: wait for the rate limit window; the app already backs off
- Excel opens without visible sheets: the app writes placeholder sheets when a segment has no data
- Port 5000 in use for OAuth: change `OAUTH_PORT` or free the port (AirPlay often uses it on macOS)

---

## Tests

```bash
pip install -r requirements.txt
pip install pytest
pytest -q
```

Run tests from the repo root so `tests/conftest.py` loads correctly. Key files include `test_excel_summary.py`, `test_rate_limiter.py`, `test_auth.py`, `test_integration_api_auth.py`, and `test_strava_api_mocked.py`.

---

## Development tips

- Work in a virtual environment and keep secrets out of source control
- Use `python -m strava_competition.oauth` for new tokens
- Use `python -m strava_competition` for production runs
- Adjust configuration values via environment variables rather than editing code when possible
