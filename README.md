# Strava Segment & Distance Competition Tool

This app reads an Excel workbook, fetches fresh Strava data, and writes a results workbook ready to share. It is aimed at club admins who want accurate segment leaderboards and distance summaries without touching the Strava UI.

You get:

- Per-segment leaderboards with attempts, fastest time, and team rankings
- Distance competition sheets covering each window and an overall summary
- Automatic refresh-token updates so the input workbook always stays current
- A segment matcher fallback that rebuilds efforts with discrete Fréchet and DTW distance checks plus accurate start/finish timing
- An activity-scan fallback that uses Strava's `include_all_efforts` payloads for runners without leaderboard access

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
USE_ACTIVITY_SCAN_FALLBACK=false
ACTIVITY_SCAN_MAX_ACTIVITY_PAGES=10
ACTIVITY_SCAN_CAPTURE_INCLUDE_ALL_EFFORTS=true
```

The app loads `.env` automatically when it starts.

#### Replay-tail refresh knobs

When `STRAVA_API_REPLAY_ENABLED` is on, the tool automatically tops up cached
`/athlete/activities` pages with live data. You can tune the behaviour via
environment variables:

| Variable                   | Default | Description                                                                                                                     |
| -------------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `REPLAY_CACHE_TTL_DAYS`    | `7`     | Maximum age for cached pages before they are discarded and fully refetched. Set to `0` to disable the TTL.                      |
| `REPLAY_MAX_LOOKBACK_DAYS` | `30`    | Guard that prevents replaying extremely old captures. When exceeded, the process logs a warning and performs a full live fetch. |
| `REPLAY_EPSILON_SECONDS`   | `60`    | Small overlap injected into the tail window so activities near the cached boundary are never skipped.                           |
| `REPLAY_MAX_PARALLELISM`   | `4`     | Caps how many runners are refreshed in parallel inside the distance service orchestration layer.                                |

The hybrid replay-tail workflow automatically persists enriched pages back to
the capture directory. When `STRAVA_API_CAPTURE_OVERWRITE` is `False` (the
default), enriched data is stored in lightweight overlay files so existing
captures remain untouched.

### Workbook layout

All sheet names are case sensitive.

- `Segment Series`: segment ID, segment name, start date, end date, default time
- `Runners`: name, Strava ID, refresh token, segment team, distance team
- Optional `Distance Series`: start date, end date, distance threshold (km)

Leave a team column blank to skip that runner for the related competition. Dates can be Excel dates or ISO strings; pandas handles both.

`Default Time` accepts `HH:MM:SS`, Excel time values, or raw seconds. Every runner without a recorded effort for a segment is assigned this fallback so rankings and summaries always include every participant.

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

## CLI tools

Two focused utilities live under `strava_competition/tools` to help with
support and debugging tasks. Both reuse the same `.env` credentials and
dependencies as the main app.

### `fetch_runner_segment_efforts`

- Exchanges a refresh token for an access token, then walks a given day or
  custom window of `/athlete/activities` results.
- Fetches `include_all_efforts=true` for every activity in that window so you
  can see exactly what Strava returns for a subscriber.
- Prints a compact JSON summary per segment effort, with an optional
  `--print-json` flag to dump each raw payload.

Run it from the repo root once your virtual environment is active:

```bash
source .venv/bin/activate && \
python -m strava_competition.tools.fetch_runner_segment_efforts \
   --runner-id 13056193 \
   --runner-name "Helen Lawrence" \
   --refresh-token abcdef5ffe85a428b5678fafe749e3a758cc3614 \
   --start 2025-11-03T00:00:00Z \
   --end 2025-11-24T00:00:00Z
```

### `deviation_map`

- Reads runners and segments from the workbook, fetches the matching activity
  stream plus segment geometry, and renders an interactive Folium map showing
  where the athlete left the official course.
- Highlights gate crossings, coverage diagnostics, and large offsets so you can
  sanity-check matcher decisions or explain unusual leaderboard results.
- Saves the output HTML wherever you choose (`maps/<slug>.html` by default, or
  a custom path when you pass `--output`).

Example invocation:

```bash
source .venv/bin/activate && \
python -m strava_competition.tools.deviation_map \
   --runner-name "Helen Lawrence" \
   --segment-name "WS25 - Fancy A Tipple After Mass" \
   --activity-id 1234567890 \
  --threshold-m 40 \
  --output maps/helen-ws25.html

The generated HTML will appear under the path you specified (or the default
`maps/` folder) so you can open it directly in a browser.
```

The legacy `helper/fetch_runner_segment_efforts.py` entry point now just calls
into the new `tools` module so older docs keep working, but future updates will
land under `strava_competition/tools/*`.

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

The fallback matcher kicks in when Strava refuses segment efforts (HTTP 402 or pre-flagged subscriptions). It rebuilds the effort from the runner’s activity stream and compares it directly to the segment geometry.

- Streams pulled: lat/lng, altitude, distance, and time. Everything is projected into metres using the segment transformer so later math stays accurate.
- Prepared geometry: `prepare_geometry` simplifies and resamples the segment, and `prepare_activity` applies the same transform to the activity so both paths share spacing and coordinate frames.
- Coverage refinement: `compute_coverage` projects every sample onto the segment. `_refine_coverage_window` keeps points inside the offset tolerance, flags the first pre-start sample and the first post-finish sample, and drops obvious detours.
- Gate clipping with full context: `_clip_activity_to_gate_window` re-evaluates start/finish gate crossings against the full resampled activity, then slices the trimmed window to just the on-segment samples. This mirrors Strava’s `entry/exit plane` logic and prevents post-finish jogs from inflating Frechet distance.
- Similarity: discrete Fréchet distance is evaluated first. If it misses the adaptive threshold we fall back to windowed DTW with a narrow band. Both operate on the already clipped resampled polylines.
- Timing: `estimate_segment_time` honours the refined bounds. `_resolve_entry_exit` interpolates timestamps for the on-segment entry and exit samples so elapsed time reflects exactly when the runner crossed the gates.
- Diagnostics: we log coverage ratios, similarity scores, `gate_trimmed` flags, sample indices, and elapsed times for every attempt.

### Activity scan fallback

Some runners only expose full Strava activities (not segment leaderboards). Enable the activity scan fallback by setting `USE_ACTIVITY_SCAN_FALLBACK=true`. When active, the service:

- Fetches every run activity inside the competition window (respecting `ACTIVITY_SCAN_MAX_ACTIVITY_PAGES` if set)
- Calls `GET /activities/{id}?include_all_efforts=true` once per activity with rate limiting, optional capture via `ACTIVITY_SCAN_CAPTURE_INCLUDE_ALL_EFFORTS`, and in-memory caching
- Counts attempts and identifies the fastest elapsed time for the target segment using the activity payload alone
- Emits diagnostics with inspected activity IDs so you can audit workbook results later

When the flag is on, the GPS matcher stays idle unless you explicitly set `MATCHING_FALLBACK_ENABLED=true`. This makes the pipeline reliable for non-subscriber athletes while keeping the legacy matcher available for debugging.

#### Operator playbook

1. Toggle the feature flags:

- `USE_ACTIVITY_SCAN_FALLBACK=true`
- `MATCHING_FORCE_FALLBACK=false` (so paid runners continue using the official Strava efforts)
- Optionally cap pagination with `ACTIVITY_SCAN_MAX_ACTIVITY_PAGES=10` while validating

2. Record captures for the current competition window by running the pipeline once with
   `STRAVA_API_CAPTURE_ENABLED=true` and `STRAVA_API_REPLAY_ENABLED=false`.
3. Switch to deterministic mode for day-to-day runs: set `STRAVA_API_CAPTURE_ENABLED=false`,
   `STRAVA_API_REPLAY_ENABLED=true`, and (if you want to block live calls) `STRAVA_OFFLINE_MODE=true`.
4. Monitor logs for entries tagged `source=activity_scan` and spot-check the emitted
   `inspected_activities` list when validating workbook outputs.

#### Capture and replay tips

- Include `ACTIVITY_SCAN_CAPTURE_INCLUDE_ALL_EFFORTS=true` so the activity detail payloads match what the scanner expects.
- When `STRAVA_OFFLINE_MODE=true`, a missing capture now raises `StravaAPIError`; the runner is skipped and the log will point at the exact request signature you need to record.
- Captures live under `strava_api_capture/` by default. Vendored regression fixtures live in `tests/strava_api_capture/` and power the pytest suite.

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
