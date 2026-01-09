# Strava Segment & Distance Competition Tool

This app reads an Excel workbook, fetches fresh Strava data, and writes a results workbook ready to share. It's built for club admins who want accurate segment leaderboards and distance summaries without living inside the Strava UI.

## Features

| Feature                    | Description                                                           |
| -------------------------- | --------------------------------------------------------------------- |
| **Segment Leaderboards**   | Per-segment sheets showing attempts, fastest times, and team rankings |
| **Split Windows**          | Run one segment across multiple date windows; best time wins          |
| **Distance Competitions**  | Track total distance per runner with threshold filtering              |
| **Birthday Bonus**         | Deduct seconds from efforts on a runner's birthday                    |
| **Time Bonus**             | Add or subtract seconds for all runners in a specific window          |
| **Activity Scan Fallback** | Rebuild results from activity payloads when segment API fails         |
| **Token Auto-Refresh**     | OAuth tokens refreshed and written back automatically                 |
| **Capture & Replay**       | Cache API responses for offline runs and debugging                    |

---

## Contents

- [What it does](#what-it-does)
- [Requirements](#requirements)
- [Configure the app](#configure-the-app)
- [Workbook layout](#workbook-layout)
- [Segment Split Windows](#segment-split-windows)
- [Birthday Bonus](#birthday-bonus)
- [Time Bonus](#time-bonus)
- [Getting refresh tokens](#getting-refresh-tokens)
- [CLI tools](#cli-tools)
- [Run it](#run-it)
- [How it works](#how-it-works)
- [Troubleshooting](#troubleshooting)
- [Tests](#tests)
- [Development tips](#development-tips)

---

## What it does

- Reads segment, runner, and optional distance data from a structured Excel workbook
- Pulls segment efforts with pagination, retries, and adaptive rate limiting
- Fetches each distance runner’s activities once and reuses them across windows
- Writes segment sheets, team/distance summaries, and persists refreshed tokens back to the workbook

Optional sheets like team or distance summaries only appear when you feed in the right data. Leave a team cell blank and that runner simply sits out that competition.

### Architecture at a glance

Here’s the workbook-to-results path that both the segment and distance flows follow.

```mermaid%%{init: {'look': 'handDrawn', 'theme': 'base', 'themeVariables': {'primaryColor': '#ddd6fe', 'primaryTextColor': '#1e1b4b', 'primaryBorderColor': '#8b5cf6', 'lineColor': '#6b7280', 'secondaryColor': '#fce7f3', 'tertiaryColor': '#e0f2fe'}}}%%flowchart LR
  subgraph Input
    Excel[(Excel workbook)]
    Config[Config & .env]
  end

  Reader[excel_reader.py]
  SegmentSvc[SegmentService]
  DistanceSvc[DistanceService]
  StravaAPI["strava_api.py<br/>(capture / replay / live)"]
  ActivityScan[Activity scan fallback]
  Aggregation["segment_aggregation.py<br/>& distance_aggregation.py"]
  Writer[excel_writer.py]
  Output[("Results workbook<br/>+ refreshed tokens")]

  Excel --> Reader --> SegmentSvc
  Excel --> Reader --> DistanceSvc
  Config --> Reader
  Config --> SegmentSvc
  Config --> DistanceSvc

  SegmentSvc -->|segment efforts| StravaAPI
  DistanceSvc -->|athlete activities| StravaAPI
  SegmentSvc --> ActivityScan --> StravaAPI

  StravaAPI -->|live fetch| StravaCloud[(Strava API)]
  StravaAPI -->|capture / replay| Capture[(Capture store)]
  Capture --> StravaAPI

  SegmentSvc --> Aggregation
  DistanceSvc --> Aggregation
  Aggregation --> Writer --> Output
```

---

## Requirements

- Python 3.10 or later (tested on 3.13)
- A Strava API application (Client ID and Client Secret)
- Strava subscriptions for any athletes whose segment efforts you need to view (Strava enforces this)

Install the dependencies from `requirements.txt` inside a virtual environment. On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip -r requirements.txt
```

On Windows use `py -3 -m venv .venv` and `.venv\Scripts\activate` instead of `source`.

---

## Configure the app

The app looks for input/output files in the `data/` folder by default. You can override
paths via command-line arguments (see [Run it](#run-it)) or by editing the defaults in
`strava_competition/config.py`:

```python
INPUT_FILE = "data/competition_input.xlsx"
OUTPUT_FILE = "data/competition_results"
OUTPUT_FILE_TIMESTAMP_ENABLED = True  # adds _YYYYMMDD_HHMMSS
```

Quick reference for the default on-disk layout inside `data/`:

- Input workbook: `data/competition_input.xlsx` (drop your Excel file here unless
  you override paths on the CLI).
- Results: `data/competition_results_<timestamp>.xlsx` when timestamping is on,
  or `data/competition_results.xlsx` when you disable it.
- GPX helpers: `data/gpx_output/` (auto-created by the CLI tools so exports live
  alongside other generated artifacts).

Performance tuning knobs—worker counts, HTTP pools, rate limits, retry strategy, and so on—also live in this file. Tweak them only when you have a concrete reason.

Create a `.env` file in the project root so credentials stay out of source control:

```dotenv
# Required credentials
STRAVA_CLIENT_ID=<your_id>
STRAVA_CLIENT_SECRET=<your_secret>

# Activity scan fallback
USE_ACTIVITY_SCAN_FALLBACK=true
FORCE_ACTIVITY_SCAN_FALLBACK=false
ACTIVITY_SCAN_MAX_ACTIVITY_PAGES=0
ACTIVITY_SCAN_CACHE_INCLUDE_ALL_EFFORTS=true

# Segment split windows
SEGMENT_SPLIT_WINDOWS_ENABLED=true

# API cache mode: live | cache | offline
STRAVA_API_CACHE_MODE=cache
STRAVA_CACHE_HASH_IDENTIFIERS=true
STRAVA_CACHE_ID_SALT=please_change_me
STRAVA_CACHE_REDACT_PII=true
STRAVA_CACHE_REDACT_FIELDS=name,email,athlete.firstname,athlete.lastname
```

| Variable                        | Default | Description                                                       |
| ------------------------------- | ------- | ----------------------------------------------------------------- |
| `USE_ACTIVITY_SCAN_FALLBACK`    | `true`  | Fall back to activity scan when segment API fails (402 errors)    |
| `FORCE_ACTIVITY_SCAN_FALLBACK`  | `false` | Always use activity scan, bypassing segment API entirely          |
| `SEGMENT_SPLIT_WINDOWS_ENABLED` | `true`  | Group duplicate segment IDs into multi-window competitions        |
| `STRAVA_API_CACHE_MODE`         | `live`  | Cache mode: `live` (no cache), `cache` (read+write), or `offline` |

The app pulls in `.env` automatically at startup.

#### Cache tail refresh knobs

When `STRAVA_API_CACHE_MODE=cache` or `offline`, the tool automatically tops up cached
`/athlete/activities` pages with live data (in `cache` mode). You can tune the behaviour via
environment variables:

| Variable                     | Default | Description                                                                                                                     |
| ---------------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `CACHE_TTL_DAYS`             | `7`     | Maximum age for cached pages before they are discarded and fully refetched. Set to `0` to disable the TTL.                      |
| `CACHE_MAX_LOOKBACK_DAYS`    | `30`    | Guard that prevents replaying extremely old captures. When exceeded, the process logs a warning and performs a full live fetch. |
| `CACHE_TAIL_OVERLAP_SECONDS` | `60`    | Small overlap injected into the tail window so activities near the cached boundary are never skipped.                           |
| `CACHE_MAX_PARALLELISM`      | `4`     | Caps how many runners are refreshed in parallel inside the distance service orchestration layer.                                |

The hybrid cache-tail workflow automatically persists enriched pages back to
the cache directory. When `STRAVA_CACHE_OVERWRITE` is `False` (the
default), enriched data is stored in lightweight overlay files so existing
caches remain untouched.

#### Cache hygiene & retention

- **Hash runner identifiers.** Turn on `STRAVA_CACHE_HASH_IDENTIFIERS=true` and give it a
  unique, non-empty `STRAVA_CACHE_ID_SALT` so cache filenames don't leak Strava IDs.
  When hashing is enabled without a salt the app simply bows out instead of producing
  predictable hashes.
- **Redact payload fields.** Enable `STRAVA_CACHE_REDACT_PII` and list comma-separated
  JSON paths in `STRAVA_CACHE_REDACT_FIELDS` to strip names, emails, GPS details, and any
  other sensitive bits before writing to disk.
- **Prune stale caches.** Use the helper in the [CLI tools](#cli-tools) section whenever
  you want to free disk space without touching fresh data.
- **Automate retention.** Set `STRAVA_CACHE_AUTO_PRUNE_DAYS=30` (or whatever window you
  prefer) to delete old cache files on startup. Pick `0` to leave files alone.

> **Note:** Token responses are never cached—they contain short-lived secrets and
> should always come from Strava directly. In `offline` mode, auth calls will fail
> with a clear error message.

### Workbook layout

All sheet names are case-sensitive; match them exactly as written below.

- `Segment Series`: Segment ID, Segment Name, Start Date, End Date, **Window Label** (optional), Default Time,
  Minimum Distance (m), **Birthday Bonus (secs)**
- `Runners`: Name, Strava ID, Refresh Token, Segment Series Team,
  Distance Series Team, **Birthday (dd-mmm)**
- Optional `Distance Series`: Start Date, End Date, Distance Threshold (km)

Leave a team column blank and that runner quietly skips the related competition. Runner birthdays must stay in `dd-MMM` form (e.g. `07-May`). The reader accepts Excel dates or ISO strings, but when the workbook is rewritten the values are normalised to that format. Other date columns can be Excel dates or ISO strings—pandas is happy with either.

`Default Time` accepts `HH:MM:SS`, Excel time values, or raw seconds. Every runner with no recorded effort picks up this fallback so rankings always account for the full roster.

### Segment Split Windows

The app supports defining **multiple date windows** for a single segment. This is useful when you want to run a segment competition across several time periods and show each runner's best effort across all windows.

#### How it works

Duplicate rows with the same **Segment ID** are automatically grouped together. Each row becomes a separate window, and the runner's fastest time across all windows appears in the output.

| Segment ID | Segment Name      | Start Date | End Date   | Window Label | Default Time | Minimum Distance (m) | Birthday Bonus (secs) | Time Bonus (secs) |
| ---------- | ----------------- | ---------- | ---------- | ------------ | ------------ | -------------------- | --------------------- | ----------------- |
| 12345678   | Hill Climb Sprint | 2026-01-01 | 2026-01-15 | Week 1       | 00:10:00     | 500                  | 30                    |                   |
| 12345678   | Hill Climb Sprint | 2026-01-16 | 2026-01-31 | Week 2       |              |                      | 30                    | -5                |
| 12345678   | Hill Climb Sprint | 2026-02-01 | 2026-02-14 | Final Push   |              |                      | 45                    | 15                |

**Key rules:**

- **Grouping key**: Rows are grouped by Segment ID (not name). All rows with the same ID must have the same Segment Name.
- **Window Label**: Optional column for human-friendly tags (e.g., "Week 1", "Final Push"). Used in sheet names when split windows is disabled.
- **Birthday Bonus**: Can vary per window. Defaults to 0 if omitted.
- **Time Bonus**: Optional per-window adjustment. Positive values subtract time (reward), negative values add time (penalty).
- **Default Time / Minimum Distance**: Apply once per group. Only specify on one row—if set on multiple rows, values must match.
- **Attempts**: Output shows total attempts across all windows.
- **Overlapping windows**: Allowed but logs a warning if windows fully overlap (likely user error).

#### Config toggle

Set `SEGMENT_SPLIT_WINDOWS_ENABLED` in `config.py` (default: `True`).

- **Enabled**: Best time across all windows → one output sheet per segment.
- **Disabled**: Each window → separate sheet. Single-window segments use just the name; multi-window use `{Name} - {Label}` or date range.

`Minimum Distance (m)` is optional per segment. Leave it blank or set it to `0`
to disable distance filtering. When populated with a positive value the runner's
effort distance (from Strava's payload) must meet or exceed that threshold to
count toward rankings.

---

### Birthday Bonus

Give runners a time advantage on their birthday. If a runner completes a segment effort on their birthday, the configured bonus (in seconds) is subtracted from their elapsed time.

**How to configure:**

1. In the `Runners` sheet, add the runner's birthday in `dd-MMM` format (e.g., `15-Jan`, `07-May`)
2. In the `Segment Series` sheet, set `Birthday Bonus (secs)` to the number of seconds to deduct

**Example:** If a runner's birthday is `15-Jan` and they complete a segment in 120 seconds on January 15th with a 30-second bonus configured, their adjusted time becomes 90 seconds.

**Notes:**

- Bonus is based on the effort's `start_date_local` (when the runner entered the segment)
- Different windows can have different bonus values (useful for split windows)
- If no bonus is configured (blank or 0), no adjustment is applied
- The output indicates whether a birthday bonus was applied via the `birthday_bonus_applied` flag

---

### Time Bonus

Apply a time bonus (positive or negative) to all runners who complete an effort within a specific window. Unlike birthday bonus, this applies to everyone—not just those on their birthday.

**How to configure:**

In the `Segment Series` sheet, add the `Time Bonus (secs)` column:

- Leave empty or blank for no adjustment
- **Positive values** subtract time (reward/bonus)
- **Negative values** add time (penalty)

**Use cases:**

- **Incentivise unlikely days:** e.g., 30-second bonus for running on Christmas Day
- **Short promotional windows:** e.g., bonus for a 1-hour window on New Year's Eve
- **Compensate for conditions:** e.g., penalty for a course diversion or adverse weather

**Example:** A window with `Time Bonus (secs) = 15` means all runners in that window get 15 seconds subtracted from their time. A value of `-5` adds 5 seconds (penalty).

**Notes:**

- Stacks with birthday bonus (applied after birthday bonus)
- Decimal values supported (e.g., `5.5`)
- Adjusted time floors at 0.0 seconds (cannot go negative)
- Does **not** apply to default time—only actual efforts

---

### Distance Series

Track cumulative running distance over time windows. This is separate from segment competitions.

**Workbook columns:**

| Column                  | Description                            |
| ----------------------- | -------------------------------------- |
| Start Date              | Beginning of the distance window       |
| End Date                | End of the distance window             |
| Distance Threshold (km) | Minimum distance to qualify (optional) |

**How it works:**

1. The app fetches each runner's activities within the date window
2. Totals the distance from all qualifying activity types (default: runs)
3. Produces a summary sheet with per-runner and per-team totals

Runners need the `Distance Series Team` column populated in the `Runners` sheet to participate.

---

## Getting refresh tokens

The first time you add a runner you’ll need a refresh token. Run the helper:

```bash
python -m strava_competition.oauth
```

It spins up a tiny web server, opens Strava’s OAuth screen, and logs masked token metadata once the runner approves. Pass `--print-tokens` if you need the plaintext values. Copy the refresh token into the `Runners` sheet. Change `OAUTH_PORT` inside `oauth.py` if you need a different port.

Already have an authorisation code? Trade it for tokens with curl:

```bash
curl -X POST https://www.strava.com/oauth/token \
  -d client_id=<your_client_id> \
  -d client_secret=<your_client_secret> \
  -d code=<authorisation_code> \
  -d grant_type=authorization_code
```

The response includes the refresh token—stash it somewhere safe.

---

## CLI tools

You’ll find the helper scripts under `strava_competition/tools`, and they pull from the
same configuration as the main app:

- `fetch_runner_segment_efforts`: dumps `/athlete/activities` windows with
  `include_all_efforts=true` so you can poke through the raw Strava payloads for
  a runner. Run `python -m strava_competition.tools.fetch_runner_segment_efforts --help`
  to check the flags.
- `fetch_activity_gps`: fetches GPS coordinates for a specific activity using
  Strava's Streams API. Outputs to `data/gpx_output/activity_<id>.gpx` by default (GPX format)
  and creates the directory if it isn't there yet.
  Altitude, time, and distance data are included by default; use `--no-altitude`,
  `--no-time`, or `--no-distance` to exclude. Use `--output-file` to override the
  output path, or `--no-file` to print to stdout. Run
  `python -m strava_competition.tools.fetch_activity_gps --help` for usage.
- `fetch_segment_gpx`: exports a Strava segment as a GPX route file for sharing
  or importing into GPS devices and mapping apps. Outputs to `data/gpx_output/segment_<id>.gpx`
  by default and creates the directory automatically. Use `--output-file` to override
  the output path, or `--no-file` to print
  to stdout. Run `python -m strava_competition.tools.fetch_segment_gpx --help` for usage.
- `clip_activity_segment`: slices a window of track points out of a GPX file so you can
  reproduce a Strava segment effort locally. Start/end indices are optional—pick
  whichever selector suits you: the zero-based indices from Strava's
  `segment_efforts` API, a time window, a captured `segment_efforts` response, or
  provide the Strava IDs (activity + segment) plus a refresh token and the
  tool will fetch the GPX stream and effort metadata for you. Auto-downloaded GPX
  files land in `data/gpx_output/` by default (override with `--download-dir`),
  and `--force-download` refreshes the cache when you need a clean pull.

  Fully automatic example (no local GPX/JSON required):

  ```bash
  python -m strava_competition.tools.clip_activity_segment \
    --activity-id 16543582334 \
    --segment-id 40422214 \
    --refresh-token 428d4533373e68e32ec57e9fae2b8fc79ed934f5 \
    --runner-id 19923466 \
    --output data/gpx_output/activity_16543582334_segment_40422214_auto.gpx
  ```

  Index-based example:

  ```bash
  python -m strava_competition.tools.clip_activity_segment \
    --input data/gpx_output/activity_16919797941.gpx \
    --start-index 1572 --end-index 2699 \
    --output data/gpx_output/activity_16919797941_segment_40641291.gpx
  ```

  JSON-based auto detection:

  ```bash
  python -m strava_competition.tools.clip_activity_segment \
    --input data/gpx_output/activity_16919797941.gpx \
    --segment-efforts-json strava_cache/c0/bd/...overlay.json \
    --activity-id 16919797941 \
    --segment-id 40641291
  ```

  You can also slice by ISO timestamps: `--start-time 2026-01-03T06:23:44+00:00 --elapsed 1127`.

- `deviation_map`: builds an interactive Folium map that highlights gate crossings
  and large deviations for a runner/segment pair. Launch it via
  `python -m strava_competition.tools.deviation_map --help` and drop the output
  wherever you need.
- `capture_gc`: deletes cache responses older than a retention window. Run
  `python -m strava_competition.tools.capture_gc --max-age 30d` to prune files
  older than 30 days (supports `d`, `h`, or raw seconds).

The legacy `helper/fetch_runner_segment_efforts.py` shim now imports these
modules so existing scripts keep working.

---

## Run it

With your virtual environment active, run from the repo root:

```bash
python -m strava_competition
```

By default, the app reads from `data/competition_input.xlsx` and writes results to
`data/competition_results_<timestamp>.xlsx`. Override these paths with command-line
arguments:

```bash
# Show available options
python -m strava_competition --help

# Use custom input and output paths
python -m strava_competition \
  --input /Users/me/Documents/strava/my_competition.xlsx \
  --output /Users/me/Documents/strava/results

# Short flags work too
python -m strava_competition -i ~/Dropbox/running/input.xlsx -o ~/Dropbox/running/output
```

| Flag       | Short | Description                                                       |
| ---------- | ----- | ----------------------------------------------------------------- |
| `--input`  | `-i`  | Path to the input Excel workbook                                  |
| `--output` | `-o`  | Output file base name (timestamp and `.xlsx` added automatically) |

The app reads the workbook, pulls the Strava data it needs, writes the results workbook named after `OUTPUT_FILE`, and updates runner tokens before it exits. Status logs stream to stdout and keep the sensitive bits redacted.

### Docker usage

To run inside Docker, build the image once and mount your host files into `/app` so inputs, captures, and results stay on your main drive:

```bash
docker build -t strava-competition .
```

#### macOS / Linux run command

```bash
docker run --rm \
  -v "$(pwd)":/app \
  -w /app \
  strava-competition
```

- `$(pwd)` should point at the folder containing `competition_input.xlsx` and `strava_cache/`.
- Outputs such as `competition_results_*.xlsx` and refreshed captures are written straight to the host directory because of the bind mount.
- Override paths by changing the `-v host_dir:/app` portion.

#### Windows run commands

PowerShell:

```powershell
docker run --rm `
  -v "${PWD}:/app" `
  -w /app `
  strava-competition
```

Command Prompt:

```cmd
docker run --rm ^
  -v %CD%:/app ^
  -w /app ^
  strava-competition
```

Tips:

- Make sure Docker Desktop can see the drive you’re mounting (Settings ▸ Resources ▸ File Sharing on Windows).
- Quote host paths that include spaces so the shell doesn’t split them.
- The container image ships with `STRAVA_API_CACHE_MODE=cache`, so cache files will show up under the mounted directory's `strava_cache/` folder.

---

## How it works

### Data flow

- `excel_reader.py` loads the workbook and validates each sheet.
- `services/segment_service.py` and `services/distance_service.py` orchestrate Strava API calls via `strava_api.py`.
- Results flow through `segment_aggregation.py`, `distance_aggregation.py`, and finally `excel_writer.py`.
- Updated refresh tokens are written back before shutdown.

### Activity scan fallback

Some runners only expose full Strava activities. Set `USE_ACTIVITY_SCAN_FALLBACK=true`
to rebuild results from `include_all_efforts` payloads. The scanner fetches each activity window once,
leans on cached pages when possible, and logs the inspected activity IDs for easy auditing.

**Playbook:**

1. Enable `USE_ACTIVITY_SCAN_FALLBACK` (and keep `FORCE_ACTIVITY_SCAN_FALLBACK=false`
   so paid athletes still use official efforts). Optionally limit pagination via
   `ACTIVITY_SCAN_MAX_ACTIVITY_PAGES` (defaults to `0`, meaning "no cap"—set a
   positive integer to stop after that many pages).
2. Prime caches with `STRAVA_API_CACHE_MODE=cache`.
3. Switch to deterministic runs using cached data (and set `STRAVA_API_CACHE_MODE=offline` if you
   want to forbid live calls). Watch for `source=activity_scan` in the logs.

Keep `ACTIVITY_SCAN_CACHE_INCLUDE_ALL_EFFORTS=true` so cached payloads match the scanner; otherwise
offline runs throw a `StravaAPIError` when caches are missing. Files live under
`strava_cache/`, and `tests/strava_cache/` holds the fixtures used by pytest.

---

## Troubleshooting

- 401 when refreshing tokens: the refresh token or client credentials are wrong—rerun the OAuth helper.
- 402 Payment Required: the athlete needs a paid Strava subscription for segment efforts.
- 429 Too Many Requests: wait for the rate-limit window; the app already backs off.
- Excel opens without visible sheets: the app writes placeholder sheets when a segment has no data.
- Port 5000 in use for OAuth: change `OAUTH_PORT` or free the port (AirPlay often uses it on macOS).

---

## Tests

```bash
pytest -q
```

Run tests from the repo root so `tests/conftest.py` wires itself up correctly. Highlights include `test_excel_summary.py`, `test_rate_limiter.py`, `test_auth.py`, `test_integration_api_auth.py`, `test_strava_api_mocked.py`, and the load/smoke suite in `tests/test_load_smoke.py` that exercises concurrency plus capture replay.

### Quality checks

Before opening a pull request, run the same tooling wired into CI:

```bash
ruff check
mypy
bandit -q -r strava_competition
pytest
python -m strava_competition.tools.capture_gc --dry-run --max-age-days 45
```

Those commands keep lint, typing, security scanning, tests, and capture-retention checks green before you ship anything.

---

## Development tips

- Work in a virtual environment and keep secrets out of source control.
- Reach for `python -m strava_competition.oauth` whenever you need fresh tokens.
- Use `python -m strava_competition` for real runs; `run.py` is just a thin wrapper.
- Prefer environment variables over code edits when you need to tweak configuration.
