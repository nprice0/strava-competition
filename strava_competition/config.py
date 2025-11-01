"""Central configuration for the Strava Segment Competition tool.

All values are constants imported by the rest of the package. Adjust as needed
for your environment. Secrets are read from environment variables (optionally
via a local `.env`).
"""

from __future__ import annotations

import importlib
import os


def _env_float(key: str, default: float) -> float:
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_int(key: str, default: int) -> int:
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    value = os.getenv(key)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


# Load .env variables when python-dotenv is available.
_load_dotenv = None
try:
    _dotenv_mod = importlib.import_module("dotenv")
    _load_dotenv = getattr(_dotenv_mod, "load_dotenv", None)
except Exception:
    _load_dotenv = None

if callable(_load_dotenv):
    # Load .env from the current directory or any parent folder.
    _load_dotenv()


# ---------------------------------------------------------------------------
# Input/Output
# ---------------------------------------------------------------------------
# Paths can be absolute or relative.
INPUT_FILE = "competition_input.xlsx"
OUTPUT_FILE = "competition_results"

# Append _YYYYMMDD_HHMMSS to the output name when True.
OUTPUT_FILE_TIMESTAMP_ENABLED = True


# ---------------------------------------------------------------------------
# Strava settings
# ---------------------------------------------------------------------------
# Base Strava API URLs.
STRAVA_BASE_URL = "https://www.strava.com/api/v3"
STRAVA_OAUTH_URL = "https://www.strava.com/oauth/token"

# Client credentials pulled from the environment. Do not hardcode secrets.
CLIENT_ID = os.getenv("STRAVA_CLIENT_ID", "")
CLIENT_SECRET = os.getenv("STRAVA_CLIENT_SECRET", "")


# ---------------------------------------------------------------------------
# Performance tuning
# ---------------------------------------------------------------------------
# Threads used per segment when fetching efforts in parallel.
MAX_WORKERS = 4

# HTTP session pool sizes for concurrent requests.
HTTP_POOL_CONNECTIONS = 20
HTTP_POOL_MAXSIZE = 20

# Request timeout in seconds.
REQUEST_TIMEOUT = 15


# Rate limiter settings.
# RATE_LIMIT_MAX_CONCURRENT caps total in-flight requests.
RATE_LIMIT_MAX_CONCURRENT = 8
# RATE_LIMIT_JITTER_RANGE adds random delay (seconds) to smooth bursts.
RATE_LIMIT_JITTER_RANGE = (0.05, 0.2)
# RATE_LIMIT_NEAR_LIMIT_BUFFER starts throttling when this close to the short-window limit.
RATE_LIMIT_NEAR_LIMIT_BUFFER = 3
# RATE_LIMIT_THROTTLE_SECONDS is the pause applied on 429s or near-limit signals.
RATE_LIMIT_THROTTLE_SECONDS = 15

# Retry/backoff behaviour for the Strava fetch loops.
# STRAVA_MAX_RETRIES covers network failures, 5xx, or bad payloads.
STRAVA_MAX_RETRIES = 3
# STRAVA_BACKOFF_MAX_SECONDS caps the exponential backoff per attempt.
STRAVA_BACKOFF_MAX_SECONDS = 4.0


# ---------------------------------------------------------------------------
# Distance competition options
# ---------------------------------------------------------------------------
# Write a distance sheet even when a window has no rows.
DISTANCE_CREATE_EMPTY_WINDOW_SHEETS = False

# Keep distance sheets in a fixed column order.
DISTANCE_ENFORCE_COLUMN_ORDER = True

# Column order used when enforcement is enabled. Missing columns are ignored.
DISTANCE_COLUMN_ORDER = [
    "Runner",
    "Team",
    "Runs",
    "Total Distance (km)",
    "Total Elev Gain (m)",
    "Avg Distance per Run (km)",  # only present in summary sheet
]


# ---------------------------------------------------------------------------
# Excel formatting
# ---------------------------------------------------------------------------
# Keep segment sheets in a fixed column order.
SEGMENT_ENFORCE_COLUMN_ORDER = True
SEGMENT_COLUMN_ORDER = [
    "Team",
    "Runner",
    "Rank",
    "Team Rank",
    "Attempts",
    "Fastest Time (sec)",
    "Fastest Date",
]

# Automatically size columns after writing each sheet (openpyxl only).
EXCEL_AUTOSIZE_COLUMNS = True
EXCEL_AUTOSIZE_MAX_WIDTH = 50  # characters
EXCEL_AUTOSIZE_MIN_WIDTH = 6  # characters
EXCEL_AUTOSIZE_PADDING = 2  # extra characters added to the detected max
EXCEL_AUTOSIZE_MAX_ROWS = (
    5000  # skip autosize for very large sheets (performance guard)
)


# ---------------------------------------------------------------------------
# Segment matching tolerances
# ---------------------------------------------------------------------------
# These values drive the GPS fallback matcher. Set an environment variable with
# the same name to override any default without editing this file.

# Maximum deviation (metres) allowed when simplifying segment geometry.
MATCHING_SIMPLIFICATION_TOLERANCE_M = _env_float(
    "MATCHING_SIMPLIFICATION_TOLERANCE_M", 7.5
)

# Target spacing (metres) for the resampled segment polyline.
MATCHING_RESAMPLE_INTERVAL_M = _env_float("MATCHING_RESAMPLE_INTERVAL_M", 5.0)

# Distance (metres) around the start used when trimming coverage and timing.
MATCHING_START_TOLERANCE_M = _env_float("MATCHING_START_TOLERANCE_M", 25.0)

# Baseline distance (metres) for discrete Fréchet similarity checks.
MATCHING_FRECHET_TOLERANCE_M = _env_float("MATCHING_FRECHET_TOLERANCE_M", 20.0)

# Minimum portion of the segment an activity must cover to count as a match.
MATCHING_COVERAGE_THRESHOLD = _env_float("MATCHING_COVERAGE_THRESHOLD", 0.99)

# Safety caps on simplified and resampled point counts.
MATCHING_MAX_SIMPLIFIED_POINTS = _env_int("MATCHING_MAX_SIMPLIFIED_POINTS", 2000)
MATCHING_MAX_RESAMPLED_POINTS = _env_int("MATCHING_MAX_RESAMPLED_POINTS", 1200)

# Cache size for prepared segment geometry objects.
MATCHING_CACHE_MAX_ENTRIES = _env_int("MATCHING_CACHE_MAX_ENTRIES", 64)

# Global switch for the fallback matcher.
MATCHING_FALLBACK_ENABLED = _env_bool("MATCHING_FALLBACK_ENABLED", True)

# Largest allowed perpendicular offset (metres) when evaluating coverage.
MATCHING_MAX_OFFSET_M = _env_float("MATCHING_MAX_OFFSET_M", 30.0)

# Accepted activity types for matcher fallback. Provide a comma-separated list or leave
# empty to disable filtering (e.g., "Run,Ride"). Values are normalised case-insensitively.
_MATCHING_ACTIVITY_REQUIRED_TYPES_RAW = os.getenv(
    "MATCHING_ACTIVITY_REQUIRED_TYPE", "Run"
)
MATCHING_ACTIVITY_REQUIRED_TYPES = [
    item.strip()
    for item in _MATCHING_ACTIVITY_REQUIRED_TYPES_RAW.split(",")
    if item.strip()
]

# Minimum fraction of the segment distance an activity must cover before running the
# matcher (e.g., 0.6 requires the activity distance to be at least 60% of the segment).
MATCHING_ACTIVITY_MIN_DISTANCE_RATIO = _env_float(
    "MATCHING_ACTIVITY_MIN_DISTANCE_RATIO", 0.6
)

# Maximum number of activity streams to keep in the in-memory matcher cache.
MATCHING_ACTIVITY_STREAM_CACHE_SIZE = _env_int(
    "MATCHING_ACTIVITY_STREAM_CACHE_SIZE", 32
)
