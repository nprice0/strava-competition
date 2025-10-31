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


# Optionally load variables from a local .env file (if python-dotenv is installed)
_load_dotenv = None
try:
    _dotenv_mod = importlib.import_module("dotenv")
    _load_dotenv = getattr(_dotenv_mod, "load_dotenv", None)
except Exception:
    _load_dotenv = None

if callable(_load_dotenv):
    # Loads .env from current working directory or parents
    _load_dotenv()


# ---------------------------------------------------------------------------
# Input/Output
# ---------------------------------------------------------------------------
# Can be absolute or relative paths
INPUT_FILE = "competition_input.xlsx"
OUTPUT_FILE = "competition_results"

# If True, appends _YYYYMMDD_HHMMSS to the output filename
OUTPUT_FILE_TIMESTAMP_ENABLED = True


# ---------------------------------------------------------------------------
# Strava settings
# ---------------------------------------------------------------------------
# API endpoints
STRAVA_BASE_URL = "https://www.strava.com/api/v3"
STRAVA_OAUTH_URL = "https://www.strava.com/oauth/token"

# Client credentials (read from environment; do not hardcode secrets)
CLIENT_ID = os.getenv("STRAVA_CLIENT_ID", "")
CLIENT_SECRET = os.getenv("STRAVA_CLIENT_SECRET", "")


# ---------------------------------------------------------------------------
# Performance tuning
# ---------------------------------------------------------------------------
# Concurrency for per-segment parallel effort fetching
MAX_WORKERS = 4  # threads per segment when fetching runner efforts

# HTTP session connection pools (for parallel requests)
HTTP_POOL_CONNECTIONS = 20
HTTP_POOL_MAXSIZE = 20

# Request timeout (seconds)
REQUEST_TIMEOUT = 15

# Rate limiter settings
RATE_LIMIT_MAX_CONCURRENT = 8  # cap in-flight HTTP requests globally
RATE_LIMIT_JITTER_RANGE = (0.05, 0.2)  # seconds; random jitter to smooth bursts
RATE_LIMIT_NEAR_LIMIT_BUFFER = (
    3  # start throttling when within this many calls of the short-window limit
)
RATE_LIMIT_THROTTLE_SECONDS = 15  # throttle window applied on 429 or when near limit

# Retry/backoff settings for Strava API (used by fetch loops)
STRAVA_MAX_RETRIES = 3  # attempts per page (network/5xx/HTML/non-JSON)
STRAVA_BACKOFF_MAX_SECONDS = 4.0  # cap for exponential backoff per attempt


# ---------------------------------------------------------------------------
# Distance competition options
# ---------------------------------------------------------------------------
# When True, create a distance window sheet even if a specific window produced zero rows.
DISTANCE_CREATE_EMPTY_WINDOW_SHEETS = False

# Enforce canonical column ordering for distance sheets.
DISTANCE_ENFORCE_COLUMN_ORDER = True

# Column order applied if enforcement enabled (missing columns ignored gracefully)
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
# Enforce canonical column ordering for segment sheets.
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

# Automatically size column widths after writing each sheet (openpyxl only)
EXCEL_AUTOSIZE_COLUMNS = True
EXCEL_AUTOSIZE_MAX_WIDTH = 50  # characters
EXCEL_AUTOSIZE_MIN_WIDTH = 6  # characters
EXCEL_AUTOSIZE_PADDING = 2  # extra chars added to detected max
EXCEL_AUTOSIZE_MAX_ROWS = 5000  # skip autosize for very large sheets (perf guard)


# ---------------------------------------------------------------------------
# Segment matching tolerances
# ---------------------------------------------------------------------------
# These configure the GPS-based matching fallback. Values can be overridden via
# environment variables if experimentation is required without code changes.
MATCHING_SIMPLIFICATION_TOLERANCE_M = _env_float(
    "MATCHING_SIMPLIFICATION_TOLERANCE_M", 7.5
)
MATCHING_RESAMPLE_INTERVAL_M = _env_float("MATCHING_RESAMPLE_INTERVAL_M", 5.0)
MATCHING_START_TOLERANCE_M = _env_float("MATCHING_START_TOLERANCE_M", 30.0)
MATCHING_FRECHET_TOLERANCE_M = _env_float("MATCHING_FRECHET_TOLERANCE_M", 20.0)
MATCHING_COVERAGE_THRESHOLD = _env_float("MATCHING_COVERAGE_THRESHOLD", 0.95)
MATCHING_MAX_SIMPLIFIED_POINTS = _env_int("MATCHING_MAX_SIMPLIFIED_POINTS", 2000)
MATCHING_MAX_RESAMPLED_POINTS = _env_int("MATCHING_MAX_RESAMPLED_POINTS", 1200)
MATCHING_CACHE_MAX_ENTRIES = _env_int("MATCHING_CACHE_MAX_ENTRIES", 64)
MATCHING_FALLBACK_ENABLED = _env_bool("MATCHING_FALLBACK_ENABLED", True)
MATCHING_MAX_OFFSET_M = _env_float("MATCHING_MAX_OFFSET_M", 100.0)
