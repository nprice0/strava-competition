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
# API capture / replay settings
# ---------------------------------------------------------------------------
# Enable writing Strava API responses to disk for offline replay. This should
# only be used in trusted environments because payloads include personal data
# such as activity names and potentially tokens.
STRAVA_API_CAPTURE_ENABLED = _env_bool("STRAVA_API_CAPTURE_ENABLED", True)

# Enable serving responses from disk instead of calling Strava. When enabled,
# missing files result in a cache miss and a live request when capture is also
# enabled; otherwise the code raises an exception.
STRAVA_API_REPLAY_ENABLED = _env_bool("STRAVA_API_REPLAY_ENABLED", True)

# Maximum age (days) before a cached activity response is considered stale and
# automatically refreshed from the live API. Set to 0 to disable the TTL.
REPLAY_CACHE_TTL_DAYS = _env_int("REPLAY_CACHE_TTL_DAYS", 90)

# When replayed activity windows contain no entries, force a live refetch after
# this many seconds to avoid missing newly recorded efforts. Set to 0 to disable.
REPLAY_EMPTY_WINDOW_REFRESH_SECONDS = _env_int(
    "REPLAY_EMPTY_WINDOW_REFRESH_SECONDS", 1 * 3600
)

# Guardrail limiting how far back a replayed cache may attempt to "tail" fill
# before falling back to a full live fetch. Set to 0 to disable.
REPLAY_MAX_LOOKBACK_DAYS = _env_int("REPLAY_MAX_LOOKBACK_DAYS", 365)

# Small overlap (seconds) applied when requesting the live tail window to avoid
# missing activities that start near the cached boundary.
REPLAY_EPSILON_SECONDS = _env_int("REPLAY_EPSILON_SECONDS", 60)

# Directory (absolute or relative) where captured responses are stored. The
# recorder organises files into subfolders using the request signature hash.
STRAVA_API_CAPTURE_DIR = os.getenv("STRAVA_API_CAPTURE_DIR", "strava_api_capture")

# Overwrite an existing capture file when recording new data. Defaults to
# False to keep the first successful response unless behaviour is explicitly
# requested otherwise.
STRAVA_API_CAPTURE_OVERWRITE = _env_bool("STRAVA_API_CAPTURE_OVERWRITE", False)

# Hash capture identifiers (runner IDs, etc.) before writing file paths.
STRAVA_CAPTURE_HASH_IDENTIFIERS = _env_bool("STRAVA_CAPTURE_HASH_IDENTIFIERS", True)
STRAVA_CAPTURE_ID_SALT = os.getenv("STRAVA_CAPTURE_ID_SALT", "")

# Redact sensitive fields before persisting payloads to disk.
STRAVA_CAPTURE_REDACT_PII = _env_bool("STRAVA_CAPTURE_REDACT_PII", True)
_redact_defaults = "access_token,refresh_token,token,athlete,email"
STRAVA_CAPTURE_REDACT_FIELDS = {
    field.strip().lower()
    for field in os.getenv("STRAVA_CAPTURE_REDACT_FIELDS", _redact_defaults).split(",")
    if field.strip()
}

# Automatically prune capture files older than this many days. Set to 0 to
# disable automatic retention (manual pruning via capture_gc remains available).
STRAVA_CAPTURE_AUTO_PRUNE_DAYS = _env_int("STRAVA_CAPTURE_AUTO_PRUNE_DAYS", 0)

# Token responses include highly sensitive data and typically do not need to
# be captured for offline analysis. Disable capture/replay for the OAuth token
# exchange unless explicitly opted in.
STRAVA_TOKEN_CAPTURE_ENABLED = _env_bool("STRAVA_TOKEN_CAPTURE_ENABLED", False)

# When enabled, the tool never makes live Strava requests; a cache miss raises
# an error instead of falling back to HTTP. Useful for deterministic offline
# runs.
STRAVA_OFFLINE_MODE = _env_bool("STRAVA_OFFLINE_MODE", False)

# ---------------------------------------------------------------------------
# Performance tuning
# ---------------------------------------------------------------------------
# Threads used per segment when fetching efforts in parallel.
MAX_WORKERS = 4

# Maximum parallel Strava runner fetches when preloading activity windows.
REPLAY_MAX_PARALLELISM = _env_int("REPLAY_MAX_PARALLELISM", 4)

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
    "Fastest Time (h:mm:ss)",
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
# Segment visualization tolerances
# ---------------------------------------------------------------------------
# These values are used for GPS-based visualization and coverage analysis.

# Maximum deviation (metres) allowed when simplifying segment geometry.
GEOMETRY_SIMPLIFICATION_TOLERANCE_M = _env_float(
    "GEOMETRY_SIMPLIFICATION_TOLERANCE_M", 7.5
)

# Target spacing (metres) for the resampled segment polyline.
GEOMETRY_RESAMPLE_INTERVAL_M = _env_float("GEOMETRY_RESAMPLE_INTERVAL_M", 5.0)

# Safety caps on simplified and resampled point counts.
GEOMETRY_MAX_SIMPLIFIED_POINTS = _env_int("GEOMETRY_MAX_SIMPLIFIED_POINTS", 2000)
GEOMETRY_MAX_RESAMPLED_POINTS = _env_int("GEOMETRY_MAX_RESAMPLED_POINTS", 1200)

# Global switch for the activity scan fallback.
USE_ACTIVITY_SCAN_FALLBACK = _env_bool("USE_ACTIVITY_SCAN_FALLBACK", True)

ACTIVITY_SCAN_MAX_ACTIVITY_PAGES: int | None = _env_int(
    "ACTIVITY_SCAN_MAX_ACTIVITY_PAGES",
    0,
)
if (
    ACTIVITY_SCAN_MAX_ACTIVITY_PAGES is not None
    and ACTIVITY_SCAN_MAX_ACTIVITY_PAGES <= 0
):
    ACTIVITY_SCAN_MAX_ACTIVITY_PAGES = None

ACTIVITY_SCAN_CAPTURE_INCLUDE_ALL_EFFORTS = _env_bool(
    "ACTIVITY_SCAN_CAPTURE_INCLUDE_ALL_EFFORTS", True
)

# When True, always bypass Strava efforts and use activity scan. Useful for debugging.
FORCE_ACTIVITY_SCAN_FALLBACK = _env_bool("FORCE_ACTIVITY_SCAN_FALLBACK", False)

# Maximum number of activity streams to keep in the in-memory cache.
ACTIVITY_STREAM_CACHE_SIZE = _env_int("ACTIVITY_STREAM_CACHE_SIZE", 64)

# Maximum number of runner activity windows cached during segment processing.
RUNNER_ACTIVITY_CACHE_SIZE = _env_int("RUNNER_ACTIVITY_CACHE_SIZE", 256)
