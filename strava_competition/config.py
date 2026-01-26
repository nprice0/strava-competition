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
# Paths can be absolute or relative. Default to current directory for distribution.
INPUT_FILE = os.getenv("INPUT_FILE", "competition_input.xlsx")
OUTPUT_FILE = os.getenv("OUTPUT_FILE", "competition_results")

# Append _YYYYMMDD_HHMMSS to the output name when True.
OUTPUT_FILE_TIMESTAMP_ENABLED = _env_bool("OUTPUT_FILE_TIMESTAMP_ENABLED", True)


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
# API cache settings
# ---------------------------------------------------------------------------
# Cache mode controls how the app interacts with Strava API responses on disk.
#   live    = Always call Strava, never use or save cache
#   cache   = Use cache + fetch new data since last run (default)
#   offline = Cache only, fail if data is missing
STRAVA_API_CACHE_MODE = os.getenv("STRAVA_API_CACHE_MODE", "cache").strip().lower()

if STRAVA_API_CACHE_MODE not in {"live", "cache", "offline"}:
    raise ValueError(
        f"Invalid STRAVA_API_CACHE_MODE: '{STRAVA_API_CACHE_MODE}'. "
        "Must be 'live', 'cache', or 'offline'."
    )

# Derived flags for internal use
_cache_mode_saves = STRAVA_API_CACHE_MODE == "cache"
_cache_mode_reads = STRAVA_API_CACHE_MODE in {"cache", "offline"}
_cache_mode_offline = STRAVA_API_CACHE_MODE == "offline"

# Maximum age (days) before a cached activity response is considered stale and
# automatically refreshed from the live API. Set to 0 to disable the TTL.
CACHE_TTL_DAYS = _env_int("CACHE_TTL_DAYS", 90)

# When replayed activity windows contain no entries, force a live refetch after
# this many seconds to avoid missing newly recorded efforts. Set to 0 to disable.
CACHE_EMPTY_REFRESH_SECONDS = _env_int("CACHE_EMPTY_REFRESH_SECONDS", 1 * 3600)

# Guardrail limiting how far back a replayed cache may attempt to "tail" fill
# before falling back to a full live fetch. Set to 0 to disable.
CACHE_MAX_LOOKBACK_DAYS = _env_int("CACHE_MAX_LOOKBACK_DAYS", 365)

# Small overlap (seconds) applied when requesting the live tail window to avoid
# missing activities that start near the cached boundary.
CACHE_TAIL_OVERLAP_SECONDS = _env_int("CACHE_TAIL_OVERLAP_SECONDS", 60)

# Directory (absolute or relative) where cached responses are stored. The
# recorder organises files into subfolders using the request signature hash.
STRAVA_CACHE_DIR = os.getenv("STRAVA_CACHE_DIR", "strava_cache")

# Overwrite an existing cache file when recording new data. Defaults to
# False to keep the first successful response unless behaviour is explicitly
# requested otherwise.
STRAVA_CACHE_OVERWRITE = _env_bool("STRAVA_CACHE_OVERWRITE", False)

# Hash cache identifiers (runner IDs, etc.) before writing file paths.
STRAVA_CACHE_HASH_IDENTIFIERS = _env_bool("STRAVA_CACHE_HASH_IDENTIFIERS", True)
STRAVA_CACHE_ID_SALT = os.getenv("STRAVA_CACHE_ID_SALT", "")

# Redact sensitive fields before persisting payloads to disk.
# Supports dot-notation for nested fields (e.g., "athlete.firstname").
# Simple field names match at any nesting level.
STRAVA_CACHE_REDACT_PII = _env_bool("STRAVA_CACHE_REDACT_PII", True)
_redact_defaults = (
    # Tokens and credentials
    "access_token,refresh_token,token,"
    # Email anywhere
    "email,"
    # Bare field names (match at any nesting level)
    "firstname,username,sex,profile,profile_medium,"
    # Specific athlete paths
    "athlete.firstname,athlete.username,athlete.email,"
    "athlete.sex,athlete.profile,athlete.profile_medium"
)
STRAVA_CACHE_REDACT_FIELDS = {
    field.strip().lower()
    for field in os.getenv("STRAVA_CACHE_REDACT_FIELDS", _redact_defaults).split(",")
    if field.strip()
}

# Automatically prune cache files older than this many days. Set to 0 to
# disable automatic retention (manual pruning via capture_gc remains available).
STRAVA_CACHE_AUTO_PRUNE_DAYS = _env_int("STRAVA_CACHE_AUTO_PRUNE_DAYS", 0)

# ---------------------------------------------------------------------------
# Performance tuning
# ---------------------------------------------------------------------------
# Threads used per segment when fetching efforts in parallel.
MAX_WORKERS = 4

# Maximum parallel Strava runner fetches when preloading activity windows.
CACHE_REFRESH_PARALLELISM = _env_int("CACHE_REFRESH_PARALLELISM", 4)

# HTTP session pool sizes for concurrent requests.
HTTP_POOL_CONNECTIONS = 20
HTTP_POOL_MAXSIZE = 20

# Request timeout in seconds.
REQUEST_TIMEOUT = 15


# Rate limiter settings.
# RATE_LIMIT_MAX_CONCURRENT caps total in-flight requests.
# Strava's 15-minute limit is 100 requests; keep this low to avoid 429 bursts.
RATE_LIMIT_MAX_CONCURRENT = 2
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
# Segment competition options
# ---------------------------------------------------------------------------
# When enabled, rows with the same Segment ID are aggregated into a single
# output sheet showing each runner's best time across all windows. When
# disabled, each row produces its own output sheet.
SEGMENT_SPLIT_WINDOWS_ENABLED = _env_bool("SEGMENT_SPLIT_WINDOWS_ENABLED", True)


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
    "Fastest Distance (m)",
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

ACTIVITY_SCAN_CACHE_INCLUDE_ALL_EFFORTS = _env_bool(
    "ACTIVITY_SCAN_CACHE_INCLUDE_ALL_EFFORTS", True
)

# When True, always bypass Strava efforts and use activity scan. Useful for debugging.
FORCE_ACTIVITY_SCAN_FALLBACK = _env_bool("FORCE_ACTIVITY_SCAN_FALLBACK", True)

# Maximum number of activity streams to keep in the in-memory cache.
ACTIVITY_STREAM_CACHE_SIZE = _env_int("ACTIVITY_STREAM_CACHE_SIZE", 64)

# Maximum number of runner activity windows cached during segment processing.
RUNNER_ACTIVITY_CACHE_SIZE = _env_int("RUNNER_ACTIVITY_CACHE_SIZE", 256)
