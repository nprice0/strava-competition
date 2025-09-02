"""Central configuration for the Strava Segment Competition tool.

All values are constants imported by the rest of the package. Adjust as needed
for your environment. Secrets are read from environment variables (optionally
via a local `.env`).
"""

from __future__ import annotations

import os
import importlib

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
MAX_WORKERS = 8  # threads per segment when fetching runner efforts

# HTTP session connection pools (for parallel requests)
HTTP_POOL_CONNECTIONS = 20
HTTP_POOL_MAXSIZE = 20

# Request timeout (seconds)
REQUEST_TIMEOUT = 15

# Rate limiter settings
RATE_LIMIT_MAX_CONCURRENT = 8  # cap in-flight HTTP requests globally
RATE_LIMIT_JITTER_RANGE = (0.05, 0.2)  # seconds; random jitter to smooth bursts
RATE_LIMIT_NEAR_LIMIT_BUFFER = 3  # start throttling when within this many calls of the short-window limit
RATE_LIMIT_THROTTLE_SECONDS = 15  # throttle window applied on 429 or when near limit

