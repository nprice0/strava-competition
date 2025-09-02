import logging
import random
import threading
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .auth import get_access_token
from .config import (
    STRAVA_BASE_URL,
    HTTP_POOL_CONNECTIONS,
    HTTP_POOL_MAXSIZE,
    REQUEST_TIMEOUT,
    RATE_LIMIT_MAX_CONCURRENT,
    RATE_LIMIT_JITTER_RANGE,
    RATE_LIMIT_NEAR_LIMIT_BUFFER,
    RATE_LIMIT_THROTTLE_SECONDS,
)

# Reusable HTTP session with retries and backoff for reliability and performance
_session = requests.Session()
_retry = Retry(
    total=3,
    backoff_factor=1.0,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET", "POST"],
)
# Increase connection pool for parallel requests
_adapter = HTTPAdapter(
    pool_connections=HTTP_POOL_CONNECTIONS,
    pool_maxsize=HTTP_POOL_MAXSIZE,
    max_retries=_retry,
)
_session.mount("https://", _adapter)
_session.mount("http://", _adapter)
_session.headers.update({"Accept-Encoding": "gzip, deflate"})

DEFAULT_TIMEOUT = REQUEST_TIMEOUT


class RateLimiter:
    """A simple, shared, rate-limit-aware gate for concurrent HTTP calls.

    - Limits concurrent in-flight requests (semaphore).
    - Adds small jitter to avoid bursts.
    - When near the short-window limit (from Strava headers) or on 429, throttles for a period.
    """

    def __init__(self, max_concurrent: int = RATE_LIMIT_MAX_CONCURRENT, jitter_range: tuple[float, float] = RATE_LIMIT_JITTER_RANGE):
        self._sem = threading.Semaphore(max_concurrent)
        self._lock = threading.Lock()
        self._throttle_until = 0.0
        self._jitter_range = jitter_range
        self._near_limit_buffer = RATE_LIMIT_NEAR_LIMIT_BUFFER

    def before_request(self):
        self._sem.acquire()
        # Respect any global throttle window
        with self._lock:
            wait_for = max(0.0, self._throttle_until - time.time())
        if wait_for > 0:
            time.sleep(wait_for)
        # Add tiny jitter to smooth spikes
        lo, hi = self._jitter_range
        if hi > 0:
            time.sleep(random.uniform(lo, hi))

    def after_response(self, headers: dict | None, status_code: int | None):
        try:
            # Throttle if we hit 429
            if status_code == 429:
                logging.warning(
                    "Rate limit: 429. Throttling %ss.",
                    RATE_LIMIT_THROTTLE_SECONDS,
                )
                self._set_throttle(RATE_LIMIT_THROTTLE_SECONDS)
            # Inspect rate-limit headers and pre-emptively slow if close
            short_used, short_limit = None, None
            if headers:
                usage = headers.get("X-RateLimit-Usage")
                limit = headers.get("X-RateLimit-Limit")
                if usage and limit:
                    try:
                        short_used = int(str(usage).split(",")[0])
                        short_limit = int(str(limit).split(",")[0])
                    except Exception:
                        short_used, short_limit = None, None
            if short_used is not None and short_limit is not None:
                if short_used >= max(short_limit - self._near_limit_buffer, 0):
                    logging.info(
                        "Approaching short-window limit (%s/%s). Throttling %ss.",
                        short_used,
                        short_limit,
                        RATE_LIMIT_THROTTLE_SECONDS,
                    )
                    self._set_throttle(RATE_LIMIT_THROTTLE_SECONDS)
        finally:
            self._sem.release()

    def _set_throttle(self, seconds: float):
        with self._lock:
            self._throttle_until = max(self._throttle_until, time.time() + seconds)


# Shared limiter instance
_limiter = RateLimiter()


def set_rate_limiter(max_concurrent: int | None = None):
    """Optionally reconfigure the global rate limiter at runtime.

    If max_concurrent is provided, replaces the limiter with a new instance.
    """
    global _limiter
    if max_concurrent is not None:
        _limiter = RateLimiter(max_concurrent=max_concurrent)


def get_segment_efforts(runner, segment_id, start_date, end_date):
    """Fetch all efforts for a segment within dates for a runner, with pagination and retry.

    - Reuses cached access_token on the runner.
    - Refreshes token once on 401 and retries.
    - Paginates through all results (per_page=200).
    - Applies retry/backoff and timeouts.
    """

    def auth_headers():
        return {"Authorization": f"Bearer {runner.access_token}"}

    def ensure_token():
        if not runner.access_token:
            access_token, new_refresh_token = get_access_token(
                runner.refresh_token, runner_name=runner.name
            )
            runner.access_token = access_token
            if new_refresh_token and new_refresh_token != runner.refresh_token:
                runner.refresh_token = new_refresh_token

    def fetch_page(page: int):
        url = f"{STRAVA_BASE_URL}/segment_efforts"
        params = {
            "segment_id": segment_id,
            "start_date_local": start_date.isoformat(),
            "end_date_local": end_date.isoformat(),
            "per_page": 200,
            "page": page,
        }
        # Enter limiter, make request, then update limiter based on response
        _limiter.before_request()
        resp = _session.get(url, headers=auth_headers(), params=params, timeout=DEFAULT_TIMEOUT)
        _limiter.after_response(resp.headers, resp.status_code)
        # If still 429 after limiter, raise to upstream for handling
        if resp.status_code == 429:
            resp.raise_for_status()
        resp.raise_for_status()
        return resp.json()

    try:
        ensure_token()
        all_efforts = []
        page = 1
        while True:
            try:
                data = fetch_page(page)
            except requests.exceptions.HTTPError as e:
                # One-time 401 retry by refreshing token
                if e.response is not None and e.response.status_code == 401:
                    logging.info(
                        f"401 for runner {runner.name}. Refreshing token and retrying page {page}."
                    )
                    runner.access_token = None
                    ensure_token()
                    data = fetch_page(page)
                else:
                    raise

            if not data:
                break
            all_efforts.extend(data)
            if len(data) < 200:
                break
            page += 1
        return all_efforts
    except requests.exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 401:
            logging.warning(
                f"Skipping runner {runner.name}: Unauthorized (invalid/expired token after retry)"
            )
            return None
        elif e.response is not None and e.response.status_code == 402:
            logging.warning(
                f"Runner {runner.name}: Payment required for segment efforts. Skipping."
            )
            return None
        else:
            logging.error(f"Error for runner {runner.name}: {e}")
            return None
