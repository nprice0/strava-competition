"""Strava API client helpers (HTTP session, rate limiting, effort fetching).

Public surface used elsewhere:
        - get_segment_efforts(runner, segment_id, start_date, end_date)
        - set_rate_limiter(max_concurrent=None)  # adjusts concurrency at runtime

Design notes (concise):
        * A single shared ``requests.Session`` provides connection pooling + retries.
        * ``RateLimiter`` enforces a soft cap on concurrent in-flight calls and can
            be resized safely without recreating global objects (important for threads
            already waiting). Resizing is immediate for increases; decreases take
            effect as in‑flight requests complete.
        * Light jitter is applied to reduce synchronized burst patterns.
        * Short-window Strava usage headers are inspected: if approaching the
            limit, a brief throttle window is set to smooth traffic.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, TypeAlias

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
from .models import Runner

# Type aliases for clarity
JSONObj: TypeAlias = Dict[str, Any]
JSONList: TypeAlias = List[JSONObj]

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
_session.headers.update({
    "Accept-Encoding": "gzip, deflate",
    "Accept": "application/json",
})

DEFAULT_TIMEOUT: int = REQUEST_TIMEOUT


class RateLimiter:
    """Concurrency + adaptive pacing primitive for API calls.

    Core behaviors:
      * Soft concurrency limit (``_max_allowed``) enforced with a condition
        variable; threads wait while ``_in_flight`` >= limit.
      * ``resize(new_max)`` changes the soft limit in-place (no swapping), so
        waiting threads continue seamlessly. Upsizing wakes all waiters; downsizing
        is naturally honored as active requests finish.
      * Optional throttle window: triggered by HTTP 429 or nearing the reported
        short-window usage limit; new requests sleep until the window expires.
      * Small random jitter further staggers request timing to avoid clumping.
    """

    def __init__(
        self,
        max_concurrent: int = RATE_LIMIT_MAX_CONCURRENT,
        jitter_range: tuple[float, float] = RATE_LIMIT_JITTER_RANGE,
    ) -> None:
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._max_allowed = max_concurrent
        self._in_flight = 0
        self._throttle_until: float = 0.0
        self._jitter_range = jitter_range
        self._near_limit_buffer = RATE_LIMIT_NEAR_LIMIT_BUFFER

    # --- Public control -------------------------------------------------
    def resize(self, new_max: int) -> None:
        """Adjust maximum concurrent requests (soft limit) at runtime."""
        if new_max < 1:
            raise ValueError("new_max must be >= 1")
        with self._cond:
            old = self._max_allowed
            self._max_allowed = new_max
            self._cond.notify_all()  # Wake waiters if limit grew
        logging.info("RateLimiter resized from %s to %s", old, new_max)

    # --- Request lifecycle ---------------------------------------------
    def before_request(self) -> None:
        with self._cond:
            while self._in_flight >= self._max_allowed:
                self._cond.wait()
            self._in_flight += 1
            wait_for = max(0.0, self._throttle_until - time.time())
        if wait_for > 0:
            time.sleep(wait_for)
        lo, hi = self._jitter_range
        if hi > 0:
            time.sleep(random.uniform(lo, hi))

    def after_response(self, headers: Optional[Dict[str, Any]], status_code: Optional[int]) -> None:
        throttle = False
        if status_code == 429:
            throttle = True
            logging.warning(
                "Rate limit: 429. Throttling %ss.", RATE_LIMIT_THROTTLE_SECONDS
            )
        else:
            short_used = short_limit = None
            if headers:
                usage = headers.get("X-RateLimit-Usage")
                limit = headers.get("X-RateLimit-Limit")
                if usage and limit:
                    try:
                        short_used = int(str(usage).split(",")[0])
                        short_limit = int(str(limit).split(",")[0])
                    except Exception:
                        short_used = short_limit = None
            if (
                short_used is not None
                and short_limit is not None
                and short_used >= max(short_limit - self._near_limit_buffer, 0)
            ):
                throttle = True
                logging.info(
                    "Approaching short-window limit (%s/%s). Throttling %ss.",
                    short_used,
                    short_limit,
                    RATE_LIMIT_THROTTLE_SECONDS,
                )
        if throttle:
            self._set_throttle(RATE_LIMIT_THROTTLE_SECONDS)
        with self._cond:  # Release slot
            self._in_flight -= 1
            self._cond.notify()

    # --- Internals ------------------------------------------------------
    def _set_throttle(self, seconds: float) -> None:
        with self._lock:
            self._throttle_until = max(self._throttle_until, time.time() + seconds)

    # --- Introspection --------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:  # pragma: no cover (not yet used)
        with self._lock:
            return {
                "max_allowed": self._max_allowed,
                "in_flight": self._in_flight,
                "throttle_until": self._throttle_until,
            }


# Shared limiter instance
_limiter = RateLimiter()


def set_rate_limiter(max_concurrent: Optional[int] = None) -> None:
    """Adjust concurrency limit at runtime.

    Passing ``max_concurrent`` calls ``RateLimiter.resize``; omitted / None
    leaves the current limit unchanged.
    """
    if max_concurrent is not None:
        try:
            _limiter.resize(max_concurrent)
        except Exception as e:
            logging.error("Failed to resize rate limiter: %s", e)


def get_segment_efforts(
    runner: Runner,
    segment_id: int,
    start_date: datetime,
    end_date: datetime,
) -> Optional[List[Dict[str, Any]]]:
    """Fetch all efforts for a segment window for one runner.

    Behavior:
      * Uses cached runner.access_token (fetches via refresh token if absent)
      * One retry on first 401 (refresh + reattempt same page)
      * Paginates (per_page=200) until fewer than 200 returned
      * Applies timeout / retries from the configured Session
    Returns a list (possibly empty) or None on unrecoverable error.
    """

    def auth_headers() -> Dict[str, str]:
        return {"Authorization": f"Bearer {runner.access_token}"}

    def ensure_token() -> None:
        if not runner.access_token:
            access_token, new_refresh_token = get_access_token(
                runner.refresh_token, runner_name=runner.name
            )
            runner.access_token = access_token
            if new_refresh_token and new_refresh_token != runner.refresh_token:
                runner.refresh_token = new_refresh_token

    def fetch_page(page: int) -> JSONList:
        url = f"{STRAVA_BASE_URL}/segment_efforts"
        params = {
            "segment_id": segment_id,
            "start_date_local": start_date.isoformat(),
            "end_date_local": end_date.isoformat(),
            "per_page": 200,
            "page": page,
        }
        _limiter.before_request()
        resp = _session.get(
            url, headers=auth_headers(), params=params, timeout=DEFAULT_TIMEOUT
        )
        _limiter.after_response(resp.headers, resp.status_code)
        if resp.status_code == 429:
            resp.raise_for_status()
        resp.raise_for_status()
        try:
            data = resp.json()
        except ValueError:
            logging.error(
                "Non-JSON response for segment efforts runner=%s segment=%s page=%s",
                runner.name,
                segment_id,
                page,
            )
            return []
        if not isinstance(data, list):  # Defensive: Strava expected list
            logging.warning(
                "Unexpected JSON shape (not list) for efforts runner=%s segment=%s page=%s type=%s",
                runner.name,
                segment_id,
                page,
                type(data).__name__,
            )
            return []
        return data  # type: ignore[return-value]

    try:
        ensure_token()
        all_efforts: List[Dict[str, Any]] = []
        page = 1
        attempted_refresh = False  # Guard against infinite 401 refresh loops
        while True:
            try:
                data = fetch_page(page)
            except requests.exceptions.HTTPError as e:
                if (
                    e.response is not None
                    and e.response.status_code == 401
                    and not attempted_refresh
                ):
                    logging.info(
                        f"401 for runner {runner.name}. Refreshing token and retrying page {page}."
                    )
                    runner.access_token = None
                    ensure_token()
                    attempted_refresh = True
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
    except requests.exceptions.HTTPError as e:  # pragma: no cover
        resp = e.response
        if resp is None:
            logging.error(
                f"HTTPError with no response object (network/transport issue) for runner {runner.name}: {e}"
            )
            return None
        # Diagnostic summary (INFO) + optional DEBUG body sample
        try:
            content_len = len(resp.content)
        except Exception:
            content_len = -1
        logging.info(
            "HTTPError summary | runner=%s | status=%s | content_type=%s | length=%s",
            runner.name,
            resp.status_code,
            resp.headers.get("Content-Type"),
            content_len,
        )
        if logging.getLogger().isEnabledFor(logging.DEBUG):  # Avoid building string unless needed
            try:
                logging.debug(
                    "HTTPError body sample (first 500 chars): %r", resp.text[:500]
                )
            except Exception:
                pass
        detail = _extract_error(resp)
        status = resp.status_code
        raw_snippet = None
        if not detail:
            try:
                txt = (resp.text or "").strip()
                if txt:
                    raw_snippet = (txt[:297] + "...") if len(txt) > 300 else txt
            except Exception:
                pass
        suffix = ""
        if detail and raw_snippet:
            suffix = f" | {detail} | raw: {raw_snippet}"
        elif detail:
            suffix = f" | {detail}"
        elif raw_snippet:
            suffix = f" | raw: {raw_snippet}"
        if status == 401:
            logging.warning(
                f"Skipping runner {runner.name}: Unauthorized (invalid/expired token after retry){suffix}"
            )
        elif status == 402:
            # Mark runner so future segment processing can skip further calls
            try:
                runner.payment_required = True  # type: ignore[attr-defined]
            except Exception:
                pass
            logging.warning(
                f"Runner {runner.name}: 402 Payment Required (likely subscription needed or access restricted){suffix}. Skipping."
            )
        else:
            logging.error(
                f"Error for runner {runner.name}: HTTP {status}{suffix}"
            )
        return None


def _extract_error(resp: Optional[requests.Response]) -> Optional[str]:
    """Return compact string with Strava error info (message + codes) if present.

    Expected JSON form:
      {"message": "...", "errors": [{"resource": .., "field": .., "code": ..}, ...]}
    Falls back to a trimmed plain-text body if JSON parse fails.
    """
    if not resp:
        return None
    try:
        data = resp.json()
    except Exception:
        try:
            txt = resp.text.strip()
            if txt:
                return (txt[:297] + "...") if len(txt) > 300 else txt
        except Exception:
            pass
        return None
    if not isinstance(data, dict):  # Unexpected shape
        return None
    parts: List[str] = []
    msg = data.get("message")
    if msg:
        parts.append(str(msg))
    errs = data.get("errors")
    if isinstance(errs, list):
        for err in errs:
            if not isinstance(err, dict):
                continue
            resource = err.get("resource")
            field = err.get("field")
            code = err.get("code")
            spec = "/".join([p for p in [resource, field] if p])
            if code and spec:
                parts.append(f"{spec}:{code}")
            elif code:
                parts.append(str(code))
    return " | ".join(parts) if parts else None
