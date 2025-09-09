"""Strava API helpers: HTTP session, rate limiting, pagination, and effort/activity fetch.

Public surface:
- get_segment_efforts(runner, segment_id, start_date, end_date)
- set_rate_limiter(max_concurrent=None)
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
    STRAVA_MAX_RETRIES,
    STRAVA_BACKOFF_MAX_SECONDS,
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
    """Soft concurrency cap with optional throttle and small jitter to smooth bursts."""

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


# --- Shared auth helpers -------------------------------------
def _ensure_runner_token(runner: "Runner") -> None:
    """Ensure the runner has a current access token.

    Performs a refresh using the runner's stored refresh_token if no access token
    is cached. If Strava rotates the refresh token, update the runner instance.
    """
    if not getattr(runner, "access_token", None):
        access_token, new_refresh_token = get_access_token(
            runner.refresh_token, runner_name=runner.name
        )
        runner.access_token = access_token
        if new_refresh_token and new_refresh_token != runner.refresh_token:
            runner.refresh_token = new_refresh_token
            # Attempt immediate persistence (best-effort, silent on failure)
            try:
                from .excel_writer import update_single_runner_refresh_token  # local import to avoid cycle
                from .config import INPUT_FILE
                update_single_runner_refresh_token(INPUT_FILE, runner)
            except Exception:
                pass


def _auth_headers(runner: "Runner") -> Dict[str, str]:
    return {"Authorization": f"Bearer {runner.access_token}"}


def _fetch_page_with_retries(
    runner: "Runner",
    url: str,
    params: Dict[str, Any],
    context_label: str,
    page: int,
    *,
    segment_id: Optional[int] = None,
) -> JSONList:
    """GET a single page with resilient retries/backoff and HTML/non-JSON handling.

    Returns a list (possibly empty). On final give-up for transient issues, returns [].
    Raises HTTPError for non-retriable errors (e.g., 429 beyond limiter, 4xx not handled).
    """
    attempts = 0
    backoff = 1.0
    while True:
        attempts += 1
        _limiter.before_request()
        resp = None
        try:
            resp = _session.get(
                url, headers=_auth_headers(runner), params=params, timeout=DEFAULT_TIMEOUT
            )
        except requests.RequestException as e:
            _limiter.after_response(None, None)
            if attempts < STRAVA_MAX_RETRIES:
                if segment_id is not None:
                    logging.warning(
                        "%s network error runner=%s segment=%s page=%s attempt=%s err=%s; backoff %.1fs",
                        context_label.capitalize(),
                        runner.name,
                        segment_id,
                        page,
                        attempts,
                        e.__class__.__name__,
                        backoff,
                    )
                else:
                    logging.warning(
                        "%s network error runner=%s page=%s attempt=%s err=%s; backoff %.1fs",
                        context_label.capitalize(),
                        runner.name,
                        page,
                        attempts,
                        e.__class__.__name__,
                        backoff,
                    )
                time.sleep(backoff)
                backoff = min(backoff * 2, STRAVA_BACKOFF_MAX_SECONDS)
                continue
            if segment_id is not None:
                logging.error(
                    "%s network error (giving up) runner=%s segment=%s page=%s attempts=%s err=%s",
                    context_label.capitalize(),
                    runner.name,
                    segment_id,
                    page,
                    attempts,
                    e,
                )
            else:
                logging.error(
                    "%s network error (giving up) runner=%s page=%s attempts=%s err=%s",
                    context_label.capitalize(),
                    runner.name,
                    page,
                    attempts,
                    e,
                )
            return []
        else:
            _limiter.after_response(resp.headers, resp.status_code)

        ct = resp.headers.get("Content-Type", "")
        if resp.status_code == 429:
            resp.raise_for_status()
        is_html = "text/html" in ct.lower()
        try:
            resp.raise_for_status()
        except requests.HTTPError:
            if attempts < STRAVA_MAX_RETRIES and (500 <= resp.status_code < 600 or is_html):
                if segment_id is not None:
                    logging.warning(
                        "Transient error (status=%s html=%s) for %s runner=%s page=%s attempt=%s; backing off %.1fs",
                        resp.status_code,
                        is_html,
                        context_label,
                        runner.name,
                        page,
                        attempts,
                        backoff,
                    )
                else:
                    logging.warning(
                        "Transient error (status=%s html=%s) for %s runner=%s page=%s attempt=%s; backing off %.1fs",
                        resp.status_code,
                        is_html,
                        context_label,
                        runner.name,
                        page,
                        attempts,
                        backoff,
                    )
                time.sleep(backoff)
                backoff = min(backoff * 2, STRAVA_BACKOFF_MAX_SECONDS)
                continue
            raise

        if is_html:
            if attempts < STRAVA_MAX_RETRIES:
                logging.warning(
                    "HTML downtime page for %s runner=%s page=%s attempt=%s; retrying in %.1fs",
                    context_label,
                    runner.name,
                    page,
                    attempts,
                    backoff,
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, STRAVA_BACKOFF_MAX_SECONDS)
                continue
            logging.error(
                "Giving up on HTML downtime page (%s) runner=%s page=%s after %s attempts",
                context_label,
                runner.name,
                page,
                attempts,
            )
            return []

        try:
            data = resp.json()
        except ValueError:
            if attempts < STRAVA_MAX_RETRIES:
                logging.warning(
                    "Non-JSON response (%s) runner=%s page=%s attempt=%s; retrying in %.1fs",
                    context_label,
                    runner.name,
                    page,
                    attempts,
                    backoff,
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, STRAVA_BACKOFF_MAX_SECONDS)
                continue
            logging.error(
                "Non-JSON response (%s) runner=%s page=%s after %s attempts; abandoning page",
                context_label,
                runner.name,
                page,
                attempts,
            )
            return []

        if not isinstance(data, list):
            if segment_id is not None:
                logging.warning(
                    "Unexpected JSON shape (not list) for efforts runner=%s segment=%s page=%s type=%s",
                    runner.name,
                    segment_id,
                    page,
                    type(data).__name__,
                )
            else:
                logging.warning(
                    "Unexpected JSON shape (not list) for activities runner=%s page=%s type=%s",
                    runner.name,
                    page,
                    type(data).__name__,
                )
            return []

        return data  # type: ignore[return-value]


def get_segment_efforts(
    runner: Runner,
    segment_id: int,
    start_date: datetime,
    end_date: datetime,
) -> Optional[List[Dict[str, Any]]]:
    """Fetch all efforts for a segment window for one runner.

    Behaviour:
      * Uses cached runner.access_token (fetches via refresh token if absent)
      * One retry on first 401 (refresh + reattempt same page)
      * Paginates (per_page=200) until fewer than 200 returned
      * Applies timeout / retries from the configured Session
    Returns a list (possibly empty) or None on unrecoverable error.
    """

    def auth_headers() -> Dict[str, str]:  # local alias for brevity
        return _auth_headers(runner)

    def ensure_token() -> None:
        _ensure_runner_token(runner)

    def fetch_page(page: int) -> JSONList:
        url = f"{STRAVA_BASE_URL}/segment_efforts"
        params = {
            "segment_id": segment_id,
            "start_date_local": start_date.isoformat(),
            "end_date_local": end_date.isoformat(),
            "per_page": 200,
            "page": page,
        }
        return _fetch_page_with_retries(
            runner,
            url,
            params,
            context_label="segment efforts",
            page=page,
            segment_id=segment_id,
        )

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
                f"Skipping runner {runner.name}: Unauthorised (invalid/expired token after retry){suffix}"
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


def get_activities(
    runner: Runner,
    start_date: datetime,
    end_date: datetime,
) -> Optional[List[Dict[str, Any]]]:
    """Fetch all activities for a runner in [start_date, end_date].

    Uses /athlete/activities with pagination (per_page=200). Filters locally to
    activity 'type' == 'Run'. Returns list (possibly empty) or None on error.
    Inclusive range based on activity start_date_local.
    """

    def auth_headers() -> Dict[str, str]:  # local alias
        return _auth_headers(runner)

    def ensure_token() -> None:
        _ensure_runner_token(runner)

    after_ts = int(start_date.timestamp())
    before_ts = int(end_date.timestamp())

    def fetch_page(page: int) -> JSONList:
        url = f"{STRAVA_BASE_URL}/athlete/activities"
        params = {
            "after": after_ts,
            "before": before_ts,
            "per_page": 200,
            "page": page,
        }
        return _fetch_page_with_retries(
            runner,
            url,
            params,
            context_label="activities",
            page=page,
        )

    try:
        ensure_token()
        all_acts: List[Dict[str, Any]] = []
        page = 1
        attempted_refresh = False
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
                        f"401 for runner {runner.name} (activities). Refreshing token and retrying page {page}."
                    )
                    runner.access_token = None
                    ensure_token()
                    attempted_refresh = True
                    data = fetch_page(page)
                else:
                    raise
            if not data:
                break
            all_acts.extend(data)
            if len(data) < 200:
                break
            page += 1
        # Local filter by time & type
        filtered: List[Dict[str, Any]] = []
        for act in all_acts:
            if act.get("type") != "Run":
                continue
            start_local = act.get("start_date_local")
            if not start_local:
                continue
            try:
                dt = datetime.fromisoformat(start_local.replace("Z", "+00:00"))
            except Exception:
                continue
            if start_date <= dt <= end_date:
                filtered.append(act)
        return filtered
    except requests.exceptions.HTTPError as e:  # pragma: no cover
        resp = e.response
        if resp is None:
            logging.error(
                f"HTTPError (activities) no response object for runner {runner.name}: {e}"
            )
            return None
        detail = _extract_error(resp)
        logging.error(
            "Activities fetch error runner=%s status=%s detail=%s", runner.name, resp.status_code, detail
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
