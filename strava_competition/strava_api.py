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

from .api_capture import record_response, replay_response
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
    STRAVA_SKIP_TOKEN_REFRESH,
)
from .errors import (
    StravaAPIError,
    StravaPaymentRequiredError,
    StravaPermissionError,
    StravaResourceNotFoundError,
    StravaStreamEmptyError,
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
    # Handle 5xx at the HTTP adapter level; 429 is handled explicitly below to
    # coordinate with our RateLimiter and avoid double retries/backoff.
    status_forcelist=[500, 502, 503, 504],
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
_session.headers.update(
    {
        "Accept-Encoding": "gzip, deflate",
        "Accept": "application/json",
    }
)

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

    def after_response(  # noqa: C901 - high cohesion guard logic
        self, headers: Optional[Dict[str, Any]], status_code: Optional[int]
    ) -> None:
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

    Performs a refresh using the runner's stored ``refresh_token`` if no access token
    is cached. When ``STRAVA_SKIP_TOKEN_REFRESH`` is enabled we assume cached API
    responses are available and quietly return without performing the refresh.
    If Strava rotates the refresh token, update the runner instance.
    """
    if STRAVA_SKIP_TOKEN_REFRESH:
        if not getattr(runner, "access_token", None) and not getattr(
            runner, "_skip_token_logged", False
        ):
            logging.info(
                "Skipping Strava token refresh for runner=%s (STRAVA_SKIP_TOKEN_REFRESH)",
                getattr(runner, "name", "?"),
            )
            setattr(runner, "_skip_token_logged", True)
        return

    if not getattr(runner, "access_token", None):
        access_token, new_refresh_token = get_access_token(
            runner.refresh_token, runner_name=runner.name
        )
        runner.access_token = access_token
        if new_refresh_token and new_refresh_token != runner.refresh_token:
            runner.refresh_token = new_refresh_token
            # Attempt immediate persistence (best-effort, silent on failure)
            try:
                from .excel_writer import (
                    update_single_runner_refresh_token,
                )  # local import to avoid cycle
                from .config import INPUT_FILE

                update_single_runner_refresh_token(INPUT_FILE, runner)
            except Exception:
                pass


def _auth_headers(runner: "Runner") -> Dict[str, str]:
    return {"Authorization": f"Bearer {runner.access_token}"}


def _runner_identity(runner: "Runner") -> str:
    """Return a stable identifier for capture/replay naming."""

    return runner.strava_id or runner.name or "unknown"


def _replay_list_response(
    runner: "Runner",
    url: str,
    params: Dict[str, Any],
    *,
    context_label: str,
    page: int,
) -> Optional[JSONList]:
    """Return cached list payload when replay mode is active."""

    cached = replay_response(
        "GET",
        url,
        _runner_identity(runner),
        params=params,
    )
    if cached is None:
        return None
    if isinstance(cached, list):
        logging.debug(
            "Replay hit for %s runner=%s page=%s entries=%s",
            context_label,
            runner.name,
            page,
            len(cached),
        )
        return cached
    logging.warning(
        "Replay payload type mismatch for %s runner=%s page=%s type=%s",
        context_label,
        runner.name,
        page,
        type(cached).__name__,
    )
    return None


def _record_list_response(
    runner: "Runner",
    url: str,
    params: Dict[str, Any],
    data: JSONList,
) -> None:
    """Persist successful list responses when capture mode is enabled."""

    record_response(
        "GET",
        url,
        _runner_identity(runner),
        response=data,
        params=params,
    )


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
                url,
                headers=_auth_headers(runner),
                params=params,
                timeout=DEFAULT_TIMEOUT,
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
            # 429 Too Many Requests: slow down and retry with increasing waits.
            if attempts < STRAVA_MAX_RETRIES:
                logging.warning(
                    "429 Too Many Requests for %s runner=%s page=%s attempt=%s; backing off %.1fs",
                    context_label,
                    runner.name,
                    page,
                    attempts,
                    backoff,
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, STRAVA_BACKOFF_MAX_SECONDS)
                continue
            resp.raise_for_status()
        is_html = "text/html" in ct.lower()
        try:
            resp.raise_for_status()
        except requests.HTTPError:
            if attempts < STRAVA_MAX_RETRIES and (
                500 <= resp.status_code < 600 or is_html
            ):
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


def get_segment_efforts(  # noqa: C901 - pagination and auth flow
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
        cache_params = dict(params)
        cached = _replay_list_response(
            runner,
            url,
            cache_params,
            context_label="segment efforts",
            page=page,
        )
        if cached is not None:
            return cached
        result = _fetch_page_with_retries(
            runner,
            url,
            params,
            context_label="segment efforts",
            page=page,
            segment_id=segment_id,
        )
        if isinstance(result, list):
            _record_list_response(runner, url, cache_params, result)
        return result

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
        if logging.getLogger().isEnabledFor(
            logging.DEBUG
        ):  # Avoid building string unless needed
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
            logging.error(f"Error for runner {runner.name}: HTTP {status}{suffix}")
        return None


def get_activities(  # noqa: C901 - pagination and auth flow
    runner: Runner,
    start_date: datetime,
    end_date: datetime,
) -> Optional[List[Dict[str, Any]]]:
    """Fetch all activities for a runner in [start_date, end_date].

    Uses /athlete/activities with pagination (per_page=200). Filters locally to
    activity 'type' == 'Run'. Returns list (possibly empty) or None on error.
    Inclusive range based on activity start_date_local.
    """

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
        cache_params = dict(params)
        cached = _replay_list_response(
            runner,
            url,
            cache_params,
            context_label="activities",
            page=page,
        )
        if cached is not None:
            return cached
        result = _fetch_page_with_retries(
            runner,
            url,
            params,
            context_label="activities",
            page=page,
        )
        if isinstance(result, list):
            _record_list_response(runner, url, cache_params, result)
        return result

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
            "Activities fetch error runner=%s status=%s detail=%s",
            runner.name,
            resp.status_code,
            detail,
        )
        return None


def _get_resource_json(
    runner: Runner,
    url: str,
    params: Optional[Dict[str, Any]],
    context: str,
) -> Any:
    """Fetch a JSON object from Strava with retry/backoff and rich errors."""

    backoff = 1.0
    attempt = 0
    attempted_refresh = False
    while True:
        attempt += 1
        can_retry = attempt < STRAVA_MAX_RETRIES
        _ensure_runner_token(runner)
        _limiter.before_request()
        response: Optional[requests.Response] = None
        try:
            response = _session.get(
                url,
                headers=_auth_headers(runner),
                params=params,
                timeout=DEFAULT_TIMEOUT,
            )
        except requests.RequestException as exc:
            _limiter.after_response(None, None)
            if can_retry:
                logging.warning(
                    "%s network error runner=%s attempt=%s err=%s; retrying in %.1fs",
                    context,
                    runner.name,
                    attempt,
                    exc.__class__.__name__,
                    backoff,
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, STRAVA_BACKOFF_MAX_SECONDS)
                continue
            message = f"{context} network error for runner={runner.name}: {exc.__class__.__name__}"
            logging.error(message)
            raise StravaAPIError(message) from exc
        else:
            _limiter.after_response(response.headers, response.status_code)

        if response.status_code == 401 and not attempted_refresh:
            logging.info(
                "%s 401 for runner %s; refreshing token and retrying.",
                context,
                runner.name,
            )
            runner.access_token = None
            attempted_refresh = True
            continue

        action, error = _classify_response_status(
            runner,
            response,
            context,
            attempt=attempt,
            backoff=backoff,
            can_retry=can_retry,
        )
        if action == "retry":
            time.sleep(backoff)
            backoff = min(backoff * 2, STRAVA_BACKOFF_MAX_SECONDS)
            continue
        if action == "raise" and error is not None:
            raise error

        try:
            return response.json()
        except ValueError as exc:
            if can_retry:
                logging.warning(
                    "Non-JSON response for %s runner=%s attempt=%s; retrying in %.1fs",
                    context,
                    runner.name,
                    attempt,
                    backoff,
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, STRAVA_BACKOFF_MAX_SECONDS)
                continue
            message = f"{context} returned non-JSON payload for runner {runner.name}"
            logging.error(message)
            raise StravaAPIError(message) from exc


def _fetch_resource_with_capture(
    runner: Runner,
    url: str,
    params: Optional[Dict[str, Any]],
    context: str,
) -> Any:
    """Wrap ``_get_resource_json`` with capture/replay semantics."""

    params_for_capture = dict(params) if params else None
    identity = _runner_identity(runner)
    cached = replay_response(
        "GET",
        url,
        identity,
        params=params_for_capture,
    )
    if cached is not None:
        logging.debug(
            "Replay hit for %s runner=%s type=%s",
            context,
            runner.name,
            type(cached).__name__,
        )
        return cached

    data = _get_resource_json(runner, url, params, context)
    record_response(
        "GET",
        url,
        identity,
        response=data,
        params=params_for_capture,
    )
    return data


def _classify_response_status(
    runner: Runner,
    response: requests.Response,
    context: str,
    *,
    attempt: int,
    backoff: float,
    can_retry: bool,
) -> tuple[str, Optional[Exception]]:
    """Return action for a non-success status: ok, retry, or raise."""

    status = response.status_code
    detail = _extract_error(response)

    def with_detail(message: str) -> str:
        return f"{message} | {detail}" if detail else message

    if status == 429 and can_retry:
        logging.warning(
            "%s rate limited (429) runner=%s attempt=%s; sleeping %.1fs",
            context,
            runner.name,
            attempt,
            backoff,
        )
        return "retry", None

    if status == 402:
        try:
            runner.payment_required = True  # type: ignore[attr-defined]
        except Exception:
            pass
        message = with_detail(
            f"{context} requires Strava subscription for runner {runner.name}"
        )
        logging.warning(message)
        return "raise", StravaPaymentRequiredError(message)

    if status in (401, 403):
        message = with_detail(f"{context} forbidden for runner {runner.name}")
        logging.warning(message)
        return "raise", StravaPermissionError(message)

    if status == 404:
        message = with_detail(f"{context} not found for runner {runner.name}")
        logging.info(message)
        return "raise", StravaResourceNotFoundError(message)

    if 500 <= status < 600 and can_retry:
        message = with_detail(
            f"{context} server error {status} for runner {runner.name}"
        )
        logging.warning("%s; retrying in %.1fs", message, backoff)
        return "retry", None

    if 400 <= status < 600:
        message = with_detail(
            f"{context} request failed (status {status}) for runner {runner.name}"
        )
        logging.error(message)
        return "raise", StravaAPIError(message)

    return "ok", None


def fetch_segment_geometry(
    runner: Runner,
    segment_id: int,
) -> Dict[str, Any]:
    """Return high-resolution geometry details for a Strava segment."""

    url = f"{STRAVA_BASE_URL}/segments/{segment_id}"
    context = f"segment:{segment_id}"
    data = _fetch_resource_with_capture(runner, url, params=None, context=context)
    if not isinstance(data, dict):
        message = f"{context} payload had unexpected type {type(data).__name__}"
        logging.error(message)
        raise StravaAPIError(message)
    map_info = data.get("map") or {}
    polyline = map_info.get("polyline") or map_info.get("summary_polyline")
    if not polyline:
        message = f"{context} missing polyline data for runner {runner.name}"
        logging.error(message)
        raise StravaAPIError(message)
    distance_val = data.get("distance")
    try:
        distance_m = float(distance_val) if distance_val is not None else 0.0
    except (TypeError, ValueError):
        distance_m = 0.0
    result: Dict[str, Any] = {
        "segment_id": int(data.get("id", segment_id)),
        "name": data.get("name"),
        "distance": distance_m,
        "polyline": polyline,
        "start_latlng": data.get("start_latlng"),
        "end_latlng": data.get("end_latlng"),
        "elevation_high": data.get("elevation_high"),
        "elevation_low": data.get("elevation_low"),
        "map": map_info,
        "raw": data,
    }
    return result


def fetch_activity_stream(
    runner: Runner,
    activity_id: int,
    *,
    stream_types: tuple[str, ...] = ("latlng", "time"),
    resolution: str = "high",
    series_type: str = "time",
) -> Dict[str, Any]:
    """Return GPS stream data for an activity (lat/lon + timestamps)."""

    keys = ",".join(stream_types)
    url = f"{STRAVA_BASE_URL}/activities/{activity_id}/streams"
    context = f"activity_stream:{activity_id}"
    params = {
        "keys": keys,
        "key_by_type": "true",
        "resolution": resolution,
        "series_type": series_type,
    }
    data = _fetch_resource_with_capture(runner, url, params=params, context=context)
    if not isinstance(data, dict):
        message = f"{context} payload had unexpected type {type(data).__name__}"
        logging.error(message)
        raise StravaAPIError(message)
    latlng_stream = data.get("latlng")
    time_stream = data.get("time")
    if not isinstance(latlng_stream, dict) or not isinstance(time_stream, dict):
        message = f"{context} missing required stream metadata"
        logging.warning(message)
        raise StravaStreamEmptyError(message)
    latlng_data = latlng_stream.get("data")
    time_data = time_stream.get("data")
    if not latlng_data or not time_data:
        message = f"{context} missing latlng or time samples"
        logging.warning(message)
        raise StravaStreamEmptyError(message)
    if len(latlng_data) != len(time_data):
        message = (
            f"{context} stream length mismatch latlng={len(latlng_data)}"
            f" time={len(time_data)}"
        )
        logging.warning(message)
        raise StravaStreamEmptyError(message)
    metadata = {
        "latlng": {k: v for k, v in latlng_stream.items() if k != "data"},
        "time": {k: v for k, v in time_stream.items() if k != "data"},
        "additional_streams": {
            key: value for key, value in data.items() if key not in {"latlng", "time"}
        },
    }
    result: Dict[str, Any] = {
        "activity_id": activity_id,
        "latlng": latlng_data,
        "time": time_data,
        "metadata": metadata,
    }
    return result


def _extract_error(resp: Optional[requests.Response]) -> Optional[str]:
    """Return compact string with Strava error info (message + codes) if present."""

    if resp is None:
        return None
    data = _safe_json(resp)
    if data is None:
        return _extract_error_text(resp)
    if not isinstance(data, dict):
        return None
    parts = _collect_error_parts(data)
    return " | ".join(parts) if parts else None


def _safe_json(resp: requests.Response) -> Optional[Any]:
    """Safely parse JSON; return None if parsing fails."""

    try:
        return resp.json()
    except Exception:
        return None


def _extract_error_text(resp: requests.Response) -> Optional[str]:
    """Best-effort plain-text extraction when JSON parsing fails."""

    text = getattr(resp, "text", "")
    if not isinstance(text, str):
        return None
    trimmed = text.strip()
    if not trimmed:
        return None
    return (trimmed[:297] + "...") if len(trimmed) > 300 else trimmed


def _collect_error_parts(data: Dict[str, Any]) -> List[str]:
    """Build error snippets from the standard Strava error response."""

    parts: List[str] = []
    message = data.get("message")
    if message:
        parts.append(str(message))
    errors = data.get("errors")
    if isinstance(errors, list):
        for err in errors:
            if not isinstance(err, dict):
                continue
            resource = err.get("resource")
            field = err.get("field")
            code = err.get("code")
            spec = "/".join(filter(None, (resource, field)))
            if code and spec:
                parts.append(f"{spec}:{code}")
            elif code:
                parts.append(str(code))
    return parts
