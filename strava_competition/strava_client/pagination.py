"""Shared pagination helpers for Strava API list endpoints."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypeAlias, cast

import requests

from ..config import (
    RATE_LIMIT_429_BACKOFF_MAX_SECONDS,
    RATE_LIMIT_429_MAX_RETRIES,
    RATE_LIMIT_THROTTLE_SECONDS,
    REQUEST_TIMEOUT,
    STRAVA_BACKOFF_MAX_SECONDS,
    STRAVA_MAX_RETRIES,
)
from ..errors import StravaAPIError, StravaRateLimitError
from ..models import Runner
from . import telemetry
from .base import auth_headers
from .rate_limiter import RateLimiter

JSONList: TypeAlias = List[Dict[str, Any]]

LOGGER = logging.getLogger(__name__)


@dataclass
class _PageRetryState:
    """Mutable retry bookkeeping for a single paginated fetch."""

    runner_name: str
    context_label: str
    page: int
    segment_id: Optional[int]
    attempts: int = 0
    backoff: float = 1.0
    rate_limit_retries: int = 0
    rate_limit_backoff: float = 0.0

    def __post_init__(self) -> None:
        self.rate_limit_backoff = float(RATE_LIMIT_THROTTLE_SECONDS)

    def can_retry(self) -> bool:
        """Return True while general retry attempts remain."""

        return self.attempts < STRAVA_MAX_RETRIES

    def sleep_and_escalate(self) -> None:
        """Sleep for the current backoff, then double it (capped)."""

        time.sleep(self.backoff)
        self.backoff = min(self.backoff * 2, STRAVA_BACKOFF_MAX_SECONDS)

    def retry_429(self, limiter: RateLimiter) -> bool:
        """Consume one 429 retry; return False once the budget is spent.

        The backoff sleep is skipped while the limiter holds an active
        window-reset deadline: ``before_request`` on the next attempt waits
        until the boundary instead.
        """

        self.rate_limit_retries += 1
        if self.rate_limit_retries > RATE_LIMIT_429_MAX_RETRIES:
            return False
        if limiter.throttle_deadline() is not None:
            LOGGER.info(
                "%s runner=%s page=%s 429 retry %s/%s; window reset wait pending",
                self.context_label,
                self.runner_name,
                self.page,
                self.rate_limit_retries,
                RATE_LIMIT_429_MAX_RETRIES,
            )
            return True
        LOGGER.info(
            "%s runner=%s page=%s 429 retry %s/%s; backing off %.0fs",
            self.context_label,
            self.runner_name,
            self.page,
            self.rate_limit_retries,
            RATE_LIMIT_429_MAX_RETRIES,
            self.rate_limit_backoff,
        )
        time.sleep(self.rate_limit_backoff)
        self.rate_limit_backoff = min(
            self.rate_limit_backoff * 2,
            RATE_LIMIT_429_BACKOFF_MAX_SECONDS,
        )
        return True


def fetch_page_with_retries(
    *,
    runner: Runner,
    url: str,
    params: Dict[str, Any],
    context_label: str,
    page: int,
    session: requests.Session,
    limiter: RateLimiter,
    segment_id: Optional[int] = None,
    timeout: int = REQUEST_TIMEOUT,
) -> JSONList:
    """GET a paginated endpoint with resilient retry/backoff logic.

    Returns:
        The JSON list payload for the page. An empty list always means a
        genuine empty page; terminal failures raise instead.

    Raises:
        StravaRateLimitError: When the 429 retry budget is exhausted.
        StravaAPIError: When the page cannot be fetched after exhausting
            retries (network errors, persistent HTML downtime, non-JSON
            payloads) or the payload is not a JSON list.
        requests.HTTPError: For non-retryable HTTP error statuses
            (e.g. 401/403), so callers can handle token refresh.
    """

    state = _PageRetryState(runner.name, context_label, page, segment_id)
    while True:
        state.attempts += 1
        resp = _request_page(
            runner=runner,
            url=url,
            params=params,
            session=session,
            limiter=limiter,
            timeout=timeout,
            state=state,
        )
        if resp is None:
            continue
        if resp.status_code == 429:
            if state.retry_429(limiter):
                continue
            message = (
                f"{context_label} runner={runner.name} page={page} rate "
                f"limited (429) after {RATE_LIMIT_429_MAX_RETRIES} retries"
            )
            LOGGER.error(message)
            raise StravaRateLimitError(message)
        data = _parse_page_response(resp, state)
        if data is None:
            continue
        return data


def _request_page(
    *,
    runner: Runner,
    url: str,
    params: Dict[str, Any],
    session: requests.Session,
    limiter: RateLimiter,
    timeout: int,
    state: _PageRetryState,
) -> Optional[requests.Response]:
    """Perform one GET attempt with limiter bookkeeping.

    Returns:
        The response, or None when a retryable network error was logged and
        the backoff sleep already applied (caller should retry).

    Raises:
        StravaAPIError: When network errors persist beyond the retry budget.
    """

    limiter.before_request()
    limiter_released = False
    try:
        resp = session.get(
            url,
            headers=auth_headers(runner),
            params=params,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        limiter.after_response(None, None)
        limiter_released = True
        if state.can_retry():
            _log_retry(
                state.runner_name,
                state.context_label,
                state.page,
                state.attempts,
                state.backoff,
                exc.__class__.__name__,
                state.segment_id,
            )
            state.sleep_and_escalate()
            return None
        _log_giveup(
            state.runner_name,
            state.context_label,
            state.page,
            state.attempts,
            exc,
            state.segment_id,
        )
        raise StravaAPIError(
            f"{state.context_label} network error for runner={state.runner_name} "
            f"page={state.page} after {state.attempts} attempts: "
            f"{exc.__class__.__name__}"
        ) from exc
    else:
        telemetry.increment(telemetry.LIVE_CALLS)
        throttled, rate_info = limiter.after_response(resp.headers, resp.status_code)
        limiter_released = True
        if throttled:
            LOGGER.warning(
                "%s runner=%s page=%s rate limited %s; throttling %ss",
                state.context_label,
                state.runner_name,
                state.page,
                rate_info,
                RATE_LIMIT_THROTTLE_SECONDS,
            )
        return resp
    finally:
        # Release the limiter slot even if an unexpected exception escapes
        # before after_response ran, preventing an in-flight count leak.
        if not limiter_released:
            limiter.after_response(None, None)


def _parse_page_response(
    resp: requests.Response,
    state: _PageRetryState,
) -> Optional[JSONList]:
    """Validate and parse a page response.

    Returns:
        The parsed JSON list, or None when a retryable condition was logged
        and the backoff sleep already applied (caller should retry).

    Raises:
        StravaAPIError: On persistent HTML downtime, non-JSON payloads, or a
            payload that is not a JSON list.
        requests.HTTPError: For non-retryable HTTP error statuses.
    """

    is_html = "text/html" in (resp.headers.get("Content-Type", "").lower())
    try:
        resp.raise_for_status()
    except requests.HTTPError:
        if state.can_retry() and (500 <= resp.status_code < 600 or is_html):
            _log_retry(
                state.runner_name,
                state.context_label,
                state.page,
                state.attempts,
                state.backoff,
                f"status={resp.status_code}",
                state.segment_id,
            )
            state.sleep_and_escalate()
            return None
        raise

    if is_html:
        if state.can_retry():
            LOGGER.warning(
                "HTML downtime page for %s runner=%s page=%s attempt=%s; retrying in %.1fs",
                state.context_label,
                state.runner_name,
                state.page,
                state.attempts,
                state.backoff,
            )
            state.sleep_and_escalate()
            return None
        raise _giveup_error(state, "persistent HTML downtime page")

    try:
        data = resp.json()
    except ValueError as exc:
        if state.can_retry():
            LOGGER.warning(
                "Non-JSON response (%s) runner=%s page=%s attempt=%s; retrying in %.1fs",
                state.context_label,
                state.runner_name,
                state.page,
                state.attempts,
                state.backoff,
            )
            state.sleep_and_escalate()
            return None
        raise _giveup_error(state, "non-JSON response") from exc

    if not isinstance(data, list):
        raise _giveup_error(
            state,
            f"unexpected JSON shape (got {type(data).__name__}, expected list)",
        )

    return cast(JSONList, data)


def _giveup_error(state: _PageRetryState, reason: str) -> StravaAPIError:
    """Log and build the terminal error for an unfetchable page."""

    message = (
        f"{state.context_label} runner={state.runner_name} "
        f"page={state.page} {reason} after {state.attempts} attempts"
    )
    LOGGER.error(message)
    return StravaAPIError(message)


def _log_retry(
    runner_name: str,
    context_label: str,
    page: int,
    attempt: int,
    backoff: float,
    reason: str,
    segment_id: Optional[int],
) -> None:
    prefix = f"{context_label.capitalize()} network error"
    if segment_id is not None:
        prefix += f" segment={segment_id}"
    LOGGER.warning(
        "%s runner=%s page=%s attempt=%s err=%s; backoff %.1fs",
        prefix,
        runner_name,
        page,
        attempt,
        reason,
        backoff,
    )


def _log_giveup(
    runner_name: str,
    context_label: str,
    page: int,
    attempts: int,
    exc: Exception,
    segment_id: Optional[int],
) -> None:
    prefix = f"{context_label.capitalize()} network error (giving up)"
    if segment_id is not None:
        prefix += f" segment={segment_id}"
    LOGGER.error(
        "%s runner=%s page=%s attempts=%s err=%s",
        prefix,
        runner_name,
        page,
        attempts,
        exc,
    )
