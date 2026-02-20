"""Shared pagination helpers for Strava API list endpoints."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, TypeAlias, cast

import requests

from ..config import (
    RATE_LIMIT_THROTTLE_SECONDS,
    REQUEST_TIMEOUT,
    STRAVA_BACKOFF_MAX_SECONDS,
    STRAVA_MAX_RETRIES,
)
from ..models import Runner
from .base import auth_headers
from .rate_limiter import RateLimiter

JSONList: TypeAlias = List[Dict[str, Any]]

LOGGER = logging.getLogger(__name__)


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
    """GET a paginated endpoint with resilient retry/backoff logic."""

    MAX_429_RETRIES = 10
    attempts = 0
    rate_limit_retries = 0
    backoff = 1.0
    while True:
        attempts += 1
        limiter.before_request()
        resp: Optional[requests.Response] = None
        try:
            resp = session.get(
                url,
                headers=auth_headers(runner),
                params=params,
                timeout=timeout,
            )
        except requests.RequestException as exc:
            limiter.after_response(None, None)
            if attempts < STRAVA_MAX_RETRIES:
                _log_retry(
                    runner.name,
                    context_label,
                    page,
                    attempts,
                    backoff,
                    exc.__class__.__name__,
                    segment_id,
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, STRAVA_BACKOFF_MAX_SECONDS)
                continue
            _log_giveup(runner.name, context_label, page, attempts, exc, segment_id)
            return []
        else:
            throttled, rate_info = limiter.after_response(
                resp.headers, resp.status_code
            )
            if throttled:
                LOGGER.warning(
                    "%s runner=%s page=%s rate limited %s; throttling %ss",
                    context_label,
                    runner.name,
                    page,
                    rate_info,
                    RATE_LIMIT_THROTTLE_SECONDS,
                )

        is_html = "text/html" in (resp.headers.get("Content-Type", "").lower())
        # Retry 429s up to a cap — rate limits are transient but may persist
        if resp.status_code == 429:
            rate_limit_retries += 1
            if rate_limit_retries > MAX_429_RETRIES:
                LOGGER.error(
                    "%s runner=%s page=%s exceeded max 429 retries (%s); giving up",
                    context_label,
                    runner.name,
                    page,
                    MAX_429_RETRIES,
                )
                return []
            time.sleep(RATE_LIMIT_THROTTLE_SECONDS)
            continue

        try:
            resp.raise_for_status()
        except requests.HTTPError:
            if attempts < STRAVA_MAX_RETRIES and (
                500 <= resp.status_code < 600 or is_html
            ):
                _log_retry(
                    runner.name,
                    context_label,
                    page,
                    attempts,
                    backoff,
                    f"status={resp.status_code}",
                    segment_id,
                )
                time.sleep(backoff)
                backoff = min(backoff * 2, STRAVA_BACKOFF_MAX_SECONDS)
                continue
            raise

        if is_html:
            if attempts < STRAVA_MAX_RETRIES:
                LOGGER.warning(
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
            LOGGER.error(
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
                LOGGER.warning(
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
            LOGGER.error(
                "Non-JSON response (%s) runner=%s page=%s after %s attempts; abandoning page",
                context_label,
                runner.name,
                page,
                attempts,
            )
            return []

        if not isinstance(data, list):
            LOGGER.warning(
                "Unexpected JSON shape (not list) for %s runner=%s page=%s type=%s",
                context_label,
                runner.name,
                page,
                type(data).__name__,
            )
            return []

        return cast(JSONList, data)


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
