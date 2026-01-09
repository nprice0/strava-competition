"""Generic JSON resource fetcher with cache support."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

import requests

from ..api_capture import get_cached_response, save_response_to_cache
from ..config import (
    REQUEST_TIMEOUT,
    STRAVA_BACKOFF_MAX_SECONDS,
    STRAVA_MAX_RETRIES,
    _cache_mode_offline,
    _cache_mode_saves,
)
from ..errors import StravaAPIError
from ..models import Runner
from .base import auth_headers, ensure_runner_token
from .capture import runner_identity
from .rate_limiter import RateLimiter
from .response_handling import classify_response_status
from .session import get_default_session

LOGGER = logging.getLogger(__name__)


class ResourceAPI:
    """Encapsulates Strava JSON resource fetching with retries and capture."""

    def __init__(
        self,
        *,
        session: requests.Session | None = None,
        limiter: RateLimiter | None = None,
        timeout: int = REQUEST_TIMEOUT,
    ) -> None:
        self._session = session or get_default_session()
        self._limiter = limiter or RateLimiter()
        self._timeout = timeout

    def fetch_json(
        self,
        runner: Runner,
        url: str,
        params: Optional[Dict[str, Any]],
        context: str,
    ) -> Any:
        backoff = 1.0
        attempt = 0
        attempted_refresh = False
        while True:
            attempt += 1
            can_retry = attempt < STRAVA_MAX_RETRIES
            ensure_runner_token(runner)
            self._limiter.before_request()
            response: Optional[requests.Response] = None
            try:
                response = self._session.get(
                    url,
                    headers=auth_headers(runner),
                    params=params,
                    timeout=self._timeout,
                )
            except requests.RequestException as exc:
                self._limiter.after_response(None, None)
                if can_retry:
                    LOGGER.warning(
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
                LOGGER.error(message)
                raise StravaAPIError(message) from exc
            else:
                self._limiter.after_response(response.headers, response.status_code)

            if response.status_code == 401 and not attempted_refresh:
                LOGGER.info(
                    "%s 401 for runner %s; refreshing token and retrying.",
                    context,
                    runner.name,
                )
                runner.access_token = None
                attempted_refresh = True
                continue

            action, error = classify_response_status(
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
                    LOGGER.warning(
                        "Non-JSON response for %s runner=%s attempt=%s; retrying in %.1fs",
                        context,
                        runner.name,
                        attempt,
                        backoff,
                    )
                    time.sleep(backoff)
                    backoff = min(backoff * 2, STRAVA_BACKOFF_MAX_SECONDS)
                    continue
                message = (
                    f"{context} returned non-JSON payload for runner {runner.name}"
                )
                LOGGER.error(message)
                raise StravaAPIError(message) from exc

    def fetch_with_capture(
        self,
        runner: Runner,
        url: str,
        params: Optional[Dict[str, Any]],
        context: str,
    ) -> Any:
        params_for_capture = dict(params) if params else None
        identity = runner_identity(runner)
        cached = get_cached_response(
            "GET",
            url,
            identity,
            params=params_for_capture,
        )
        if cached is not None:
            LOGGER.debug(
                "Cache hit for %s runner=%s type=%s",
                context,
                runner.name,
                type(cached).__name__,
            )
            return cached
        if _cache_mode_offline:
            message = (
                f"{context} cache miss for runner {runner.name} while "
                "STRAVA_API_CACHE_MODE=offline is enabled"
            )
            LOGGER.error(message)
            raise StravaAPIError(message)

        data = self.fetch_json(runner, url, params, context)
        if _cache_mode_saves:
            save_response_to_cache(
                "GET",
                url,
                identity,
                response=data,
                params=params_for_capture,
            )
        return data
