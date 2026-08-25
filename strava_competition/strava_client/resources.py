"""Generic JSON resource fetcher with cache support.

Telemetry semantics: ``live_calls`` is incremented once per completed HTTP
round-trip (including retries that reach the server); attempts that fail
before a response arrives (network errors) are not counted.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Dict, Optional

import requests

from ..api_capture import (
    _redact_payload,
    get_cached_response,
    save_overlay_to_cache,
    save_response_to_cache,
)
from ..config import (
    RATE_LIMIT_429_BACKOFF_MAX_SECONDS,
    RATE_LIMIT_429_MAX_RETRIES,
    RATE_LIMIT_THROTTLE_SECONDS,
    REQUEST_TIMEOUT,
    STRAVA_BACKOFF_MAX_SECONDS,
    STRAVA_MAX_RETRIES,
    _cache_mode_offline,
    _cache_mode_saves,
)
from ..errors import StravaAPIError, StravaRateLimitError
from ..models import Runner
from . import telemetry
from .base import auth_headers, ensure_runner_token
from .cache_helpers import runner_identity
from .rate_limiter import RateLimiter
from ..utils import json_dumps_sorted
from .response_handling import classify_response_status, extract_error
from . import session as session_mod

LOGGER = logging.getLogger(__name__)

# Per-key locks deduplicating concurrent refetches of invalid cached entries.
# The registry grows with distinct invalid keys; that is bounded and acceptable.
_refetch_locks: dict[str, threading.Lock] = {}
_refetch_locks_guard = threading.Lock()


def _refetch_lock(key: str) -> threading.Lock:
    """Return (creating if needed) the refetch dedup lock for a capture key."""
    with _refetch_locks_guard:
        return _refetch_locks.setdefault(key, threading.Lock())


def _extract_error_detail(response: requests.Response) -> str | None:
    """Extract error detail from a response for inclusion in error messages."""
    return extract_error(response)


class ResourceAPI:
    """Encapsulates Strava JSON resource fetching with retries and capture."""

    def __init__(
        self,
        *,
        session: requests.Session | None = None,
        limiter: RateLimiter | None = None,
        timeout: int = REQUEST_TIMEOUT,
    ) -> None:
        # An explicitly injected session is honoured as-is (tests); otherwise
        # the thread-local default session is resolved per call so worker
        # threads never share one requests.Session instance.
        self._session = session
        self._limiter = limiter or RateLimiter()
        self._timeout = timeout

    def _resolve_session(self) -> requests.Session:
        """Return the injected session or this thread's default session."""

        return self._session or session_mod.get_default_session()

    def _sleep_429_backoff(self, backoff: float) -> float:
        """Sleep the escalating 429 backoff and return the next backoff value.

        Skipped (backoff unchanged) while the limiter holds an active
        window-reset deadline: ``before_request`` on the next attempt waits
        until the boundary instead, so an extra sleep here would only delay
        recovery.
        """

        if self._limiter.throttle_deadline() is not None:
            return backoff
        time.sleep(backoff)
        return min(backoff * 2, RATE_LIMIT_429_BACKOFF_MAX_SECONDS)

    def _raise_if_offline(self, runner: Runner, context: str) -> None:
        """Raise when offline mode forbids live HTTP calls.

        Raises:
            StravaAPIError: When STRAVA_API_CACHE_MODE=offline is enabled.
        """

        if not _cache_mode_offline:
            return
        message = (
            f"{context} live fetch requested for runner {runner.name} "
            "while STRAVA_API_CACHE_MODE=offline is enabled"
        )
        LOGGER.error(message)
        raise StravaAPIError(message)

    def fetch_json(
        self,
        runner: Runner,
        url: str,
        params: Optional[Dict[str, Any]],
        context: str,
    ) -> Any:
        """Fetch a JSON resource over live HTTP with retry/backoff.

        Raises:
            StravaAPIError: When offline mode is enabled (live calls are
                forbidden) or the request ultimately fails.
            StravaRateLimitError: When the 429 retry budget is exhausted.
        """
        self._raise_if_offline(runner, context)
        http_session = self._resolve_session()
        backoff = 1.0
        attempt = 0
        attempted_refresh = False
        rate_limit_retries = 0
        rate_limit_backoff = float(RATE_LIMIT_THROTTLE_SECONDS)
        while True:
            attempt += 1
            can_retry = attempt < STRAVA_MAX_RETRIES
            ensure_runner_token(runner)
            self._limiter.before_request()
            response: Optional[requests.Response] = None
            limiter_released = False
            try:
                response = http_session.get(
                    url,
                    headers=auth_headers(runner),
                    params=params,
                    timeout=self._timeout,
                )
            except requests.RequestException as exc:
                self._limiter.after_response(None, None)
                limiter_released = True
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
                telemetry.increment(telemetry.LIVE_CALLS)
                throttled, rate_info = self._limiter.after_response(
                    response.headers, response.status_code
                )
                limiter_released = True
                if throttled:
                    LOGGER.warning(
                        "%s runner=%s rate limited %s; throttling %ss",
                        context,
                        runner.name,
                        rate_info,
                        RATE_LIMIT_THROTTLE_SECONDS,
                    )
            finally:
                # Guard against an unexpected exception escaping before the
                # limiter slot was released, which would otherwise leak the
                # in-flight count and eventually deadlock all workers.
                if not limiter_released:
                    self._limiter.after_response(None, None)

            if response.status_code == 401 and not attempted_refresh:
                LOGGER.info(
                    "%s 401 for runner %s; refreshing token and retrying.",
                    context,
                    runner.name,
                )
                runner.access_token = None
                attempted_refresh = True
                continue

            # Handle 429 rate limits with a dedicated retry budget and
            # escalating backoff so transient limits don't consume the
            # general retry counter.
            if response.status_code == 429:
                rate_limit_retries += 1
                if rate_limit_retries <= RATE_LIMIT_429_MAX_RETRIES:
                    LOGGER.info(
                        "%s runner=%s 429 retry %s/%s; backing off %.0fs",
                        context,
                        runner.name,
                        rate_limit_retries,
                        RATE_LIMIT_429_MAX_RETRIES,
                        rate_limit_backoff,
                    )
                    rate_limit_backoff = self._sleep_429_backoff(rate_limit_backoff)
                    continue
                detail = _extract_error_detail(response)
                message = (
                    f"{context} rate limited (429) after "
                    f"{RATE_LIMIT_429_MAX_RETRIES} retries "
                    f"for runner {runner.name}"
                )
                if detail:
                    message = f"{message} | {detail}"
                LOGGER.error(message)
                raise StravaRateLimitError(message)

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
        *,
        validate: Optional[Callable[[Any], bool]] = None,
    ) -> Any:
        """Fetch a JSON resource, serving and persisting via the capture cache.

        Args:
            runner: Participant whose credentials authorise the request.
            url: Absolute resource URL.
            params: Optional query parameters.
            context: Label used in log messages and errors.
            validate: Optional predicate applied to cached and live payloads.
                A cached payload that fails validation is treated as a cache
                miss and re-fetched; a valid replacement is persisted as an
                overlay so it supersedes the bad base file. A live payload
                that fails validation is returned but never cached. The
                callable must not raise; any exception it raises propagates
                to the caller.

        Returns:
            The cached or freshly fetched JSON payload.

        Raises:
            StravaAPIError: On a cache miss or invalid cached payload while
                offline mode is enabled, or if the live fetch fails.
        """
        params_for_capture = dict(params) if params else None
        identity = runner_identity(runner)
        cached = get_cached_response(
            "GET",
            url,
            identity,
            params=params_for_capture,
        )
        cached_invalid = False
        if cached is not None:
            if validate is None or validate(cached):
                telemetry.increment(telemetry.CACHE_HITS)
                LOGGER.debug(
                    "Cache hit for %s runner=%s type=%s",
                    context,
                    runner.name,
                    type(cached).__name__,
                )
                return cached
            cached_invalid = True
            LOGGER.warning(
                "Cached payload failed validation for %s runner=%s; refetching",
                context,
                runner.name,
            )
        if _cache_mode_offline:
            reason = "invalid cached payload" if cached_invalid else "cache miss"
            message = (
                f"{context} {reason} for runner {runner.name} while "
                "STRAVA_API_CACHE_MODE=offline is enabled"
            )
            LOGGER.error(message)
            raise StravaAPIError(message)

        if cached_invalid:
            return self._refetch_invalid_cached(
                runner,
                url,
                params,
                context,
                identity=identity,
                params_for_capture=params_for_capture,
                validate=validate,
            )

        data = self.fetch_json(runner, url, params, context)
        self._persist_validated(
            data,
            runner=runner,
            url=url,
            identity=identity,
            params_for_capture=params_for_capture,
            context=context,
            validate=validate,
            overlay=False,
        )
        return data

    def _refetch_invalid_cached(
        self,
        runner: Runner,
        url: str,
        params: Optional[Dict[str, Any]],
        context: str,
        *,
        identity: str,
        params_for_capture: Optional[Dict[str, Any]],
        validate: Optional[Callable[[Any], bool]],
    ) -> Any:
        """Refetch an invalid cached entry, deduplicating concurrent callers.

        Holding a per-key lock, the cache is re-checked first: another worker
        may already have refetched and persisted a valid replacement, in
        which case that payload is served without a duplicate live fetch.
        """
        key = f"{url}|{identity}|{json_dumps_sorted(params_for_capture or {})}"
        with _refetch_lock(key):
            cached = get_cached_response(
                "GET",
                url,
                identity,
                params=params_for_capture,
            )
            if cached is not None and (validate is None or validate(cached)):
                telemetry.increment(telemetry.CACHE_HITS)
                LOGGER.debug(
                    "Cache healed by concurrent refetch for %s runner=%s",
                    context,
                    runner.name,
                )
                return cached
            telemetry.increment(telemetry.VALIDATION_REFETCHES)
            data = self.fetch_json(runner, url, params, context)
            self._persist_validated(
                data,
                runner=runner,
                url=url,
                identity=identity,
                params_for_capture=params_for_capture,
                context=context,
                validate=validate,
                overlay=True,
            )
            return data

    def _persist_validated(
        self,
        data: Any,
        *,
        runner: Runner,
        url: str,
        identity: str,
        params_for_capture: Optional[Dict[str, Any]],
        context: str,
        validate: Optional[Callable[[Any], bool]],
        overlay: bool,
    ) -> None:
        """Persist a live payload unless validation (pre/post-redaction) fails.

        Saving redacts the payload first, so a redaction config that strips a
        structurally required key would persist a payload that fails
        validation on every future read (a permanent refetch loop). Guard by
        validating the redacted payload before persisting.
        """
        if not _cache_mode_saves:
            return
        if validate is not None:
            if not validate(data):
                LOGGER.warning(
                    "Live payload failed validation for %s runner=%s; not caching",
                    context,
                    runner.name,
                )
                return
            if not validate(_redact_payload(data)):
                LOGGER.error(
                    "Redaction breaks payload validation for %s runner=%s; "
                    "not persisting (check STRAVA_CACHE_REDACT_FIELDS)",
                    context,
                    runner.name,
                )
                return
        save = save_overlay_to_cache if overlay else save_response_to_cache
        save("GET", url, identity, response=data, params=params_for_capture)
