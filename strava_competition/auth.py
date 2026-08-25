"""OAuth / token refresh utilities for Strava API.

This module focuses on securely exchanging a refresh token for an access token
using Strava's OAuth endpoint. It adds resiliency (HTTP retries), safe logging
that avoids leaking secrets, and defensive JSON parsing.
"""

from __future__ import annotations
import logging
import threading
import time
from typing import TYPE_CHECKING, NoReturn, Tuple

from requests.exceptions import RequestException

from .config import (
    CLIENT_ID,
    CLIENT_SECRET,
    RATE_LIMIT_THROTTLE_SECONDS,
    REQUEST_TIMEOUT,
    STRAVA_OAUTH_URL,
    _cache_mode_offline,
)

if TYPE_CHECKING:
    from requests import Response, Session


class TokenError(Exception):
    """Raised when token refresh fails (after retries)."""


# Retry policy for 429 responses from the token endpoint.
MAX_429_RETRIES = 10
_RATE_LIMIT_MAX_BACKOFF_SECONDS = 120.0

# Process-wide cooldown shared by all refresh attempts: when any thread is
# rate limited, every other thread waits out the same cooldown instead of
# burning its own retries against a known-throttled endpoint.
_rate_limit_lock = threading.Lock()
_rate_limit_cooldown_until = 0.0  # time.monotonic() timestamp


def _wait_for_shared_cooldown() -> None:
    """Sleep until the shared rate-limit cooldown (if any) has elapsed."""
    with _rate_limit_lock:
        wait = _rate_limit_cooldown_until - time.monotonic()
    if wait > 0:
        time.sleep(wait)


def _set_shared_cooldown(wait_seconds: float) -> None:
    """Extend the shared cooldown so concurrent refreshes back off together."""
    global _rate_limit_cooldown_until
    until = time.monotonic() + wait_seconds
    with _rate_limit_lock:
        if until > _rate_limit_cooldown_until:
            _rate_limit_cooldown_until = until


def _retry_after_seconds(resp: "Response") -> float | None:
    """Return a valid Retry-After header value in seconds, else None."""
    raw = resp.headers.get("Retry-After")
    if raw is None:
        return None
    try:
        seconds = float(raw)
    except (TypeError, ValueError):
        return None
    if seconds < 0:
        return None
    return seconds


def _get_session() -> "Session":
    """Lazy import to avoid circular dependency with strava_client."""
    from .strava_client.session import get_default_session

    return get_default_session()


def _mask_tail(value: str | None, visible: int = 4) -> str:
    """Mask all but the trailing ``visible`` chars; short values are fully masked."""
    if not value:
        return ""
    if len(value) <= visible:
        return "*" * len(value)
    return "****" + value[-visible:]


def _post_token_with_retry(payload: dict, logger: logging.Logger) -> "Response":
    """POST to the token endpoint, retrying capped 429s with shared cooldown.

    Honors the response's ``Retry-After`` header when present (falling back
    to exponential backoff) and records the wait in a process-wide cooldown
    so concurrent refreshes do not burn their own retries.

    Raises:
        TokenError: On transport failure or when the 429 retry cap is hit.
    """
    rate_limit_attempts = 0
    while True:
        _wait_for_shared_cooldown()
        try:
            resp = _get_session().post(
                STRAVA_OAUTH_URL, data=payload, timeout=REQUEST_TIMEOUT
            )
        except RequestException as e:  # Network / connection / timeout
            logger.exception("Token request transport error")
            raise TokenError("Transport failure during token refresh") from e

        if resp.status_code != 429:
            return resp

        rate_limit_attempts += 1
        if rate_limit_attempts > MAX_429_RETRIES:
            raise TokenError(
                f"Token refresh rate limited (429) after "
                f"{MAX_429_RETRIES} retries — giving up"
            )
        backoff = min(
            RATE_LIMIT_THROTTLE_SECONDS * (2 ** (rate_limit_attempts - 1)),
            _RATE_LIMIT_MAX_BACKOFF_SECONDS,
        )
        retry_after = _retry_after_seconds(resp)
        if retry_after is not None:
            backoff = min(retry_after, _RATE_LIMIT_MAX_BACKOFF_SECONDS)
        _set_shared_cooldown(backoff)
        logger.warning(
            "Token refresh rate limited (429); backing off %0.1fs (attempt %s/%s)",
            backoff,
            rate_limit_attempts,
            MAX_429_RETRIES,
        )
        time.sleep(backoff)


def _error_detail_from_json(resp: "Response", logger: logging.Logger) -> str | None:
    """Extract a human-readable error detail from a token error response."""
    try:
        data_err = resp.json()
    except ValueError as exc:
        logger.debug("Failed to parse token error response: %s", exc)
        return None
    if not isinstance(data_err, dict):
        return None
    joined: list[str] = []
    errors = data_err.get("errors")
    if isinstance(errors, list):
        for err in errors:
            if not isinstance(err, dict):
                continue
            code = err.get("code")
            field = err.get("field")
            if code and field:
                joined.append(f"{field}:{code}")
            elif code:
                joined.append(str(code))
    parts = [p for p in (data_err.get("message"), " ".join(joined) or None) if p]
    return " | ".join(parts) if parts else None


def _snippet_from_text(resp: "Response") -> str | None:
    """Return a short body snippet for diagnostics when no JSON detail exists."""
    text = getattr(resp, "text", "")
    if not isinstance(text, str):
        return None
    text = text.strip()
    if not text:
        return None
    return (text[:197] + "...") if len(text) > 200 else text


def _raise_token_http_error(resp: "Response", logger: logging.Logger) -> "NoReturn":
    """Log a token endpoint HTTP error and raise ``TokenError``."""
    status = resp.status_code
    detail = _error_detail_from_json(resp, logger)
    snippet = None if detail else _snippet_from_text(resp)
    logger.error(
        "Token refresh failed status=%s%s%s",
        status,
        f" detail={detail}" if detail else "",
        f" snippet={snippet}" if snippet else "",
    )
    raise TokenError(f"Token refresh failed with status {status}")


def get_access_token(
    refresh_token: str, runner_name: str | None = None
) -> Tuple[str, str | None]:
    """Exchange a refresh token for a new access (and possibly new refresh) token.

    Args:
        refresh_token: The existing Strava refresh token.
        runner_name: Optional runner identifier for contextual logging.

    Returns:
        (access_token, new_refresh_token) tuple. ``new_refresh_token`` is
        None when Strava did not rotate the refresh token.

    Raises:
        TokenError: If the HTTP request fails, the response JSON is invalid,
            or no access token is present in the response.
    """
    if not CLIENT_ID or not CLIENT_SECRET:
        raise TokenError(
            "Client credentials not configured (CLIENT_ID / CLIENT_SECRET missing)"
        )
    if not refresh_token:
        raise TokenError("Missing refresh token")

    payload = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
    }
    logger = logging.getLogger(__name__)
    masked_refresh = _mask_tail(refresh_token)
    runner_context = f" for runner={runner_name}" if runner_name else ""
    logger.info(
        "Refreshing Strava token%s refresh_token=%s", runner_context, masked_refresh
    )
    logger.debug("Token endpoint: %s", STRAVA_OAUTH_URL)
    logger.debug({"client_id": CLIENT_ID, "grant_type": payload["grant_type"]})

    # Tokens cannot be cached - they're short-lived and contain secrets.
    # In offline mode, fail immediately with a clear message.
    if _cache_mode_offline:
        raise TokenError(
            "Cannot refresh tokens in offline mode. "
            "Use STRAVA_API_CACHE_MODE=cache or live to authenticate."
        )

    resp = _post_token_with_retry(payload, logger)

    status = resp.status_code
    logger.debug("Token endpoint status=%s", status)
    if status >= 400:
        _raise_token_http_error(resp, logger)

    # Parse JSON success
    try:
        data = resp.json()
    except ValueError as e:
        logger.exception("Invalid JSON in token response")
        raise TokenError("Invalid JSON in token response") from e

    if not isinstance(data, dict):
        logger.error("Unexpected token response shape: %s", type(data).__name__)
        raise TokenError("Unexpected token response shape")

    access_token_raw = data.get("access_token")
    new_refresh_token_raw = data.get("refresh_token")
    access_len = len(access_token_raw) if isinstance(access_token_raw, str) else 0
    refresh_changed = (
        isinstance(new_refresh_token_raw, str)
        and new_refresh_token_raw
        and new_refresh_token_raw != refresh_token
    )
    logger.info(
        "Token refresh complete access_token_len=%s refresh_token_changed=%s",
        access_len,
        bool(refresh_changed),
    )
    if not isinstance(access_token_raw, str) or not access_token_raw:
        # Treat missing access token as error for downstream certainty
        logger.error("No access_token in token response")
        raise TokenError("No access_token in response")
    new_refresh_token: str | None
    if isinstance(new_refresh_token_raw, str) and new_refresh_token_raw:
        new_refresh_token = new_refresh_token_raw
    else:
        new_refresh_token = None
    return access_token_raw, new_refresh_token
