"""Shared authenticated HTTP helper for CLI tools.

Provides a single canonical ``http_get`` used by every standalone tool
so that auth headers, logging, timeout handling, and 429 retry behaviour
are consistent.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

from strava_competition.config import REQUEST_TIMEOUT

LOGGER = logging.getLogger(__name__)

# Capped retry budget for HTTP 429 responses.
_RATE_LIMIT_MAX_RETRIES = 3
_RATE_LIMIT_MAX_WAIT_SECONDS = 60.0
_RATE_LIMIT_BACKOFF_BASE_SECONDS = 2.0


def _retry_after_seconds(response: requests.Response, attempt: int) -> float:
    """Return the wait before retrying a 429, honouring ``Retry-After``.

    Falls back to exponential backoff when the header is missing or
    non-numeric; the wait is always capped.
    """
    header = response.headers.get("Retry-After")
    wait: float | None = None
    if header is not None:
        try:
            wait = float(header)
        except ValueError:
            wait = None
    if wait is None or wait < 0:
        wait = _RATE_LIMIT_BACKOFF_BASE_SECONDS * (2**attempt)
    return min(wait, _RATE_LIMIT_MAX_WAIT_SECONDS)


def http_get(
    url: str,
    token: str,
    *,
    params: dict[str, Any] | None = None,
) -> Any:
    """Perform an authenticated GET request against the Strava API.

    HTTP 429 responses are retried up to ``_RATE_LIMIT_MAX_RETRIES`` times,
    honouring a numeric ``Retry-After`` header (capped exponential backoff
    otherwise). All other non-2xx responses raise immediately.

    Args:
        url: Full endpoint URL.
        token: Bearer access token.
        params: Optional query parameters.

    Returns:
        Parsed JSON response body.

    Raises:
        requests.HTTPError: On non-2xx responses (including a 429 that
            persists after the retry budget is exhausted).
    """
    headers = {"Authorization": f"Bearer {token}"}
    for attempt in range(_RATE_LIMIT_MAX_RETRIES + 1):
        LOGGER.debug("GET %s params=%s (attempt %d)", url, params, attempt + 1)
        response = requests.get(
            url,
            headers=headers,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
        if response.status_code == 429 and attempt < _RATE_LIMIT_MAX_RETRIES:
            wait = _retry_after_seconds(response, attempt)
            LOGGER.warning(
                "Rate limited (429) on %s; retrying in %.1fs (attempt %d/%d)",
                url,
                wait,
                attempt + 1,
                _RATE_LIMIT_MAX_RETRIES,
            )
            time.sleep(wait)
            continue
        response.raise_for_status()
        return response.json()
    raise RuntimeError("unreachable")  # pragma: no cover - loop always returns/raises
