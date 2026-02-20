"""Shared authenticated HTTP helper for CLI tools.

Provides a single canonical ``http_get`` used by every standalone tool
so that auth headers, logging, and timeout handling are consistent.
"""

from __future__ import annotations

import logging
from typing import Any

import requests

from strava_competition.config import REQUEST_TIMEOUT

LOGGER = logging.getLogger(__name__)


def http_get(
    url: str,
    token: str,
    *,
    params: dict[str, Any] | None = None,
) -> Any:
    """Perform an authenticated GET request against the Strava API.

    Args:
        url: Full endpoint URL.
        token: Bearer access token.
        params: Optional query parameters.

    Returns:
        Parsed JSON response body.

    Raises:
        requests.HTTPError: On non-2xx responses.
    """
    headers = {"Authorization": f"Bearer {token}"}
    LOGGER.debug("GET %s params=%s", url, params)
    response = requests.get(
        url,
        headers=headers,
        params=params,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()
