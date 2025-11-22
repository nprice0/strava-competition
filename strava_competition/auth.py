"""OAuth / token refresh utilities for Strava API.

This module focuses on securely exchanging a refresh token for an access token
using Strava's OAuth endpoint. It adds resiliency (HTTP retries), safe logging
that avoids leaking secrets, and defensive JSON parsing.
"""

from __future__ import annotations
import logging
from hashlib import sha256
from typing import Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import (
    CLIENT_ID,
    CLIENT_SECRET,
    STRAVA_OAUTH_URL,
    REQUEST_TIMEOUT,
    STRAVA_API_CAPTURE_ENABLED,
    STRAVA_API_REPLAY_ENABLED,
    STRAVA_TOKEN_CAPTURE_ENABLED,
)
from .api_capture import record_response, replay_response

# Reusable session with limited retry for transient network/server issues.
_token_retry = Retry(
    total=3,
    backoff_factor=1.0,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["POST"],
)
_session = requests.Session()
_session.mount("https://", HTTPAdapter(max_retries=_token_retry))
_session.mount("http://", HTTPAdapter(max_retries=_token_retry))


class TokenError(Exception):
    """Raised when token refresh fails (after retries)."""


def _mask_tail(value: str | None, visible: int = 4) -> str:
    if not value:
        return ""
    tail = value[-visible:]
    return f"****{tail}" if len(value) > visible else "****" + tail


def get_access_token(
    refresh_token: str, runner_name: str | None = None
) -> Tuple[str | None, str | None]:
    """Exchange a refresh token for a new access (and possibly new refresh) token.

    Args:
        refresh_token: The existing Strava refresh token.
        runner_name: Optional runner identifier for contextual logging.

    Returns:
        (access_token, refresh_token) tuple. Either element may be None on unexpected response.

    Raises:
        TokenError: If the HTTP request fails or JSON is invalid / lacks tokens.
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
    if runner_name:
        logger.info(
            "Refreshing Strava token for runner=%s refresh_token=%s",
            runner_name,
            masked_refresh,
        )
    else:
        logger.info("Refreshing Strava token refresh_token=%s", masked_refresh)
    logger.debug("Token endpoint: %s", STRAVA_OAUTH_URL)
    logger.debug({"client_id": CLIENT_ID, "grant_type": payload["grant_type"]})

    capture_body = None
    identity = runner_name or "token"
    cached = None
    if STRAVA_TOKEN_CAPTURE_ENABLED:
        hashed_refresh = sha256(refresh_token.encode("utf-8")).hexdigest()
        capture_body = {
            "client_id": CLIENT_ID,
            "grant_type": payload["grant_type"],
            "refresh_token_hash": hashed_refresh,
        }
        if STRAVA_API_REPLAY_ENABLED:
            cached = replay_response(
                "POST",
                STRAVA_OAUTH_URL,
                identity,
                body=capture_body,
            )
    response_source = "cache" if cached is not None else "live"
    data = None
    if cached is not None:
        if not isinstance(cached, dict):
            logger.error(
                "Cached token response had unexpected type %s", type(cached).__name__
            )
            raise TokenError("Cached token response is invalid")
        data = cached
    else:
        try:
            resp = _session.post(
                STRAVA_OAUTH_URL, data=payload, timeout=REQUEST_TIMEOUT
            )
        except (
            requests.exceptions.RequestException
        ) as e:  # Network / connection / timeout
            logger.error("Token request transport error: %s", e)
            raise TokenError("Transport failure during token refresh") from e

        status = resp.status_code
        logger.debug("Token endpoint status=%s", status)
        if status >= 400:
            # Attempt to parse error JSON for context
            detail = None
            try:
                data_err = resp.json()
                if isinstance(data_err, dict):
                    msg = data_err.get("message")
                    errors = data_err.get("errors")
                    if msg:
                        detail = msg
                    if isinstance(errors, list):
                        joined = []
                        for err in errors:
                            if isinstance(err, dict):
                                code = err.get("code")
                                field = err.get("field")
                                if code and field:
                                    joined.append(f"{field}:{code}")
                                elif code:
                                    joined.append(str(code))
                        if joined:
                            detail = (
                                f"{detail} | {' '.join(joined)}"
                                if detail
                                else " ".join(joined)
                            )
            except Exception:
                detail = None
            snippet = None
            if not detail:
                try:
                    text = resp.text.strip()
                    if text:
                        snippet = (text[:197] + "...") if len(text) > 200 else text
                except Exception:
                    pass
            logger.error(
                "Token refresh failed status=%s%s%s",
                status,
                f" detail={detail}" if detail else "",
                f" snippet={snippet}" if snippet else "",
            )
            raise TokenError(f"Token refresh failed with status {status}")

        # Parse JSON success
        try:
            data = resp.json()
        except ValueError as e:
            logger.error("Invalid JSON in token response: %s", e)
            raise TokenError("Invalid JSON in token response") from e

        if not isinstance(data, dict):
            logger.error("Unexpected token response shape: %s", type(data).__name__)
            raise TokenError("Unexpected token response shape")

        if STRAVA_TOKEN_CAPTURE_ENABLED and STRAVA_API_CAPTURE_ENABLED:
            record_response(
                "POST",
                STRAVA_OAUTH_URL,
                identity,
                response=data,
                body=capture_body,
            )

    access_token = data.get("access_token") if isinstance(data, dict) else None
    new_refresh_token = data.get("refresh_token") if isinstance(data, dict) else None
    logger.info(
        "Token refresh %s access_token_len=%s refresh_token_changed=%s",
        response_source,
        len(access_token) if access_token else 0,
        bool(new_refresh_token and new_refresh_token != refresh_token),
    )
    if not access_token:
        # Treat missing access token as error for downstream certainty
        logger.error("No access_token in token response")
        raise TokenError("No access_token in response")
    return access_token, new_refresh_token
