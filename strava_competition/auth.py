"""OAuth / token refresh utilities for Strava API.

This module focuses on securely exchanging a refresh token for an access token
using Strava's OAuth endpoint. It adds resiliency (HTTP retries), safe logging
that avoids leaking secrets, and defensive JSON parsing.
"""

from __future__ import annotations
import logging
from hashlib import sha256
from typing import TYPE_CHECKING, Tuple

from requests.exceptions import RequestException

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

if TYPE_CHECKING:
    from requests import Session


class TokenError(Exception):
    """Raised when token refresh fails (after retries)."""


def _get_session() -> "Session":
    """Lazy import to avoid circular dependency with strava_client."""
    from .strava_client.session import get_default_session

    return get_default_session()


def _mask_tail(value: str | None, visible: int = 4) -> str:
    if not value:
        return ""
    tail = value[-visible:]
    return f"****{tail}" if len(value) > visible else "****" + tail


def get_access_token(
    refresh_token: str, runner_name: str | None = None
) -> Tuple[str, str | None]:
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
            resp = _get_session().post(
                STRAVA_OAUTH_URL, data=payload, timeout=REQUEST_TIMEOUT
            )
        except RequestException as e:  # Network / connection / timeout
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
            except ValueError as exc:
                detail = None
                logger.debug("Failed to parse token error response: %s", exc)
            snippet = None
            if not detail:
                text = getattr(resp, "text", "")
                if isinstance(text, str):
                    text = text.strip()
                    if text:
                        snippet = (text[:197] + "...") if len(text) > 200 else text
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

    access_token_raw = data.get("access_token") if isinstance(data, dict) else None
    new_refresh_token_raw = (
        data.get("refresh_token") if isinstance(data, dict) else None
    )
    access_len = len(access_token_raw) if isinstance(access_token_raw, str) else 0
    refresh_changed = (
        isinstance(new_refresh_token_raw, str)
        and new_refresh_token_raw
        and new_refresh_token_raw != refresh_token
    )
    logger.info(
        "Token refresh %s access_token_len=%s refresh_token_changed=%s",
        response_source,
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
