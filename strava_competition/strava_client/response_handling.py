"""Shared HTTP response helpers for Strava API interactions."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Type

import requests

from ..errors import (
    StravaAPIError,
    StravaPaymentRequiredError,
    StravaPermissionError,
    StravaResourceNotFoundError,
)
from ..models import Runner

RequestsJSONDecodeError: Type[Exception]
if hasattr(requests.exceptions, "JSONDecodeError"):
    RequestsJSONDecodeError = requests.exceptions.JSONDecodeError
else:  # pragma: no cover - fallback for old requests versions
    RequestsJSONDecodeError = ValueError


__all__ = [
    "classify_response_status",
    "extract_error",
]


def classify_response_status(
    runner: Runner,
    response: requests.Response,
    context: str,
    *,
    attempt: int,
    backoff: float,
    can_retry: bool,
) -> Tuple[str, Optional[Exception]]:
    """Return action for a non-success status: ok, retry, or raise."""

    status = response.status_code
    detail = extract_error(response)

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
        runner.payment_required = True
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


def extract_error(resp: Optional[requests.Response]) -> Optional[str]:
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
    except (
        ValueError,
        RequestsJSONDecodeError,
    ) as exc:  # pragma: no cover - logging path
        logging.debug(
            "Failed to decode JSON from %s: %s", getattr(resp, "url", "?"), exc
        )
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
