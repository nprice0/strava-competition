"""Capture/replay helpers shared by Strava API modules."""

from __future__ import annotations

import logging
from hashlib import sha256
from typing import TYPE_CHECKING, Any, Dict, List, Optional, TypeAlias

from ..api_capture import (
    CaptureRecord,
    record_response,
    replay_response,
    replay_response_with_meta,
)
from ..config import (
    STRAVA_API_CAPTURE_ENABLED,
    STRAVA_API_REPLAY_ENABLED,
    STRAVA_OFFLINE_MODE,
    STRAVA_CAPTURE_HASH_IDENTIFIERS,
    STRAVA_CAPTURE_ID_SALT,
)
from ..errors import StravaAPIError

if TYPE_CHECKING:  # pragma: no cover
    from ..models import Runner

JSONList: TypeAlias = List[Dict[str, Any]]

__all__ = [
    "runner_identity",
    "replay_list_response",
    "replay_list_response_with_meta",
    "record_list_response",
]


def runner_identity(
    runner: "Runner",
    *,
    hash_identifiers: Optional[bool] = None,
    salt: Optional[str] = None,
) -> str:
    """Return a stable, privacy-safe identifier for capture/replay naming."""

    raw_source = runner.strava_id or runner.name or "unknown"
    raw = str(raw_source)
    hashed = (
        STRAVA_CAPTURE_HASH_IDENTIFIERS
        if hash_identifiers is None
        else hash_identifiers
    )
    if not hashed:
        return raw
    salt_value = STRAVA_CAPTURE_ID_SALT if salt is None else salt
    if not salt_value:
        raise RuntimeError(
            "STRAVA_CAPTURE_HASH_IDENTIFIERS requires STRAVA_CAPTURE_ID_SALT to be set"
        )
    digest = sha256(f"{raw}:{salt_value}".encode("utf-8")).hexdigest()
    return digest


def _handle_offline_miss(
    context_label: str,
    runner_name: str,
    *,
    offline_mode: Optional[bool] = None,
) -> None:
    offline = STRAVA_OFFLINE_MODE if offline_mode is None else offline_mode
    if not offline:
        return
    message = f"{context_label} cache miss for runner {runner_name} while STRAVA_OFFLINE_MODE is enabled"
    logging.error(message)
    raise StravaAPIError(message)


def replay_list_response(
    runner: "Runner",
    url: str,
    params: Dict[str, Any],
    *,
    context_label: str,
    page: int,
    replay_enabled: Optional[bool] = None,
    offline_mode: Optional[bool] = None,
    hash_identifiers: Optional[bool] = None,
    salt: Optional[str] = None,
) -> Optional[JSONList]:
    """Return cached list payload when replay mode is active."""

    enabled = STRAVA_API_REPLAY_ENABLED if replay_enabled is None else replay_enabled
    if not enabled:
        return None
    cached = replay_response(
        "GET",
        url,
        runner_identity(
            runner,
            hash_identifiers=hash_identifiers,
            salt=salt,
        ),
        params=params,
    )
    if cached is None:
        _handle_offline_miss(
            context_label,
            runner.name,
            offline_mode=offline_mode,
        )
        return None
    if isinstance(cached, list):
        logging.debug(
            "Replay hit for %s runner=%s page=%s entries=%s",
            context_label,
            runner.name,
            page,
            len(cached),
        )
        return cached
    logging.warning(
        "Replay payload type mismatch for %s runner=%s page=%s type=%s",
        context_label,
        runner.name,
        page,
        type(cached).__name__,
    )
    return None


def replay_list_response_with_meta(
    runner: "Runner",
    url: str,
    params: Dict[str, Any],
    *,
    context_label: str,
    page: int,
    replay_enabled: Optional[bool] = None,
    offline_mode: Optional[bool] = None,
    hash_identifiers: Optional[bool] = None,
    salt: Optional[str] = None,
) -> Optional[CaptureRecord]:
    """Return cached payload with metadata when replaying list endpoints."""

    enabled = STRAVA_API_REPLAY_ENABLED if replay_enabled is None else replay_enabled
    if not enabled:
        return None
    record = replay_response_with_meta(
        "GET",
        url,
        runner_identity(
            runner,
            hash_identifiers=hash_identifiers,
            salt=salt,
        ),
        params=params,
    )
    if record is None:
        _handle_offline_miss(
            context_label,
            runner.name,
            offline_mode=offline_mode,
        )
        return None
    if isinstance(record.response, list):
        logging.debug(
            "Replay hit(meta) for %s runner=%s page=%s entries=%s",
            context_label,
            runner.name,
            page,
            len(record.response),
        )
        return record
    logging.warning(
        "Replay payload type mismatch for %s runner=%s page=%s type=%s",
        context_label,
        runner.name,
        page,
        type(record.response).__name__,
    )
    return None


def record_list_response(
    runner: "Runner",
    url: str,
    params: Dict[str, Any],
    data: JSONList,
    *,
    capture_enabled: Optional[bool] = None,
    hash_identifiers: Optional[bool] = None,
    salt: Optional[str] = None,
) -> None:
    """Persist successful list responses when capture mode is enabled."""

    enabled = STRAVA_API_CAPTURE_ENABLED if capture_enabled is None else capture_enabled
    if not enabled:
        return
    record_response(
        "GET",
        url,
        runner_identity(
            runner,
            hash_identifiers=hash_identifiers,
            salt=salt,
        ),
        response=data,
        params=params,
    )
