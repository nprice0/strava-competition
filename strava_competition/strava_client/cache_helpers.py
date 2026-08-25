"""Cache helpers shared by Strava API modules."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, TypeAlias

from ..api_capture import (
    CaptureRecord,
    save_response_to_cache,
    get_cached_response,
    get_cached_response_with_meta,
)
from ..config import (
    _cache_mode_saves,
    _cache_mode_reads,
    _cache_mode_offline,
)
from ..errors import StravaAPIError
from . import telemetry

if TYPE_CHECKING:  # pragma: no cover
    from ..models import Runner

JSONList: TypeAlias = List[Dict[str, Any]]

__all__ = [
    "runner_identity",
    "get_cached_list",
    "get_cached_list_with_meta",
    "save_list_to_cache",
]


def runner_identity(runner: "Runner") -> str:
    """Return a stable identifier for cache file naming."""

    return str(runner.strava_id or runner.name or "unknown")


def _handle_offline_miss(
    context_label: str,
    runner_name: str,
    *,
    require_cache: Optional[bool] = None,
) -> None:
    offline = _cache_mode_offline if require_cache is None else require_cache
    if not offline:
        return
    message = f"{context_label} cache miss for runner {runner_name} in cache-only mode"
    logging.error(message)
    raise StravaAPIError(message)


def get_cached_list(
    runner: "Runner",
    url: str,
    params: Dict[str, Any],
    *,
    context_label: str,
    page: int,
    use_cache: Optional[bool] = None,
    require_cache: Optional[bool] = None,
) -> Optional[JSONList]:
    """Return cached list payload when cache reading is active."""

    enabled = _cache_mode_reads if use_cache is None else use_cache
    if not enabled:
        return None
    cached = get_cached_response(
        "GET",
        url,
        runner_identity(runner),
        params=params,
    )
    if cached is None:
        _handle_offline_miss(
            context_label,
            runner.name,
            require_cache=require_cache,
        )
        return None
    if isinstance(cached, list):
        telemetry.increment(telemetry.CACHE_HITS)
        logging.debug(
            "Cache hit for %s runner=%s page=%s entries=%s",
            context_label,
            runner.name,
            page,
            len(cached),
        )
        return cached
    logging.warning(
        "Cache payload type mismatch for %s runner=%s page=%s type=%s",
        context_label,
        runner.name,
        page,
        type(cached).__name__,
    )
    _handle_offline_miss(
        context_label,
        runner.name,
        require_cache=require_cache,
    )
    return None


def get_cached_list_with_meta(
    runner: "Runner",
    url: str,
    params: Dict[str, Any],
    *,
    context_label: str,
    page: int,
    use_cache: Optional[bool] = None,
    require_cache: Optional[bool] = None,
) -> Optional[CaptureRecord]:
    """Return cached payload with metadata when reading list endpoints."""

    enabled = _cache_mode_reads if use_cache is None else use_cache
    if not enabled:
        return None
    record = get_cached_response_with_meta(
        "GET",
        url,
        runner_identity(runner),
        params=params,
    )
    if record is None:
        _handle_offline_miss(
            context_label,
            runner.name,
            require_cache=require_cache,
        )
        return None
    if isinstance(record.response, list):
        telemetry.increment(telemetry.CACHE_HITS)
        logging.debug(
            "Cache hit(meta) for %s runner=%s page=%s entries=%s",
            context_label,
            runner.name,
            page,
            len(record.response),
        )
        return record
    logging.warning(
        "Cache payload type mismatch for %s runner=%s page=%s type=%s",
        context_label,
        runner.name,
        page,
        type(record.response).__name__,
    )
    _handle_offline_miss(
        context_label,
        runner.name,
        require_cache=require_cache,
    )
    return None


def save_list_to_cache(
    runner: "Runner",
    url: str,
    params: Dict[str, Any],
    data: JSONList,
    *,
    save_to_cache: Optional[bool] = None,
) -> None:
    """Persist successful list responses when cache saving is enabled."""

    enabled = _cache_mode_saves if save_to_cache is None else save_to_cache
    if not enabled:
        return
    save_response_to_cache(
        "GET",
        url,
        runner_identity(runner),
        response=data,
        params=params,
    )
