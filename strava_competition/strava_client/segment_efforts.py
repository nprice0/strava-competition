"""Segment efforts pagination and caching orchestration."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, TypeAlias, cast

import requests

from ..api_capture import CaptureRecord, save_overlay_to_cache, save_response_to_cache
from ..config import (
    _cache_mode_saves,
    STRAVA_CACHE_OVERWRITE,
    _cache_mode_reads,
    STRAVA_BASE_URL,
    _cache_mode_offline,
)
from ..errors import StravaAPIError
from ..models import Runner
from ..replay_tail import chunk_activities, summarize_activities, merge_activity_lists
from .base import ensure_runner_token
from .capture import (
    save_list_to_cache,
    get_cached_list_with_meta,
    runner_identity,
)
from .pagination import fetch_page_with_retries
from .rate_limiter import RateLimiter
from .response_handling import extract_error
from .session import get_default_session

JSONList: TypeAlias = List[Dict[str, object]]

LOGGER = logging.getLogger(__name__)
SEGMENT_PAGE_SIZE = 200


@dataclass(slots=True)
class CachedPage:
    params: Dict[str, Any]
    data: JSONList
    record: Optional[CaptureRecord]


class SegmentEffortsAPI:
    """Low-level helper responsible for getting segment efforts."""

    def __init__(
        self,
        *,
        session: requests.Session | None = None,
        limiter: RateLimiter | None = None,
    ) -> None:
        self._session = session or get_default_session()
        self._limiter = limiter or RateLimiter()

    def get_segment_efforts(
        self,
        runner: Runner,
        segment_id: int,
        start_date: datetime,
        end_date: datetime,
    ) -> Optional[List[Dict[str, object]]]:
        """Fetch all efforts for ``segment_id`` between ``start_date`` and ``end_date``."""

        cache_available = _cache_mode_reads
        base_params = {
            "segment_id": segment_id,
            "start_date_local": start_date.isoformat(),
            "end_date_local": end_date.isoformat(),
            "per_page": SEGMENT_PAGE_SIZE,
        }
        url = f"{STRAVA_BASE_URL}/segment_efforts"
        cached_pages: List[CachedPage] = []
        used_cache = False

        def fetch_page(page: int) -> JSONList:
            nonlocal cache_available, used_cache
            params = dict(base_params)
            params["page"] = page
            if cache_available:
                record = get_cached_list_with_meta(
                    runner,
                    url,
                    params,
                    context_label="segment_efforts",
                    page=page,
                    require_cache=_cache_mode_offline,
                )
                if record is not None:
                    used_cache = True
                    cached_data = cast(JSONList, record.response)
                    cached_pages.append(
                        CachedPage(
                            params=dict(params),
                            data=cached_data,
                            record=record,
                        )
                    )
                    return cached_data
                if _cache_mode_offline:
                    raise StravaAPIError(
                        "Cache-only mode miss for segment efforts"
                        f" (runner={runner.name} segment={segment_id} page={page})"
                    )
            ensure_runner_token(runner)
            data = fetch_page_with_retries(
                runner=runner,
                url=url,
                params=params,
                context_label="segment_efforts",
                page=page,
                session=self._session,
                limiter=self._limiter,
                segment_id=segment_id,
            )
            save_list_to_cache(
                runner,
                url,
                dict(params),
                data,
                save_to_cache=_cache_mode_saves,
            )
            return data

        try:
            ensure_runner_token(runner)
            page = 1
            attempts_refresh = False
            all_efforts: List[Dict[str, object]] = []
            while True:
                try:
                    batch = fetch_page(page)
                except requests.exceptions.HTTPError as exc:
                    if (
                        exc.response is not None
                        and exc.response.status_code == 401
                        and not attempts_refresh
                    ):
                        LOGGER.info(
                            "Segment efforts 401 for runner %s; refreshing token and retrying page %s",
                            runner.name,
                            page,
                        )
                        runner.access_token = None
                        ensure_runner_token(runner)
                        attempts_refresh = True
                        batch = fetch_page(page)
                    else:
                        raise
                all_efforts.extend(batch)
                if len(batch) < SEGMENT_PAGE_SIZE:
                    break
                page += 1
            if used_cache and _cache_mode_saves:
                all_efforts, refreshed = _maybe_refresh_segment_tail(
                    runner,
                    segment_id,
                    url,
                    base_params,
                    cached_pages,
                    all_efforts,
                    self._session,
                    self._limiter,
                    start_date,
                    end_date,
                )
                if refreshed:
                    LOGGER.info(
                        "Segment tail refresh runner=%s segment=%s performed incremental update",
                        runner.name,
                        segment_id,
                    )
            return all_efforts
        except (
            requests.exceptions.HTTPError
        ) as exc:  # pragma: no cover - defensive logging
            self._log_http_error(runner, exc)
            return None

    @staticmethod
    def _log_http_error(
        runner: Runner,
        exc: requests.exceptions.HTTPError,
    ) -> None:
        resp = exc.response
        if resp is None:
            LOGGER.error(
                "HTTPError with no response object (network/transport issue) for runner %s: %s",
                runner.name,
                exc,
            )
            return
        detail = extract_error(resp)
        status = resp.status_code
        raw_snippet: Optional[str] = None
        try:
            text = (resp.text or "").strip()
            if text:
                raw_snippet = (text[:297] + "...") if len(text) > 300 else text
        except Exception:  # pragma: no cover - diagnostics only
            raw_snippet = None
        suffix = ""
        if detail and raw_snippet:
            suffix = f" | {detail} | raw: {raw_snippet}"
        elif detail:
            suffix = f" | {detail}"
        elif raw_snippet:
            suffix = f" | raw: {raw_snippet}"
        if status == 401:
            LOGGER.warning(
                "Skipping runner %s: Unauthorised (invalid/expired token after retry)%s",
                runner.name,
                suffix,
            )
        elif status == 402:
            runner.payment_required = True
            LOGGER.warning(
                "Runner %s: 402 Payment Required%s. Skipping.",
                runner.name,
                suffix,
            )
        else:
            LOGGER.error(
                "Error for runner %s: HTTP %s%s",
                runner.name,
                status,
                suffix,
            )


def _maybe_refresh_segment_tail(
    runner: Runner,
    segment_id: int,
    url: str,
    base_params: Dict[str, Any],
    cached_pages: List[CachedPage],
    existing_efforts: List[Dict[str, object]],
    session: requests.Session,
    limiter: RateLimiter,
    start_date: datetime,
    end_date: datetime,
) -> tuple[List[Dict[str, object]], bool]:
    if not cached_pages:
        return existing_efforts, False
    cached_payloads = _flatten_pages(page.data for page in cached_pages)
    stats = summarize_activities(cached_payloads)
    latest_cached = stats.latest
    if latest_cached is None:
        return existing_efforts, False
    window_start = _to_utc(start_date)
    window_end = _to_utc(end_date)
    if latest_cached >= window_end:
        return existing_efforts, False
    tail_start = max(latest_cached, window_start)
    tail_pages = _fetch_segment_tail_pages(
        runner,
        segment_id,
        url,
        base_params,
        tail_start,
        window_end,
        session,
        limiter,
    )
    if not tail_pages:
        return existing_efforts, False
    tail_flat = _flatten_pages(tail_pages)
    merged = merge_activity_lists(tail_flat, existing_efforts)
    paged = chunk_activities(merged, chunk_size=SEGMENT_PAGE_SIZE)
    _persist_segment_pages(runner, url, base_params, paged)
    return merged, True


def _fetch_segment_tail_pages(
    runner: Runner,
    segment_id: int,
    url: str,
    base_params: Dict[str, Any],
    tail_start: datetime,
    window_end: datetime,
    session: requests.Session,
    limiter: RateLimiter,
) -> List[JSONList]:
    boundary = tail_start + timedelta(seconds=1)
    if boundary >= window_end:
        return []
    per_page = int(base_params.get("per_page", SEGMENT_PAGE_SIZE))
    tail_pages: List[JSONList] = []
    page = 1
    while True:
        params = dict(base_params)
        params["start_date_local"] = boundary.isoformat()
        params["end_date_local"] = window_end.isoformat()
        params["page"] = page
        data = fetch_page_with_retries(
            runner=runner,
            url=url,
            params=params,
            context_label="segment_efforts_tail",
            page=page,
            session=session,
            limiter=limiter,
            segment_id=segment_id,
        )
        if not data:
            break
        tail_pages.append(data)
        save_list_to_cache(
            runner,
            url,
            dict(params),
            data,
            save_to_cache=_cache_mode_saves,
        )
        if len(data) < per_page:
            break
        page += 1
    return tail_pages


def _persist_segment_pages(
    runner: Runner,
    url: str,
    base_params: Dict[str, Any],
    page_payloads: List[JSONList],
) -> None:
    if not _cache_mode_saves:
        LOGGER.warning(
            "Cache saving disabled; cannot persist enriched segment data for runner=%s",
            runner.name,
        )
        return
    identity = runner_identity(runner)
    template = {
        "segment_id": base_params.get("segment_id"),
        "start_date_local": base_params.get("start_date_local"),
        "end_date_local": base_params.get("end_date_local"),
        "per_page": base_params.get("per_page", SEGMENT_PAGE_SIZE),
    }
    if not page_payloads:
        params = dict(template)
        params["page"] = 1
        _write_segment_overlay(identity, url, params, [])
        return
    for idx, payload in enumerate(page_payloads, start=1):
        params = dict(template)
        params["page"] = idx
        _write_segment_overlay(identity, url, params, payload)
    params = dict(template)
    params["page"] = len(page_payloads) + 1
    _write_segment_overlay(identity, url, params, [])


def _write_segment_overlay(
    identity: str,
    url: str,
    params: Dict[str, Any],
    payload: JSONList,
) -> None:
    if STRAVA_CACHE_OVERWRITE:
        save_response_to_cache("GET", url, identity, payload, params=params)
    else:
        save_overlay_to_cache("GET", url, identity, payload, params=params)


def _flatten_pages(pages: Iterable[JSONList]) -> List[Dict[str, object]]:
    flattened: List[Dict[str, object]] = []
    for page in pages:
        flattened.extend(page)
    return flattened


def _to_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)
