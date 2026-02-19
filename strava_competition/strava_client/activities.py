"""Activities fetcher with cache tail refresh logic."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, TypeAlias, cast

import requests
from cachetools import TTLCache

from ..activity_types import activity_type_matches, normalize_activity_type
from ..api_capture import save_overlay_to_cache, save_response_to_cache, CaptureRecord
from ..config import (
    CACHE_TAIL_OVERLAP_SECONDS,
    CACHE_EMPTY_REFRESH_SECONDS,
    CACHE_MAX_LOOKBACK_DAYS,
    _cache_mode_saves,
    STRAVA_CACHE_OVERWRITE,
    _cache_mode_reads,
    STRAVA_BASE_URL,
    STRAVA_CACHE_HASH_IDENTIFIERS,
    STRAVA_CACHE_ID_SALT,
    _cache_mode_offline,
)
from ..models import Runner
from ..replay_tail import (
    chunk_activities,
    clamp_window,
    exceeds_lookback,
    merge_activity_lists,
    summarize_activities,
)
from ..utils import to_utc_aware
from .base import ensure_runner_token
from .capture import (
    save_list_to_cache,
    get_cached_list_with_meta,
    runner_identity,
)
from .pagination import fetch_page_with_retries
from .response_handling import extract_error
from .rate_limiter import RateLimiter
from .session import get_default_session

JSONList: TypeAlias = List[Dict[str, Any]]

LOGGER = logging.getLogger(__name__)
ACTIVITY_PAGE_SIZE = 200


@dataclass
class CachedPage:
    params: Dict[str, Any]
    data: JSONList
    record: Optional[CaptureRecord]


_runner_tail_lock = threading.Lock()
# Use TTLCache to prevent unbounded growth - entries expire after 1 hour
_runner_tail_refreshed_until: TTLCache[str, datetime] = TTLCache(maxsize=1000, ttl=3600)


def _to_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _runner_refresh_deadline(runner_id: str) -> Optional[datetime]:
    with _runner_tail_lock:
        return _runner_tail_refreshed_until.get(runner_id)


def _mark_runner_refreshed(runner_id: str, refresh_until: datetime) -> None:
    with _runner_tail_lock:
        current = _runner_tail_refreshed_until.get(runner_id)
        if current is None or refresh_until > current:
            _runner_tail_refreshed_until[runner_id] = refresh_until


def _flatten_pages(pages: List[JSONList]) -> List[Dict[str, Any]]:
    flattened: List[Dict[str, Any]] = []
    for page in pages:
        flattened.extend(page)
    return flattened


class ActivitiesAPI:
    def __init__(
        self,
        *,
        session: requests.Session | None = None,
        limiter: RateLimiter | None = None,
    ) -> None:
        self._session = session or get_default_session()
        self._limiter = limiter or RateLimiter()

    def get_activities(
        self,
        runner: Runner,
        start_date: datetime,
        end_date: datetime,
        *,
        activity_types: Optional[Iterable[str]] = ("Run",),
        max_pages: Optional[int] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """Fetch activities for a runner in [start_date, end_date]."""

        after_ts = int(start_date.timestamp())
        before_ts = int(end_date.timestamp())
        url = f"{STRAVA_BASE_URL}/athlete/activities"
        base_params = {
            "after": after_ts,
            "before": before_ts,
            "per_page": ACTIVITY_PAGE_SIZE,
        }

        raw_pages: List[JSONList] = []
        cached_pages: List[CachedPage] = []
        used_cache = False
        cache_available = _cache_mode_reads

        def fetch_page(page: int) -> JSONList:
            nonlocal used_cache, cache_available
            params = dict(base_params)
            params["page"] = page
            cache_record: Optional[CaptureRecord] = None
            if cache_available:
                cache_record = get_cached_list_with_meta(
                    runner,
                    url,
                    params,
                    context_label="activities",
                    page=page,
                    use_cache=_cache_mode_reads,
                    require_cache=_cache_mode_offline,
                    hash_identifiers=STRAVA_CACHE_HASH_IDENTIFIERS,
                    salt=STRAVA_CACHE_ID_SALT,
                )
                if cache_record is not None:
                    used_cache = True
                    cached_data = cast(JSONList, cache_record.response)
                    cached_pages.append(
                        CachedPage(
                            params=dict(params),
                            data=cached_data,
                            record=cache_record,
                        )
                    )
                    return cached_data
            result = fetch_page_with_retries(
                runner=runner,
                url=url,
                params=params,
                context_label="activities",
                page=page,
                session=self._session,
                limiter=self._limiter,
            )
            if isinstance(result, list):
                save_list_to_cache(
                    runner,
                    url,
                    dict(params),
                    result,
                    save_to_cache=_cache_mode_saves,
                    hash_identifiers=STRAVA_CACHE_HASH_IDENTIFIERS,
                    salt=STRAVA_CACHE_ID_SALT,
                )
                return result
            return []

        try:
            ensure_runner_token(runner)
            normalized_types: Optional[set[str]] = None
            if activity_types:
                normalized_types = {
                    normalized
                    for normalized in (
                        normalize_activity_type(value) for value in activity_types
                    )
                    if normalized
                }
            page = 1
            attempted_refresh = False
            while True:
                try:
                    data = fetch_page(page)
                except requests.exceptions.HTTPError as exc:
                    if (
                        exc.response is not None
                        and exc.response.status_code == 401
                        and not attempted_refresh
                    ):
                        LOGGER.info(
                            "401 for runner %s (activities). Refreshing token and retrying page %s.",
                            runner.name,
                            page,
                        )
                        runner.access_token = None
                        ensure_runner_token(runner)
                        attempted_refresh = True
                        data = fetch_page(page)
                    else:
                        raise
                raw_pages.append(data)
                if not data:
                    break
                if len(data) < ACTIVITY_PAGE_SIZE:
                    break
                if max_pages is not None and page >= max_pages:
                    break
                page += 1

            raw_activities = _flatten_pages(raw_pages)
            if used_cache and _cache_mode_saves:
                raw_activities, refreshed = _maybe_refresh_cache_tail(
                    runner,
                    url,
                    base_params,
                    cached_pages,
                    raw_activities,
                    self._session,
                    self._limiter,
                    start_date=start_date,
                    end_date=end_date,
                )
                if refreshed:
                    raw_pages = chunk_activities(
                        raw_activities,
                        chunk_size=ACTIVITY_PAGE_SIZE,
                    )

            filtered: List[Dict[str, Any]] = []
            start_utc = to_utc_aware(start_date)
            end_utc = to_utc_aware(end_date)
            for act in raw_activities:
                if normalized_types and not activity_type_matches(
                    act, normalized_types
                ):
                    continue
                start_local = act.get("start_date_local")
                if not start_local:
                    continue
                try:
                    dt = to_utc_aware(
                        datetime.fromisoformat(start_local.replace("Z", "+00:00"))
                    )
                except ValueError:
                    continue
                if start_utc <= dt <= end_utc:
                    filtered.append(act)
            return filtered
        except requests.exceptions.HTTPError as exc:  # pragma: no cover
            resp = exc.response
            if resp is None:
                LOGGER.error(
                    "HTTPError (activities) no response object for runner %s: %s",
                    runner.name,
                    exc,
                )
                return None
            detail = extract_error(resp)
            LOGGER.error(
                "Activities fetch error runner=%s status=%s detail=%s",
                runner.name,
                resp.status_code,
                detail,
            )
            return None


def _maybe_refresh_cache_tail(
    runner: Runner,
    url: str,
    base_params: Dict[str, Any],
    cached_pages: List[CachedPage],
    raw_activities: List[Dict[str, Any]],
    session: requests.Session,
    limiter: RateLimiter,
    *,
    start_date: datetime,
    end_date: datetime,
) -> tuple[List[Dict[str, Any]], bool]:
    if not cached_pages:
        return raw_activities, False
    cached_payloads = _flatten_pages([page.data for page in cached_pages])
    stats = summarize_activities(cached_payloads)
    start_utc, end_utc = clamp_window(_to_utc(start_date), _to_utc(end_date))
    latest_cached = stats.latest
    if latest_cached is None:
        stale_age = _stale_empty_cache_age(cached_pages)
        if stale_age is None:
            return raw_activities, False
        latest_cached = start_utc
        LOGGER.info(
            (
                "Cache for runner=%s had no activities and is stale "
                "(age=%ds); refreshing full window"
            ),
            runner.name,
            int(stale_age),
        )
    if latest_cached >= end_utc:
        return raw_activities, False
    if exceeds_lookback(latest_cached, CACHE_MAX_LOOKBACK_DAYS):
        LOGGER.info(
            "Cache for runner=%s exceeds max lookback; skipping tail refresh",
            runner.name,
        )
        return raw_activities, False
    runner_id = runner_identity(
        runner,
        hash_identifiers=STRAVA_CACHE_HASH_IDENTIFIERS,
        salt=STRAVA_CACHE_ID_SALT,
    )
    refreshed_until = _runner_refresh_deadline(runner_id)
    if refreshed_until and refreshed_until >= end_utc:
        return raw_activities, False
    tail_pages = _fetch_tail_pages(
        runner,
        url,
        base_params,
        latest_cached,
        start_utc,
        end_utc,
        session,
        limiter,
    )
    if not tail_pages:
        return raw_activities, False
    tail_flat = _flatten_pages(tail_pages)
    merged = merge_activity_lists(tail_flat, raw_activities)
    paged = chunk_activities(merged, chunk_size=ACTIVITY_PAGE_SIZE)
    _persist_enriched_pages(runner, url, base_params, paged)
    _mark_runner_refreshed(runner_id, end_utc)
    LOGGER.info(
        "Cache tail refresh runner=%s cached_latest=%s tail_end=%s live_pages=%s",
        runner.name,
        latest_cached,
        end_utc,
        len(tail_pages),
    )
    return merged, True


def _fetch_tail_pages(
    runner: Runner,
    url: str,
    base_params: Dict[str, Any],
    latest_cached: datetime,
    window_start: datetime,
    window_end: datetime,
    session: requests.Session,
    limiter: RateLimiter,
) -> List[JSONList]:
    per_page = int(base_params.get("per_page", ACTIVITY_PAGE_SIZE))
    base_after = max(int(base_params.get("after", 0)), 0)
    base_after_dt = datetime.fromtimestamp(base_after, tz=timezone.utc)
    after_dt = max(
        base_after_dt, latest_cached - timedelta(seconds=CACHE_TAIL_OVERLAP_SECONDS)
    )
    after_dt = max(after_dt, window_start)
    if after_dt >= window_end:
        return []
    after_ts = int(after_dt.timestamp())
    before_ts = int(window_end.timestamp())
    tail_pages: List[JSONList] = []
    page = 1
    while True:
        params = {
            "after": after_ts,
            "before": before_ts,
            "per_page": per_page,
            "page": page,
        }
        data = fetch_page_with_retries(
            runner=runner,
            url=url,
            params=params,
            context_label="activities_tail",
            page=page,
            session=session,
            limiter=limiter,
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
            hash_identifiers=STRAVA_CACHE_HASH_IDENTIFIERS,
            salt=STRAVA_CACHE_ID_SALT,
        )
        if len(data) < per_page:
            break
        page += 1
    return tail_pages


def _persist_enriched_pages(
    runner: Runner,
    url: str,
    base_params: Dict[str, Any],
    page_payloads: List[JSONList],
) -> None:
    if not _cache_mode_saves:
        LOGGER.warning(
            "Cache saving disabled; cannot persist enriched cache data for runner=%s",
            runner.name,
        )
        return
    identity = runner_identity(
        runner,
        hash_identifiers=STRAVA_CACHE_HASH_IDENTIFIERS,
        salt=STRAVA_CACHE_ID_SALT,
    )
    template = {
        "after": base_params.get("after"),
        "before": base_params.get("before"),
        "per_page": base_params.get("per_page", ACTIVITY_PAGE_SIZE),
    }
    if not page_payloads:
        params = dict(template)
        params["page"] = 1
        _write_page_overlay(identity, url, params, [])
        return
    for idx, payload in enumerate(page_payloads, start=1):
        params = dict(template)
        params["page"] = idx
        _write_page_overlay(identity, url, params, payload)
    params = dict(template)
    params["page"] = len(page_payloads) + 1
    _write_page_overlay(identity, url, params, [])


def _write_page_overlay(
    identity: str,
    url: str,
    params: Dict[str, Any],
    payload: JSONList,
) -> None:
    if STRAVA_CACHE_OVERWRITE:
        save_response_to_cache("GET", url, identity, payload, params=params)
    else:
        save_overlay_to_cache("GET", url, identity, payload, params=params)


def _stale_empty_cache_age(pages: Sequence[CachedPage]) -> float | None:
    if CACHE_EMPTY_REFRESH_SECONDS <= 0:
        return None
    capture_ts = _latest_capture_timestamp(pages)
    if capture_ts is None:
        return None
    age = (datetime.now(timezone.utc) - capture_ts).total_seconds()
    if age >= CACHE_EMPTY_REFRESH_SECONDS:
        return age
    return None


def _latest_capture_timestamp(pages: Sequence[CachedPage]) -> datetime | None:
    latest: datetime | None = None
    for page in pages:
        record = page.record
        if record is None or record.captured_at is None:
            continue
        if latest is None or record.captured_at > latest:
            latest = record.captured_at
    return latest
