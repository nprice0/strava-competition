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
    _cache_mode_offline,
)
from ..errors import StravaAPIError
from ..models import Runner
from ..replay_tail import (
    chunk_activities,
    clamp_window,
    dedupe_activities,
    exceeds_lookback,
    merge_activity_lists,
    parse_activity_timestamp,
    summarize_activities,
)
from ..utils import to_utc_aware
from .base import ensure_runner_token
from .cache_helpers import (
    save_list_to_cache,
    get_cached_list_with_meta,
    runner_identity,
)
from .pagination import fetch_page_with_retries
from .response_handling import extract_error
from .rate_limiter import RateLimiter
from . import session as session_mod

JSONList: TypeAlias = List[Dict[str, Any]]

LOGGER = logging.getLogger(__name__)
ACTIVITY_PAGE_SIZE = 200


@dataclass
class CachedPage:
    params: Dict[str, Any]
    data: JSONList
    record: Optional[CaptureRecord]


# Tail-refresh suppression is keyed per (runner, window) because cache
# entries are per (after, before) window; a runner-only key would let one
# window's refresh suppress another window's for the TTL duration.
_TailRefreshKey = tuple[str, int, int]

_runner_tail_lock = threading.Lock()
# Use TTLCache to prevent unbounded growth - entries expire after 1 hour
_runner_tail_refreshed_until: TTLCache[_TailRefreshKey, datetime] = TTLCache(
    maxsize=1000, ttl=3600
)


def _to_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _runner_refresh_deadline(key: _TailRefreshKey) -> Optional[datetime]:
    with _runner_tail_lock:
        return _runner_tail_refreshed_until.get(key)


def _mark_runner_refreshed(key: _TailRefreshKey, refresh_until: datetime) -> None:
    with _runner_tail_lock:
        current = _runner_tail_refreshed_until.get(key)
        if current is None or refresh_until > current:
            _runner_tail_refreshed_until[key] = refresh_until


def _flatten_pages(pages: List[JSONList]) -> List[Dict[str, Any]]:
    flattened: List[Dict[str, Any]] = []
    for page in pages:
        flattened.extend(page)
    return flattened


_EPOCH_UTC = datetime.min.replace(tzinfo=timezone.utc)


def _activity_sort_key(act: Dict[str, Any]) -> datetime:
    """Chronological sort key; unparseable entries sort first (stable)."""

    ts = parse_activity_timestamp(act)
    return ts if ts is not None else _EPOCH_UTC


class ActivitiesAPI:
    def __init__(
        self,
        *,
        session: requests.Session | None = None,
        limiter: RateLimiter | None = None,
    ) -> None:
        # An explicitly injected session is honoured as-is (tests); otherwise
        # the thread-local default session is resolved per call so worker
        # threads never share one requests.Session instance.
        self._session = session
        self._limiter = limiter or RateLimiter()

    def _resolve_session(self) -> requests.Session:
        """Return the injected session or this thread's default session."""

        return self._session or session_mod.get_default_session()

    def get_activities(
        self,
        runner: Runner,
        start_date: datetime,
        end_date: datetime,
        *,
        activity_types: Optional[Iterable[str]] = ("Run",),
        max_pages: Optional[int] = None,
    ) -> Optional[List[Dict[str, Any]]]:
        """Fetch activities for a runner in [start_date, end_date].

        Returns:
            The filtered activities, or None when the listing could not be
            fetched (network/API failure). A failed listing page is never
            persisted to the cache.
        """

        start_utc = _to_utc(start_date)
        end_utc = _to_utc(end_date)
        url = f"{STRAVA_BASE_URL}/athlete/activities"
        base_params = {
            "after": int(start_utc.timestamp()),
            "before": int(end_utc.timestamp()),
            "per_page": ACTIVITY_PAGE_SIZE,
        }
        http_session = self._resolve_session()

        try:
            ensure_runner_token(runner)
            normalized_types = _normalize_types(activity_types)
            raw_pages, cached_pages, used_cache = self._collect_pages(
                runner, url, base_params, http_session, max_pages
            )
            raw_activities = dedupe_activities(_flatten_pages(raw_pages))
            if used_cache and _cache_mode_saves:
                raw_activities, _ = _maybe_refresh_cache_tail(
                    runner,
                    url,
                    base_params,
                    cached_pages,
                    raw_activities,
                    http_session,
                    self._limiter,
                    start_date=start_date,
                    end_date=end_date,
                )
            return _filter_activities(
                raw_activities, normalized_types, start_utc, end_utc
            )
        except StravaAPIError:
            # Terminal listing-page failure: nothing was cached for the
            # failed page and the caller's failure contract is None.
            LOGGER.exception("Activities fetch failed runner=%s", runner.name)
            return None
        except requests.exceptions.HTTPError as exc:  # pragma: no cover
            resp = exc.response
            if resp is None:
                LOGGER.exception(
                    "HTTPError (activities) no response object for runner %s",
                    runner.name,
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

    def _collect_pages(
        self,
        runner: Runner,
        url: str,
        base_params: Dict[str, Any],
        http_session: requests.Session,
        max_pages: Optional[int],
    ) -> tuple[List[JSONList], List[CachedPage], bool]:
        """Page through the listing endpoint until an incomplete page.

        Returns:
            Tuple of (raw pages, cache-sourced pages, whether any page came
            from the cache).
        """

        raw_pages: List[JSONList] = []
        cached_pages: List[CachedPage] = []
        used_cache = False
        attempted_refresh = False
        page = 1
        while True:
            try:
                data, from_cache = self._fetch_listing_page(
                    runner, url, base_params, page, http_session, cached_pages
                )
            except requests.exceptions.HTTPError as exc:
                if not _is_401(exc) or attempted_refresh:
                    raise
                LOGGER.info(
                    "401 for runner %s (activities). Refreshing token and retrying page %s.",
                    runner.name,
                    page,
                )
                runner.access_token = None
                ensure_runner_token(runner)
                attempted_refresh = True
                data, from_cache = self._fetch_listing_page(
                    runner, url, base_params, page, http_session, cached_pages
                )
            used_cache = used_cache or from_cache
            raw_pages.append(data)
            if len(data) < ACTIVITY_PAGE_SIZE:
                break
            if max_pages is not None and page >= max_pages:
                break
            page += 1
        return raw_pages, cached_pages, used_cache

    def _fetch_listing_page(
        self,
        runner: Runner,
        url: str,
        base_params: Dict[str, Any],
        page: int,
        http_session: requests.Session,
        cached_pages: List[CachedPage],
    ) -> tuple[JSONList, bool]:
        """Fetch a single listing page, preferring the cache.

        Returns:
            Tuple of (page data, whether the page was served from cache).
            A cache hit is appended to ``cached_pages``.

        Raises:
            StravaAPIError: When the page cannot be fetched (nothing is
                persisted to the cache in that case).
        """

        params = dict(base_params)
        params["page"] = page
        if _cache_mode_reads:
            cache_record: Optional[CaptureRecord] = get_cached_list_with_meta(
                runner,
                url,
                params,
                context_label="activities",
                page=page,
                use_cache=_cache_mode_reads,
                require_cache=_cache_mode_offline,
            )
            if cache_record is not None:
                cached_data = cast(JSONList, cache_record.response)
                cached_pages.append(
                    CachedPage(
                        params=dict(params),
                        data=cached_data,
                        record=cache_record,
                    )
                )
                return cached_data, True
        result = fetch_page_with_retries(
            runner=runner,
            url=url,
            params=params,
            context_label="activities",
            page=page,
            session=http_session,
            limiter=self._limiter,
        )
        save_list_to_cache(
            runner,
            url,
            dict(params),
            result,
            save_to_cache=_cache_mode_saves,
        )
        return result, False


def _is_401(exc: requests.exceptions.HTTPError) -> bool:
    """Return True when the HTTP error carries a 401 response."""

    return exc.response is not None and exc.response.status_code == 401


def _normalize_types(activity_types: Optional[Iterable[str]]) -> Optional[set[str]]:
    """Normalize the requested activity types into a filter set."""

    if not activity_types:
        return None
    return {
        normalized
        for normalized in (normalize_activity_type(value) for value in activity_types)
        if normalized
    }


def _filter_activities(
    raw_activities: List[Dict[str, Any]],
    normalized_types: Optional[set[str]],
    start_utc: datetime,
    end_utc: datetime,
) -> List[Dict[str, Any]]:
    """Filter activities by type and by the [start_utc, end_utc] window."""

    filtered: List[Dict[str, Any]] = []
    for act in raw_activities:
        if normalized_types and not activity_type_matches(act, normalized_types):
            continue
        # Prefer start_date (true UTC).  Strava's start_date_local
        # carries a misleading "Z" suffix but is actually the
        # athlete's local time — treating it as UTC shifts the
        # timestamp by the athlete's timezone offset.  Fall back to
        # start_date_local only when start_date is absent (e.g.
        # incomplete cache data) and accept the approximation.
        raw_start = act.get("start_date") or act.get("start_date_local")
        if not raw_start:
            continue
        try:
            dt = to_utc_aware(datetime.fromisoformat(raw_start.replace("Z", "+00:00")))
        except ValueError:
            continue
        if start_utc <= dt <= end_utc:
            filtered.append(act)
    return filtered


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
    runner_id = runner_identity(runner)
    refresh_key: _TailRefreshKey = (
        runner_id,
        int(base_params.get("after", 0)),
        int(base_params.get("before", 0)),
    )
    refreshed_until = _runner_refresh_deadline(refresh_key)
    if refreshed_until and refreshed_until >= end_utc:
        return raw_activities, False
    try:
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
    except (StravaAPIError, requests.exceptions.HTTPError) as exc:
        # Degrade gracefully: serve the already-loaded cached listing
        # rather than discarding it because the tail refresh failed.
        LOGGER.warning(
            "Cache tail refresh failed runner=%s; serving cached data: %s",
            runner.name,
            exc,
        )
        return raw_activities, False
    if not tail_pages:
        return raw_activities, False
    tail_flat = _flatten_pages(tail_pages)
    merged = merge_activity_lists(tail_flat, raw_activities)
    merged.sort(key=_activity_sort_key)
    paged = chunk_activities(merged, chunk_size=ACTIVITY_PAGE_SIZE)
    persisted = _persist_enriched_pages(runner, url, base_params, paged)
    if persisted:
        _mark_runner_refreshed(refresh_key, end_utc)
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
        # No per-page cache write here: these orphan (after, before) cache
        # signatures are never read back; _persist_enriched_pages persists
        # the merged result under the caller's window signature.
        if len(data) < per_page:
            break
        page += 1
    return tail_pages


def _persist_enriched_pages(
    runner: Runner,
    url: str,
    base_params: Dict[str, Any],
    page_payloads: List[JSONList],
) -> bool:
    """Persist merged activity pages to the cache overlay.

    Returns:
        True if pages were persisted successfully, False otherwise.
    """
    if not _cache_mode_saves:
        LOGGER.warning(
            "Cache saving disabled; cannot persist enriched cache data for runner=%s",
            runner.name,
        )
        return False
    identity = runner_identity(runner)
    template = {
        "after": base_params.get("after"),
        "before": base_params.get("before"),
        "per_page": base_params.get("per_page", ACTIVITY_PAGE_SIZE),
    }
    if not page_payloads:
        params = dict(template)
        params["page"] = 1
        _write_page_overlay(identity, url, params, [])
        return True
    for idx, payload in enumerate(page_payloads, start=1):
        params = dict(template)
        params["page"] = idx
        _write_page_overlay(identity, url, params, payload)
    params = dict(template)
    params["page"] = len(page_payloads) + 1
    _write_page_overlay(identity, url, params, [])
    return True


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
