"""Segment efforts pagination, caching, and replay orchestration."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional, TypeAlias, cast

import requests

from ..config import (
    REPLAY_CACHE_TTL_DAYS,
    STRAVA_API_CAPTURE_ENABLED,
    STRAVA_API_REPLAY_ENABLED,
    STRAVA_BASE_URL,
    STRAVA_OFFLINE_MODE,
)
from ..errors import StravaAPIError
from ..models import Runner
from ..replay_tail import cache_is_stale
from .capture import record_list_response, replay_list_response_with_meta
from .pagination import fetch_page_with_retries
from .rate_limiter import RateLimiter
from .response_handling import extract_error
from .base import ensure_runner_token
from .session import get_default_session

JSONList: TypeAlias = List[Dict[str, object]]

LOGGER = logging.getLogger(__name__)
SEGMENT_PAGE_SIZE = 200


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

        replay_allowed = STRAVA_API_REPLAY_ENABLED
        base_params = {
            "segment_id": segment_id,
            "start_date_local": start_date.isoformat(),
            "end_date_local": end_date.isoformat(),
            "per_page": SEGMENT_PAGE_SIZE,
        }
        url = f"{STRAVA_BASE_URL}/segment_efforts"

        def fetch_page(page: int) -> JSONList:
            nonlocal replay_allowed
            params = dict(base_params)
            params["page"] = page
            if replay_allowed:
                record = replay_list_response_with_meta(
                    runner,
                    url,
                    params,
                    context_label="segment_efforts",
                    page=page,
                    offline_mode=STRAVA_OFFLINE_MODE,
                )
                if record is not None:
                    if not STRAVA_OFFLINE_MODE and cache_is_stale(
                        record.captured_at,
                        REPLAY_CACHE_TTL_DAYS,
                    ):
                        LOGGER.info(
                            "Replay cache TTL expired for runner=%s segment=%s page=%s; switching to live fetch",
                            runner.name,
                            segment_id,
                            page,
                        )
                        replay_allowed = False
                    else:
                        return cast(JSONList, record.response)
                if STRAVA_OFFLINE_MODE:
                    raise StravaAPIError(
                        "Offline mode cache miss for segment efforts"
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
            record_list_response(
                runner,
                url,
                dict(params),
                data,
                capture_enabled=STRAVA_API_CAPTURE_ENABLED,
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
