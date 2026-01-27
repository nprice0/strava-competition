"""Rate limiting utilities shared across Strava API helpers."""

from __future__ import annotations

import logging
import random
import threading
import time
from typing import Mapping

from ..config import (
    RATE_LIMIT_JITTER_RANGE,
    RATE_LIMIT_MAX_CONCURRENT,
    RATE_LIMIT_NEAR_LIMIT_BUFFER,
    RATE_LIMIT_THROTTLE_SECONDS,
)

__all__ = ["RateLimiter"]


class RateLimiter:
    """Soft concurrency cap with optional throttle and jitter to smooth bursts."""

    def __init__(
        self,
        max_concurrent: int = RATE_LIMIT_MAX_CONCURRENT,
        jitter_range: tuple[float, float] = RATE_LIMIT_JITTER_RANGE,
    ) -> None:
        if max_concurrent < 1:
            raise ValueError("max_concurrent must be >= 1")
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._max_allowed = max_concurrent
        self._in_flight = 0
        self._throttle_until: float = 0.0
        self._jitter_range = jitter_range
        self._near_limit_buffer = RATE_LIMIT_NEAR_LIMIT_BUFFER

    def resize(self, new_max: int) -> None:
        """Adjust maximum concurrent requests (soft limit) at runtime."""

        if new_max < 1:
            raise ValueError("new_max must be >= 1")
        with self._cond:
            old = self._max_allowed
            self._max_allowed = new_max
            self._cond.notify_all()
        logging.info("RateLimiter resized from %s to %s", old, new_max)

    def before_request(self) -> None:
        with self._cond:
            while self._in_flight >= self._max_allowed:
                self._cond.wait()
            self._in_flight += 1
            wait_for = max(0.0, self._throttle_until - time.time())
        if wait_for > 0:
            time.sleep(wait_for)
        lo, hi = self._jitter_range
        if hi > 0:
            jitter = random.uniform(lo, hi)  # nosec B311
            # Random jitter smooths bursts; not used for security-sensitive logic.
            time.sleep(jitter)

    def after_response(
        self,
        headers: Mapping[str, object] | None,
        status_code: int | None,
    ) -> tuple[bool, str]:
        """Process response and apply throttling if needed.

        Returns:
            Tuple of (throttled, rate_info) where:
            - throttled: True if rate limited (429 or approaching limit)
            - rate_info: String with rate limit status for logging, e.g.
              "(15-min: 95/100, daily: 450/1000)" or empty string if unavailable
        """
        throttle = False

        def _parse_limits(
            usage_val: object | None, limit_val: object | None
        ) -> tuple[int | None, int | None, int | None, int | None]:
            """Parse comma-separated short/daily usage and limits."""

            if not usage_val or not limit_val:
                return None, None, None, None
            try:
                usage_parts = str(usage_val).split(",")
                limit_parts = str(limit_val).split(",")
                short_u = int(usage_parts[0])
                short_l = int(limit_parts[0])
                daily_u = daily_l = None
                if len(usage_parts) > 1 and len(limit_parts) > 1:
                    daily_u = int(usage_parts[1])
                    daily_l = int(limit_parts[1])
                return short_u, short_l, daily_u, daily_l
            except (ValueError, TypeError, IndexError) as exc:
                logging.debug(
                    "Failed to parse rate limit headers usage=%s limit=%s: %s",
                    usage_val,
                    limit_val,
                    exc,
                )
                return None, None, None, None

        short_used = short_limit = daily_used = daily_limit = None
        read_used = read_limit = read_daily_used = read_daily_limit = None
        if headers:
            rate_usage = headers.get("X-RateLimit-Usage")
            rate_limit = headers.get("X-RateLimit-Limit")
            (
                short_used,
                short_limit,
                daily_used,
                daily_limit,
            ) = _parse_limits(rate_usage, rate_limit)

            read_usage = headers.get("X-ReadRateLimit-Usage")
            read_limit_header = headers.get("X-ReadRateLimit-Limit")
            (
                read_used,
                read_limit,
                read_daily_used,
                read_daily_limit,
            ) = _parse_limits(read_usage, read_limit_header)

        # Build rate limit info for caller's log messages
        parts: list[str] = []
        if short_used is not None and short_limit is not None:
            part = f"app 15-min: {short_used}/{short_limit}"
            if daily_used is not None and daily_limit is not None:
                part += f", daily: {daily_used}/{daily_limit}"
            parts.append(part)
        if read_used is not None and read_limit is not None:
            part = f"read 15-min: {read_used}/{read_limit}"
            if read_daily_used is not None and read_daily_limit is not None:
                part += f", daily: {read_daily_used}/{read_daily_limit}"
            parts.append(part)
        rate_info = f"({'; '.join(parts)})" if parts else ""

        if status_code == 429:
            throttle = True
        else:
            # Throttle if either app or read limits are near the short-window cap.
            if (
                short_used is not None
                and short_limit is not None
                and short_used >= max(short_limit - self._near_limit_buffer, 0)
            ):
                throttle = True
                logging.debug(
                    "Near-limit throttle (app): used=%s limit=%s buffer=%s threshold=%s",
                    short_used,
                    short_limit,
                    self._near_limit_buffer,
                    max(short_limit - self._near_limit_buffer, 0),
                )
            if (
                read_used is not None
                and read_limit is not None
                and read_used >= max(read_limit - self._near_limit_buffer, 0)
            ):
                throttle = True
                logging.debug(
                    "Near-limit throttle (read): used=%s limit=%s buffer=%s threshold=%s",
                    read_used,
                    read_limit,
                    self._near_limit_buffer,
                    max(read_limit - self._near_limit_buffer, 0),
                )

        if throttle:
            with self._cond:
                self._throttle_until = time.time() + RATE_LIMIT_THROTTLE_SECONDS
        with self._cond:
            self._in_flight = max(0, self._in_flight - 1)
            if self._in_flight < self._max_allowed:
                self._cond.notify()

        return throttle, rate_info

    def snapshot(self) -> dict[str, float | int]:  # pragma: no cover - debug helper
        """Return current limiter stats (used by tests and diagnostics)."""

        with self._lock:
            return {
                "max_allowed": self._max_allowed,
                "in_flight": self._in_flight,
                "throttle_until": self._throttle_until,
            }
