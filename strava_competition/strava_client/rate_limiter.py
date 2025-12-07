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
        self, headers: Mapping[str, object] | None, status_code: int | None
    ) -> None:
        throttle = False
        if status_code == 429:
            throttle = True
            logging.warning(
                "Rate limit: 429. Throttling %ss.", RATE_LIMIT_THROTTLE_SECONDS
            )
        else:
            short_used = short_limit = None
            if headers:
                usage = headers.get("X-RateLimit-Usage")
                limit = headers.get("X-RateLimit-Limit")
                if usage and limit:
                    try:
                        short_used = int(str(usage).split(",")[0])
                        short_limit = int(str(limit).split(",")[0])
                    except (ValueError, TypeError) as exc:
                        short_used = short_limit = None
                        logging.debug(
                            "Failed to parse rate limit headers usage=%s limit=%s: %s",
                            usage,
                            limit,
                            exc,
                        )
            if (
                short_used is not None
                and short_limit is not None
                and short_used >= max(short_limit - self._near_limit_buffer, 0)
            ):
                throttle = True
                logging.info(
                    "Approaching short-window limit (%s/%s). Throttling %ss.",
                    short_used,
                    short_limit,
                    RATE_LIMIT_THROTTLE_SECONDS,
                )
        if throttle:
            with self._cond:
                self._throttle_until = time.time() + RATE_LIMIT_THROTTLE_SECONDS
        with self._cond:
            self._in_flight = max(0, self._in_flight - 1)
            if self._in_flight < self._max_allowed:
                self._cond.notify()

    def snapshot(self) -> dict[str, float | int]:  # pragma: no cover - debug helper
        """Return current limiter stats (used by tests and diagnostics)."""

        with self._lock:
            return {
                "max_allowed": self._max_allowed,
                "in_flight": self._in_flight,
                "throttle_until": self._throttle_until,
            }
