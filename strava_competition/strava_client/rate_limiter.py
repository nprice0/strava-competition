"""Rate limiting utilities shared across Strava API helpers."""

from __future__ import annotations

import logging
import random
import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Mapping

from ..config import (
    RATE_LIMIT_JITTER_RANGE,
    RATE_LIMIT_MAX_CONCURRENT,
    RATE_LIMIT_NEAR_LIMIT_BUFFER,
    RATE_LIMIT_THROTTLE_SECONDS,
    RATE_LIMIT_WAIT_FOR_RESET,
)
from ..errors import StravaRateLimitError
from . import telemetry

__all__ = ["RateLimiter", "seconds_until_window_reset"]

# Strava's short rate-limit window length: 15 minutes.
_WINDOW_SECONDS = 15 * 60
# Extra seconds slept past the boundary so the new window is definitely open.
_RESET_SAFETY_BUFFER_SECONDS = 5.0
# Maximum single sleep slice while waiting for a reset, so cancellation via
# the limiter's cancel event is honoured promptly.
_WAIT_POLL_SLICE_SECONDS = 1.0


def seconds_until_window_reset(now: datetime | None = None) -> float:
    """Return seconds until the next quarter-hour UTC boundary plus a buffer.

    Strava's 15-minute rate-limit windows reset at fixed :00/:15/:30/:45 UTC
    boundaries. An input exactly on a boundary waits for the *next* boundary
    plus the safety buffer so callers never resume inside a window that is
    still exhausted.

    Args:
        now: Reference time, injectable for tests. Defaults to current UTC.

    Returns:
        Seconds to sleep before the window is guaranteed to have reset.
    """
    current = now if now is not None else datetime.now(timezone.utc)
    elapsed = current.timestamp() % _WINDOW_SECONDS
    return (_WINDOW_SECONDS - elapsed) + _RESET_SAFETY_BUFFER_SECONDS


def _midnight_utc_epoch(now: datetime | None = None) -> float:
    """Return the epoch timestamp of the most recent midnight UTC.

    Strava's daily rate-limit window resets at midnight UTC; usage readings
    captured before this instant describe an already-reset budget.

    Args:
        now: Reference time, injectable for tests. Defaults to current UTC.

    Returns:
        Epoch seconds of the start of the current UTC day.
    """
    current = now if now is not None else datetime.now(timezone.utc)
    return current.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()


def _window_start_epoch(now: datetime | None = None) -> float:
    """Return the epoch timestamp of the most recent quarter-hour UTC boundary.

    Usage readings captured before this instant describe a 15-minute window
    that has already reset.

    Args:
        now: Reference time, injectable for tests. Defaults to current UTC.

    Returns:
        Epoch seconds of the start of the current 15-minute window.
    """
    current = now if now is not None else datetime.now(timezone.utc)
    epoch = current.timestamp()
    return epoch - (epoch % _WINDOW_SECONDS)


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
        # Last-seen X-RateLimit usage/limits, for end-of-run diagnostics.
        self._last_short_used: int | None = None
        self._last_short_limit: int | None = None
        self._last_daily_used: int | None = None
        self._last_daily_limit: int | None = None
        # Last-seen X-ReadRateLimit short-window usage, for exhaustion checks.
        self._last_read_used: int | None = None
        self._last_read_limit: int | None = None
        # Wall-clock capture times of the readings above, so exhaustion
        # checks can ignore readings that predate a window reset boundary.
        self._last_short_at: float | None = None
        self._last_daily_at: float | None = None
        self._last_read_at: float | None = None
        # True when the current throttle deadline targets a window reset.
        self._throttle_is_reset = False
        # Optional event that aborts reset waits promptly when set.
        self._cancel_event: threading.Event | None = None

    def set_cancel_event(self, event: threading.Event | None) -> None:
        """Install (or clear) the event that aborts reset waits when set."""

        self._cancel_event = event

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
        self._raise_if_daily_exhausted()
        with self._cond:
            while self._in_flight >= self._max_allowed:
                self._cond.wait()
            self._in_flight += 1
            wait_for = max(0.0, self._throttle_until - time.time())
            throttle_is_reset = self._throttle_is_reset
            reset_wait = self._proactive_reset_wait_locked()
        if reset_wait is not None or (wait_for > 0 and throttle_is_reset):
            self._wait_for_window_reset(max(reset_wait or 0.0, wait_for))
            self._clear_short_usage()
        elif wait_for > 0:
            time.sleep(wait_for)
        lo, hi = self._jitter_range
        if hi > 0:
            jitter = random.uniform(lo, hi)  # nosec B311
            # Random jitter smooths bursts; not used for security-sensitive logic.
            time.sleep(jitter)

    @staticmethod
    def _exhausted(used: int | None, limit: int | None) -> bool:
        """Return True when a tracked usage figure has reached a real limit."""

        return used is not None and limit is not None and limit > 0 and used >= limit

    def _raise_if_daily_exhausted(self) -> None:
        """Fail fast when the last-seen daily usage has hit the daily limit.

        Readings captured before the most recent midnight UTC describe a
        budget that has already reset and are ignored, so a run that spans
        midnight recovers without a restart.

        Raises:
            StravaRateLimitError: When the daily budget is exhausted; waiting
                until the midnight UTC reset is not practical.
        """

        if not RATE_LIMIT_WAIT_FOR_RESET:
            return
        with self._lock:
            used, limit = self._last_daily_used, self._last_daily_limit
            captured_at = self._last_daily_at
        if not self._exhausted(used, limit):
            return
        if captured_at is None or captured_at < _midnight_utc_epoch():
            return
        raise StravaRateLimitError(
            f"Daily Strava rate limit exhausted ({used}/{limit}); "
            "it resets at midnight UTC — aborting instead of waiting"
        )

    def _proactive_reset_wait_locked(self) -> float | None:
        """Return the boundary wait when the short window is exhausted.

        Must be called with the limiter lock held. Returns None when the
        reset-wait feature is disabled, the short window has headroom, or
        every exhausted reading predates the current quarter-hour boundary
        (the window it measured has already reset; stored usage is cleared).
        """

        if not RATE_LIMIT_WAIT_FOR_RESET:
            return None
        exhaustion = self._classify_exhaustion_locked(_window_start_epoch())
        if exhaustion is None:
            return None
        if not exhaustion:
            self._clear_short_usage_locked()
            return None
        return seconds_until_window_reset()

    def _classify_exhaustion_locked(self, window_start: float) -> bool | None:
        """Classify short-window exhaustion against the current window.

        Must be called with the limiter lock held.

        Args:
            window_start: Epoch of the most recent quarter-hour UTC boundary.

        Returns:
            True when an exhausted reading was captured inside the current
            window (a wait is warranted), False when every exhausted reading
            predates the boundary (that window already reset), or None when
            nothing is exhausted.
        """

        fresh = stale = False
        for used, limit, captured_at in (
            (self._last_short_used, self._last_short_limit, self._last_short_at),
            (self._last_read_used, self._last_read_limit, self._last_read_at),
        ):
            if not self._exhausted(used, limit):
                continue
            if captured_at is not None and captured_at >= window_start:
                fresh = True
            else:
                stale = True
        if fresh:
            return True
        return False if stale else None

    def _wait_for_window_reset(self, wait_seconds: float) -> None:
        """Log, count, and perform an interruptible wait for the window reset."""

        resume = datetime.now(timezone.utc) + timedelta(seconds=wait_seconds)
        logging.info(
            "15-min rate-limit window exhausted; waiting %.0fs for reset "
            "(resuming ~%s UTC)",
            wait_seconds,
            resume.strftime("%H:%M:%S"),
        )
        telemetry.increment(telemetry.RESET_WAITS)
        self._interruptible_sleep(wait_seconds)

    def _interruptible_sleep(self, seconds: float) -> None:
        """Sleep in short slices, aborting promptly if the cancel event fires."""

        deadline = time.time() + seconds
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                return
            slice_seconds = min(_WAIT_POLL_SLICE_SECONDS, remaining)
            event = self._cancel_event
            if event is None:
                time.sleep(slice_seconds)
            elif event.wait(slice_seconds):
                logging.info("Rate-limit reset wait cancelled; proceeding")
                return

    def _clear_short_usage(self) -> None:
        """Optimistically zero short-window usage after a reset wait.

        Lets requests go out again after the boundary; the first response
        refreshes the real usage headers.
        """

        with self._lock:
            self._clear_short_usage_locked()

    def _clear_short_usage_locked(self) -> None:
        """Zero short-window usage; must be called with the limiter lock held."""

        if self._last_short_used is not None:
            self._last_short_used = 0
        if self._last_read_used is not None:
            self._last_read_used = 0

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

        self._record_last_usage(
            short_used,
            short_limit,
            daily_used,
            daily_limit,
            read_used,
            read_limit,
        )

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
            self._apply_throttle(status_code == 429)
        with self._cond:
            self._in_flight = max(0, self._in_flight - 1)
            if self._in_flight < self._max_allowed:
                self._cond.notify()

        return throttle, rate_info

    def _apply_throttle(self, is_429: bool) -> None:
        """Set the throttle deadline: window boundary for 429s, fixed otherwise."""

        use_reset = is_429 and RATE_LIMIT_WAIT_FOR_RESET
        seconds = (
            seconds_until_window_reset()
            if use_reset
            else float(RATE_LIMIT_THROTTLE_SECONDS)
        )
        with self._cond:
            self._throttle_until = time.time() + seconds
            self._throttle_is_reset = use_reset
        if use_reset:
            logging.info(
                "429 received; throttling all requests %.0fs until the "
                "15-min window reset",
                seconds,
            )

    def throttle_deadline(self) -> float | None:
        """Return the epoch deadline of an active window-reset throttle.

        Returns:
            The deadline set by a 429 while reset-waiting is enabled, or
            ``None`` when no boundary throttle is active. Retry loops use
            this to skip their own backoff sleep — ``before_request`` on the
            next attempt performs the boundary wait instead.
        """

        with self._lock:
            if self._throttle_is_reset and self._throttle_until > time.time():
                return self._throttle_until
        return None

    def _record_last_usage(
        self,
        short_used: int | None,
        short_limit: int | None,
        daily_used: int | None,
        daily_limit: int | None,
        read_used: int | None = None,
        read_limit: int | None = None,
    ) -> None:
        """Remember the most recent rate-limit usage headers for diagnostics.

        Each captured reading is timestamped so exhaustion checks can ignore
        readings that predate a window reset boundary.
        """

        now = time.time()
        with self._lock:
            if short_used is not None and short_limit is not None:
                self._last_short_used = short_used
                self._last_short_limit = short_limit
                self._last_short_at = now
            if daily_used is not None and daily_limit is not None:
                self._last_daily_used = daily_used
                self._last_daily_limit = daily_limit
                self._last_daily_at = now
            if read_used is not None and read_limit is not None:
                self._last_read_used = read_used
                self._last_read_limit = read_limit
                self._last_read_at = now

    def snapshot(self) -> dict[str, float | int | None]:
        """Return current limiter stats plus last-seen rate-limit usage.

        The ``short_*``/``daily_*`` keys reflect the most recent
        ``X-RateLimit-Usage``/``X-RateLimit-Limit`` headers observed, or
        ``None`` when no rate-limit headers were ever seen (fully cached run).
        """

        with self._lock:
            return {
                "max_allowed": self._max_allowed,
                "in_flight": self._in_flight,
                "throttle_until": self._throttle_until,
                "short_used": self._last_short_used,
                "short_limit": self._last_short_limit,
                "daily_used": self._last_daily_used,
                "daily_limit": self._last_daily_limit,
            }
