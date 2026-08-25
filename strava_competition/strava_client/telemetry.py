"""Thread-safe in-process counters for Strava API usage telemetry.

Tracks how much work a run performed live versus served from the disk
cache so an end-of-run summary can be logged. Counters:

- ``live_calls``: completed HTTP round-trips to the Strava API (each retry
  that reaches the server counts; network errors that never complete do not).
- ``cache_hits``: responses served from the disk cache.
- ``validation_refetches``: live refetches triggered by a cached payload
  failing validation (counted in addition to the ``live_calls`` the refetch
  itself performs).
- ``reset_waits``: pauses taken to wait for the next quarter-hour rate-limit
  window reset (one per thread that actually waited).

OAuth token refreshes are intentionally not counted — data calls only.
"""

from __future__ import annotations

import threading

LIVE_CALLS = "live_calls"
CACHE_HITS = "cache_hits"
VALIDATION_REFETCHES = "validation_refetches"
RESET_WAITS = "reset_waits"

_KNOWN_COUNTERS: tuple[str, ...] = (
    LIVE_CALLS,
    CACHE_HITS,
    VALIDATION_REFETCHES,
    RESET_WAITS,
)

_lock = threading.Lock()
_counters: dict[str, int] = dict.fromkeys(_KNOWN_COUNTERS, 0)


def increment(name: str) -> None:
    """Increment counter ``name`` by one, creating it at zero if unknown.

    Args:
        name: Counter name, typically one of the module constants
            ``LIVE_CALLS``, ``CACHE_HITS`` or ``VALIDATION_REFETCHES``.
    """
    with _lock:
        _counters[name] = _counters.get(name, 0) + 1


def snapshot() -> dict[str, int]:
    """Return a point-in-time copy of all counters.

    Returns:
        Mapping of counter name to its current value. Mutating the returned
        dict does not affect the registry.
    """
    with _lock:
        return dict(_counters)


def reset() -> None:
    """Reset the registry to the known counters at zero.

    Exists so tests and ``main()`` can start from a clean slate.
    """
    with _lock:
        _counters.clear()
        _counters.update(dict.fromkeys(_KNOWN_COUNTERS, 0))
