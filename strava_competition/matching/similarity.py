"""Similarity scoring utilities for comparing segment and activity tracks."""

from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock
from typing import Hashable, Iterator, Optional, Sequence, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from ..config import MATCHING_CACHE_MAX_ENTRIES
from .preprocessing import PreparedSegmentGeometry

MetricArray = NDArray[np.float64]

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from shapely.geometry import LineString


def discrete_frechet_distance(
    activity_points: Sequence[Sequence[float]],
    segment_points: Sequence[Sequence[float]],
) -> float:
    """Compute the discrete Fréchet distance between two sequences of points."""

    a = _as_metric_array(activity_points)
    b = _as_metric_array(segment_points)
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    n, m = a.shape[0], b.shape[0]
    ca = np.empty((n, m), dtype=float)
    for i in range(n):
        for j in range(m):
            dist = float(np.linalg.norm(a[i] - b[j]))
            if i == 0 and j == 0:
                ca[i, j] = dist
            elif i == 0:
                ca[i, j] = max(ca[i, j - 1], dist)
            elif j == 0:
                ca[i, j] = max(ca[i - 1, j], dist)
            else:
                ca[i, j] = max(
                    min(ca[i - 1, j], ca[i - 1, j - 1], ca[i, j - 1]),
                    dist,
                )
    return float(ca[-1, -1])


def windowed_dtw(
    activity_points: Sequence[Sequence[float]],
    segment_points: Sequence[Sequence[float]],
    window_size: int,
) -> float:
    """Compute a constrained DTW score between two tracks."""

    a = _as_metric_array(activity_points)
    b = _as_metric_array(segment_points)
    n, m = a.shape[0], b.shape[0]
    if n == 0 or m == 0:
        return float("inf")
    window = max(window_size, abs(n - m)) if window_size > 0 else max(n, m)
    window = max(window, 1)
    cost = np.full((n + 1, m + 1), np.inf, dtype=float)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        start_j = max(1, i - window)
        end_j = min(m, i + window)
        for j in range(start_j, end_j + 1):
            dist = float(np.linalg.norm(a[i - 1] - b[j - 1]))
            cost[i, j] = dist + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    dtw_distance = cost[n, m]
    return float(dtw_distance)


@dataclass(slots=True)
class CachedSegment:
    """Cached representation of a segment for fast similarity comparisons."""

    prepared: PreparedSegmentGeometry
    linestring: "LineString"


class SegmentCache:
    """In-memory cache holder for preprocessed segment geometry."""

    def __init__(self, max_entries: int = MATCHING_CACHE_MAX_ENTRIES) -> None:
        self._cache: "OrderedDict[Hashable, CachedSegment]" = OrderedDict()
        self._lock = RLock()
        self._max_entries = max(1, max_entries)

    def get(self, key: Hashable) -> Optional[CachedSegment]:
        """Return cached segment data if available."""
        with self._lock:
            value = self._cache.get(key)
            if value is not None:
                self._cache.move_to_end(key)
            return value

    def set(self, key: Hashable, value: CachedSegment) -> None:
        """Persist preprocessed segment data in the cache."""
        with self._lock:
            self._cache[key] = value
            self._cache.move_to_end(key)
            while len(self._cache) > self._max_entries:
                self._cache.popitem(last=False)

    def clear(self) -> None:
        """Empty the cache (primarily for testing)."""
        with self._lock:
            self._cache.clear()


@contextmanager
def segment_cache_scope(clear_on_enter: bool = True) -> Iterator[SegmentCache]:
    """Provide a request-scoped cache and ensure cleanup afterwards."""

    if clear_on_enter:
        SEGMENT_CACHE.clear()
    try:
        yield SEGMENT_CACHE
    finally:
        SEGMENT_CACHE.clear()


SEGMENT_CACHE = SegmentCache()


def build_cached_segment(prepared: PreparedSegmentGeometry) -> CachedSegment:
    """Create a cached segment entry from prepared geometry."""

    try:
        from shapely.geometry import LineString as _LineString  # type: ignore[import]
    except ImportError as exc:  # pragma: no cover - handled at runtime
        raise RuntimeError("shapely is required to cache segment geometry") from exc

    line = _LineString(prepared.resampled_points.tolist())
    return CachedSegment(prepared=prepared, linestring=line)


def _as_metric_array(points: Sequence[Sequence[float]]) -> MetricArray:
    array = np.asarray(points, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("Expected a sequence of 2D coordinates")
    return array


__all__ = [
    "SEGMENT_CACHE",
    "CachedSegment",
    "SegmentCache",
    "build_cached_segment",
    "discrete_frechet_distance",
    "segment_cache_scope",
    "windowed_dtw",
]
