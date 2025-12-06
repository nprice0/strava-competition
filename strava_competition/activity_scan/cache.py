"""Simple in-memory caches used by the activity scan workflow."""

from __future__ import annotations

from collections import OrderedDict
from threading import RLock
from typing import Any, Dict, Hashable, Tuple


class ActivityDetailCache:
    """Thread-safe LRU cache keyed by (runner_id, activity_id)."""

    def __init__(self, max_entries: int = 128) -> None:
        self._max_entries = max(0, max_entries)
        self._lock = RLock()
        self._store: OrderedDict[Tuple[Hashable, Hashable], Dict[str, Any]] = (
            OrderedDict()
        )

    def get(self, key: Tuple[Hashable, Hashable]) -> Dict[str, Any] | None:
        if self._max_entries <= 0:
            return None
        with self._lock:
            value = self._store.get(key)
            if value is not None:
                self._store.move_to_end(key)
            return value

    def put(self, key: Tuple[Hashable, Hashable], value: Dict[str, Any]) -> None:
        if self._max_entries <= 0:
            return
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = value
            while len(self._store) > self._max_entries:
                self._store.popitem(last=False)
