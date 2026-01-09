"""Distance competition service.

Encapsulates fetching runner activities over the union date window and
building per-window + summary distance outputs. Uses the pure
`build_distance_outputs` function (now in `distance_aggregation` module)
to keep aggregation logic testable and decoupled from I/O.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
import logging
import random
import threading
import time
from typing import Any, Callable, Dict, List, Sequence, Tuple

from ..models import Runner
from ..strava_api import get_activities
from ..auth import TokenError
from ..distance_aggregation import build_distance_outputs
from ..config import CACHE_REFRESH_PARALLELISM

DistanceWindow = Tuple[datetime, datetime, float | None]
ActivityRow = Dict[str, Any]
ActivityList = List[ActivityRow]
ActivityCache = Dict[int | str, ActivityList]
SheetRows = List[ActivityRow]

# Jitter range (seconds) between batches to avoid API rate limiting
_BATCH_JITTER_MIN = 0.05
_BATCH_JITTER_MAX = 0.2


def _default_activity_fetcher(
    runner: Runner, start_date: datetime, end_date: datetime
) -> ActivityList | None:
    """Default fetcher that calls the Strava API for runner activities."""
    return get_activities(runner, start_date, end_date)


@dataclass(slots=True)
class DistanceServiceConfig:
    """Configuration for DistanceService dependency injection."""

    fetcher: Callable[[Runner, datetime, datetime], ActivityList | None] = (
        _default_activity_fetcher
    )
    logger: logging.Logger | None = None


class DistanceService:
    """Fetches runner activities and builds distance competition outputs."""

    def __init__(self, config: DistanceServiceConfig | None = None):
        """Initialize with optional config for testing/customization."""
        self.config = config or DistanceServiceConfig()
        self._log = self.config.logger or logging.getLogger(self.__class__.__name__)

    def process(
        self,
        runners: Sequence[Runner],
        windows: Sequence[DistanceWindow],
    ) -> List[Tuple[str, SheetRows]]:
        """Fetch activities for all runners and build distance output sheets.

        Returns a list of (sheet_name, rows) tuples for each window plus summary.
        """
        if not runners or not windows:
            return []

        distance_runners = [r for r in runners if r.distance_team]
        if not distance_runners:
            return []

        earliest, latest = self._compute_date_range(windows)
        self._log.info(
            "Fetching activities for %d runners over union window %s -> %s",
            len(distance_runners),
            earliest,
            latest,
        )

        cache, failed_runners = self._fetch_all_activities(
            distance_runners, earliest, latest
        )
        outputs = build_distance_outputs(distance_runners, list(windows), cache)
        if failed_runners:
            runner_list = ", ".join(sorted(failed_runners)) or "unknown"
            self._log.warning(
                "Suppressed %d distance fetch errors (runners: %s)",
                len(failed_runners),
                runner_list,
            )
        self._log.info("Prepared %d distance sheets (including summary)", len(outputs))
        return outputs

    def _compute_date_range(
        self, windows: Sequence[DistanceWindow]
    ) -> Tuple[datetime, datetime]:
        """Compute the union date range across all windows."""
        earliest = min(w[0] for w in windows)
        latest = max(w[1] for w in windows)
        return earliest, latest

    def _fetch_all_activities(
        self,
        runners: Sequence[Runner],
        earliest: datetime,
        latest: datetime,
    ) -> Tuple[ActivityCache, set[str]]:
        """Fetch activities for all runners in parallel batches.

        Returns:
            Tuple of (cache, failed_runner_names).
        """
        cache: ActivityCache = {}
        max_parallel = max(1, CACHE_REFRESH_PARALLELISM)
        batch_size = min(max_parallel, len(runners))
        failed_runners: set[str] = set()
        failure_lock = threading.Lock()

        def record_failure(name: str) -> None:
            with failure_lock:
                failed_runners.add(name)

        for batch_index in range(0, len(runners), batch_size):
            batch = runners[batch_index : batch_index + batch_size]
            self._fetch_batch(
                batch, earliest, latest, cache, record_failure, batch_index, batch_size
            )
            if batch_index + batch_size < len(runners):
                # Bandit B311 false positive: randomness introduces pacing jitter only.
                time.sleep(random.uniform(_BATCH_JITTER_MIN, _BATCH_JITTER_MAX))  # nosec B311

        return cache, failed_runners

    def _fetch_batch(
        self,
        batch: Sequence[Runner],
        earliest: datetime,
        latest: datetime,
        cache: ActivityCache,
        record_failure: Callable[[str], None],
        batch_index: int,
        batch_size: int,
    ) -> None:
        """Fetch activities for a single batch of runners concurrently."""
        with ThreadPoolExecutor(max_workers=len(batch)) as executor:
            future_map = {
                executor.submit(
                    self._fetch_runner_activities, runner, earliest, latest
                ): runner
                for runner in batch
            }
            for future in as_completed(future_map):
                runner = future_map[future]
                try:
                    acts = future.result()
                except Exception as exc:  # pragma: no cover - defensive
                    record_failure(runner.name)
                    self._log.error(
                        "Runner %s distance fetch failed: %s",
                        runner.name,
                        exc,
                        exc_info=True,
                    )
                    cache[runner.strava_id] = []
                    continue
                cache[runner.strava_id] = acts
                self._log.debug(
                    "Cached %d activities for runner=%s team=%s (batch %d)",
                    len(acts),
                    runner.name,
                    runner.distance_team,
                    batch_index // batch_size + 1,
                )

    def _fetch_runner_activities(
        self,
        runner: Runner,
        earliest: datetime,
        latest: datetime,
    ) -> ActivityList:
        """Fetch activities for a single runner, handling errors gracefully."""
        try:
            return self.config.fetcher(runner, earliest, latest) or []
        except TokenError as exc:
            self._log.warning(
                "Skipping runner=%s distance processing: %s", runner.name, exc
            )
            return []


__all__ = ["DistanceService", "DistanceServiceConfig"]
