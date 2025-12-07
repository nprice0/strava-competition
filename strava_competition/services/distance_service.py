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
from ..config import REPLAY_MAX_PARALLELISM

DistanceWindow = Tuple[datetime, datetime, float | None]
ActivityRow = Dict[str, Any]
ActivityList = List[ActivityRow]
ActivityCache = Dict[int | str, ActivityList]
SheetRows = List[ActivityRow]


def _default_activity_fetcher(
    runner: Runner, start_date: datetime, end_date: datetime
) -> ActivityList | None:
    return get_activities(runner, start_date, end_date)


@dataclass(slots=True)
class DistanceServiceConfig:
    fetcher: Callable[[Runner, datetime, datetime], ActivityList | None] = (
        _default_activity_fetcher
    )
    logger: logging.Logger | None = None


class DistanceService:
    def __init__(self, config: DistanceServiceConfig | None = None):
        self.config = config or DistanceServiceConfig()
        self._log = self.config.logger or logging.getLogger(self.__class__.__name__)

    def process(
        self,
        runners: Sequence[Runner],
        windows: Sequence[DistanceWindow],
    ) -> List[Tuple[str, SheetRows]]:
        if not runners or not windows:
            return []

        distance_runners = [r for r in runners if r.distance_team]
        if not distance_runners:
            return []

        earliest = min(w[0] for w in windows)
        latest = max(w[1] for w in windows)
        self._log.info(
            "Fetching activities for %d runners over union window %s -> %s",
            len(distance_runners),
            earliest,
            latest,
        )
        cache: ActivityCache = {}
        max_parallel = max(1, REPLAY_MAX_PARALLELISM)
        batch_size = min(max_parallel, len(distance_runners))
        suppressed_errors = 0
        failed_runners: set[str] = set()
        failure_lock = threading.Lock()

        def _record_failure(name: str) -> None:
            nonlocal suppressed_errors
            with failure_lock:
                suppressed_errors += 1
                failed_runners.add(name)

        def fetch_runner(runner: Runner) -> ActivityList:
            try:
                return self.config.fetcher(runner, earliest, latest) or []
            except TokenError as exc:
                self._log.warning(
                    "Skipping runner=%s distance processing: %s", runner.name, exc
                )
                return []
            except Exception as exc:  # pragma: no cover - defensive logging
                _record_failure(runner.name)
                self._log.error(
                    "Runner %s distance fetch failed due to unexpected error: %s",
                    runner.name,
                    exc,
                    exc_info=True,
                )
                return []

        for batch_index in range(0, len(distance_runners), batch_size):
            batch = distance_runners[batch_index : batch_index + batch_size]
            with ThreadPoolExecutor(max_workers=len(batch)) as executor:
                future_map = {
                    executor.submit(fetch_runner, runner): runner for runner in batch
                }
                for future in as_completed(future_map):
                    runner = future_map[future]
                    try:
                        acts = future.result()
                    except Exception as exc:  # pragma: no cover - defensive
                        _record_failure(runner.name)
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
            if batch_index + batch_size < len(distance_runners):
                # Bandit B311 false positive: randomness introduces pacing jitter only.
                time.sleep(random.uniform(0.05, 0.2))  # nosec B311
        outputs = build_distance_outputs(distance_runners, list(windows), cache)
        if suppressed_errors:
            runner_list = ", ".join(sorted(failed_runners)) or "unknown"
            self._log.warning(
                "Suppressed %d distance fetch errors (runners: %s)",
                suppressed_errors,
                runner_list,
            )
        self._log.info("Prepared %d distance sheets (including summary)", len(outputs))
        return outputs


__all__ = ["DistanceService", "DistanceServiceConfig"]
