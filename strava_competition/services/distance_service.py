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
import time
from typing import Callable, Dict, List, Sequence, Tuple

from ..models import Runner
from ..strava_api import get_activities
from ..auth import TokenError
from ..distance_aggregation import build_distance_outputs
from ..config import REPLAY_MAX_PARALLELISM

DistanceWindow = Tuple[datetime, datetime, float | None]
SheetRows = List[dict]


@dataclass(slots=True)
class DistanceServiceConfig:
    fetcher: Callable[[Runner, datetime, datetime], List[dict]] = get_activities
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
        cache: Dict[str, List[dict]] = {}
        max_parallel = max(1, REPLAY_MAX_PARALLELISM)
        batch_size = min(max_parallel, len(distance_runners))

        def fetch_runner(runner: Runner) -> List[dict]:
            try:
                return self.config.fetcher(runner, earliest, latest) or []
            except TokenError as exc:
                self._log.warning(
                    "Skipping runner=%s distance processing: %s", runner.name, exc
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
                    acts = future.result()
                    cache[runner.strava_id] = acts
                    self._log.debug(
                        "Cached %d activities for runner=%s team=%s (batch %d)",
                        len(acts),
                        runner.name,
                        runner.distance_team,
                        batch_index // batch_size + 1,
                    )
            if batch_index + batch_size < len(distance_runners):
                time.sleep(random.uniform(0.05, 0.2))
        outputs = build_distance_outputs(distance_runners, list(windows), cache)
        self._log.info("Prepared %d distance sheets (including summary)", len(outputs))
        return outputs


__all__ = ["DistanceService", "DistanceServiceConfig"]
