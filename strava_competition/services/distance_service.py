"""Distance competition service.

Encapsulates fetching runner activities over the union date window and
building per-window + summary distance outputs. Uses the pure
`build_distance_outputs` function (now in `distance_aggregation` module)
to keep aggregation logic testable and decoupled from I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Sequence, Tuple
import logging

from ..models import Runner
from ..strava_api import get_activities
from ..auth import TokenError
from ..distance_aggregation import build_distance_outputs

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
        for r in distance_runners:
            try:
                acts = self.config.fetcher(r, earliest, latest) or []
            except TokenError as exc:
                # Skip runners whose refresh tokens are invalid so the batch can continue.
                self._log.warning(
                    "Skipping runner=%s distance processing: %s", r.name, exc
                )
                acts = []
            cache[r.strava_id] = acts
            self._log.debug(
                "Cached %d activities for runner=%s team=%s",
                len(acts),
                r.name,
                r.distance_team,
            )
        outputs = build_distance_outputs(distance_runners, list(windows), cache)
        self._log.info("Prepared %d distance sheets (including summary)", len(outputs))
        return outputs


__all__ = ["DistanceService", "DistanceServiceConfig"]
