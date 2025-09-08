"""Deprecated module retaining backward compatibility.

The original process_segments orchestration has been migrated into the
application-layer service: services.segment_service.SegmentService.

This module now only exposes a thin wrapper so existing imports continue to
function. It will be removed in a future major version.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable, Dict, List, Sequence

from .models import SegmentResult, Segment, Runner
from .config import MAX_WORKERS


def process_segments(  # pragma: no cover - shim
    segments: Sequence[Segment],
    runners: Sequence[Runner],
    max_workers: int | None = None,
    cancel_event: threading.Event | None = None,
    progress: Callable[[str, int, int], None] | None = None,
) -> Dict[str, Dict[str, List[SegmentResult]]]:
    """Delegate to SegmentService.process (deprecated path)."""
    from .services.segment_service import SegmentService  # Local import to avoid cycle

    if max_workers is None:
        max_workers = MAX_WORKERS
    logging.getLogger(__name__).warning(
        "process_segments is deprecated; instantiate SegmentService and call .process instead"
    )
    service = SegmentService(max_workers=max_workers)
    # SegmentService ignores cancel_event / progress for now; could be wired later if needed.
    return service.process(segments, runners)

__all__ = ["process_segments"]
