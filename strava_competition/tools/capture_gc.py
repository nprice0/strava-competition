"""Utility for pruning stale cache payloads."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import logging
from pathlib import Path
from typing import Iterable, Optional

from ..config import STRAVA_CACHE_DIR

LOGGER = logging.getLogger(__name__)


def _resolve_base(path: str | Path | None) -> Path:
    if path is None:
        candidate = Path(STRAVA_CACHE_DIR)
    else:
        candidate = Path(path)
    return candidate if candidate.is_absolute() else Path.cwd() / candidate


def _iter_capture_files(base: Path) -> Iterable[Path]:
    if not base.exists():
        return []
    return base.rglob("*.json")


def prune_directory(
    *,
    base: str | Path | None = None,
    max_age: Optional[timedelta] = None,
    max_age_days: int = 30,
    dry_run: bool = False,
) -> dict[str, int]:
    resolved = _resolve_base(base)
    if not resolved.exists():
        LOGGER.info("Capture directory %s does not exist; nothing to prune.", resolved)
        return {"deleted": 0, "skipped": 0}

    window = max_age if max_age is not None else timedelta(days=max(0, max_age_days))
    cutoff = datetime.now(timezone.utc) - window
    deleted = skipped = 0
    for file_path in _iter_capture_files(resolved):
        try:
            modified = datetime.fromtimestamp(
                file_path.stat().st_mtime, tz=timezone.utc
            )
        except OSError:
            skipped += 1
            continue
        if modified >= cutoff:
            continue
        if dry_run:
            LOGGER.info("[dry-run] Would delete %s (age=%s)", file_path, modified)
            skipped += 1
            continue
        try:
            file_path.unlink()
            deleted += 1
        except OSError as exc:  # pragma: no cover - filesystem race
            LOGGER.warning("Failed to delete %s: %s", file_path, exc)
            skipped += 1
    LOGGER.info(
        "Cache GC complete base=%s deleted=%s skipped=%s cutoff=%s",
        resolved,
        deleted,
        skipped,
        cutoff,
    )
    return {"deleted": deleted, "skipped": skipped}


def _parse_duration(spec: str) -> timedelta:
    """Parse strings such as ``30d``/``12h``/``90m``/``3600`` into a timedelta."""

    spec = spec.strip().lower()
    if not spec:
        raise ValueError("empty duration spec")
    multiplier = 1
    suffix = spec[-1]
    if suffix in {"d", "h", "m", "s"}:
        spec = spec[:-1]
        if suffix == "d":
            multiplier = 24 * 3600
        elif suffix == "h":
            multiplier = 3600
        elif suffix == "m":
            multiplier = 60
        elif suffix == "s":
            multiplier = 1
    seconds = float(spec) * multiplier
    if seconds <= 0:
        raise ValueError("duration must be positive (got 0)")
    return timedelta(seconds=seconds)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Delete stale Strava cache payloads")
    parser.add_argument("--path", help="Cache directory (defaults to STRAVA_CACHE_DIR)")
    parser.add_argument(
        "--max-age",
        help="Delete files older than this duration (e.g. 30d, 12h, 90m, 3600)",
    )
    parser.add_argument(
        "--max-age-days",
        type=int,
        default=30,
        help="Delete files older than this many days (default: 30)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log deletions without removing files",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not LOGGER.handlers:
        logging.basicConfig(
            level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s"
        )
    window = None
    if args.max_age:
        try:
            window = _parse_duration(args.max_age)
        except ValueError as exc:  # pragma: no cover - argument validation
            LOGGER.error("Invalid --max-age value %s: %s", args.max_age, exc)
            raise SystemExit(2) from exc
    prune_directory(
        base=args.path,
        max_age=window,
        max_age_days=args.max_age_days,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
