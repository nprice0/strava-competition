"""CLI tool to purge cached Strava API responses matching filters.

Evicts cache entries after an API incident or to force re-fetch of a date
range. Runs in dry-run mode by default; deletion requires ``--delete``.

Exit codes: 0 on success or dry run, 1 when ``--delete`` was given and one
or more matched files failed to delete, 2 on usage errors.

Usage::

    python -m strava_competition.tools.purge_cache --captured-after 2026-08-01
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ..config import STRAVA_CACHE_DIR

LOGGER = logging.getLogger(__name__)

# Only files named like capture-cache entries are eligible for scanning and
# deletion; guards against deleting arbitrary JSON when --cache-dir is
# mis-pointed at a non-cache directory. (*.tmp only scanned with --include-tmp.)
_CACHE_NAME_PATTERN = re.compile(r"^[a-f0-9]{64}(\.overlay)?\.(json|tmp)$")


@dataclass(frozen=True)
class PurgeFilters:
    """Filter criteria for selecting cache files; combined with AND."""

    captured_after: datetime | None = None
    captured_before: datetime | None = None
    activity_after: datetime | None = None
    activity_before: datetime | None = None
    url_pattern: re.Pattern[str] | None = None
    all_entries: bool = False

    def any_set(self) -> bool:
        """Return True when at least one filter is active."""
        return any(
            value is not None
            for value in (
                self.captured_after,
                self.captured_before,
                self.activity_after,
                self.activity_before,
                self.url_pattern,
            )
        )


def _parse_cli_datetime(value: str) -> datetime:
    """Parse an ISO date or datetime from the CLI; naive values become UTC.

    Raises:
        ValueError: If the value is not a valid ISO date/datetime.
    """
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _parse_iso_timestamp(value: Any) -> datetime | None:
    """Parse an ISO timestamp from cache data (may end in "Z"); None if invalid."""
    if not isinstance(value, str):
        return None
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _in_range(
    value: datetime,
    after: datetime | None,
    before: datetime | None,
) -> bool:
    """Return True when ``value`` falls within the (inclusive) bounds."""
    if after is not None and value < after:
        return False
    if before is not None and value > before:
        return False
    return True


def _activity_start_dates(response: Any) -> list[datetime]:
    """Extract parsable ``start_date`` values from a cached response payload.

    Dict responses yield at most one date; list responses (activity listing
    pages) yield one per entry that carries a valid ``start_date``.
    """
    candidates: list[Any] = []
    if isinstance(response, dict):
        candidates = [response.get("start_date")]
    elif isinstance(response, list):
        candidates = [
            entry.get("start_date") for entry in response if isinstance(entry, dict)
        ]
    parsed = (_parse_iso_timestamp(value) for value in candidates)
    return [value for value in parsed if value is not None]


def _captured_ok(record: dict[str, Any], filters: PurgeFilters) -> bool:
    """Return True when the record passes the captured-at filters."""
    if filters.captured_after is None and filters.captured_before is None:
        return True
    captured = _parse_iso_timestamp(record.get("captured_at"))
    if captured is None:
        return False
    return _in_range(captured, filters.captured_after, filters.captured_before)


def _activity_ok(record: dict[str, Any], filters: PurgeFilters) -> bool:
    """Return True when the record passes the activity start-date filters."""
    if filters.activity_after is None and filters.activity_before is None:
        return True
    response = record.get("response")
    in_range = [
        _in_range(start, filters.activity_after, filters.activity_before)
        for start in _activity_start_dates(response)
    ]
    if filters.all_entries and isinstance(response, list):
        # All dated entries must be in range, and at least one must exist.
        return bool(in_range) and all(in_range)
    return any(in_range)


def _url_ok(record: dict[str, Any], filters: PurgeFilters) -> bool:
    """Return True when the record passes the URL pattern filter."""
    if filters.url_pattern is None:
        return True
    request = record.get("request")
    url = request.get("url") if isinstance(request, dict) else None
    return isinstance(url, str) and bool(filters.url_pattern.search(url))


def _matches(record: dict[str, Any], filters: PurgeFilters) -> bool:
    """Return True when a cache record satisfies every active filter."""
    return (
        _captured_ok(record, filters)
        and _activity_ok(record, filters)
        and _url_ok(record, filters)
    )


def _resolve_base(path: str | Path | None) -> Path:
    """Resolve the cache directory, defaulting to STRAVA_CACHE_DIR."""
    candidate = Path(path) if path is not None else Path(STRAVA_CACHE_DIR)
    return candidate if candidate.is_absolute() else Path.cwd() / candidate


def _iter_cache_files(base: Path, *, include_tmp: bool = False) -> Iterable[Path]:
    """Yield candidate cache files under ``base`` (``*.tmp`` when requested)."""
    if not base.exists():
        return []
    files = list(base.rglob("*.json"))
    if include_tmp:
        files.extend(base.rglob("*.tmp"))
    return sorted(files)


def _load_record(path: Path) -> tuple[dict[str, Any] | None, str]:
    """Load a cache record from ``path``.

    Returns:
        ``(record, "ok")`` on success, ``(None, "unreadable")`` for IO or
        JSON errors, and ``(None, "non_record")`` for valid JSON that is
        not an object.
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        LOGGER.warning("Skipping unreadable cache file %s: %s", path, exc)
        return None, "unreadable"
    if not isinstance(payload, dict):
        LOGGER.debug("Ignoring non-record cache file %s", path)
        return None, "non_record"
    return payload, "ok"


def _report_match(path: Path, record: dict[str, Any], *, delete: bool) -> None:
    """Print a one-line description of a matching cache file."""
    request = record.get("request")
    url = request.get("url") if isinstance(request, dict) else None
    prefix = "[delete]" if delete else "[dry-run]"
    print(f"{prefix} captured_at={record.get('captured_at')} url={url} {path}")


@dataclass
class PurgeStats:
    """Counters accumulated during a purge scan."""

    matched: int = 0
    deleted: int = 0
    failed: int = 0
    ignored: int = 0
    non_records: int = 0
    unreadable: int = 0


def purge_directory(
    *,
    base: str | Path | None = None,
    filters: PurgeFilters,
    delete: bool = False,
    include_tmp: bool = False,
) -> dict[str, int]:
    """Scan the cache directory and purge (or list) matching files.

    Args:
        base: Cache directory; defaults to ``STRAVA_CACHE_DIR``.
        filters: Selection criteria; every active filter must match.
        delete: When False (default) run a dry run that only lists matches.
        include_tmp: Also scan ``*.tmp`` files; a ``.tmp`` file that cannot
            be parsed as a cache record (orphaned partial write) matches
            unconditionally.

    Returns:
        Counts of matched, deleted, failed, ignored (non-cache filenames),
        non-record (valid JSON but not a cache record), and unreadable files.
    """
    resolved = _resolve_base(base)
    stats = PurgeStats()
    for file_path in _iter_cache_files(resolved, include_tmp=include_tmp):
        if not _CACHE_NAME_PATTERN.match(file_path.name):
            stats.ignored += 1
            LOGGER.debug("Ignoring non-cache file name %s", file_path)
            continue
        _process_file(file_path, filters=filters, delete=delete, stats=stats)
    _print_summary(stats, delete=delete)
    return asdict(stats)


def _process_file(
    path: Path,
    *,
    filters: PurgeFilters,
    delete: bool,
    stats: PurgeStats,
) -> None:
    """Classify one candidate file, updating ``stats`` and deleting matches."""
    record, status = _load_record(path)
    if record is None:
        if path.suffix == ".tmp":
            # Unparseable .tmp = orphaned partial write; always purge.
            stats.matched += 1
            prefix = "[delete]" if delete else "[dry-run]"
            print(f"{prefix} orphaned tmp {path}")
            _delete_matched(path, delete=delete, stats=stats)
        elif status == "unreadable":
            stats.unreadable += 1
        else:
            stats.non_records += 1
        return
    if not _matches(record, filters):
        return
    stats.matched += 1
    _report_match(path, record, delete=delete)
    _delete_matched(path, delete=delete, stats=stats)


def _delete_matched(path: Path, *, delete: bool, stats: PurgeStats) -> None:
    """Delete a matched file (when deleting), recording success or failure."""
    if not delete:
        return
    if _delete_file(path):
        stats.deleted += 1
    else:
        stats.failed += 1


def _print_summary(stats: PurgeStats, *, delete: bool) -> None:
    """Print the scan summary to stdout."""
    if delete:
        print(f"Matched {stats.matched} file(s); deleted {stats.deleted}.")
        if stats.failed:
            print(f"Failed to delete {stats.failed} file(s).")
    else:
        print(
            f"Matched {stats.matched} file(s); would delete {stats.matched} "
            "(dry run; pass --delete)."
        )
    if stats.ignored:
        print(f"Ignored {stats.ignored} file(s) (not cache filenames).")
    if stats.non_records:
        print(f"Ignored {stats.non_records} file(s) (not cache records).")
    if stats.unreadable:
        print(f"Skipped {stats.unreadable} file(s) (unreadable).")


def _delete_file(path: Path) -> bool:
    """Delete a file, returning False (with a warning) on filesystem errors."""
    try:
        path.unlink()
    except OSError as exc:
        LOGGER.warning("Failed to delete %s: %s", path, exc)
        return False
    return True


def _datetime_arg(value: str) -> datetime:
    """argparse type converter for ISO date/datetime values."""
    try:
        return _parse_cli_datetime(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid ISO date/datetime: {value!r}"
        ) from exc


def _regex_arg(value: str) -> re.Pattern[str]:
    """argparse type converter for regular expressions."""
    try:
        return re.compile(value)
    except re.error as exc:
        raise argparse.ArgumentTypeError(f"invalid regex: {value!r} ({exc})") from exc


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments; errors out unless at least one filter is given."""
    parser = argparse.ArgumentParser(
        description="Purge cached Strava API responses matching filters "
        "(dry run by default)"
    )
    parser.add_argument(
        "--cache-dir",
        help="Cache directory (defaults to STRAVA_CACHE_DIR)",
    )
    parser.add_argument(
        "--captured-after",
        type=_datetime_arg,
        help="Match files captured at/after this ISO date or datetime (UTC)",
    )
    parser.add_argument(
        "--captured-before",
        type=_datetime_arg,
        help="Match files captured at/before this ISO date or datetime (UTC)",
    )
    parser.add_argument(
        "--activity-after",
        type=_datetime_arg,
        help="Match files whose activity start_date is at/after this instant",
    )
    parser.add_argument(
        "--activity-before",
        type=_datetime_arg,
        help="Match files whose activity start_date is at/before this instant",
    )
    parser.add_argument(
        "--url-pattern",
        type=_regex_arg,
        help="Regex matched (re.search) against the captured request URL",
    )
    parser.add_argument(
        "--include-tmp",
        action="store_true",
        help="Also scan *.tmp files (orphaned partial writes); a .tmp file "
        "that cannot be parsed as a cache record always matches",
    )
    parser.add_argument(
        "--all-entries",
        action="store_true",
        help="For list responses, match only when all entries with a "
        "parseable start_date are within the activity date range "
        "(default: any entry)",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Actually delete matches (default is a dry run)",
    )
    args = parser.parse_args(argv)
    filters = PurgeFilters(
        captured_after=args.captured_after,
        captured_before=args.captured_before,
        activity_after=args.activity_after,
        activity_before=args.activity_before,
        url_pattern=args.url_pattern,
        all_entries=args.all_entries,
    )
    if not filters.any_set():
        parser.error(
            "at least one filter is required (--captured-after/before, "
            "--activity-after/before, or --url-pattern)"
        )
    args.filters = filters
    return args


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Returns:
        0 on success or dry run; 1 when ``--delete`` was given and one or
        more matched files failed to delete.
    """
    args = parse_args(argv)
    if not LOGGER.handlers:
        logging.basicConfig(
            level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s"
        )
    stats = purge_directory(
        base=args.cache_dir,
        filters=args.filters,
        delete=args.delete,
        include_tmp=args.include_tmp,
    )
    return 1 if args.delete and stats["failed"] else 0


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
