from datetime import datetime, timedelta, timezone
import os
from pathlib import Path

import pytest

from strava_competition.tools.capture_gc import (
    prune_directory,
    _parse_duration,
    parse_args,
)

_HEX_A = "a" * 64
_HEX_B = "b" * 64
_HEX_C = "c" * 64


def _create_file(path: Path, age_days: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}", encoding="utf-8")
    ts = datetime.now(timezone.utc) - timedelta(days=age_days)
    epoch = ts.timestamp()
    os.utime(path, (epoch, epoch))
    return path


def test_prune_directory_deletes_older_files(tmp_path: Path) -> None:
    old_file = _create_file(tmp_path / "aa" / "bb" / f"{_HEX_A}.json", age_days=40)
    fresh_file = _create_file(tmp_path / "aa" / "cc" / f"{_HEX_B}.json", age_days=5)

    stats = prune_directory(base=tmp_path, max_age_days=30, dry_run=False)

    assert not old_file.exists()
    assert fresh_file.exists()
    assert stats["deleted"] == 1


def test_prune_directory_dry_run(tmp_path: Path) -> None:
    old_file = _create_file(tmp_path / "dd" / "ee" / f"{_HEX_A}.json", age_days=40)

    stats = prune_directory(base=tmp_path, max_age_days=30, dry_run=True)

    assert old_file.exists()
    assert stats["deleted"] == 0
    assert stats["skipped"] >= 1


def test_prune_directory_with_timedelta(tmp_path: Path) -> None:
    old_file = _create_file(tmp_path / f"{_HEX_A}.json", age_days=31)
    prune_directory(base=tmp_path, max_age=timedelta(days=30), dry_run=False)
    assert not old_file.exists()


def test_prune_directory_deletes_overlay_files(tmp_path: Path) -> None:
    overlay = _create_file(tmp_path / f"{_HEX_C}.overlay.json", age_days=40)
    stats = prune_directory(base=tmp_path, max_age_days=30, dry_run=False)
    assert not overlay.exists()
    assert stats["deleted"] == 1


def test_prune_directory_ignores_non_cache_filenames(tmp_path: Path) -> None:
    """Arbitrary JSON files under the path must never be deleted."""
    precious = _create_file(tmp_path / "important-data.json", age_days=100)
    nested = _create_file(tmp_path / "sub" / "notes.json", age_days=100)
    short_hex = _create_file(tmp_path / "abc123.json", age_days=100)

    stats = prune_directory(base=tmp_path, max_age_days=30, dry_run=False)

    assert precious.exists()
    assert nested.exists()
    assert short_hex.exists()
    assert stats["deleted"] == 0
    assert stats["ignored"] == 3


def test_parse_duration_variants() -> None:
    assert _parse_duration("30d") == timedelta(days=30)
    assert _parse_duration("24h") == timedelta(days=1)
    assert _parse_duration("90m") == timedelta(minutes=90)
    assert _parse_duration("3600") == timedelta(hours=1)


def test_cli_defaults_to_dry_run() -> None:
    args = parse_args([])
    assert args.delete is False


def test_cli_delete_flag() -> None:
    args = parse_args(["--delete"])
    assert args.delete is True


def test_cli_delete_and_dry_run_conflict() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--delete", "--dry-run"])
