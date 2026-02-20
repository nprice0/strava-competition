from datetime import datetime, timedelta, timezone
import os
from pathlib import Path

from strava_competition.tools.capture_gc import prune_directory, _parse_duration


def _create_file(path: Path, age_days: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}", encoding="utf-8")
    ts = datetime.now(timezone.utc) - timedelta(days=age_days)
    epoch = ts.timestamp()
    os.utime(path, (epoch, epoch))
    return path


def test_prune_directory_deletes_older_files(tmp_path: Path) -> None:
    old_file = _create_file(tmp_path / "aa" / "bb" / "old.json", age_days=40)
    fresh_file = _create_file(tmp_path / "aa" / "cc" / "fresh.json", age_days=5)

    stats = prune_directory(base=tmp_path, max_age_days=30, dry_run=False)

    assert not old_file.exists()
    assert fresh_file.exists()
    assert stats["deleted"] == 1


def test_prune_directory_dry_run(tmp_path: Path) -> None:
    old_file = _create_file(tmp_path / "dd" / "ee" / "dry.json", age_days=40)

    stats = prune_directory(base=tmp_path, max_age_days=30, dry_run=True)

    assert old_file.exists()
    assert stats["deleted"] == 0
    assert stats["skipped"] >= 1


def test_prune_directory_with_timedelta(tmp_path: Path) -> None:
    old_file = _create_file(tmp_path / "older.json", age_days=31)
    prune_directory(base=tmp_path, max_age=timedelta(days=30), dry_run=False)
    assert not old_file.exists()


def test_parse_duration_variants() -> None:
    assert _parse_duration("30d") == timedelta(days=30)
    assert _parse_duration("24h") == timedelta(days=1)
    assert _parse_duration("90m") == timedelta(minutes=90)
    assert _parse_duration("3600") == timedelta(hours=1)
