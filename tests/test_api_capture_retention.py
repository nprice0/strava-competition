from datetime import datetime, timedelta, timezone
import importlib
import os
from pathlib import Path

import pytest

import strava_competition.config as config_module
import strava_competition.api_capture as api_capture_module


@pytest.fixture()
def reload_capture_modules():
    """Reload config + capture modules with current environment."""

    def _reload() -> None:
        importlib.reload(config_module)
        importlib.reload(api_capture_module)

    return _reload


def _write_capture(path: Path, *, age_days: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}", encoding="utf-8")
    ts = datetime.now(timezone.utc) - timedelta(days=age_days)
    epoch = ts.timestamp()
    os.utime(path, (epoch, epoch))
    return path


def test_auto_prune_removes_old_files(monkeypatch, tmp_path, reload_capture_modules):
    old_file = _write_capture(tmp_path / "aa" / "bb" / "old.json", age_days=20)
    fresh_file = _write_capture(tmp_path / "aa" / "bb" / "fresh.json", age_days=1)

    monkeypatch.setenv("STRAVA_API_CACHE_MODE", "cache")
    monkeypatch.setenv("STRAVA_CACHE_AUTO_PRUNE_DAYS", "7")
    monkeypatch.setenv("STRAVA_CACHE_DIR", str(tmp_path))

    reload_capture_modules()

    assert not old_file.exists()
    assert fresh_file.exists()

    monkeypatch.delenv("STRAVA_CACHE_AUTO_PRUNE_DAYS", raising=False)
    monkeypatch.delenv("STRAVA_CACHE_DIR", raising=False)
    monkeypatch.setenv("STRAVA_API_CACHE_MODE", "live")

    reload_capture_modules()
