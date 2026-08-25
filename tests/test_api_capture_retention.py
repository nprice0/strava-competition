from datetime import datetime, timedelta, timezone
import importlib
import os
from pathlib import Path

import pytest

import strava_competition.config as config_module
import strava_competition.api_capture as api_capture_module
from typing import Any


@pytest.fixture
def reload_capture_modules() -> Any:
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


_OLD_NAME = "a" * 64 + ".json"
_FRESH_NAME = "b" * 64 + ".json"


def test_auto_prune_removes_old_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, reload_capture_modules: Any
) -> None:
    old_file = _write_capture(tmp_path / "aa" / "bb" / _OLD_NAME, age_days=20)
    fresh_file = _write_capture(tmp_path / "aa" / "bb" / _FRESH_NAME, age_days=1)

    monkeypatch.setenv("STRAVA_API_CACHE_MODE", "cache")
    monkeypatch.setenv("STRAVA_CACHE_AUTO_PRUNE_DAYS", "7")
    monkeypatch.setenv("STRAVA_CACHE_DIR", str(tmp_path))

    reload_capture_modules()

    # Import must be side-effect free: nothing pruned until first use.
    assert old_file.exists()
    api_capture_module.get_capture()

    assert not old_file.exists()
    assert fresh_file.exists()

    monkeypatch.delenv("STRAVA_CACHE_AUTO_PRUNE_DAYS", raising=False)
    monkeypatch.delenv("STRAVA_CACHE_DIR", raising=False)
    monkeypatch.setenv("STRAVA_API_CACHE_MODE", "live")

    reload_capture_modules()


def test_no_prune_when_cache_disabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, reload_capture_modules: Any
) -> None:
    """Live mode (no save/read) must never run the retention policy."""

    old_file = _write_capture(tmp_path / "aa" / "bb" / _OLD_NAME, age_days=20)

    monkeypatch.setenv("STRAVA_API_CACHE_MODE", "live")
    monkeypatch.setenv("STRAVA_CACHE_AUTO_PRUNE_DAYS", "7")
    monkeypatch.setenv("STRAVA_CACHE_DIR", str(tmp_path))

    reload_capture_modules()
    api_capture_module.get_capture()

    assert old_file.exists()

    monkeypatch.delenv("STRAVA_CACHE_AUTO_PRUNE_DAYS", raising=False)
    monkeypatch.delenv("STRAVA_CACHE_DIR", raising=False)

    reload_capture_modules()
