"""Utilities for recording and replaying Strava API responses.

The Strava competition tooling occasionally needs to run without making live
HTTP requests. This module provides a lightweight capture/replay layer that can
persist JSON responses to disk and optionally serve them back on subsequent
runs. The behaviour is controlled through configuration flags declared in
``config.py``.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Optional

from .config import (
    STRAVA_API_CAPTURE_DIR,
    STRAVA_API_CAPTURE_ENABLED,
    STRAVA_API_CAPTURE_OVERWRITE,
    STRAVA_API_REPLAY_ENABLED,
)
from .utils import json_dumps_sorted

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CaptureKey:
    """Represents the signature of a Strava API request."""

    method: str
    url: str
    identity: str
    params: Optional[Dict[str, Any]]
    body: Optional[Dict[str, Any]]

    def to_serialisable(self) -> Dict[str, Any]:
        """Return a JSON-friendly representation."""
        return {
            "method": self.method.upper(),
            "url": self.url,
            "identity": self.identity,
            "params": self.params or {},
            "body": self.body or {},
        }


class APICapture:
    """Persistent store for Strava API JSON responses."""

    def __init__(self) -> None:
        base = Path(STRAVA_API_CAPTURE_DIR)
        self._base_dir = base if base.is_absolute() else Path.cwd() / base
        self._record_enabled = STRAVA_API_CAPTURE_ENABLED
        self._replay_enabled = STRAVA_API_REPLAY_ENABLED
        self._overwrite = STRAVA_API_CAPTURE_OVERWRITE
        self._lock = threading.Lock()
        if self._record_enabled or self._replay_enabled:
            self._base_dir.mkdir(parents=True, exist_ok=True)
            _LOGGER.info(
                "API capture initialised dir=%s record=%s replay=%s overwrite=%s",
                self._base_dir,
                self._record_enabled,
                self._replay_enabled,
                self._overwrite,
            )

    def enabled_for_record(self) -> bool:
        """Return True when recording responses to disk."""

        return self._record_enabled

    def enabled_for_replay(self) -> bool:
        """Return True when replaying responses from disk."""

        return self._replay_enabled

    def _file_path(self, signature: str) -> Path:
        return self._base_dir / signature[0:2] / signature[2:4] / f"{signature}.json"

    def _ensure_parent(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

    def _signature(self, key: CaptureKey) -> str:
        payload = json_dumps_sorted(key.to_serialisable())
        # sha256 keeps the signature deterministic while avoiding collisions that
        # could arise from Python's randomised object hashing.
        return sha256(payload.encode("utf-8")).hexdigest()

    def _read_file(self, path: Path) -> Optional[Dict[str, Any]]:
        try:
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except FileNotFoundError:
            return None
        except Exception as exc:  # pragma: no cover - defensive logging
            _LOGGER.error("Failed reading capture file %s: %s", path, exc)
            return None

    def _write_file(self, path: Path, payload: Dict[str, Any]) -> None:
        self._ensure_parent(path)
        temp_path = path.with_suffix(".tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2)
        temp_path.replace(path)

    def fetch(self, key: CaptureKey) -> Optional[Any]:
        """Return cached response when replay is active."""

        if not self._replay_enabled:
            return None
        signature = self._signature(key)
        path = self._file_path(str(signature))
        record = self._read_file(path)
        if record is None:
            return None
        return record.get("response")

    def store(self, key: CaptureKey, response: Any) -> None:
        """Persist a JSON response when recording is active."""

        if not self._record_enabled:
            return
        signature = self._signature(key)
        path = self._file_path(str(signature))
        if not self._overwrite and path.exists():
            return
        payload = {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "request": key.to_serialisable(),
            "response": response,
        }
        with self._lock:
            self._write_file(path, payload)
        _LOGGER.debug("Captured response signature=%s path=%s", signature, path)


_CAPTURE = APICapture()


def replay_response(
    method: str,
    url: str,
    identity: str,
    *,
    params: Optional[Dict[str, Any]] = None,
    body: Optional[Dict[str, Any]] = None,
) -> Optional[Any]:
    """Return a cached response if replay mode is active."""

    return _CAPTURE.fetch(
        CaptureKey(method=method, url=url, identity=identity, params=params, body=body)
    )


def record_response(
    method: str,
    url: str,
    identity: str,
    response: Any,
    *,
    params: Optional[Dict[str, Any]] = None,
    body: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist a JSON serialisable response when capture mode is active."""

    _CAPTURE.store(
        CaptureKey(method=method, url=url, identity=identity, params=params, body=body),
        response,
    )


def capture_modes() -> Dict[str, bool]:
    """Expose current capture flags for diagnostics."""

    return {
        "record": _CAPTURE.enabled_for_record(),
        "replay": _CAPTURE.enabled_for_replay(),
    }
