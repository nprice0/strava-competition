"""Utilities for caching and retrieving Strava API responses.

The Strava competition tooling occasionally needs to run without making live
HTTP requests. This module provides a lightweight cache layer that can
persist JSON responses to disk and optionally serve them back on subsequent
runs. The behaviour is controlled through the STRAVA_API_CACHE_MODE setting
declared in ``config.py``.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

from .config import (
    STRAVA_CACHE_DIR,
    _cache_mode_saves,
    STRAVA_CACHE_OVERWRITE,
    _cache_mode_reads,
    STRAVA_CACHE_REDACT_PII,
    STRAVA_CACHE_REDACT_FIELDS,
    STRAVA_CACHE_AUTO_PRUNE_DAYS,
)
from .tools.capture_gc import prune_directory
from .utils import json_dumps_sorted

_LOGGER = logging.getLogger(__name__)
# Validate that signatures contain only lowercase hex characters (SHA256 output)
_SAFE_SIGNATURE = re.compile(r"^[a-f0-9]+$")


@dataclass(frozen=True)
class CaptureKey:
    """Represents the signature of a Strava API request."""

    method: str
    url: str
    identity: str
    params: dict[str, Any] | None
    body: dict[str, Any] | None

    def to_serialisable(self) -> dict[str, Any]:
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
        base = Path(STRAVA_CACHE_DIR)
        self._base_dir = base if base.is_absolute() else Path.cwd() / base
        self._save_to_cache = _cache_mode_saves
        self._use_cache = _cache_mode_reads
        self._overwrite = STRAVA_CACHE_OVERWRITE
        self._lock = threading.Lock()
        self._apply_retention_policy()
        if self._save_to_cache or self._use_cache:
            self._base_dir.mkdir(parents=True, exist_ok=True)
            _LOGGER.info(
                "API cache initialised dir=%s save=%s read=%s overwrite=%s",
                self._base_dir,
                self._save_to_cache,
                self._use_cache,
                self._overwrite,
            )

    def enabled_for_record(self) -> bool:
        """Return True when saving responses to disk."""

        return self._save_to_cache

    def enabled_for_replay(self) -> bool:
        """Return True when reading responses from disk."""

        return self._use_cache

    def _validate_signature(self, signature: str) -> None:
        """Validate signature contains only safe hex characters and length."""
        if not _SAFE_SIGNATURE.match(signature):
            raise ValueError(f"Invalid capture signature format: {signature!r}")
        if len(signature) != 64:
            raise ValueError(
                f"Capture signature must be 64 hex chars (SHA-256), "
                f"got {len(signature)}"
            )

    def _file_path(self, signature: str) -> Path:
        self._validate_signature(signature)
        return self._base_dir / signature[0:2] / signature[2:4] / f"{signature}.json"

    def _overlay_path(self, signature: str) -> Path:
        self._validate_signature(signature)
        return (
            self._base_dir
            / signature[0:2]
            / signature[2:4]
            / f"{signature}.overlay.json"
        )

    def _ensure_parent(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)

    def _signature(self, key: CaptureKey) -> str:
        payload = json_dumps_sorted(key.to_serialisable())
        # sha256 keeps the signature deterministic while avoiding collisions that
        # could arise from Python's randomised object hashing.
        return sha256(payload.encode("utf-8")).hexdigest()

    def _read_file(self, path: Path) -> dict[str, Any] | None:
        payload: Any
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except FileNotFoundError:
            return None
        except Exception as exc:  # pragma: no cover - defensive logging
            _LOGGER.error("Failed reading capture file %s: %s", path, exc)
            return None
        if isinstance(payload, dict):
            return payload
        _LOGGER.error(
            "Capture file %s contained unexpected payload type %s",
            path,
            type(payload).__name__,
        )
        return None

    def _write_file(self, path: Path, payload: dict[str, Any]) -> None:
        self._ensure_parent(path)
        temp_path = path.with_suffix(".tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2)
        try:
            temp_path.replace(path)
        except PermissionError:  # pragma: no cover - Windows file locking
            # On Windows another process may still hold the target file
            # open; fall back to removing before renaming.
            try:
                path.unlink(missing_ok=True)
                temp_path.replace(path)
            except OSError as exc:
                _LOGGER.error("Failed to persist cache file %s: %s", path, exc)
                temp_path.unlink(missing_ok=True)
                raise

    def _load_record(
        self,
        path: Path,
        signature: str,
        *,
        source: str,
    ) -> "CaptureRecord | None":
        payload = self._read_file(path)
        if payload is None:
            return None
        captured_at = payload.get("captured_at")
        try:
            captured_dt = (
                datetime.fromisoformat(captured_at)
                if isinstance(captured_at, str)
                else None
            )
        except Exception:  # pragma: no cover - defensive fallback
            captured_dt = None
        return CaptureRecord(
            signature=signature,
            response=payload.get("response"),
            captured_at=captured_dt,
            source=source,
        )

    def fetch_record(self, key: CaptureKey) -> "CaptureRecord | None":
        """Return cached response + metadata when cache reading is active."""

        if not self._use_cache:
            return None
        signature = self._signature(key)
        overlay = self._load_record(
            self._overlay_path(signature), signature, source="overlay"
        )
        if overlay is not None:
            return overlay
        return self._load_record(self._file_path(signature), signature, source="base")

    def fetch(self, key: CaptureKey) -> Any | None:
        """Return cached response when replay is active (legacy helper)."""

        record = self.fetch_record(key)
        return None if record is None else record.response

    def store(self, key: CaptureKey, response: Any) -> None:
        """Persist a JSON response when cache saving is active."""

        if not self._save_to_cache:
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

    def store_overlay(self, key: CaptureKey, response: Any) -> None:
        """Persist an enriched response that should override the base cache."""

        if not self._save_to_cache:
            return
        signature = self._signature(key)
        path = self._overlay_path(signature)
        payload = {
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "request": key.to_serialisable(),
            "response": response,
            "source": "overlay",
        }
        with self._lock:
            self._write_file(path, payload)
        _LOGGER.debug("Overlay stored signature=%s path=%s", signature, path)

    def _apply_retention_policy(self) -> None:
        days = STRAVA_CACHE_AUTO_PRUNE_DAYS
        if days <= 0:
            return
        try:
            stats = prune_directory(
                base=self._base_dir,
                max_age_days=days,
                dry_run=False,
            )
            _LOGGER.info(
                "Capture retention pruned deleted=%s skipped=%s window_days=%s",
                stats.get("deleted", 0),
                stats.get("skipped", 0),
                days,
            )
        except Exception as exc:  # pragma: no cover - retention best-effort
            _LOGGER.warning("Capture retention pruning failed: %s", exc)


_CAPTURE = APICapture()


def _redact_payload(value: Any, path: str = "") -> Any:
    """Recursively redact sensitive fields from a payload.

    Supports both simple field names (match at any level) and dot-notation
    paths (e.g., "athlete.firstname" only matches firstname inside athlete).
    """
    if not STRAVA_CACHE_REDACT_PII:
        return value
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            lowered = key.lower() if isinstance(key, str) else key
            if not isinstance(lowered, str):
                redacted[key] = _redact_payload(item, path)
                continue
            # Build the full dot-path for this key
            full_path = f"{path}.{lowered}" if path else lowered
            # Check if this field should be redacted:
            # 1. Exact path match (e.g., "athlete.firstname")
            # 2. Simple field name match at any level (e.g., "email")
            should_redact = (
                full_path in STRAVA_CACHE_REDACT_FIELDS
                or lowered in STRAVA_CACHE_REDACT_FIELDS
            )
            if should_redact:
                redacted[key] = "***redacted***"
            else:
                redacted[key] = _redact_payload(item, full_path)
        return redacted
    if isinstance(value, list):
        return [_redact_payload(item, path) for item in value]
    return value


def get_cached_response(
    method: str,
    url: str,
    identity: str,
    *,
    params: dict[str, Any] | None = None,
    body: dict[str, Any] | None = None,
) -> Any | None:
    """Return a cached response if cache reading is active."""

    return _CAPTURE.fetch(
        CaptureKey(method=method, url=url, identity=identity, params=params, body=body)
    )


@dataclass(frozen=True)
class CaptureRecord:
    """Container for cached payloads and their metadata."""

    signature: str
    response: Any
    captured_at: datetime | None
    source: str


def get_cached_response_with_meta(
    method: str,
    url: str,
    identity: str,
    *,
    params: dict[str, Any] | None = None,
    body: dict[str, Any] | None = None,
) -> CaptureRecord | None:
    """Return cached response and metadata when cache reading is active."""

    return _CAPTURE.fetch_record(
        CaptureKey(method=method, url=url, identity=identity, params=params, body=body)
    )


def save_response_to_cache(
    method: str,
    url: str,
    identity: str,
    response: Any,
    *,
    params: dict[str, Any] | None = None,
    body: dict[str, Any] | None = None,
) -> None:
    """Persist a JSON serialisable response when cache saving is active."""

    _CAPTURE.store(
        CaptureKey(method=method, url=url, identity=identity, params=params, body=body),
        _redact_payload(response),
    )


def save_overlay_to_cache(
    method: str,
    url: str,
    identity: str,
    response: Any,
    *,
    params: dict[str, Any] | None = None,
    body: dict[str, Any] | None = None,
) -> None:
    """Persist an enriched response that should override the cached payload."""

    _CAPTURE.store_overlay(
        CaptureKey(method=method, url=url, identity=identity, params=params, body=body),
        _redact_payload(response),
    )


def cache_modes() -> dict[str, bool]:
    """Expose current cache flags for diagnostics."""

    return {
        "save": _CAPTURE.enabled_for_record(),
        "read": _CAPTURE.enabled_for_replay(),
    }
