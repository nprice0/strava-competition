"""Shared Strava client helpers (auth, headers, session utilities)."""

from __future__ import annotations

import logging
from typing import Dict

from ..auth import get_access_token
from ..config import STRAVA_OFFLINE_MODE
from ..models import Runner

LOGGER = logging.getLogger(__name__)


def ensure_runner_token(runner: Runner) -> None:
    """Ensure the runner has a valid access token, refreshing when needed."""

    if STRAVA_OFFLINE_MODE:
        if not getattr(runner, "access_token", None) and not getattr(
            runner, "_skip_token_logged", False
        ):
            LOGGER.info(
                "Skipping Strava token refresh for runner=%s (STRAVA_OFFLINE_MODE)",
                getattr(runner, "name", "?"),
            )
            setattr(runner, "_skip_token_logged", True)
        return

    if getattr(runner, "access_token", None):
        return

    access_token, new_refresh_token = get_access_token(
        runner.refresh_token, runner_name=runner.name
    )
    runner.access_token = access_token
    if new_refresh_token and new_refresh_token != runner.refresh_token:
        runner.refresh_token = new_refresh_token
        try:
            from ..excel_writer import (
                update_single_runner_refresh_token,
            )  # local import
            from ..config import INPUT_FILE

            update_single_runner_refresh_token(INPUT_FILE, runner)
        except Exception as exc:  # pragma: no cover - best-effort persistence
            LOGGER.debug(
                "Failed to persist refresh token for runner %s: %s",
                runner.name,
                exc,
                exc_info=True,
            )


def auth_headers(runner: Runner) -> Dict[str, str]:
    """Return bearer auth headers for the runner (token assumed valid)."""

    token = getattr(runner, "access_token", None) or ""
    return {"Authorization": f"Bearer {token}"}
