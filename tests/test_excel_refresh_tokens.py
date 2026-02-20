"""Regression tests for refresh-token persistence helpers."""

from __future__ import annotations

import threading
from pathlib import Path

import pandas as pd

from strava_competition.excel_writer import (
    REFRESH_TOKEN_COLUMN,
    RUNNERS_SHEET,
    update_runner_refresh_tokens,
    update_single_runner_refresh_token,
)
from strava_competition.models import Runner


def _runner_row(name: str, runner_id: str, refresh: str) -> dict[str, str]:
    return {
        "Name": name,
        "Strava ID": runner_id,
        REFRESH_TOKEN_COLUMN: refresh,
        "Segment Series Team": "Team",
        "Distance Series Team": "Team",
    }


def _write_workbook(path: Path, rows: list[dict[str, str]]) -> None:
    df = pd.DataFrame(rows)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=RUNNERS_SHEET, index=False)


def _read_tokens(path: Path) -> dict[str, str]:
    df = pd.read_excel(path, sheet_name=RUNNERS_SHEET)
    tokens: dict[str, str] = {}
    for row in df[["Strava ID", REFRESH_TOKEN_COLUMN]].itertuples(index=False):
        runner_id, token = row
        tokens[str(runner_id).strip()] = str(token)
    return tokens


def test_update_runner_refresh_tokens_updates_all_rows(tmp_path: Path) -> None:
    workbook = tmp_path / "runners.xlsx"
    _write_workbook(
        workbook,
        [
            _runner_row("Alice", "101", "old-a"),
            _runner_row("Bob", "102", "old-b"),
        ],
    )

    runners = [
        Runner(name="Alice", strava_id="101", refresh_token="new-a"),
        Runner(name="Bob", strava_id="102", refresh_token="new-b"),
    ]

    update_runner_refresh_tokens(workbook, runners)

    tokens = _read_tokens(workbook)
    assert tokens["101"] == "new-a"
    assert tokens["102"] == "new-b"


def test_update_single_runner_refresh_token_is_thread_safe(tmp_path: Path) -> None:
    workbook = tmp_path / "runners.xlsx"
    _write_workbook(
        workbook,
        [
            _runner_row("Carol", "201", "old-c"),
            _runner_row("Dave", "202", "old-d"),
            _runner_row("Eve", "203", "old-e"),
        ],
    )

    runners = [
        Runner(name="Carol", strava_id="201", refresh_token="base-c"),
        Runner(name="Dave", strava_id="202", refresh_token="base-d"),
        Runner(name="Eve", strava_id="203", refresh_token="base-e"),
    ]

    barrier = threading.Barrier(len(runners) + 1)

    def _worker(runner: Runner, suffix: str) -> None:
        barrier.wait()
        update_single_runner_refresh_token(
            workbook,
            Runner(
                name=runner.name,
                strava_id=runner.strava_id,
                refresh_token=f"{runner.refresh_token}:{suffix}",
            ),
        )

    threads = []
    for idx, runner in enumerate(runners):
        thread = threading.Thread(
            target=_worker,
            args=(runner, f"thread-{idx}"),
        )
        threads.append(thread)
        thread.start()

    barrier.wait()
    for thread in threads:
        thread.join()

    tokens = _read_tokens(workbook)
    for idx, runner in enumerate(runners):
        expected = f"{runner.refresh_token}:thread-{idx}"
        assert tokens[str(runner.strava_id)] == expected
