import concurrent.futures
from pathlib import Path

import pandas as pd
import pytest

from strava_competition.excel_writer import (
    RUNNERS_SHEET,
    STRAVA_ID_COLUMN,
    REFRESH_TOKEN_COLUMN,
    SEGMENT_TEAM_COLUMN,
    DISTANCE_TEAM_COLUMN,
    BIRTHDAY_COLUMN,
    _atomic_replace_sheet,
    update_runner_refresh_tokens,
    update_single_runner_refresh_token,
)
from strava_competition.models import Runner
from typing import Any


@pytest.fixture()
def runners_sheet(tmp_path: Path) -> Path:
    path = Path(tmp_path) / "tokens.xlsx"
    df = pd.DataFrame(
        {
            "Name": ["Ana", "Ben", "Cara"],
            STRAVA_ID_COLUMN: ["101", "202", "303"],
            REFRESH_TOKEN_COLUMN: ["tok1", "tok2", "tok3"],
            SEGMENT_TEAM_COLUMN: [None, None, None],
            DISTANCE_TEAM_COLUMN: [None, None, None],
            BIRTHDAY_COLUMN: [
                pd.Timestamp("2001-05-10"),
                "3-Nov",
                pd.Timestamp("1999-01-01 08:00:00"),
            ],
        }
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name=RUNNERS_SHEET, index=False)
    return path


def _refresh_token_for(df: pd.DataFrame, strava_id: str) -> Any:
    ids = df[STRAVA_ID_COLUMN].astype(str).str.strip()
    series = df.loc[ids == str(strava_id), REFRESH_TOKEN_COLUMN]
    assert not series.empty
    return series.iat[0]


def test_update_runner_refresh_tokens_updates_all_rows(runners_sheet: Any) -> None:
    runners = [
        Runner("Ana", "101", "new-ana"),
        Runner("Ben", "202", "new-ben"),
        Runner("Cara", "303", "new-cara"),
    ]

    update_runner_refresh_tokens(str(runners_sheet), runners)

    result = pd.read_excel(runners_sheet, sheet_name=RUNNERS_SHEET)
    assert _refresh_token_for(result, "101") == "new-ana"
    assert _refresh_token_for(result, "202") == "new-ben"
    assert _refresh_token_for(result, "303") == "new-cara"


def test_update_runner_refresh_tokens_formats_birthdays(runners_sheet: Any) -> None:
    runners = [Runner("Ana", "101", "tok-new")]

    update_runner_refresh_tokens(str(runners_sheet), runners)

    result = pd.read_excel(runners_sheet, sheet_name=RUNNERS_SHEET)
    birthdays = result[BIRTHDAY_COLUMN].tolist()
    assert birthdays[0] == "10-May"
    assert birthdays[1] == "03-Nov"
    assert birthdays[2] == "01-Jan"


def test_update_single_runner_refresh_token_threadsafe(runners_sheet: Any) -> None:
    runners = [
        Runner("Ana", "101", "tok-ana-1"),
        Runner("Ben", "202", "tok-ben-1"),
        Runner("Cara", "303", "tok-cara-1"),
    ]

    def rotate(runner: Runner, suffix: str) -> None:
        update_single_runner_refresh_token(
            str(runners_sheet),
            Runner(runner.name, runner.strava_id, f"{runner.refresh_token}:{suffix}"),
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(runners)) as executor:
        futures = [executor.submit(rotate, runner, "thread") for runner in runners]
        concurrent.futures.wait(futures)

    result = pd.read_excel(runners_sheet, sheet_name=RUNNERS_SHEET)
    for runner in runners:
        expected = f"{runner.refresh_token}:thread"
        actual = _refresh_token_for(result, str(runner.strava_id))
        assert actual == expected


def test_atomic_replace_sheet_failure_preserves_workbook(
    runners_sheet: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash mid-write must leave the original workbook untouched.

    Simulates a failure while writing the temporary copy and asserts the
    original tokens survive and no stray temp file is left behind.
    """

    def boom(*_a: Any, **_kw: Any) -> Any:
        raise RuntimeError("simulated crash mid-write")

    monkeypatch.setattr("strava_competition.excel_writer.pd.ExcelWriter", boom)

    df = pd.DataFrame({"Name": ["Ana"], STRAVA_ID_COLUMN: ["101"]})
    with pytest.raises(RuntimeError, match="simulated crash"):
        _atomic_replace_sheet(str(runners_sheet), RUNNERS_SHEET, df)

    # Original workbook intact
    result = pd.read_excel(runners_sheet, sheet_name=RUNNERS_SHEET)
    assert _refresh_token_for(result, "101") == "tok1"
    assert _refresh_token_for(result, "202") == "tok2"
    assert _refresh_token_for(result, "303") == "tok3"

    # No leftover temp files in the workbook's directory
    workbook = Path(runners_sheet)
    leftovers = [
        p
        for p in workbook.parent.iterdir()
        if p != workbook and p.name.startswith(f".{workbook.stem}.")
    ]
    assert not leftovers, f"Temp files leaked: {leftovers}"


def test_update_single_runner_token_write_failure_preserves_workbook(
    runners_sheet: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A write error during single-runner persistence must not corrupt data."""

    def boom(*_a: Any, **_kw: Any) -> Any:
        raise OSError("disk full")

    monkeypatch.setattr("strava_competition.excel_writer.pd.ExcelWriter", boom)

    # Swallowed (OSError) and logged — should not raise.
    update_single_runner_refresh_token(
        str(runners_sheet), Runner("Ana", "101", "rotated-token")
    )

    result = pd.read_excel(runners_sheet, sheet_name=RUNNERS_SHEET)
    assert _refresh_token_for(result, "101") == "tok1"

    workbook = Path(runners_sheet)
    leftovers = [
        p
        for p in workbook.parent.iterdir()
        if p != workbook and p.name.startswith(f".{workbook.stem}.")
    ]
    assert not leftovers, f"Temp files leaked: {leftovers}"
