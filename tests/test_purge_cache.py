"""Tests for the cache purge CLI tool."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from strava_competition.tools.purge_cache import main

# Capture-cache files are named <sha256-hex>[.overlay].(json|tmp).
DETAIL_NAME = "a" * 64
LISTING_NAME = "b" * 64
MALFORMED_NAME = "e" * 64
TMP_ORPHAN_NAME = "f" * 64
TMP_RECORD_NAME = "c" * 64


def _write_record(
    path: Path,
    *,
    captured_at: str,
    url: str,
    response: Any,
    overlay: bool = False,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "captured_at": captured_at,
        "request": {
            "method": "GET",
            "url": url,
            "identity": "runner:7",
            "params": None,
            "body": None,
        },
        "response": response,
    }
    if overlay:
        record["source"] = "overlay"
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


@pytest.fixture
def cache_tree(tmp_path: Path) -> dict[str, Path]:
    """Build a small fake cache tree covering the main record shapes."""
    detail = _write_record(
        tmp_path / "aa" / "bb" / f"{DETAIL_NAME}.json",
        captured_at="2026-08-20T10:00:00+00:00",
        url="https://www.strava.com/api/v3/activities/123",
        response={"id": 123, "start_date": "2026-08-19T06:00:00Z"},
    )
    listing = _write_record(
        tmp_path / "cc" / "dd" / f"{LISTING_NAME}.json",
        captured_at="2026-08-10T09:00:00+00:00",
        url="https://www.strava.com/api/v3/athlete/activities",
        response=[
            {"id": 1, "start_date": "2026-08-01T07:00:00Z"},
            {"id": 2, "start_date": "2026-08-09T07:00:00Z"},
        ],
    )
    overlay = _write_record(
        tmp_path / "aa" / "bb" / f"{DETAIL_NAME}.overlay.json",
        captured_at="2026-08-21T11:00:00+00:00",
        url="https://www.strava.com/api/v3/activities/123",
        response={"id": 123, "start_date": "2026-08-19T06:00:00Z"},
        overlay=True,
    )
    malformed = tmp_path / "ee" / f"{MALFORMED_NAME}.json"
    malformed.parent.mkdir(parents=True, exist_ok=True)
    malformed.write_text("{not valid json", encoding="utf-8")
    return {
        "base": tmp_path,
        "detail": detail,
        "listing": listing,
        "overlay": overlay,
        "malformed": malformed,
    }


def test_dry_run_lists_matches_but_keeps_files(
    cache_tree: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "--cache-dir",
            str(cache_tree["base"]),
            "--captured-after",
            "2026-08-01",
        ]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert str(cache_tree["detail"]) in out
    assert str(cache_tree["listing"]) in out
    assert str(cache_tree["overlay"]) in out
    assert "Matched 3 file(s)" in out
    for key in ("detail", "listing", "overlay", "malformed"):
        assert cache_tree[key].exists()


def test_delete_removes_only_matches(
    cache_tree: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = main(
        [
            "--cache-dir",
            str(cache_tree["base"]),
            "--captured-after",
            "2026-08-15",
            "--delete",
        ]
    )

    assert exit_code == 0
    assert not cache_tree["detail"].exists()
    assert not cache_tree["overlay"].exists()
    assert cache_tree["listing"].exists()
    assert cache_tree["malformed"].exists()
    assert "Matched 2 file(s)" in capsys.readouterr().out


def test_captured_window_filters(
    cache_tree: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(
        [
            "--cache-dir",
            str(cache_tree["base"]),
            "--captured-after",
            "2026-08-09",
            "--captured-before",
            "2026-08-11T00:00:00",
        ]
    )

    out = capsys.readouterr().out
    assert str(cache_tree["listing"]) in out
    assert str(cache_tree["detail"]) not in out
    assert "Matched 1 file(s)" in out


def test_activity_after_matches_dict_and_list_responses(
    cache_tree: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(
        [
            "--cache-dir",
            str(cache_tree["base"]),
            "--activity-after",
            "2026-08-05",
        ]
    )

    out = capsys.readouterr().out
    # Detail (start 2026-08-19) and listing (one entry 2026-08-09) match.
    assert str(cache_tree["detail"]) in out
    assert str(cache_tree["listing"]) in out
    assert str(cache_tree["overlay"]) in out
    assert "Matched 3 file(s)" in out


def test_activity_before_excludes_later_activities(
    cache_tree: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(
        [
            "--cache-dir",
            str(cache_tree["base"]),
            "--activity-before",
            "2026-08-05",
        ]
    )

    out = capsys.readouterr().out
    assert str(cache_tree["listing"]) in out
    assert str(cache_tree["detail"]) not in out
    assert "Matched 1 file(s)" in out


def test_url_pattern_filter(
    cache_tree: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(
        [
            "--cache-dir",
            str(cache_tree["base"]),
            "--url-pattern",
            r"/activities/\d+$",
        ]
    )

    out = capsys.readouterr().out
    assert str(cache_tree["detail"]) in out
    assert str(cache_tree["overlay"]) in out
    assert str(cache_tree["listing"]) not in out
    assert "Matched 2 file(s)" in out


def test_no_filter_invocation_errors(cache_tree: dict[str, Path]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--cache-dir", str(cache_tree["base"])])

    assert excinfo.value.code == 2
    for key in ("detail", "listing", "overlay", "malformed"):
        assert cache_tree[key].exists()


def test_malformed_files_are_skipped_gracefully(
    cache_tree: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    main(
        [
            "--cache-dir",
            str(cache_tree["base"]),
            "--url-pattern",
            "broken",
            "--delete",
        ]
    )

    assert cache_tree["malformed"].exists()
    out = capsys.readouterr().out
    assert "Matched 0 file(s)" in out
    assert "Skipped 1 file(s) (unreadable)" in out


def test_partial_delete_failure_exits_nonzero(
    cache_tree: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """M3: a deletion failure on any matched file yields exit code 1."""
    protected = cache_tree["detail"]
    original_unlink = Path.unlink

    def flaky_unlink(self: Path, *args: Any, **kwargs: Any) -> None:
        if self == protected:
            raise OSError("simulated deletion failure")
        original_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", flaky_unlink)

    exit_code = main(
        [
            "--cache-dir",
            str(cache_tree["base"]),
            "--captured-after",
            "2026-08-15",
            "--delete",
        ]
    )

    assert exit_code == 1
    assert protected.exists()
    assert not cache_tree["overlay"].exists()
    assert "Failed to delete 1 file(s)" in capsys.readouterr().out


def test_orphan_tmp_untouched_without_include_tmp(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """L1: *.tmp files are not scanned unless --include-tmp is given."""
    orphan = tmp_path / "ff" / f"{TMP_ORPHAN_NAME}.tmp"
    orphan.parent.mkdir(parents=True, exist_ok=True)
    orphan.write_text("{partial write", encoding="utf-8")

    exit_code = main(
        ["--cache-dir", str(tmp_path), "--captured-after", "2026-01-01", "--delete"]
    )

    assert exit_code == 0
    assert orphan.exists()
    assert "Matched 0 file(s)" in capsys.readouterr().out


def test_orphan_tmp_matches_unconditionally_with_include_tmp(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """L1: an unparseable .tmp always matches when --include-tmp is set."""
    orphan = tmp_path / "ff" / f"{TMP_ORPHAN_NAME}.tmp"
    orphan.parent.mkdir(parents=True, exist_ok=True)
    orphan.write_text("{partial write", encoding="utf-8")

    exit_code = main(
        [
            "--cache-dir",
            str(tmp_path),
            "--captured-after",
            "2030-01-01",  # No record could match this; orphan still does.
            "--include-tmp",
            "--delete",
        ]
    )

    assert exit_code == 0
    assert not orphan.exists()
    out = capsys.readouterr().out
    assert "orphaned tmp" in out
    assert "Matched 1 file(s)" in out


def test_wellformed_tmp_is_subject_to_filters(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """L1: a parseable .tmp record obeys the normal filters."""
    record_tmp = _write_record(
        tmp_path / "cc" / f"{TMP_RECORD_NAME}.tmp",
        captured_at="2026-08-10T09:00:00+00:00",
        url="https://www.strava.com/api/v3/activities/55",
        response={"id": 55, "start_date": "2026-08-09T07:00:00Z"},
    )

    exit_code = main(
        [
            "--cache-dir",
            str(tmp_path),
            "--captured-after",
            "2026-08-15",
            "--include-tmp",
            "--delete",
        ]
    )
    assert exit_code == 0
    assert record_tmp.exists()
    assert "Matched 0 file(s)" in capsys.readouterr().out

    exit_code = main(
        [
            "--cache-dir",
            str(tmp_path),
            "--captured-after",
            "2026-08-01",
            "--include-tmp",
            "--delete",
        ]
    )
    assert exit_code == 0
    assert not record_tmp.exists()
    assert "Matched 1 file(s)" in capsys.readouterr().out


def test_all_entries_requires_every_listing_entry_in_range(
    cache_tree: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """L2: --all-entries excludes listings with any out-of-range entry."""
    main(
        [
            "--cache-dir",
            str(cache_tree["base"]),
            "--activity-after",
            "2026-08-05",
            "--all-entries",
        ]
    )

    out = capsys.readouterr().out
    # Listing has one entry (2026-08-01) outside the range -> excluded.
    assert str(cache_tree["listing"]) not in out
    # Dict (detail) responses are unaffected by --all-entries.
    assert str(cache_tree["detail"]) in out
    assert str(cache_tree["overlay"]) in out
    assert "Matched 2 file(s)" in out


def test_all_entries_matches_when_every_entry_in_range(
    cache_tree: dict[str, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """L2: --all-entries still matches listings fully inside the range."""
    main(
        [
            "--cache-dir",
            str(cache_tree["base"]),
            "--activity-after",
            "2026-07-31",
            "--all-entries",
        ]
    )

    out = capsys.readouterr().out
    assert str(cache_tree["listing"]) in out
    assert "Matched 3 file(s)" in out


def test_non_cache_filenames_are_never_matched_or_deleted(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """L3: files without the capture-cache filename shape are ignored."""
    notes = _write_record(
        tmp_path / "aa" / "notes.json",
        captured_at="2026-08-20T10:00:00+00:00",
        url="https://www.strava.com/api/v3/activities/9",
        response={"id": 9, "start_date": "2026-08-19T06:00:00Z"},
    )

    exit_code = main(
        ["--cache-dir", str(tmp_path), "--captured-after", "2026-08-01", "--delete"]
    )

    assert exit_code == 0
    assert notes.exists()
    out = capsys.readouterr().out
    assert "Matched 0 file(s)" in out
    assert "Ignored 1 file(s) (not cache filenames)" in out


def test_valid_json_non_record_reported_distinctly(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Quick-win 4: valid-JSON-but-non-record files reported separately."""
    non_record = tmp_path / "aa" / f"{DETAIL_NAME}.json"
    non_record.parent.mkdir(parents=True, exist_ok=True)
    non_record.write_text(json.dumps([1, 2, 3]), encoding="utf-8")

    main(["--cache-dir", str(tmp_path), "--captured-after", "2026-08-01"])

    out = capsys.readouterr().out
    assert "Matched 0 file(s)" in out
    assert "Ignored 1 file(s) (not cache records)" in out


def test_invalid_url_regex_exits_with_usage_error(tmp_path: Path) -> None:
    """L5: a malformed --url-pattern regex exits cleanly with code 2."""
    with pytest.raises(SystemExit) as excinfo:
        main(["--cache-dir", str(tmp_path), "--url-pattern", "["])

    assert excinfo.value.code == 2
