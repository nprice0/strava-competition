"""Utilities for comparing Excel workbooks.

This module provides a command-line interface for comparing two Excel workbooks
sheet by sheet and emitting a concise difference report. It is intentionally
lightweight so that analysts can validate regression outputs without opening
Excel manually.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, cast

import pandas as pd


@dataclass(frozen=True)
class CellDiff:
    """Represents a cell-level difference between two workbooks."""

    sheet: str
    row_label: object
    column: str
    left_value: object
    right_value: object


@dataclass(frozen=True)
class WorkbookDiff:
    """Aggregates sheet-level comparison results."""

    missing_in_left: List[str]
    missing_in_right: List[str]
    cell_differences: List[CellDiff]


@dataclass(frozen=True)
class TimingDiff:
    """Represents a timing difference for a runner within a segment sheet."""

    sheet: str
    runner: object
    column: str
    baseline_seconds: Optional[float]
    candidate_seconds: Optional[float]
    delta_seconds: Optional[float]


def _load_workbook(path: Path) -> Dict[str, pd.DataFrame]:
    """Load every sheet from an Excel workbook into pandas DataFrames."""
    sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")
    return cast(Dict[str, pd.DataFrame], sheets)


def _align_frames(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Return both frames with matching columns and indexes for comparison."""
    columns: List[str] = list(left.columns)
    for col in right.columns:
        if col not in columns:
            columns.append(col)

    left_aligned = left.copy()
    right_aligned = right.copy()
    for df in (left_aligned, right_aligned):
        for col in columns:
            if col not in df.columns:
                df[col] = pd.NA
        df.sort_index(
            axis=1, inplace=True, key=lambda idx: [columns.index(c) for c in idx]
        )

    combined_index = left_aligned.index.union(right_aligned.index)
    left_aligned = left_aligned.reindex(combined_index)
    right_aligned = right_aligned.reindex(combined_index)
    return left_aligned.convert_dtypes(), right_aligned.convert_dtypes()


def _values_close(left: object, right: object, float_tol: float) -> bool:
    """Return True when two scalar values should be treated as equal."""
    if pd.isna(left) and pd.isna(right):
        return True
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(
            float(left), float(right), rel_tol=float_tol, abs_tol=float_tol
        )
    return left == right


def compare_workbooks(
    left_path: Path,
    right_path: Path,
    float_tol: float = 1e-9,
) -> WorkbookDiff:
    """Compare two Excel workbooks and report sheet and cell discrepancies."""
    left_sheets = _load_workbook(left_path)
    right_sheets = _load_workbook(right_path)

    left_only = sorted(set(left_sheets) - set(right_sheets))
    right_only = sorted(set(right_sheets) - set(left_sheets))

    diffs: List[CellDiff] = []
    shared_sheets = sorted(set(left_sheets) & set(right_sheets))
    for sheet in shared_sheets:
        diffs.extend(
            _diff_sheet(sheet, left_sheets[sheet], right_sheets[sheet], float_tol)
        )

    return WorkbookDiff(
        missing_in_left=left_only,
        missing_in_right=right_only,
        cell_differences=diffs,
    )


def compare_segment_times(
    left_path: Path,
    right_path: Path,
    float_tol: float = 1e-9,
    runner_column: str = "Runner",
    time_columns: Optional[Sequence[str]] = None,
) -> List[TimingDiff]:
    """Compare runner timing columns across workbooks on a sheet-by-sheet basis."""
    left_sheets = _load_workbook(left_path)
    right_sheets = _load_workbook(right_path)
    shared_sheets = sorted(set(left_sheets) & set(right_sheets))

    timing_diffs: List[TimingDiff] = []
    for sheet in shared_sheets:
        sheet_diffs = _diff_sheet_timings(
            sheet_name=sheet,
            left_df=left_sheets[sheet],
            right_df=right_sheets[sheet],
            runner_column=runner_column,
            time_columns=time_columns,
            float_tol=float_tol,
        )
        timing_diffs.extend(sheet_diffs)
    return timing_diffs


def _diff_sheet(
    sheet_name: str,
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    float_tol: float,
) -> List[CellDiff]:
    """Return cell-level diffs for a single sheet."""
    aligned_left, aligned_right = _align_frames(left_df, right_df)
    comparison = aligned_left.compare(
        aligned_right,
        align_axis=0,
        keep_shape=True,
        keep_equal=False,
        result_names=("left", "right"),
    )
    if comparison.empty:
        return []

    base_columns = comparison.columns.get_level_values(0)
    unique_columns = dict.fromkeys(base_columns).keys()
    sheet_diffs: List[CellDiff] = []
    for row_index, row in comparison.iterrows():
        for column in unique_columns:
            left_value = row.get((column, "left"), pd.NA)
            right_value = row.get((column, "right"), pd.NA)
            if _values_close(left_value, right_value, float_tol):
                continue
            sheet_diffs.append(
                CellDiff(
                    sheet=sheet_name,
                    row_label=row_index,
                    column=column,
                    left_value=None if pd.isna(left_value) else left_value,
                    right_value=None if pd.isna(right_value) else right_value,
                )
            )
    return sheet_diffs


def _diff_sheet_timings(
    sheet_name: str,
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    runner_column: str,
    time_columns: Optional[Sequence[str]],
    float_tol: float,
) -> List[TimingDiff]:
    """Return timing diffs for runners within a single sheet."""
    prepared = _prepare_runner_frames(left_df, right_df, runner_column)
    if prepared is None:
        return []
    left_prepared, right_prepared = prepared

    candidate_columns = _select_time_columns(
        left_prepared.columns,
        right_prepared.columns,
        time_columns,
    )
    if not candidate_columns:
        return []

    left_subset = left_prepared.reindex(columns=candidate_columns)
    right_subset = right_prepared.reindex(columns=candidate_columns)
    aligned_left, aligned_right = _align_frames(left_subset, right_subset)

    timing_diffs: List[TimingDiff] = []
    all_runners = aligned_left.index.union(aligned_right.index)
    left_all = aligned_left.reindex(all_runners)
    right_all = aligned_right.reindex(all_runners)
    for runner in all_runners:
        timing_diffs.extend(
            _diff_runner_timings(
                sheet_name=sheet_name,
                runner=runner,
                columns=candidate_columns,
                left_row=left_all.loc[runner],
                right_row=right_all.loc[runner],
                float_tol=float_tol,
            )
        )
    return timing_diffs


def _diff_runner_timings(
    sheet_name: str,
    runner: object,
    columns: Sequence[str],
    left_row: pd.Series,
    right_row: pd.Series,
    float_tol: float,
) -> List[TimingDiff]:
    """Return timing diffs for an individual runner."""
    runner_diffs: List[TimingDiff] = []
    for column in columns:
        left_value = _coerce_seconds(left_row.get(column, pd.NA))
        right_value = _coerce_seconds(right_row.get(column, pd.NA))
        if left_value is None and right_value is None:
            continue
        if (
            left_value is not None
            and right_value is not None
            and math.isclose(
                left_value, right_value, rel_tol=float_tol, abs_tol=float_tol
            )
        ):
            continue
        delta = (
            None
            if left_value is None or right_value is None
            else right_value - left_value
        )
        runner_diffs.append(
            TimingDiff(
                sheet=sheet_name,
                runner=runner,
                column=column,
                baseline_seconds=left_value,
                candidate_seconds=right_value,
                delta_seconds=delta,
            )
        )
    return runner_diffs


def _prepare_runner_frames(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    runner_column: str,
) -> Optional[tuple[pd.DataFrame, pd.DataFrame]]:
    """Set the runner column as index for both sheets, skipping invalid layouts."""
    if runner_column not in left_df.columns or runner_column not in right_df.columns:
        return None

    left_prepared = _dedupe_runner_rows(left_df, runner_column)
    right_prepared = _dedupe_runner_rows(right_df, runner_column)
    if left_prepared.empty and right_prepared.empty:
        return None
    return left_prepared, right_prepared


def _dedupe_runner_rows(df: pd.DataFrame, runner_column: str) -> pd.DataFrame:
    """Return a frame indexed by runner with duplicate names collapsed."""
    unique = df.drop_duplicates(subset=runner_column).set_index(runner_column)
    return unique


def _select_time_columns(
    left_columns: Iterable[str],
    right_columns: Iterable[str],
    explicit: Optional[Sequence[str]],
) -> List[str]:
    """Determine which columns represent runner timings to evaluate."""
    if explicit:
        return [col for col in explicit if col in left_columns or col in right_columns]

    def matches(column: str) -> bool:
        return "time" in column.lower()

    candidates = {col for col in left_columns if matches(col)} | {
        col for col in right_columns if matches(col)
    }
    return sorted(candidates)


def _coerce_seconds(value: object) -> Optional[float]:
    """Convert a cell value into seconds when possible."""
    if pd.isna(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        candidate = value.strip()
        if not candidate:
            return None
        try:
            return float(candidate)
        except ValueError:
            return None
    return None


def _format_diff(diff: WorkbookDiff, max_rows: Optional[int]) -> str:
    """Build a human-readable report from the raw difference data."""
    lines: List[str] = []
    if diff.missing_in_left:
        lines.append(
            "Sheets only in right workbook: " + ", ".join(diff.missing_in_left)
        )
    if diff.missing_in_right:
        lines.append(
            "Sheets only in left workbook: " + ", ".join(diff.missing_in_right)
        )

    if not diff.cell_differences:
        if not lines:
            return "No differences detected."
        lines.append("No cell-level differences detected in shared sheets.")
        return "\n".join(lines)

    lines.append("Cell-level differences:")
    displayed = 0
    for cell_diff in diff.cell_differences:
        if max_rows is not None and displayed >= max_rows:
            lines.append("...")
            break
        lines.append(
            f"- Sheet '{cell_diff.sheet}', row {cell_diff.row_label}, column '{cell_diff.column}': "
            f"left={cell_diff.left_value!r} right={cell_diff.right_value!r}"
        )
        displayed += 1

    total = len(diff.cell_differences)
    if max_rows is not None and total > max_rows:
        lines.append(f"Displayed {max_rows} of {total} differences.")
    else:
        lines.append(f"Total differences: {total}.")
    return "\n".join(lines)


def _format_segment_diffs(
    timing_diffs: List[TimingDiff],
    max_rows: Optional[int],
) -> str:
    """Render runner timing differences in a readable format."""
    if not timing_diffs:
        return "No timing differences detected."

    lines: List[str] = ["Runner timing differences:"]
    ordered = sorted(
        timing_diffs,
        key=lambda diff: (diff.sheet, str(diff.runner), diff.column),
    )
    displayed = 0

    def fmt(value: Optional[float]) -> str:
        if value is None:
            return "—"
        text = f"{value:.3f}"
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return text

    for diff in ordered:
        if max_rows is not None and displayed >= max_rows:
            lines.append("...")
            break
        lines.append(
            "- Sheet '{sheet}', runner '{runner}', column '{column}': "
            "baseline={base} candidate={cand} delta={delta}".format(
                sheet=diff.sheet,
                runner=diff.runner,
                column=diff.column,
                base=fmt(diff.baseline_seconds),
                cand=fmt(diff.candidate_seconds),
                delta=fmt(diff.delta_seconds),
            )
        )
        displayed += 1

    total = len(ordered)
    if max_rows is not None and total > max_rows:
        lines.append(f"Displayed {max_rows} of {total} timing differences.")
    else:
        lines.append(f"Total timing differences: {total}.")
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for workbook comparison."""
    parser = argparse.ArgumentParser(description="Compare two Excel workbooks.")
    parser.add_argument("left", type=Path, help="Path to the baseline workbook.")
    parser.add_argument("right", type=Path, help="Path to the workbook under test.")
    parser.add_argument(
        "--float-tol",
        type=float,
        default=1e-9,
        help="Absolute and relative tolerance for float comparison (default: 1e-9).",
    )
    parser.add_argument(
        "--max-diffs",
        type=int,
        default=100,
        help="Maximum number of cell differences to display (default: 100).",
    )
    parser.add_argument(
        "--segment-times",
        action="store_true",
        help="Emit runner timing differences for each shared sheet.",
    )
    parser.add_argument(
        "--runner-column",
        default="Runner",
        help="Column name that identifies runners (default: Runner).",
    )
    parser.add_argument(
        "--time-columns",
        nargs="+",
        help="Specific timing columns to compare (defaults to any column containing 'time').",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point for comparing two Excel workbooks."""
    args = _parse_args()
    diff = compare_workbooks(args.left, args.right, args.float_tol)
    print(_format_diff(diff, args.max_diffs))
    if args.segment_times:
        timing_diffs = compare_segment_times(
            left_path=args.left,
            right_path=args.right,
            float_tol=args.float_tol,
            runner_column=args.runner_column,
            time_columns=args.time_columns,
        )
        print()
        print(
            _format_segment_diffs(
                timing_diffs=timing_diffs,
                max_rows=args.max_diffs,
            )
        )


if __name__ == "__main__":  # pragma: no cover
    main()
