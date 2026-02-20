import os
import tempfile
import pandas as pd

from strava_competition.excel_writer import write_results


def test_write_results_empty_summary() -> None:
    # Write an empty results workbook and ensure a Summary sheet exists
    with tempfile.TemporaryDirectory() as td:
        out_path = os.path.join(td, "results.xlsx")
        write_results(out_path, results={})
        assert os.path.exists(out_path)
        # Verify Summary sheet present
        with pd.ExcelFile(out_path) as xf:
            assert "Summary" in xf.sheet_names
