from datetime import datetime
import logging

from .config import INPUT_FILE, OUTPUT_FILE, OUTPUT_FILE_TIMESTAMP_ENABLED, MAX_WORKERS
from .excel_io import (
    read_runners,
    read_segments,
    update_runner_refresh_tokens,
    write_results,
)
from .processor import process_segments


def main():
    # Central logging setup (once)
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s"
        )
    if OUTPUT_FILE_TIMESTAMP_ENABLED:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{OUTPUT_FILE}_{timestamp}.xlsx"
    else:
        output_file = f"{OUTPUT_FILE}.xlsx"

    segments = read_segments(INPUT_FILE)
    runners = read_runners(INPUT_FILE)
    results = process_segments(segments, runners, max_workers=MAX_WORKERS)

    write_results(output_file, results)
    update_runner_refresh_tokens(INPUT_FILE, runners)
    logging.info(f"Results saved to {output_file}")
