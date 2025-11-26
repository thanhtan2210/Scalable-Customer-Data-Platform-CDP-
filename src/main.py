"""Main orchestration for the ETL pipeline.

This module provides an entrypoint (CLI) and the `run_pipeline` function which
is tested by integration tests. Keep business logic in `src.etl.cleaning`.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.etl.cleaning import (
    load_csv,
    convert_types,
    drop_invalid_rows,
    map_booleans,
    create_features,
    save_parquet,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ETL pipeline")
    parser.add_argument("--input", required=True,
                        help="Path to input CSV file")
    parser.add_argument("--output", required=True,
                        help="Directory to write parquet output")
    parser.add_argument("--partition", default=None,
                        help="Column to partition on (optional)")
    parser.add_argument("--dry-run", action="store_true",
                        help="If provided, don't write output to disk")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    return parser.parse_args()


def run_pipeline(input_path: str, output_dir: str, partition_col: Optional[str] = None, dry_run: bool = False) -> pd.DataFrame:
    """Run the ETL pipeline end-to-end and optionally persist results.

    Args:
        input_path: path to the CSV input file.
        output_dir: directory where parquet files will be written.
        partition_col: optional column to partition by (writes one file per partition value).
        dry_run: if True, pipeline performs transformations but does not save.

    Returns:
        Transformed DataFrame that resulted from the pipeline.
    """
    logger.info("Starting ETL pipeline: input=%s output=%s partition=%s dry_run=%s",
                input_path, output_dir, partition_col, dry_run)
    # Load
    df = load_csv(input_path)

    # Transformations
    df = convert_types(df)
    df = drop_invalid_rows(df)
    df = map_booleans(df)
    df = create_features(df)

    # Persist
    if not dry_run:
        save_parquet(df, output_dir, partition_col=partition_col)
        logger.info("Saved parquet files to %s", output_dir)
    else:
        logger.info("Dry-run mode; no files written.")

    return df


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=args.log_level)
    try:
        run_pipeline(args.input, args.output,
                     partition_col=args.partition, dry_run=args.dry_run)
        return 0
    except Exception as exc:  # pragma: no cover - orchestration only
        logger.exception("ETL pipeline failed: %s", exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
