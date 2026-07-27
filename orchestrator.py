#!/usr/bin/env python
"""
ETL Pipeline Orchestrator — Nigerian SME Loans
Runs extract → transform → load in sequence with logging and error handling.
"""

import sys
import logging
import argparse

print("[orchestrator] Orchestrator module loaded")

from config import Config
from extract import extract
from transform import transform, analyze
from load import load

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("orchestrator")


def main():
    parser = argparse.ArgumentParser(
        description="ETL pipeline for Nigerian SME loan data"
    )
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip download and use cached CSV files",
    )
    parser.add_argument(
        "--skip-load",
        action="store_true",
        help="Skip PostgreSQL load step",
    )
    args = parser.parse_args()

    cfg = Config()

    logger.info("=" * 60)
    logger.info("ETL PIPELINE STARTED")
    logger.info("=" * 60)

    # -- Extract -----------------------------------------------------------
    if not args.skip_extract:
        df = extract(cfg)
    else:
        import pandas as pd
        clean_path = cfg.OUTPUT_DIR / cfg.CLEAN_FILE
        if clean_path.exists():
            df = pd.read_parquet(clean_path)
            logger.info("Loaded existing clean dataset (%s rows)", df.shape[0])
        else:
            full_path = cfg.OUTPUT_DIR / cfg.FULL_FILE
            if full_path.exists():
                df = pd.read_parquet(full_path)
                logger.info("Loaded existing full dataset (%s rows)", df.shape[0])
            else:
                logger.error("No cached dataset found. Remove --skip-extract to download.")
                sys.exit(1)

    # -- Transform ---------------------------------------------------------
    df = transform(df, cfg)
    analyze(df)

    # -- Load --------------------------------------------------------------
    if not args.skip_load:
        load(df, cfg)

    logger.info("=" * 60)
    logger.info("ETL PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
