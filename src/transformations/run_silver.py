"""
Silver Layer Pipeline Orchestrator
====================================

Reads from Bronze Delta tables, applies transformations, validates data quality,
and writes Silver Delta tables.

Pipeline:
  bronze_loans → validate → clean → silver_loans
  bronze_market → validate → standardize → silver_market_rates
  silver_loans + silver_market → feature_engineering → silver_loan_features

CLI Usage:
    python -m src.transformations.run_silver
    python -m src.transformations.run_silver --bronze-loans data/bronze/loans --output data/silver
"""

from __future__ import annotations

import argparse
import logging
import time
from datetime import date
from pathlib import Path
from typing import Any

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.delta_writer import DeltaReader, DeltaWriter
from src.transformations.silver_loans import transform_silver_loans
from src.transformations.silver_market import transform_silver_market
from src.transformations.feature_engineering import compute_loan_features
from src.transformations.data_quality import (
    validate_bronze_loans,
    validate_bronze_market,
    validate_silver_features,
)

logger = logging.getLogger(__name__)

DEFAULT_BRONZE_LOANS = Path("data/bronze/loans")
DEFAULT_BRONZE_MARKET = Path("data/bronze/market")
DEFAULT_SILVER_DIR = Path("data/silver")


def run_silver_pipeline(
    bronze_loans_path: Path = DEFAULT_BRONZE_LOANS,
    bronze_market_path: Path = DEFAULT_BRONZE_MARKET,
    silver_dir: Path = DEFAULT_SILVER_DIR,
    reference_date: date | None = None,
    halt_on_quality_failure: bool = True,
) -> dict[str, Any]:
    """
    Execute the full Silver layer pipeline.

    Returns:
        Summary dict with record counts and timing
    """
    start_time = time.time()
    summary: dict[str, Any] = {}

    logger.info("=" * 60)
    logger.info("SILVER LAYER PIPELINE — Starting")
    logger.info("=" * 60)

    # ─── Step 1: Read Bronze Data ─────────────────────────────────────────────
    logger.info("\n[1/6] Reading Bronze tables...")

    bronze_loan_reader = DeltaReader(bronze_loans_path)
    bronze_loans = bronze_loan_reader.read()
    logger.info(f"  Bronze loans: {len(bronze_loans)} records")

    bronze_market_reader = DeltaReader(bronze_market_path)
    bronze_market = bronze_market_reader.read()
    logger.info(f"  Bronze market: {len(bronze_market)} records")

    summary["bronze_loans_count"] = len(bronze_loans)
    summary["bronze_market_count"] = len(bronze_market)

    # ─── Step 2: Validate Bronze Data ─────────────────────────────────────────
    logger.info("\n[2/6] Validating Bronze data quality...")

    loan_report = validate_bronze_loans(
        bronze_loans, halt_on_failure=halt_on_quality_failure
    )
    market_report = validate_bronze_market(
        bronze_market, halt_on_failure=halt_on_quality_failure
    )

    summary["bronze_loan_quality"] = "PASS" if loan_report.passed else "FAIL"
    summary["bronze_market_quality"] = "PASS" if market_report.passed else "FAIL"

    # ─── Step 3: Transform Silver Loans ───────────────────────────────────────
    logger.info("\n[3/6] Transforming silver loans...")

    silver_loans = transform_silver_loans(bronze_loans, reference_date=reference_date)
    summary["silver_loans_count"] = len(silver_loans)

    # Write silver_loans
    silver_loans_path = silver_dir / "loans"
    writer = DeltaWriter(silver_loans_path)
    writer.write(silver_loans, partition_by="origination_year", mode="overwrite")
    logger.info(f"  Written silver_loans to {silver_loans_path}")

    # ─── Step 4: Transform Silver Market ──────────────────────────────────────
    logger.info("\n[4/6] Transforming silver market rates...")

    silver_market = transform_silver_market(bronze_market)
    summary["silver_market_count"] = len(silver_market)

    # Write silver_market_rates
    silver_market_path = silver_dir / "market_rates"
    writer = DeltaWriter(silver_market_path)
    writer.write(silver_market, partition_by="data_type", mode="overwrite")
    logger.info(f"  Written silver_market_rates to {silver_market_path}")

    # ─── Step 5: Feature Engineering ──────────────────────────────────────────
    logger.info("\n[5/6] Computing loan features...")

    silver_features = compute_loan_features(
        silver_loans, silver_market, reference_date=reference_date
    )
    summary["silver_features_count"] = len(silver_features)

    # Write silver_loan_features
    silver_features_path = silver_dir / "loan_features"
    writer = DeltaWriter(silver_features_path)
    writer.write(silver_features, partition_by="origination_year", mode="overwrite")
    logger.info(f"  Written silver_loan_features to {silver_features_path}")

    # ─── Step 6: Validate Silver Output ───────────────────────────────────────
    logger.info("\n[6/6] Validating silver output quality...")

    features_report = validate_silver_features(
        silver_features, halt_on_failure=halt_on_quality_failure
    )
    summary["silver_features_quality"] = "PASS" if features_report.passed else "FAIL"

    # ─── Summary ──────────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    summary["elapsed_seconds"] = round(elapsed, 2)

    logger.info("\n" + "=" * 60)
    logger.info("SILVER LAYER PIPELINE — Complete")
    logger.info("=" * 60)
    logger.info(f"  Bronze loans:         {summary['bronze_loans_count']}")
    logger.info(f"  Silver loans:         {summary['silver_loans_count']}")
    logger.info(f"  Silver market rates:  {summary['silver_market_count']}")
    logger.info(f"  Silver loan features: {summary['silver_features_count']}")
    logger.info(f"  Total time:           {elapsed:.1f}s")

    return summary


# ─── CLI ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Silver layer transformation pipeline."
    )
    parser.add_argument(
        "--bronze-loans", type=Path, default=DEFAULT_BRONZE_LOANS,
        help="Path to bronze loans Delta table",
    )
    parser.add_argument(
        "--bronze-market", type=Path, default=DEFAULT_BRONZE_MARKET,
        help="Path to bronze market Delta table",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_SILVER_DIR,
        help="Output directory for silver tables",
    )
    parser.add_argument(
        "--no-halt", action="store_true",
        help="Continue pipeline on quality check failures (don't halt)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    run_silver_pipeline(
        bronze_loans_path=args.bronze_loans,
        bronze_market_path=args.bronze_market,
        silver_dir=args.output,
        halt_on_quality_failure=not args.no_halt,
    )


if __name__ == "__main__":
    main()
