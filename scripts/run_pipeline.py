#!/usr/bin/env python3
"""
CRE Distress Warning System — Pipeline Orchestrator

Runs the full end-to-end pipeline or individual stages:
    Bronze → Silver → Gold → Model → Stress → Export

Usage:
    python scripts/run_pipeline.py                  # Full pipeline
    python scripts/run_pipeline.py --stage bronze   # Single stage
    python scripts/run_pipeline.py --stage silver --stage gold  # Multiple stages
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from enum import Enum

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("cre_pipeline")


class PipelineStage(str, Enum):
    """Pipeline execution stages in dependency order."""

    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"
    MODEL = "model"
    STRESS = "stress"
    EXPORT = "export"


# Ordered list of all stages for full pipeline execution
ALL_STAGES = [
    PipelineStage.BRONZE,
    PipelineStage.SILVER,
    PipelineStage.GOLD,
    PipelineStage.MODEL,
    PipelineStage.STRESS,
    PipelineStage.EXPORT,
]


def run_bronze() -> None:
    """Bronze layer: ingest synthetic loans and market data into Delta Lake."""
    logger.info("=" * 60)
    logger.info("STAGE: BRONZE — Raw Data Ingestion")
    logger.info("=" * 60)
    # TODO: Implement in Milestone 2
    #   - Generate synthetic CMBS loan data
    #   - Fetch Treasury rates from FRED API
    #   - Fetch/generate cap rate data
    #   - Write to Delta Lake bronze tables
    logger.info("Bronze layer ingestion complete.")


def run_silver() -> None:
    """Silver layer: clean, validate, and enrich bronze data."""
    logger.info("=" * 60)
    logger.info("STAGE: SILVER — Data Cleaning & Feature Engineering")
    logger.info("=" * 60)
    # TODO: Implement in Milestone 3
    #   - Clean and validate loan data (nulls, types, outliers)
    #   - Standardize market rate data
    #   - Join loans with market data
    #   - Compute derived features (current_ltv, rate_gap, new_dscr, etc.)
    #   - Write to Delta Lake silver tables
    logger.info("Silver layer transformations complete.")


def run_gold() -> None:
    """Gold layer: run dbt transformations for consumption models."""
    logger.info("=" * 60)
    logger.info("STAGE: GOLD — dbt Aggregations (Silver → Gold)")
    logger.info("=" * 60)
    # TODO: Implement in Milestone 4
    #   - Execute dbt run (staging + marts models)
    #   - Produce gold_loan_scores, gold_market_indices
    #   - Run dbt tests for data quality validation
    logger.info("Gold layer dbt transformations complete.")


def run_model() -> None:
    """Model training: XGBoost distress classifier + survival analysis."""
    logger.info("=" * 60)
    logger.info("STAGE: MODEL — ML Training & Scoring")
    logger.info("=" * 60)
    # TODO: Implement in Milestones 5 & 6
    #   - Train XGBoost distress classifier with Optuna HPO
    #   - Train Cox PH survival model
    #   - Log experiments to MLflow
    #   - Register best model to MLflow Model Registry
    #   - Score all loans with distress probability + time-to-distress
    #   - Compute SHAP values for explainability
    #   - Write scored results to gold layer
    logger.info("Model training and scoring complete.")


def run_stress() -> None:
    """Stress testing: apply rate/cap-rate shocks and re-score portfolio."""
    logger.info("=" * 60)
    logger.info("STAGE: STRESS — Scenario Stress Testing")
    logger.info("=" * 60)
    # TODO: Implement in Milestone 7
    #   - Load scenario definitions from config/stress_scenarios.yaml
    #   - For each scenario: apply shocks, recompute features, re-score
    #   - Validate monotonicity (distress increases with shock severity)
    #   - Write gold_stress_results partitioned by scenario_name
    logger.info("Stress testing complete.")


def run_export() -> None:
    """Export: produce Parquet/CSV files for Power BI dashboard."""
    logger.info("=" * 60)
    logger.info("STAGE: EXPORT — Power BI Data Export")
    logger.info("=" * 60)
    # TODO: Implement in Milestone 8
    #   - Export gold tables as partitioned Parquet files
    #   - Export flat CSVs for Excel/Power BI fallback
    #   - Build star schema tables (fact + dimensions)
    #   - Write to data/exports/powerbi/
    logger.info("Power BI export complete.")


# Stage function dispatch
STAGE_RUNNERS = {
    PipelineStage.BRONZE: run_bronze,
    PipelineStage.SILVER: run_silver,
    PipelineStage.GOLD: run_gold,
    PipelineStage.MODEL: run_model,
    PipelineStage.STRESS: run_stress,
    PipelineStage.EXPORT: run_export,
}


def run_pipeline(stages: list[PipelineStage]) -> None:
    """Execute the specified pipeline stages in order."""
    logger.info("CRE Distress Warning System — Pipeline Starting")
    logger.info(f"Stages to execute: {[s.value for s in stages]}")
    logger.info("")

    total_start = time.time()

    for stage in stages:
        stage_start = time.time()
        try:
            STAGE_RUNNERS[stage]()
        except Exception as e:
            logger.error(f"FAILED at stage '{stage.value}': {e}")
            raise
        elapsed = time.time() - stage_start
        logger.info(f"  [{stage.value}] completed in {elapsed:.1f}s")
        logger.info("")

    total_elapsed = time.time() - total_start
    logger.info("=" * 60)
    logger.info(f"PIPELINE COMPLETE — Total time: {total_elapsed:.1f}s")
    logger.info("=" * 60)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="CRE Distress Warning System — Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_pipeline.py                    # Run full pipeline
  python scripts/run_pipeline.py --stage bronze     # Bronze only
  python scripts/run_pipeline.py --stage silver --stage gold  # Multiple stages
        """,
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=[s.value for s in PipelineStage],
        action="append",
        default=None,
        help="Pipeline stage(s) to run. Omit for full pipeline.",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point for the pipeline orchestrator."""
    args = parse_args()

    if args.stage is None:
        # Run all stages in order
        stages = ALL_STAGES
    else:
        # Run only specified stages (in canonical order)
        requested = {PipelineStage(s) for s in args.stage}
        stages = [s for s in ALL_STAGES if s in requested]

    if not stages:
        logger.error("No valid stages specified.")
        sys.exit(1)

    try:
        run_pipeline(stages)
    except Exception:
        sys.exit(1)


if __name__ == "__main__":
    main()
