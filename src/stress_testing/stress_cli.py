"""
Stress Testing CLI
===================

Usage:
    python -m src.stress_testing.stress_cli run-all
    python -m src.stress_testing.stress_cli run-all --scenarios baseline,combined_severe
"""

from __future__ import annotations

import argparse
import logging
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser(description="CRE Distress Stress Testing Engine")
    parser.add_argument("command", choices=["run-all"], help="Command to execute")
    parser.add_argument(
        "--scenarios", type=str, default=None,
        help="Comma-separated scenario names to run (default: all 8)",
    )
    parser.add_argument("--gold-path", default="data/gold/loan_current_state")
    parser.add_argument("--config", default="config/stress_scenarios.yaml")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Fail fast on missing deps
    try:
        import xgboost  # noqa: F401
        import mlflow  # noqa: F401
        import pandas  # noqa: F401
        import numpy  # noqa: F401
    except ImportError as e:
        print(f"ERROR: {e.name} not installed. pip install xgboost mlflow pandas numpy", file=sys.stderr)
        sys.exit(1)

    from src.stress_testing.scenario_engine import load_scenarios, run_all_scenarios, load_classifier
    from src.stress_testing.aggregate_report import compute_aggregate_report, save_report

    logger = logging.getLogger(__name__)
    start = time.time()

    logger.info("=" * 60)
    logger.info("STRESS TESTING ENGINE — Starting")
    logger.info("=" * 60)

    # Verify model is loadable before proceeding
    logger.info("Verifying classifier is loadable...")
    try:
        model = load_classifier()
        logger.info("  ✓ Classifier loaded successfully")
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # Load scenarios
    scenarios = load_scenarios(args.config)
    logger.info(f"Loaded {len(scenarios)} scenarios from {args.config}")

    # Filter scenarios if specified
    if args.scenarios:
        requested = set(args.scenarios.split(","))
        scenarios = [s for s in scenarios if s.get("name") in requested]
        logger.info(f"  Filtered to {len(scenarios)} scenarios: {[s['name'] for s in scenarios]}")

    if not scenarios:
        print("ERROR: No scenarios to run.", file=sys.stderr)
        sys.exit(1)

    # Run all scenarios
    results = run_all_scenarios(
        scenarios=scenarios,
        gold_path=args.gold_path,
    )

    # Compute and save aggregate report
    logger.info("\nComputing aggregate report...")
    summary_df = compute_aggregate_report(results)
    save_report(summary_df)

    # Print summary table
    elapsed = time.time() - start
    logger.info(f"\n{'='*60}")
    logger.info(f"STRESS TESTING — Complete ({elapsed:.1f}s)")
    logger.info(f"{'='*60}")
    logger.info(f"\n{'Scenario':<25} {'% Distressed':>13} {'Avg PD':>8} {'Delta pp':>9}")
    logger.info("-" * 58)
    for _, row in summary_df.iterrows():
        logger.info(
            f"{row['scenario_name']:<25} {row['pct_loans_distressed']:>12.1f}% "
            f"{row['avg_predicted_pd']:>7.3f} {row.get('avg_delta_pd', 0):>+8.3f}"
        )


if __name__ == "__main__":
    main()
