"""
Training CLI for the CRE Distress Classifier
==============================================

Entry point for running model training from the command line.

Usage:
    python -m src.models.train_cli --experiment-name cre_distress
    python -m src.models.train_cli --n-trials 50 --seed 123

Why a separate CLI:
  Keeps the training logic (distress_classifier.py) importable as a library
  while providing a clean command-line interface for manual runs and CI/CD.
"""

from __future__ import annotations

import argparse
import logging
import sys


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training."""
    parser = argparse.ArgumentParser(
        description="Train CRE Distress Classifier with Optuna HPO + MLflow tracking.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.models.train_cli
  python -m src.models.train_cli --experiment-name cre_distress --n-trials 20
  python -m src.models.train_cli --n-trials 50 --seed 123
        """,
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="cre_distress",
        help="MLflow experiment name (default: cre_distress)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Number of Optuna optimization trials (default: 20)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--gold-path",
        type=str,
        default="data/gold/loan_current_state",
        help="Path to Gold loan_current_state table",
    )
    parser.add_argument(
        "--market-path",
        type=str,
        default="data/silver/market_rates",
        help="Path to Silver market_rates table",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )
    return parser.parse_args()


def main() -> None:
    """Run distress classifier training pipeline."""
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Fail fast if required libraries are missing
    try:
        import xgboost  # noqa: F401
        import optuna  # noqa: F401
        import mlflow  # noqa: F401
        import shap  # noqa: F401
    except ImportError as e:
        print(
            f"ERROR: Missing required library: {e.name}\n"
            f"Install with: pip install xgboost optuna mlflow shap scikit-learn pandas numpy",
            file=sys.stderr,
        )
        sys.exit(1)

    from src.models.distress_classifier import train_and_log

    run_id = train_and_log(
        experiment_name=args.experiment_name,
        n_trials=args.n_trials,
        seed=args.seed,
        gold_path=args.gold_path,
        market_path=args.market_path,
    )

    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  MLflow run_id: {run_id}")
    print(f"  Inspect results: mlflow ui  (opens at http://localhost:5000)")
    print(f"  Metrics JSON:   models/evaluation/distress_classifier_metrics.json")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
