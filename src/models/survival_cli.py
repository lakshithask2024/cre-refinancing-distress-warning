"""
Survival Model Training CLI
=============================

Usage:
    python -m src.models.survival_cli
    python -m src.models.survival_cli --experiment-name cre_distress --penalizer 0.1
"""

from __future__ import annotations

import argparse
import logging
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Cox PH survival model")
    parser.add_argument("--experiment-name", default="cre_distress")
    parser.add_argument("--penalizer", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gold-path", default="data/gold/loan_current_state")
    parser.add_argument("--market-path", default="data/silver/market_rates")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        from lifelines import CoxPHFitter  # noqa: F401
    except ImportError:
        print("ERROR: lifelines not installed. Run: pip install lifelines", file=sys.stderr)
        sys.exit(1)

    from src.models.survival_model import train_survival_model

    run_id = train_survival_model(
        experiment_name=args.experiment_name,
        seed=args.seed,
        gold_path=args.gold_path,
        market_path=args.market_path,
        penalizer=args.penalizer,
    )
    print(f"\nSurvival model trained. MLflow run_id: {run_id}")


if __name__ == "__main__":
    main()
