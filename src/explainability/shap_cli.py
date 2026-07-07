"""
SHAP Explainability CLI
========================

Usage:
    python -m src.explainability.shap_cli
    python -m src.explainability.shap_cli --n-sample 500
"""

from __future__ import annotations

import argparse
import logging
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute SHAP explanations for distress classifier"
    )
    parser.add_argument("--n-sample", type=int, default=1000)
    parser.add_argument("--n-top-risk", type=int, default=100)
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
        import shap  # noqa: F401
        import xgboost  # noqa: F401
        import mlflow  # noqa: F401
    except ImportError as e:
        print(
            f"ERROR: {e.name} not installed. "
            f"Run: pip install shap xgboost mlflow",
            file=sys.stderr,
        )
        sys.exit(1)

    import mlflow
    from src.models.features import build_training_frame
    from src.explainability.shap_explainer import compute_shap_explanations

    # Load the trained model from MLflow
    mlflow.set_tracking_uri("sqlite:///mlruns.db")

    # Build test data
    _, _, _, _, X_test, y_test, feature_names = build_training_frame(
        gold_path=args.gold_path, market_path=args.market_path
    )

    # Load latest registered model
    try:
        model = mlflow.xgboost.load_model("models:/cre_distress_classifier/Staging")
    except Exception:
        logging.warning("Could not load from registry, trying latest run...")
        runs = mlflow.search_runs(experiment_names=["cre_distress"])
        if len(runs) == 0:
            print("ERROR: No trained model found. Run train_cli first.", file=sys.stderr)
            sys.exit(1)
        latest_run = runs.iloc[0]
        model = mlflow.xgboost.load_model(f"runs:/{latest_run.run_id}/model")

    # Compute SHAP
    summary = compute_shap_explanations(
        model=model,
        X_test=X_test,
        feature_names=feature_names,
        n_sample=args.n_sample,
        n_top_risk=args.n_top_risk,
    )

    print(f"\nSHAP computation complete.")
    print(f"  Samples analyzed: {summary['n_sample']}")
    print(f"  Top risk loans:   {summary['n_top_risk']}")
    print(f"  Top feature:      {summary['top_feature']}")
    print(f"  Plots saved to:   docs/model_risk_management/figures/")


if __name__ == "__main__":
    main()
