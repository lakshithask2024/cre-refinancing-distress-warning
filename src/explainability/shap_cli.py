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

    import os
    import mlflow
    import mlflow.xgboost
    from src.models.features import build_training_frame
    from src.explainability.shap_explainer import compute_shap_explanations

    # Use the same tracking URI as the classifier (env-driven, consistent)
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    mlflow.set_tracking_uri(tracking_uri)
    logger = logging.getLogger(__name__)
    logger.info(f"MLflow tracking URI: {tracking_uri}")

    # Build test data
    _, _, _, _, X_test, y_test, feature_names = build_training_frame(
        gold_path=args.gold_path, market_path=args.market_path
    )

    # Load the trained model — three strategies in priority order
    model = None
    client = mlflow.tracking.MlflowClient()

    # Strategy 1: Load via alias (MLflow 3.x @ syntax)
    try:
        model = mlflow.xgboost.load_model("models:/cre_distress_classifier@Staging")
        logger.info("✓ Loaded model via alias: models:/cre_distress_classifier@Staging")
    except Exception as e:
        logger.info(f"Alias load failed ({e}), trying version-based load...")

    # Strategy 2: Load the highest-numbered registered version explicitly
    if model is None:
        try:
            versions = client.search_model_versions("name='cre_distress_classifier'")
            if versions:
                latest = max(versions, key=lambda v: int(v.version))
                model_uri = f"models:/cre_distress_classifier/{latest.version}"
                model = mlflow.xgboost.load_model(model_uri)
                logger.info(f"✓ Loaded model via version: {model_uri} (v{latest.version})")
        except Exception as e:
            logger.info(f"Version-based load failed ({e}), trying run artifact search...")

    # Strategy 3: Search recent classifier runs for 'model' artifact
    if model is None:
        try:
            runs = mlflow.search_runs(
                experiment_names=["cre_distress"],
                filter_string="tags.mlflow.runName LIKE 'distress_classifier%'",
                order_by=["start_time DESC"],
                max_results=10,
            )
            # Fallback: if tag filter returns nothing, try without filter
            if len(runs) == 0:
                runs = mlflow.search_runs(
                    experiment_names=["cre_distress"],
                    order_by=["start_time DESC"],
                    max_results=10,
                )
        except Exception:
            runs = pd.DataFrame()

        if len(runs) == 0:
            print("ERROR: No trained model found. Run train_cli first.", file=sys.stderr)
            sys.exit(1)

        import pandas as pd_  # ensure available for iterrows

        for _, run_row in runs.iterrows():
            run_id = run_row["run_id"]
            try:
                artifacts = client.list_artifacts(run_id)
                artifact_paths = [a.path for a in artifacts]

                if "model" in artifact_paths:
                    model = mlflow.xgboost.load_model(f"runs:/{run_id}/model")
                    logger.info(f"✓ Loaded model from run {run_id[:8]}... artifact 'model'")
                    break
            except Exception as e:
                logger.debug(f"  Run {run_id[:8]}... failed: {e}")
                continue

    if model is None:
        print("ERROR: Could not load model from any strategy. Check MLflow artifacts.", file=sys.stderr)
        sys.exit(1)

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
