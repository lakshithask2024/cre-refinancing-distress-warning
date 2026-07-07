"""
XGBoost Distress Classifier with Optuna HPO and MLflow Tracking
=================================================================

Trains an XGBoost binary classifier to predict CRE refinancing distress,
using Optuna for hyperparameter optimization and MLflow for experiment
tracking, artifact logging, and model registry.

Why XGBoost:
  - Tabular data with mixed feature types → tree-based models dominate
  - Built-in handling of class imbalance via scale_pos_weight
  - Fast training (minutes, not hours) for rapid iteration
  - SHAP TreeExplainer produces exact Shapley values (Milestone 6)

Why Optuna:
  - Bayesian optimization (TPE sampler) is more sample-efficient than grid/random
  - Pruning support (MedianPruner) kills bad trials early
  - Native MLflow integration for logging each trial

Architecture:
  train_and_log() → Optuna study (N trials) → best model → evaluate on test
                  → log everything to MLflow → register to Model Registry
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # headless backend — no display required
import matplotlib.pyplot as plt
import mlflow
import mlflow.xgboost
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    auc,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.models.features import build_training_frame

logger = logging.getLogger(__name__)

# Default paths
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
METRICS_OUTPUT_DIR = Path("models/evaluation")
MODEL_REGISTRY_NAME = "cre_distress_classifier"


def train_and_log(
    experiment_name: str = "cre_distress",
    n_trials: int = 20,
    seed: int = 42,
    gold_path: str | Path = "data/gold/loan_current_state",
    market_path: str | Path = "data/silver/market_rates",
) -> str:
    """
    Train XGBoost distress classifier with Optuna HPO and log to MLflow.

    Why this is one function:
      The training lifecycle (data → HPO → eval → log → register) is atomic.
      Either the full run succeeds and is logged, or it fails loudly.
      Partial runs are never registered to the Model Registry.

    Args:
        experiment_name: MLflow experiment name
        n_trials: Number of Optuna optimization trials
        seed: Random seed for reproducibility across numpy, optuna, xgboost
        gold_path: Path to Gold loan_current_state Delta table
        market_path: Path to Silver market_rates Delta table

    Returns:
        MLflow parent run_id (str)
    """
    # Set seeds for reproducibility
    np.random.seed(seed)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Configure MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)

    logger.info("=" * 60)
    logger.info("DISTRESS CLASSIFIER — Training Pipeline")
    logger.info("=" * 60)
    start_time = time.time()

    # ─── Step 1: Load and prepare data ────────────────────────────────────────
    logger.info("\n[1/5] Building training frame...")
    X_train, y_train, X_valid, y_valid, X_test, y_test, feature_names = (
        build_training_frame(gold_path=gold_path, market_path=market_path, seed=seed)
    )

    # Class imbalance handling
    n_positive = int(y_train.sum())
    n_negative = int(len(y_train) - n_positive)
    scale_pos_weight = n_negative / max(n_positive, 1)
    logger.info(
        f"  Class balance — positive: {n_positive}, negative: {n_negative}, "
        f"scale_pos_weight: {scale_pos_weight:.2f}"
    )

    # ─── Step 2: Optuna hyperparameter optimization ───────────────────────────
    logger.info(f"\n[2/5] Running Optuna study ({n_trials} trials)...")

    with mlflow.start_run(run_name=f"distress_classifier_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}") as parent_run:
        parent_run_id = parent_run.info.run_id

        # Log dataset metadata
        mlflow.log_params({
            "train_size": len(X_train),
            "valid_size": len(X_valid),
            "test_size": len(X_test),
            "n_features": len(feature_names),
            "scale_pos_weight": round(scale_pos_weight, 4),
            "n_optuna_trials": n_trials,
            "seed": seed,
        })

        # Define Optuna objective
        def objective(trial: optuna.Trial) -> float:
            params = {
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
            }

            model = xgb.XGBClassifier(
                **params,
                scale_pos_weight=scale_pos_weight,
                objective="binary:logistic",
                eval_metric="auc",
                use_label_encoder=False,
                random_state=seed,
                verbosity=0,
            )

            model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                verbose=False,
            )

            y_pred_proba = model.predict_proba(X_valid)[:, 1]
            val_auc = roc_auc_score(y_valid, y_pred_proba)

            # Log trial as nested MLflow run
            with mlflow.start_run(
                run_name=f"trial_{trial.number:03d}",
                nested=True,
            ):
                mlflow.log_params(params)
                mlflow.log_metric("valid_auc", val_auc)

            return val_auc

        # Run optimization
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        best_params = study.best_params
        best_val_auc = study.best_value
        logger.info(f"  Best validation AUC: {best_val_auc:.4f}")
        logger.info(f"  Best params: {best_params}")

        # ─── Step 3: Train final model with best params ───────────────────────
        logger.info("\n[3/5] Training final model with best hyperparameters...")

        best_model = xgb.XGBClassifier(
            **best_params,
            scale_pos_weight=scale_pos_weight,
            objective="binary:logistic",
            eval_metric="auc",
            use_label_encoder=False,
            random_state=seed,
            verbosity=0,
        )
        best_model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False,
        )

        # ─── Step 4: Evaluate on test set ─────────────────────────────────────
        logger.info("\n[4/5] Evaluating on test set...")

        y_test_proba = best_model.predict_proba(X_test)[:, 1]
        y_test_pred = (y_test_proba >= 0.5).astype(int)

        test_metrics = _compute_test_metrics(y_test, y_test_proba, y_test_pred)
        logger.info(f"  Test AUC:      {test_metrics['test_auc']:.4f}")
        logger.info(f"  Test PR-AUC:   {test_metrics['test_pr_auc']:.4f}")
        logger.info(f"  Test Brier:    {test_metrics['test_brier_score']:.4f}")
        logger.info(f"  Test Log Loss: {test_metrics['test_log_loss']:.4f}")

        # Log best params and test metrics
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        mlflow.log_metrics(test_metrics)
        mlflow.log_metric("best_valid_auc", best_val_auc)

        # ─── Step 5: Log artifacts ────────────────────────────────────────────
        logger.info("\n[5/5] Logging artifacts to MLflow...")

        # Feature importance plot
        _log_feature_importance_plot(best_model, feature_names)

        # Calibration curve
        _log_calibration_plot(y_test, y_test_proba)

        # Confusion matrix
        _log_confusion_matrix_plot(y_test, y_test_pred)

        # Log the model
        mlflow.xgboost.log_model(
            best_model,
            artifact_path="model",
            registered_model_name=MODEL_REGISTRY_NAME,
        )

        # Write metrics JSON
        _write_metrics_json(test_metrics, best_params, parent_run_id)

    elapsed = time.time() - start_time
    logger.info(f"\nTraining complete in {elapsed:.1f}s")
    logger.info(f"MLflow run_id: {parent_run_id}")
    logger.info(f"Model registered as '{MODEL_REGISTRY_NAME}' (Staging)")

    return parent_run_id


# ─── Evaluation Helpers ───────────────────────────────────────────────────────


def _compute_test_metrics(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """
    Compute the full suite of test metrics.

    Why these specific metrics:
      - AUC-ROC: threshold-independent discrimination power
      - PR-AUC: better for imbalanced classes than ROC-AUC
      - Brier score: calibration quality (lower = better calibrated probabilities)
      - Log loss: proper scoring rule for probabilistic predictions
    """
    return {
        "test_auc": float(roc_auc_score(y_true, y_proba)),
        "test_pr_auc": float(average_precision_score(y_true, y_proba)),
        "test_brier_score": float(brier_score_loss(y_true, y_proba)),
        "test_log_loss": float(log_loss(y_true, y_proba)),
    }


# ─── Artifact Logging ─────────────────────────────────────────────────────────


def _log_feature_importance_plot(
    model: xgb.XGBClassifier,
    feature_names: list[str],
    top_n: int = 20,
) -> None:
    """
    Log a bar chart of the top-N most important features.

    Why: Feature importance gives domain experts an immediate sanity check.
    If 'rate_gap_bps' and 'current_ltv' aren't in the top 5, something is wrong.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, 8))
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]

    ax.barh(range(len(top_features)), top_importances[::-1], align="center")
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features[::-1])
    ax.set_xlabel("Feature Importance (Gain)")
    ax.set_title(f"Top {top_n} Feature Importances — Distress Classifier")
    plt.tight_layout()

    path = "/tmp/feature_importance.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(path, "plots")


def _log_calibration_plot(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
) -> None:
    """
    Log a reliability diagram (calibration curve).

    Why: AUC measures ranking, not calibration. A well-calibrated model means
    "60% predicted probability" actually corresponds to ~60% observed distress rate.
    This matters for portfolio-level loss estimation.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy="uniform")

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(prob_pred, prob_true, marker="o", label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly Calibrated")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve — Distress Classifier")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()

    path = "/tmp/calibration_curve.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(path, "plots")


def _log_confusion_matrix_plot(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
) -> None:
    """
    Log confusion matrix heatmap at 0.5 threshold.

    Why: Provides the raw TP/FP/TN/FN counts that risk managers need to
    understand false-positive (unnecessary early action) vs false-negative
    (missed distress) tradeoffs.
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    labels = ["Not Distressed", "Distressed"]
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted",
        ylabel="Actual",
        title="Confusion Matrix (threshold=0.5)",
    )

    # Annotate cells
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")

    plt.tight_layout()
    path = "/tmp/confusion_matrix.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    mlflow.log_artifact(path, "plots")


def _write_metrics_json(
    test_metrics: dict[str, float],
    best_params: dict[str, Any],
    run_id: str,
) -> None:
    """
    Write evaluation summary to JSON for downstream consumption.

    Why: MLflow UI is great for interactive exploration, but CI/CD pipelines
    and the SR 11-7 documentation need a machine-readable metrics file.
    """
    METRICS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = METRICS_OUTPUT_DIR / "distress_classifier_metrics.json"

    summary = {
        "model_name": MODEL_REGISTRY_NAME,
        "mlflow_run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "test_metrics": test_metrics,
        "best_hyperparameters": best_params,
    }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    logger.info(f"  Metrics written to {output_path}")
    mlflow.log_artifact(str(output_path), "evaluation")
